#!/usr/bin/env python3

import argparse
import functools
import html
import io
import json
import logging
from multiprocessing import cpu_count
import os
from pathlib import Path
import subprocess
import sys

import yaml

from pipeline_common import get_pipeline_by_name, Pipeline, PipelineContext, resolve_variables

logger = logging.getLogger(__name__)


if sys.version_info >= (3, 9):
    def with_stem(path, stem):
        return path.with_stem(stem)
else:
    def with_stem(path, stem):
        return path.with_name(stem + path.suffix)


def update_file_contents(path, content):
    content_bin = content.encode() if isinstance(content, str) else content

    try:
        if path.read_bytes() == content_bin:
            return
        else:
            os.rename(path, path.with_suffix(path.suffix + ".bak"))
        #     print("OLD:", path.read_bytes())
        #     print("NEW:", content_bin)
    except FileNotFoundError:
        pass

    logger.info("Updating %s", path)
    path.write_bytes(content_bin)


def exec_pipeline(pipeline_name: str, p: Pipeline, spec_path: Path, /, cache_dir: Path, parallel_tasks: int):
    with open(spec_path) as f:
        spec = yaml.safe_load(f)

    project_name = spec_path.stem
    work_dir = Path("work") / project_name
    work_dir.mkdir(parents=True, exist_ok=True)
    options_dir = work_dir / "_tasks"
    options_dir.mkdir(parents=True, exist_ok=True)

    # TODO: is this class still useful?
    context = PipelineContext(work_dir=work_dir,
                              spec_dir=spec_path.parent,
                              )

    warnings = []

    # Determine pipeline tasks. This is not very smart right now as custom tasks can only be put at the beginning.
    pipeline_tasks = []
    would_be_triggered = set()

    for spec_key, options in spec.items():
        # TODO: need to also instantiate sub-operators
        if "/" in spec_key:
            operator_name, task_name = spec_key.split("/")
            operator = p.get_operator_by_name(operator_name)
            pipeline_tasks.append((task_name, operator, spec_key))
        elif spec_key != "variants":
            operator_name = spec_key
            operator = p.get_operator_by_name(operator_name)

            for leaf_operator in p.expand_triggered_operators(operator):
                would_be_triggered.add(leaf_operator.NAME)

    for operator in p.default_steps:
        operator_sanitized_name = operator.NAME.replace(":", "_")
        options_key = operator.NAME.split(":")[0]
        pipeline_tasks.append((operator_sanitized_name, operator, options_key))

        if operator.NAME not in would_be_triggered:
            warnings.append(f"Implicitly instantiated operator {operator.NAME}")

    # observation: erosion is wrong in that it doesn't take into account difference in depth (for the same surface level)
    #              -- it should not apply equally, but instead towards flattening to bottom

    # TODO: should not write Makefile manually now that we have goeieDAG
    makefile_path = work_dir / "Makefile"
    with open(makefile_path, "wt") as mk_f:
        # Generate Make goal for each step
        # - update .OPTIONS file (if changed)
        # - collect dependencies
        # - collect list of outputs
        # - emit rule
        # - add to list of "ALL"
        # - repeat per step, per variant

        mk_f.write(f"# This file has been automatically generated from {spec_path.name}\n\n")
        mk_f.write("SHELL=/bin/bash -o pipefail\n") # necessary since we pipe every command's output to `tee`; see https://stackoverflow.com/a/31362286
        mk_f.write(f'WORLDMAKER_FLAGS={str(context.work_dir)} {str(context.spec_dir)} --cache-dir={str(cache_dir)}\n\n')

        all_outputs = []
        all_outputs_by_task_and_variant = {}
        paths_affected_by_variant = set() # set of tuples (path, variant)

        for task_def_name, step, options_key in pipeline_tasks:
            all_outputs_by_task_and_variant[task_def_name] = {}

            try:
                options = spec[options_key]
            except KeyError:
                options = {}

            """
            Variants:
            - for each step:
            - iterate all variants
              - if non-default variant:
                - generate merged options
                - for each input:
                    - find out if this input affected by variation
                - is any input, OR our options affected by the variation?
                    - if yes, add suffix to all outputs (mark them as affectd), and emit an additional task
            """

            # https://stackoverflow.com/a/20666342
            def merge(source, destination):
                """
                run me with nosetests --with-doctest file.py

                >>> a = { 'first' : { 'all_rows' : { 'pass' : 'dog', 'number' : '1' } } }
                >>> b = { 'first' : { 'all_rows' : { 'fail' : 'cat', 'number' : '5' } } }
                >>> merge(b, a) == { 'first' : { 'all_rows' : { 'pass' : 'dog', 'fail' : 'cat', 'number' : '5' } } }
                True
                """
                for key, value in source.items():
                    if isinstance(value, dict):
                        # get node or create one
                        node = destination.setdefault(key, {})
                        merge(value, node)
                    else:
                        destination[key] = value

                return destination

            for variant_obj in [None] + spec.get("variants", []):
                if variant_obj is None:
                    task_name = task_def_name
                    # variant_name will be undefined in this case, but variant_suffix always exists -- this allows some DRY later.
                    variant_suffix = ""
                    variant_options = options
                else:
                    variant_name = variant_obj["name"]
                    variant_name_sanitized = "".join(ch if ch.isalnum() else "_" for ch in variant_name)
                    variant_suffix = "@" + variant_name_sanitized
                    task_name = task_def_name + variant_suffix

                    # Options varied?
                    if options_key in variant_obj:
                        new_options = {}
                        merge(options, new_options)
                        merge(variant_obj[options_key], new_options)
                        variant_options = new_options
                        step_varied = True
                    else:
                        variant_options = options
                        step_varied = False

                # Instantiate step
                inst = step(context, variant_options)

                # Collect inputs & outputs
                nominal_inputs = inst.get_inputs()
                nominal_outputs = inst.get_outputs()
                # Why are variables only resolved in the input/output map values, but not in the keys?
                # Because this way, the operator can trivially use references like output_paths["$output"]
                resolve = functools.partial(resolve_variables, options=variant_options, location_hint=options_key)

                def bind_input(filename_or_path):
                    if isinstance(filename_or_path, Path):
                        if not filename_or_path.is_file():
                            raise FileNotFoundError(str(filename_or_path))

                        return filename_or_path
                    else:
                        return work_dir / filename_or_path

                def add_variant_suffix(path, variant_suffix):
                    return with_stem(path, path.stem + variant_suffix)

                if variant_obj is None:
                    input_map = {str(filename): bind_input(resolve(filename)) for filename in sorted(nominal_inputs)}
                else:
                    input_map = {}

                    # Map inputs & detect if any input is affected by this variant
                    for filename_or_path in sorted(nominal_inputs):
                        bound_path = bind_input(resolve(filename_or_path))

                        if (filename_or_path, variant_name) in paths_affected_by_variant:
                            # Here can assume filename_or_path is string
                            input_map[filename_or_path] = add_variant_suffix(bound_path, variant_suffix)
                            step_varied = True
                        else:
                            input_map[str(filename_or_path)] = bound_path

                    # If neither the options, nor any inputs, are varied, there is nothing to do (default variant already covers this step fully)
                    if not step_varied:
                        continue

                    # Mark all outputs as varied, so variants will be emitted for all subsequent steps
                    for filename in nominal_outputs:
                        paths_affected_by_variant.add((resolve(filename), variant_name))

                output_map = {filename: add_variant_suffix(work_dir / resolve(filename), variant_suffix) for filename in sorted(nominal_outputs)}
                all_outputs += output_map.values()
                all_outputs_by_task_and_variant[task_def_name][task_name] = output_map.values()

                # Update OPTIONS file (if changed)
                # The operator tag is encoded here so that incrementing an operator version forces re-evaluation
                # TODO: is there any value in encoding also inputs & outputs, as opposed to passing them on the command line?
                variant_options["__operator"] = f"{pipeline_name}.{step.NAME}:{step.VERSION}"
                variant_options["__inputs"] = {k: str(v) for k, v in input_map.items()}
                variant_options["__outputs"] = {k: str(v) for k, v in output_map.items()}
                options_path = options_dir / f"{task_name}.json"
                task_log_path = options_dir / f"{task_name}.log"
                update_file_contents(options_path, json.dumps(variant_options, indent=2))

                # Generate target
                # TODO: in Make 4.3+ we can do this cleanly: https://stackoverflow.com/a/59877127
                # Afterwards, Make might even correctly remove outputs from interrupted tasks (see https://stackoverflow.com/questions/588550/why-does-gnu-make-delete-a-file)
                outputs_list = list(output_map.values())
                mk_f.write(f"#### {task_name}\n")
                dependencies = [options_path] + list(input_map.values())
                mk_f.write(f"{str(outputs_list[0])}: {' '.join([str(p) for p in dependencies])}\n")
                mk_f.write(f"\t$(PYTHON) worldmaker_execstep.py {str(options_path)} $(WORLDMAKER_FLAGS) 2>&1 | tee {task_log_path}\n")
                mk_f.write("\n")

                # Generate rules for additional outputs
                for path in outputs_list[1:]:
                    mk_f.write(f"{str(path)}: {str(outputs_list[0])}\n")
                    mk_f.write("\n")

                # TODO (nice to have): DAG visualization -- https://github.com/chadsgilbert/makefile2dot/blob/master/makefile2dot/__init__.py#L36

        # Ensure no duplicates among output paths
        if len(set(all_outputs)) != len(all_outputs):
            s = set(all_outputs)
            l = [o for o in all_outputs]
            for el in s:
                l.remove(el)
            print("DUPLICATES:", l)
        assert len(set(all_outputs)) == len(all_outputs)

        # Generate rules for additional outputs
        all_outputs_str = "\\\n        ".join([str(p) for p in all_outputs])
        mk_f.write(f"_all: {all_outputs_str}\n")

    # Now generate plots.html
    f = io.StringIO()
    f.write("<html>\n")

    if len(warnings):
        f.write('<div style="border-left: 0.5em solid #ffc107; padding-left: 1em">')
        for message in warnings:
            f.write(f'<div><b>Warning:</b>&ensp;{html.escape(message)}</div>')
        f.write('</div>')

    for task_def_name, outputs_by_variant in all_outputs_by_task_and_variant.items():
        f.write(f"<h3>\n")

        # Radios for variant selection
        if len(outputs_by_variant) > 1:
            for i, (task_name, outputs) in enumerate(outputs_by_variant.items()):
                statements = []
                for task_name_tgt in outputs_by_variant.keys():
                    statements.append("document.getElementById('" + task_name_tgt + "').style.display = " + ("'block'" if task_name_tgt == task_name else "'none'"))
                f.write(f'  <label><input name="{task_def_name}" type="radio" onclick="{"; ".join(statements)}"{" checked" if i == 0 else ""} autocomplete="off">&ensp;{task_name}</label>\n')
        else:
            f.write(f'  {task_def_name}\n')

        f.write("</h3>\n")

        for i, (task_name, outputs) in enumerate(outputs_by_variant.items()):
            f.write(f'  <div id="{task_name}" style="border-left: 0.5em solid #ccc; padding-left: 1em' + ('; display: none' if i > 0 else "") + '">\n')
            f.write(f"    <div><a href=\"_tasks/{task_name}.log\"><img src=\"_tasks/{task_name}.status.png\" alt=\"status badge\"></a></div>\n")

            for plot_path in sorted(outputs):
                if plot_path.suffix.lower() not in {".jpg", ".png"}:
                    continue

                # plot_path is a string and is relative to root directory of worldmaker
                f.write(f'    <img src="{str(Path("../..") / plot_path)}" alt="{plot_path.stem}" style="max-width: 1600px; max-height: 500px">\n')

            f.write(f'  </div>\n')

    update_file_contents(work_dir / "aa_plots.html", f.getvalue())

    # Run the make process
    # We must call python with -u, otherwise it will buffer stdout, as it is being piped to `tee` instead of a TTY
    rc = subprocess.call(["make", "-j" + str(parallel_tasks), "-f", makefile_path, "_all"], env={**os.environ, "PYTHON": sys.executable + " -u"})
    if rc != 0:
        sys.exit(rc)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("spec", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=Path("work") / "zz_cache")
    parser.add_argument("--pipeline", type=str, default="world")
    parser.add_argument("-j", dest="parallel_tasks", type=int, default=cpu_count())
    args = parser.parse_args()

    # Import the pipeline
    pipeline = get_pipeline_by_name(args.pipeline)

    exec_pipeline(args.pipeline, pipeline, args.spec,
                  cache_dir=args.cache_dir,
                  parallel_tasks=args.parallel_tasks
                  )
