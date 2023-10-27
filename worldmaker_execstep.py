import argparse
import copy
from datetime import datetime
import hashlib
import io
import json
import logging
from pathlib import Path
from socket import gethostname
import tarfile
import tempfile
from typing import Callable
import shutil
import subprocess
import sys

from pipeline_common import get_pipeline_by_name, PipelineContext, sanitize_file_name

logging.basicConfig(level=logging.INFO)

# ---

def hash_path(path: Path, algorithm: Callable = hashlib.md5, chunk_size: int = 65536):
    # Adapted from https://stackoverflow.com/a/59974585
    h = algorithm()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h


def text_to_image(path, text, bgcolor, fgcolor):
    # usually provided by 'pango' package
    subprocess.call(["pango-view", "--font=mono bold 10", "--margin=4 8", "-qo", path, "-t", text, "--background", bgcolor, "--foreground", fgcolor])


def make_unique_descriptor(options_in):
    # Clone options & substitute input paths with hashes of the files
    options_resolved = copy.deepcopy(options_in)

    for key in options_resolved["__inputs"].keys():
        options_resolved["__inputs"][key] = hash_path(Path(options_resolved["__inputs"][key])).hexdigest()

    # the set of outputs is fully determined by the operator tag + options
    del options_resolved["__outputs"]

    return json.dumps(options_resolved)


def cache_store(new_cache_path, output_map, unique_descriptor):
    my_cache_path = new_cache_path.with_suffix(".tar.xz")

    with tempfile.NamedTemporaryFile(suffix=my_cache_path.suffix, dir=my_cache_path.parent, delete=False) as tmpf:
        # TODO: tarfile idempotent? https://stackoverflow.com/questions/32997526/how-to-create-a-tar-file-that-omits-timestamps-for-its-contents
        with tarfile.open(name=tmpf.name, mode="w:xz", fileobj=tmpf, format=tarfile.PAX_FORMAT) as tar:
            for key in output_map.keys():
                tar.add(output_map[key], arcname=key)

            blob = unique_descriptor.encode()
            tarinfo = tarfile.TarInfo("__unique_descriptor.json")
            tarinfo.size = len(blob)
            tar.addfile(tarinfo, io.BytesIO(blob))

    logging.debug("cache_store: %s --> %s", tmpf.name, my_cache_path)
    Path(tmpf.name).rename(my_cache_path)


def cache_try_restore(new_cache_path, output_map):
    my_cache_path = new_cache_path.with_suffix(".tar.xz")

    try:
        tar = tarfile.open(my_cache_path, mode="r")
    except FileNotFoundError:
        return False
    else:
        # TAR found
        try:
            for member in tar.getmembers():
                logging.debug("cache_try_restore: %s --> %s", member, output_map.get(member.name))

                if member.name == "__unique_descriptor.json":
                    continue

                dest = output_map[member.name]

                try:
                    with tar.extractfile(member) as f_cached, open(dest, "wb") as f_out:
                        shutil.copyfileobj(f_cached, f_out)
                except:
                    # Make sure that partial file doesn't linger around
                    dest.unlink(missing_ok=True)
                    raise

            return True
        finally:
            tar.close()


# ---

parser = argparse.ArgumentParser()
parser.add_argument("options", type=Path)
parser.add_argument("work_dir", type=Path)
parser.add_argument("spec_dir", type=Path)
parser.add_argument("--cache-dir", type=Path)
args = parser.parse_args()

with open(args.options) as f:
    options = json.load(f)

pipeline_name, operator_tag = options["__operator"].split(".")

input_map = {k: Path(v) for k, v in options["__inputs"].items()}
output_map = {k: Path(v) for k, v in options["__outputs"].items()}

status_img_path = args.options.with_suffix(".status.png")

# Generate unique descriptor, total input hash & cache dir path
# Note that the operator tag slug in the cache path is *not* necessary for uniqueness -- it is only there for easier manual inspection
unique_descriptor = make_unique_descriptor(options)
inputs_hash = hashlib.md5(unique_descriptor.encode()).hexdigest()[0:8]
new_cache_path = args.cache_dir / sanitize_file_name(operator_tag) / inputs_hash

if cache_try_restore(new_cache_path, output_map):
    logging.info("%s: Using cached result '%s/%s'", args.options, sanitize_file_name(operator_tag), inputs_hash)
    text_to_image(status_img_path, f"CACHED {gethostname()} {datetime.now()}", bgcolor="#198754", fgcolor="#ffffff")
    sys.exit(0)

# work-around for broken HydroStep
del options["__operator"]
del options["__inputs"]
del options["__outputs"]

pipeline = get_pipeline_by_name(pipeline_name)
operator = pipeline.get_operator_by_name_with_version(operator_tag)

context = PipelineContext(work_dir=args.work_dir,
                          spec_dir=args.spec_dir,
                          )

inst = operator(context, options)

try:
    try:
        inst.run(input_paths=input_map, output_paths=output_map)
    except:
        # Make sure that partial files don't linger around
        for path in output_map.values():
            path.unlink(missing_ok=True)
        raise

    for p in output_map.values():
        if not p.is_file():
            raise FileNotFoundError(f"{args.options}: Expected output {p} has not been created")
except:
    text_to_image(status_img_path, f"FAILED {gethostname()} {datetime.now()}", bgcolor="#dc3545", fgcolor="#ffffff")
    raise

text_to_image(status_img_path, f"OK     {gethostname()} {datetime.now()}", bgcolor="#198754", fgcolor="#ffffff")

if cacheable := (not hasattr(operator, "CACHEABLE") or operator.CACHEABLE): # cannot use .get here, it's a class, not a dict
    # Note: there might be a race condition between multiple tasks solving the same problem in parallel
    # However, this is benign (aside from the wasted effort), since cache_store ultimately puts the file in place via an atomic rename operation
    new_cache_path.parent.mkdir(exist_ok=True, parents=True)
    cache_store(new_cache_path, output_map, unique_descriptor)
