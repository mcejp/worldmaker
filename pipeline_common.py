import abc
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Set, Union

logger = logging.getLogger(__name__)


@dataclass
class PipelineContext:
    spec_dir: Path
    work_dir: Path


@dataclass
class Operator(abc.ABC):
    context: PipelineContext
    spec: dict

    @abc.abstractmethod
    def run(self, input_paths, output_paths):
        pass


@dataclass
class Pipeline:
    all_steps: List[type]       # might as well be Set, but we want consistent iteration order

    def expand_triggered_operators(self, operator: Operator) -> List[Operator]:
        queue = [operator]
        result = []

        while len(queue):
            operator = queue.pop()
            result.append(operator)

            if hasattr(operator, "TRIGGERS"):
                for name in operator.TRIGGERS:
                    queue.append(self.get_operator_by_name(name))

        return result

    def get_operator_by_name(self, name: str) -> Operator:
        matching = [op for op in self.all_steps if op.NAME == name]
        if len(matching) != 1:
            raise Exception(f"No such operator: '{name}'")
        op, = matching
        return op

    def get_operator_by_name_with_version(self, name_and_version: str) -> Operator:
        matching = [op for op in self.all_steps if f"{op.NAME}:{op.VERSION}" == name_and_version]
        if len(matching) != 1:
            raise Exception(f"Operator look-up error: {name_and_version}")
        op, = matching
        return op


def get_pipeline_by_name(name: str) -> Pipeline:
    import importlib

    module = importlib.import_module("pipeline_" + name)
    return module.__dict__["pipeline_" + name]


# This function resolves variables like "$input.png" -> options["input"] + ".png
def resolve_variables(filename_or_path, options, location_hint) -> Union[str, Path]:
    if isinstance(filename_or_path, Path):
        return filename_or_path
    else:
        processed_filename = ""
        variable_name = None
        i = 0
        while i < len(filename_or_path):
            ch = filename_or_path[i]
            i += 1

            if ch == "$":
                variable_name = ""

                while i < len(filename_or_path) and (filename_or_path[i].isalnum() or filename_or_path[i] == "_"):
                    variable_name += filename_or_path[i]
                    i += 1

                try:
                    processed_filename += options[variable_name]
                except KeyError:
                    raise Exception(f"No such variable: '{variable_name}' in '{location_hint}'")
            else:
                processed_filename += ch

        return processed_filename


def sanitize_file_name(filename: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"@", "+", "."} else "_" for ch in filename)
