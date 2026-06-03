import shlex
from pathlib import Path

from .utils import ensure_dir, run_command


class SingularityRunner:
    def __init__(self, image, binds=None):
        self.image = image
        self.binds = binds or []

    def build_command(self, input_dir, output_dir, work_dir=None, extra_args=""):
        command = ["singularity", "run"]
        for path in [input_dir, output_dir, work_dir]:
            if path is not None:
                command.extend(["--bind", str(Path(path).resolve())])

        command.append(self.image)
        command.append(str(Path(input_dir).resolve()))
        command.append(str(Path(output_dir).resolve()))

        if extra_args:
            command.extend(shlex.split(extra_args))

        return command

    def run(self, input_dir, output_dir, work_dir=None, extra_args="", dry_run=False):
        ensure_dir(output_dir)
        if work_dir is not None:
            ensure_dir(work_dir)
        command = self.build_command(input_dir, output_dir, work_dir, extra_args)
        return run_command(command, dry_run=dry_run)
