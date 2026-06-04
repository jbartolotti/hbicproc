import shlex
from pathlib import Path

from .utils import ensure_dir, run_command


class SingularityRunner:
    def __init__(self, image, binds=None, clean_env=False):
        self.image = image
        self.binds = binds or []
        self.clean_env = clean_env

    def build_command(
        self,
        input_dir=None,
        output_dir=None,
        work_dir=None,
        extra_args="",
        binds=None,
        clean_env=None,
    ):
        command = ["singularity", "run"]
        if clean_env is None:
            clean_env = self.clean_env
        if clean_env:
            command.append("--cleanenv")

        bind_list = list(self.binds)
        if binds:
            bind_list.extend(binds)

        if bind_list:
            for host_path, container_path in bind_list:
                command.extend(["--bind", f"{Path(host_path).resolve()}:{container_path}"])
        elif input_dir is not None or output_dir is not None or work_dir is not None:
            for path in [input_dir, output_dir, work_dir]:
                if path is not None:
                    command.extend(["--bind", str(Path(path).resolve())])

        command.append(self.image)

        if binds is None and input_dir is not None and output_dir is not None:
            command.append(str(Path(input_dir).resolve()))
            command.append(str(Path(output_dir).resolve()))

        if extra_args:
            command.extend(shlex.split(extra_args))

        return command

    def run(
        self,
        input_dir=None,
        output_dir=None,
        work_dir=None,
        extra_args="",
        dry_run=False,
        binds=None,
        clean_env=None,
    ):
        if output_dir is not None:
            ensure_dir(output_dir)
        if work_dir is not None:
            ensure_dir(work_dir)
        command = self.build_command(
            input_dir=input_dir,
            output_dir=output_dir,
            work_dir=work_dir,
            extra_args=extra_args,
            binds=binds,
            clean_env=clean_env,
        )
        return run_command(command, dry_run=dry_run)
