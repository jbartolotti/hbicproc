from dataclasses import dataclass, field
from typing import Optional


@dataclass
class StageResult:
    success: bool
    message: str
    skipped: bool = False
    needs_review: bool = False
    next_command: Optional[str] = None
    details: dict = field(default_factory=dict)


class BaseStage:
    name = "base"
    state_key = None
    human_step = False

    def run(self, subject, config, state, dry_run=False, rerun=False):
        raise NotImplementedError("Stage implementations must override run().")

    def execute(self, subject, config, state, dry_run=False, rerun=False):
        if self.state_key and state.get(self.state_key) and not rerun:
            return StageResult(
                success=True,
                skipped=True,
                message=f"Stage '{self.name}' already complete for {subject}.",
            )

        result = self.run(subject, config, state, dry_run=dry_run, rerun=rerun)

        if result.success and not result.skipped and self.state_key and not self.human_step:
            state[self.state_key] = True

        return result
