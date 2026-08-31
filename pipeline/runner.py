from .state import load_subject_state, save_subject_state
from .stages import STAGE_CLASSES
from .stages.base import StageResult
from .core.paths import list_subjects


class PipelineRunner:
    stage_order = [
        "download",
        "bidsify",
        "validate",
        "qc",
        "qc_review",
        "preprocess",
    ]

    def __init__(self, config):
        self.config = config
        self.stages = {name: cls() for name, cls in STAGE_CLASSES.items()}

    def load_state(self, subject):
        return load_subject_state(self.config, subject)

    def save_state(self, subject, state):
        save_subject_state(self.config, subject, state)

    def get_next_stage(self, state):
        for stage_name in self.stage_order:
            stage = self.stages[stage_name]
            if stage.state_key and state.get(stage.state_key):
                continue
            return stage_name
        return None

    def run_stage(self, stage_name, subject, dry_run=False, rerun=False, **kwargs):
        if stage_name not in self.stages:
            return StageResult(success=False, message=f"Unknown stage: {stage_name}")

        state = self.load_state(subject)
        stage = self.stages[stage_name]
        result = stage.execute(subject, self.config, state, dry_run=dry_run, rerun=rerun, **kwargs)

        if result.success and not result.skipped and stage.state_key and not stage.human_step:
            self.save_state(subject, state)

        return result

    def run_resume(self, subject, dry_run=False, rerun=False):
        for stage_name in self.stage_order:
            result = self.run_stage(stage_name, subject, dry_run=dry_run, rerun=rerun)
            if not result.success:
                return result

        return StageResult(success=True, message=f"Pipeline complete for {subject}.")

    def get_subject_for_stage(self, stage_name, rerun=False):
        if stage_name not in self.stages:
            return None

        subjects = self.all_subjects()
        if not subjects:
            return None

        stage_index = self.stage_order.index(stage_name)
        previous_stages = self.stage_order[:stage_index]
        for subject in subjects:
            state = self.load_state(subject)
            stage = self.stages[stage_name]
            if stage.state_key and state.get(stage.state_key) and not rerun:
                continue

            ready = True
            for prev_stage_name in previous_stages:
                prev_stage = self.stages[prev_stage_name]
                if prev_stage.state_key and not state.get(prev_stage.state_key):
                    ready = False
                    break
            if ready:
                return subject

        return None

    def all_subjects(self):
        return list_subjects(self.config)
