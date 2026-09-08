from .analysis import AnalysisStage
from .bidsify import BidsifyStage
from .behavior import BehaviorStage
from .download import DownloadStage
from .preprocess import PreprocessStage
from .qc import QcStage
from .qc_review import QcReviewStage
from .validate import ValidateStage
from .group import GroupStage

STAGE_CLASSES = {
    "download": DownloadStage,
    "bidsify": BidsifyStage,
    "validate": ValidateStage,
    "qc": QcStage,
    "qc_review": QcReviewStage,
    "preprocess": PreprocessStage,
    "analysis": AnalysisStage,
    "behavior": BehaviorStage,
    "group": GroupStage,
}
