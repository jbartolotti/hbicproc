from .bidsify import BidsifyStage
from .download import DownloadStage
from .preprocess import PreprocessStage
from .qc import QcStage
from .qc_review import QcReviewStage
from .validate import ValidateStage

STAGE_CLASSES = {
    "download": DownloadStage,
    "bidsify": BidsifyStage,
    "validate": ValidateStage,
    "qc": QcStage,
    "qc_review": QcReviewStage,
    "preprocess": PreprocessStage,
}
