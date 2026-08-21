from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from ..config import TaskConfig


class TaskParser(ABC):
    @abstractmethod
    def parse(self, raw_df: pd.DataFrame, config: TaskConfig) -> pd.DataFrame:
        """Convert an E-Prime dataframe into an events dataframe."""
