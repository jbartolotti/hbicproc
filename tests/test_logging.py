import logging

import pytest

from pipeline.config import _apply_defaults, validate_config
from pipeline.logger import setup_logging


def test_logging_defaults_include_dependency_level() -> None:
    config = _apply_defaults({})

    assert config["logging"]["level"] == "INFO"
    assert config["logging"]["dependency_level"] == "INFO"
    validate_config(config)


def test_setup_logging_separates_pipeline_and_dependency_levels() -> None:
    root_logger = logging.getLogger()
    pipeline_logger = logging.getLogger("pipeline")
    original_root_level = root_logger.level
    original_pipeline_level = pipeline_logger.level
    original_handlers = list(root_logger.handlers)
    try:
        setup_logging("DEBUG", "WARNING")

        assert root_logger.level == logging.WARNING
        assert pipeline_logger.level == logging.DEBUG
        console_handlers = [
            handler
            for handler in root_logger.handlers
            if getattr(handler, "_hbicproc_console", False)
        ]
        assert console_handlers
        assert all(handler.level == logging.NOTSET for handler in console_handlers)
    finally:
        root_logger.setLevel(original_root_level)
        pipeline_logger.setLevel(original_pipeline_level)
        for handler in list(root_logger.handlers):
            if handler not in original_handlers:
                root_logger.removeHandler(handler)
                handler.close()


def test_invalid_dependency_level_is_rejected() -> None:
    config = _apply_defaults({"logging": {"dependency_level": "TRACE"}})

    with pytest.raises(ValueError, match="dependency_level"):
        validate_config(config)
