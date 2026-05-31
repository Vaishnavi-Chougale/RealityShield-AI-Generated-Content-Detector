import logging

from .config import config

logger = logging.getLogger(__name__)

# Keep package initialization lightweight so callers can import specific models
# without pulling in optional training and augmentation dependencies.
__all__ = ["config", "logger"]
