"""Local AI-generated-image detection.

    from aidetect import Detector
    d = Detector.load()
    print(d.predict("photo.jpg").as_dict())
"""

from .calibration import FusionModel, PlattCalibrator, metrics, threshold_at_fpr
from .detector import Detector, Prediction
from .imageio import ImageLoadError, load_rgb, open_image
from .metadata import MetadataVerdict, read_metadata
from .probe import LinearProbe

__version__ = "0.1.0"
__all__ = [
    "Detector", "Prediction", "FusionModel", "PlattCalibrator", "LinearProbe",
    "MetadataVerdict", "read_metadata", "open_image", "load_rgb", "ImageLoadError",
    "metrics", "threshold_at_fpr",
]
