import os
import numpy as np
import pydicom
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import defaultdict



@dataclass
class CT_series:
    """
    Class to hold the DICOM CT series information for one series.
    Attributes:
        series_id (str): The unique identifier for the series.
        series_description (str): A description of the series.
        images (List[pydicom.Dataset]): A list of DICOM datasets representing the images in the series.
    """
    series_id: str
    series_description: str
    images: List[pydicom.Dataset] = field(default_factory=list)