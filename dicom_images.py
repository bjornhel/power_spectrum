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







@dataclass
class study_data:
    """
    A class to hold an array of CT_series objects and their associated metadata.
    Attributes:
        series (Dict[str, CT_series]): A dictionary mapping series IDs to CT_series objects.
        series_description (Dict[str, str]): A dictionary mapping series IDs to their descriptions.
        series_images (Dict[str, List[pydicom.Dataset]]): A dictionary mapping series IDs to lists of DICOM datasets.
    """
    root_dir: str = field(default_factory=str)
    
    series: Dict[str, CT_series] = field(default_factory=dict)
    series_description: Dict[str, str] = field(default_factory=dict)
    series_images: Dict[str, List[pydicom.Dataset]] = field(default_factory=dict)


    def __post_init__(self):
        """
        Initialize the study_data object with empty dictionaries for series, descriptions, and images.
        """

    def add_series(self, series_id: str, series_description: str, images: List[pydicom.Dataset]) -> None:
        """
        Add a new CT series to the study data.
        Args:
            series_id (str): The unique identifier for the series.
            series_description (str): A description of the series.
            images (List[pydicom.Dataset]): A list of DICOM datasets representing the images in the series.
        """
        self.series[series_id] = CT_series(series_id, series_description, images)
        self.series_description[series_id] = series_description
        self.series_images[series_id] = images