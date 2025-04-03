import logging
import os
import pydicom as dcm
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

# Set up logger for this module
logger = logging.getLogger(__name__)

@dataclass
class StudyData:
    """Represents a DICOM study containing multiple series."""
    
    def __init__(self, study_instance_uid: str, study_description: str = None):
        self.study_instance_uid = study_instance_uid
        self.study_description = study_description
        self.series: Dict[str, CTSeries] = {}  # Dictionary of SeriesInstanceUID -> CTSeries
        self.metadata: Dict = {}
        logger.info(f"Created new StudyData with UID: {study_instance_uid}")
    
    def add_series(self, series: CTSeries):
        """Add a series to this study."""
        self.series[series.series_instance_uid] = series
        logger.info(f"Added series {series.series_instance_uid} to study {self.study_instance_uid}")
    
    def get_or_create_series(self, series_instance_uid: str, series_description: str = None) -> CTSeries:
        """Get an existing series or create a new one if it doesn't exist."""
        if series_instance_uid not in self.series:
            self.series[series_instance_uid] = CTSeries(series_instance_uid, series_description)
        return self.series[series_instance_uid]
    
    def extract_metadata(self):
        """Extract study-level metadata from any series in the study."""
        if not self.series:
            return
            
        # Take first series
        first_series = next(iter(self.series.values()))
        if not first_series.dicom_datasets:
            return
            
        ds = first_series.dicom_datasets[0]
        
        # Extract key study metadata
        self.metadata = {
            'StudyDescription': getattr(ds, 'StudyDescription', None),
            'StudyDate': getattr(ds, 'StudyDate', None),
            'StudyTime': getattr(ds, 'StudyTime', None),
            'PatientName': str(getattr(ds, 'PatientName', '')),
            'PatientID': getattr(ds, 'PatientID', None),
            'PatientBirthDate': getattr(ds, 'PatientBirthDate', None),
            'PatientSex': getattr(ds, 'PatientSex', None),
            'AccessionNumber': getattr(ds, 'AccessionNumber', None),
            'NumSeries': len(self.series)
        }
        
        logger.info(f"Extracted metadata for study {self.study_instance_uid}")