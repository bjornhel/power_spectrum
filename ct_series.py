import logging
import os
import pydicom as dcm
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

# Set up logger for this module
logger = logging.getLogger(__name__)

@dataclass
class CTSeries:
    """Represents a series of CT images with the same Series Instance UID."""
    
    def __init__(self, series_instance_uid: str, series_description: str = None):
        self.series_instance_uid = series_instance_uid
        self.series_description = series_description
        self.image_paths: List[str] = []
        self.dicom_datasets: List[dcm.Dataset] = []
        self.pixel_data: Optional[np.ndarray] = None
        self.metadata: Dict = {}
        logger.info(f"Created new CTSeries with UID: {series_instance_uid}")
    
    def add_image(self, dicom_path: str, dataset: dcm.Dataset):
        """Add a DICOM image to this series."""
        self.image_paths.append(dicom_path)
        self.dicom_datasets.append(dataset)
        logger.debug(f"Added image to series {self.series_instance_uid}: {dicom_path}")
    
    def load_pixel_data(self):
        """Load pixel data for all images in this series."""
        if not self.dicom_datasets:
            logger.warning(f"No datasets to load pixel data from in series {self.series_instance_uid}")
            return
            
        # Sort images by Instance Number or Position
        sorted_datasets = self._sort_datasets()
        
        # Get dimensions from first image
        rows = sorted_datasets[0].Rows
        cols = sorted_datasets[0].Columns
        depth = len(sorted_datasets)
        
        # Create 3D array
        self.pixel_data = np.zeros((rows, cols, depth), dtype=np.int16)
        
        # Fill array with pixel data
        for i, ds in enumerate(sorted_datasets):
            full_ds = dcm.dcmread(self.image_paths[i])  # Read with pixels
            self.pixel_data[:, :, i] = full_ds.pixel_array
            
        logger.info(f"Loaded pixel data for series {self.series_instance_uid}, shape: {self.pixel_data.shape}")
    
    def _sort_datasets(self):
        """Sort datasets by instance number or slice location."""
        if hasattr(self.dicom_datasets[0], 'InstanceNumber'):
            return sorted(self.dicom_datasets, key=lambda x: x.InstanceNumber)
        elif hasattr(self.dicom_datasets[0], 'SliceLocation'):
            return sorted(self.dicom_datasets, key=lambda x: x.SliceLocation)
        return self.dicom_datasets
    
    def extract_metadata(self):
        """Extract common metadata from the series."""
        if not self.dicom_datasets:
            return
            
        ds = self.dicom_datasets[0]  # Use first dataset for common tags
        
        # Extract key metadata fields
        self.metadata = {
            'SeriesDescription': getattr(ds, 'SeriesDescription', None),
            'SeriesDate': getattr(ds, 'SeriesDate', None),
            'SeriesTime': getattr(ds, 'SeriesTime', None),
            'Modality': getattr(ds, 'Modality', None),
            'ConvolutionKernel': getattr(ds, 'ConvolutionKernel', None),
            'SliceThickness': getattr(ds, 'SliceThickness', None),
            'ImageType': getattr(ds, 'ImageType', None),
            'PixelSpacing': getattr(ds, 'PixelSpacing', None),
            'NumImages': len(self.dicom_datasets)
        }
        
        logger.info(f"Extracted metadata for series {self.series_instance_uid}")
