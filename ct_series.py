import os
import pydicom as dcm
import numpy as np
import pandas as pd
from dataclasses import dataclass

import logging
# Set up logger for this module
if __name__ == "__main__":
    logger = logging.getLogger('ct_series')
else:
    logger = logging.getLogger(__name__)

@dataclass
class CTSeries:
    """Represents a series of CT images with the same Series Instance UID."""
    data: pd.DataFrame
    has_pixel_data: bool = False
    pixel_data: np.ndarray = None       # Placeholder for pixel data
    pixel_data_shape: tuple = None      
    z_location: list = None             # A list of z locations for each slice
    mA_curve: list = None               # A list of mA values for each slice
    ctdi_vol_curve: list = None               # A list of CTDIvol values for each slice

    def __post_init__(self):
        # Fill in the z_location
        if 'SliceLocation' in self.data.columns:
            # Check if the data is sorted by SliceLocation
            if not self.data['SliceLocation'].is_monotonic_increasing:
                logger.warning("SliceLocation is not sorted. Sorting it now.")
                self.data = self.data.sort_values(by='SliceLocation')
            # Fill in the z_location
            self.z_location = self.data['SliceLocation'].tolist()
        if 'XRayTubeCurrent' in self.data.columns: 
            self.mA_curve = self.data['XRayTubeCurrent'].tolist()
        # Get the CTDIvol curve if it exists
        if 'CTDIvol' in self.data.columns:
            self.ctdi_vol_curve = self.data['CTDIvol'].tolist()
            
        

        
    
   

