import os
import pydicom as dcm
import numpy as np
from typing import Dict, List, Optional
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
    
   # Create a Series.
   # Store the dataframe witht the info.
   # Read the pixel data.
   

