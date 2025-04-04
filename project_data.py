import pandas as pd
from typing import Dict
from ct_series import CTSeries

import logging
# Set up logger for this module
if __name__ == "__main__":
    logger = logging.getLogger('project_data')
else:
    logger = logging.getLogger(__name__)

class ProjectData:
    """Represents a DICOM study containing multiple series."""
    
    def __init__(self, name: str):
        self.project_name = name
        logger.info(f"Created new ProjectData with name: {name}")
        self.series: Dict[str, 'CTSeries'] = {} # A dictonary to hold the series.
        
    
    def add_series(self, df: pd.DataFrame):
        # Create a CT series object form the dataframe.
        series = CTSeries(df)
        # Add the series to the dictionary with a running number as the key
        series_number = str(len(self.series) + 1)
        self.series[series_number] = series
        logger.info(f"Added series {series_number} to project {self.project_name}")
        
        # Add some relevant metadata for this series to a dataframe containing summary information of all the series along with the key.
        # This is a placeholder for the actual implementation.

    
