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
    
    def __init__(self, name: str, df: pd.DataFrame):
        
        self.project_name = name
        logger.info(f"Created new ProjectData with name: {name}")
        # Initialize an empty dataframe to hold the metadata that is common for the series along with an index:
        self.series_overview = self.add_series(df=pd.DataFrame)
        logger.info(f"Imported metadata for {len(self.series_overview)} series into project {self.project_name}")

    def add_series(self, df: pd.DataFrame):
        """
        Create several CT series from a dataframe containing dicom metadata.
        All CT series are kept 
        """
        # Check if the dataframe is empty
        if df.empty:
            logger.warning("The provided dataframe is empty. No series will be added.")
            return

        # Create a CTSeries object for each unique Series Instance UID in the dataframe
        for series_uid in df['SeriesInstanceUID'].unique():
            # If the series already exists, skip it
            if series_uid in self.series_overview:
                logger.warning(f"Series {series_uid} already exists in the project. Skipping.")
                continue
                
            # Add the value of all the columns for each parameter that does not change for all the imagen in the series.
            # If the index does not exist, create it.
            # Increment the index by 1 form the previous length of the dataframe

            
            



            series_df = df[df['SeriesInstanceUID'] == series_uid]
            

            series = CTSeries(series_df)
            # Add the series to the dictionary with a running number as the key
            series_number = str(len(self.series_overview) + 1)
            self.series_overview[series_number] = series
            logger.info(f"Added series {series_number} to project {self.project_name}")
        
        return self.series_overview
        # Optionally, you can add some relevant metadata for this series to a dataframe containing summary information of all the series along with the key.
        # This is a placeholder for the actual implementation.



    #     series = CTSeries(df)
    #     # Add the series to the dictionary with a running number as the key
    #     series_number = str(len(self.series) + 1)
    #     self.series[series_number] = series
    #     logger.info(f"Added series {series_number} to project {self.project_name}")
        
        # Add some relevant metadata for this series to a dataframe containing summary information of all the series along with the key.
        # This is a placeholder for the actual implementation.

    
