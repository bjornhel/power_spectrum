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
        
        # Create the dataframe to store the overview of the series
        self._initialize_series_overview(df)
        # create a list of the series in the project:
        self.list_of_series = []
        # Add the data from the series to the overview:
        self.add_series(df)
        
    def _initialize_series_overview(self, df: pd.DataFrame):
        """
        Initialize the series overview dataframe.
        This method is called in the constructor to set up the initial state of the series overview.
        """
        # Check if the dataframe is empty
        if df.empty:
            logger.warning("The provided dataframe is empty. No series overview will be created.")
            return None
        # Check if the dataframe contains study instance UID
        if 'series_uid' not in df.columns:
            logger.error("The dataframe does not contain 'StudyInstanceUID' column. Cannot initialize series overview.")
            return None
        
        # Get a list of all the columns in the input ataframe
        potential_columns = df.columns.tolist()
        # Go through all the study instance UID's in the dataframe
        for uid in df['series_uid'].unique():
            study  = df[df['series_uid'] == uid]
            # Go through all potential columns in the dataframe and check whether they have exacly one unique value:
            for column in potential_columns:
                # Check if the column exists in the dataframe
                if column not in df.columns:
                    logger.warning(f"Column {column} does not exist in the dataframe. Skipping.")
                    continue
                
                # Check if the column has exactly one unique value
                unique_values = study[column].unique()
                # if the length of the unique values is not 1, remove from the list of potential columns
                if len(unique_values) != 1:
                    potential_columns.remove(column)
                    logger.warning(f"Column {column} has {len(unique_values)} unique values. Removing from overview.")
                
        # Initialize a dataframe with the relevant columns with series index first:
        self.series_overview = pd.DataFrame(columns=['series_index'] + potential_columns)

        for column in potential_columns:
            logger.info(f"{column} was added to the series overview dataframe.")

    
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
        for series_uid in df['series_uid'].unique():
            # If the series already exists, skip it
            if series_uid in self.series_overview:
                logger.warning(f"Series {series_uid} already exists in the project. Skipping.")
                continue
            
            series_df = df[df['series_uid'] == series_uid]

            # Create a ct_series object and store it in the series_overview list.
            new_series = CTSeries(series_df)
            self.list_of_series.append(new_series)
            # Get the index of the series from the list of series
            series_index = self.list_of_series.index(new_series)
            logger.info(f"Added series {series_index} with series uid: {series_uid} to project {self.project_name}")
            # Add the series index to the series overview dataframe
            new_row = {'series_index': series_index}
            # Add all the other columns to the new row
            for column in self.series_overview.columns:
                if column in 'series_index':
                    continue # Already added the series index
                if column in series_df.columns:
                    # check if the column has exactly one unique value
                    unique_values = series_df[column].unique()
                    if len(unique_values) == 1:
                        new_row[column] = unique_values[0]
                    else:
                        new_row[column] = None
                        logger.warning(f"Column {column} has {len(unique_values)} unique values. Skipping.")
                else:
                    new_row[column] = None
                    logger.warning(f"Column {column} does not exist in the dataframe. Skipping.")
            # Append the new row to the series overview dataframe
            new_row_df = pd.DataFrame([new_row])
            self.series_overview = pd.concat([self.series_overview, new_row_df], ignore_index=True)
            logger.info(f"Added series {series_index} to project {self.project_name}")
        
        logger.info(f"Added {len(self.list_of_series)} series to project {self.project_name}")
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

    
