"""DICOM CT series project data management.

This module provides functionality for organizing and analyzing CT series data from DICOM files.
It categorizes metadata into series-level and instance-level information, facilitating analysis
across multiple series within a project.

Classes:
    ProjectData: Container for multiple CT series with shared metadata
    
Dependencies:
    pandas: For data manipulation and storage
    logging: For activity logging
    ct_series: For individual series handling
"""

import pandas as pd
from ct_series import CTSeries

import logging
# Set up logger for this module
if __name__ == "__main__":
    logger = logging.getLogger('project_data')
else:
    logger = logging.getLogger(__name__)

class ProjectData:
    """Represents a DICOM study containing multiple series.
    
    This class serves as a container for CT series data from DICOM files,
    organizing them into overview information (consistent across a series) and
    series-specific details. It provides methods to import, categorize and
    analyze DICOM metadata.
    
    Attributes:
        project_name (str): Name identifier for this project
        list_of_series (list[CTSeries]): Collection of CT series objects
        overview_columns (list[str]): Names of columns with consistent values per series
        series_only_columns (list[str]): Names of columns with varying values within series
        series_overview (pd.DataFrame): Summary table with one row per series
        
    Methods:
        add_series: Adds new CT series from a DataFrame to this project
        _initialize_series_overview: Sets up the series overview data structure
        _check_columns: Categorizes columns as overview or series-specific
    """
    project_name: str
    list_of_series: list[CTSeries]
    overview_columns: list[str]
    series_only_columns: list[str]
    series_overview: pd.DataFrame

    def __init__(self, name: str, df: pd.DataFrame):
        """Initialize a new ProjectData instance.
    
        Creates a project to manage multiple CT series with shared metadata.
        Processes the input DataFrame to identify series-level and instance-level
        attributes, and organizes them into appropriate data structures.
        
        Parameters
        ----------
        name : str
            Name identifier for this project
        df : pd.DataFrame
            DataFrame containing DICOM metadata with at least a 'series_uid' column
            
        Notes
        -----
        - Initializes empty lists for series objects and column categorizations
        - Creates an empty series_overview DataFrame
        - Calls _initialize_series_overview() to process the input DataFrame
        - Logs the creation of the project
        """
        self.project_name = name
        logger.info(f"Created new ProjectData with name: {name}")
        
        # Initialize an empty list to store CTSeries objects
        self.list_of_series = []
        self.overview_columns = []      # These are the columns that are in the overview dataframe
        self.series_only_columns = []   # These are the columns that only exists in the series dataframe
        self.series_overview = pd.DataFrame() # Initialize empty DataFrame

        # Create the dataframe to store the overview of the series
        self._initialize_series_overview(df)    
                
    def _initialize_series_overview(self, df: pd.DataFrame) -> None:
        """Initialize the series overview dataframe.
    
        This method prepares the data structure that will store summary information
        about each series in the project. It verifies the input dataframe has the
        required columns, extracts column metadata, and then calls add_series() 
        to populate the overview with data.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing DICOM metadata with at least a 'series_uid' column.
            Each unique series_uid will become a row in the overview table.
            
        Returns
        -------
        None
            This method modifies the project state but doesn't return a value.
            
        Notes
        -----
        - Called automatically during ProjectData initialization
        - Sets the initial overview_columns based on all columns in the input dataframe
        - Creates series objects for each unique series_uid through add_series()
        - Returns early if the dataframe is empty or missing required columns
        """
        # Check if the dataframe is empty
        if df.empty:
            logger.warning("The provided dataframe is empty. No series overview will be created.")
            return None
        # Check if the dataframe contains study instance UID
        if 'series_uid' not in df.columns:
            logger.error("The dataframe does not contain 'StudyInstanceUID' column. Cannot initialize series overview.")
            return None
        
        # Get a list of all the columns in the input dataframe
        self.overview_columns = df.columns.tolist()
        self.add_series(df) # Add all the series in the dataframe to the project.
    
    def _check_columns(self, df: pd.DataFrame) -> None:
        """Categorize dataframe columns as either overview or series-only.
        
        This method analyzes each column in the dataframe to determine whether it 
        contains consistent values within each series (suitable for overview) or
        varies within series (series-only). It modifies the class attributes
        'overview_columns' and 'series_only_columns' accordingly, and updates the
        'series_overview' dataframe structure.
        
        For each unique series_uid in the dataframe, this method checks if each column
        has exactly one unique value (making it an overview column) or multiple unique
        values (making it a series-only column). The series_overview dataframe is then
        updated to include only overview columns.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing DICOM metadata with at least a 'series_uid' column.
            
        Returns
        -------
        None
            This method modifies class attributes in-place but returns no value.
            
        Notes
        -----
        - Modifies self.overview_columns, self.series_only_columns, and self.series_overview
        - Logs warnings when columns are reclassified or modified
        - Creates the series_overview dataframe if it doesn't exist
        - For columns with exactly one value per series, they are added to overview_columns
        - For columns with multiple values per series, they are added to series_only_columns
        """
        # Get all the columns in the dataframe
        all_df_columns = df.columns.tolist()

        # Go through all the study instance UID's in the dataframe
        for uid in df['series_uid'].unique():
            study  = df[df['series_uid'] == uid]
            # Go through all potential columns in the dataframe and check whether they have exacly one unique value:
            for column in all_df_columns:
                # Check if the column has exactly one unique value
                unique_values = study[column].unique()
                # if the length of the unique values is not 1, put the column in the series only column and remove it from the overview columns
                if len(unique_values) != 1:
                    # Check if the column is in the overview column
                    if column in self.overview_columns:
                        # If the column is in the overview columns, remove it from the overview columns
                        self.overview_columns.remove(column)
                        self.series_only_columns.append(column)
                        logger.warning(f"Column {column} has {len(unique_values)} unique values. Removing from overview and adding to series only.")
                    if column not in self.series_only_columns: 
                        self.series_only_columns.append(column)
                        logger.warning(f"Column {column} has {len(unique_values)} unique values. But was not found in series only or overview columns. Adding to series only.")
                if len(unique_values) == 1:
                    if column not in (self.overview_columns + self.series_only_columns):
                        # If the column is not in the overview columns, add it to the overview columns
                        self.overview_columns.append(column)
                        logger.warning(f"Column {column} has {len(unique_values)} unique values. Adding to overview.")
        
        # If the series overview is empty, create it with the overview columns
        if len(self.series_overview) == 0:
            self.series_overview = pd.DataFrame(columns=['series_index'] + self.overview_columns)
        # If it exists, check all columns in the overview columns and add or remove them as needed
        else:
            for column in self.overview_columns:
                if column not in self.series_overview.columns:
                    # If the column is not in the series overview, add it with None values
                    self.series_overview[column] = None
                    logger.warning(f"Column {column} not found in series overview. Adding it.")
            for column in self.series_only_columns:
                if column in self.series_overview.columns:
                    # If the column is in the series overview, remove it
                    self.series_overview.drop(column, axis=1, inplace=True)
                    logger.warning(f"Column {column} was found to have multiple values for single series. Removing it from the overview.")

    def add_series(self, df: pd.DataFrame) -> None:
        """Add CT series from a DataFrame to this project.
        
        This method creates CTSeries objects from a DataFrame containing DICOM metadata.
        It performs the following steps:
        1. Validates the input DataFrame
        2. Categorizes columns as overview or series-specific via _check_columns()
        3. For each unique series_uid, creates a CTSeries object
        4. Adds a summary row to the series_overview DataFrame
        5. Logs the addition of each series
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing DICOM metadata with at least a 'series_uid' column.
            Each unique series_uid will become a separate CTSeries object.
            
        Returns
        -------
        None
            This method modifies the project state but doesn't return a value.
            
        Notes
        -----
        - Skips series that already exist in the project
        - Creates CTSeries objects stored in list_of_series
        - Updates series_overview with summary information
        - Maintains series_index values to reference CTSeries objects
        - Returns early if the DataFrame is empty or lacks required columns
        """
        # Check if the dataframe is empty
        if df.empty:
            logger.warning("The provided dataframe is empty. No series will be added.")
            return

        # Check if the dataframe contains study instance UID
        if 'series_uid' not in df.columns:
            logger.error("The dataframe does not contain 'series_uid' column. Cannot add series.")
            return
        
        self._check_columns(df) # Check all the columns in the dataframe and add the relevant data to the overview.

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
            series_index = len(self.list_of_series) -1
            # Add the series index to the series overview dataframe
            new_row = {'series_index': series_index}
            # Add all the other columns to the new row
            for column in self.series_overview.columns:
                if column in 'series_index':
                    continue # Already added the series index
                if column in series_df.columns:
                    new_row[column] = series_df[column].iloc[0]
                else:
                    new_row[column] = None
            
            new_row_df = pd.DataFrame([new_row])
            self.series_overview = pd.concat([self.series_overview, new_row_df], ignore_index=True)
            logger.info(f"Added series {series_index} with series uid: {series_uid} to project {self.project_name}")
        
        logger.info(f"Added {len(self.list_of_series)} series to project {self.project_name}")

    def get_list(self) -> list[CTSeries]:
        """Get the list of CTSeries objects in this project.
        
        Returns
        -------
        list[CTSeries]
            List of CTSeries objects associated with this project.
        """
        return self.list_of_series
    
    def get_overview(self) -> pd.DataFrame:
        """Get the series overview DataFrame.
        
        Returns
        -------
        pd.DataFrame
            DataFrame containing summary information about each CTSeries in this project.
        """
        return self.series_overview
    
