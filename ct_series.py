import os
import shutil
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
    series_instance_uid: str = None   # Unique identifier for the series
    series_index: int = None          # Index of the series in the dataset
    has_pixel_data: bool = False
    pixel_data: np.ndarray = None       # Placeholder for pixel data
    pixel_data_shape: tuple = None      
    pixel_data_stored: bool = False     # Flag to indicate if pixel data is stored in a pkl.
    pixel_data_path: str = None         # Path to the pixel data file if stored
    z_location: list = None             # A list of z locations for each slice
    mA_curve: list = None               # A list of mA values for each slice
    ctdi_vol_curve: list = None               # A list of CTDIvol values for each slice

    def __post_init__(self):
        self.series_index
        self.series_instance_uid = self.data['SeriesInstanceUID'].iloc[0]
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

    def _check_consistency_pixeldata_memory(self):
        """Verify and correct consistency of in-memory pixel data state.

        Inspects `self.pixel_data` and `self.has_pixel_data` to ensure they
        accurately reflect whether pixel data (a NumPy array) is loaded.
        The method attempts to correct inconsistencies by updating
        `self.has_pixel_data` and logs these actions.

        Scenarios Handled:
        1.  `self.pixel_data` is a NumPy array and `self.has_pixel_data`
            is `True`: Consistent state, no action.
        2.  `self.pixel_data` is `None` and `self.has_pixel_data` is `False`:
            Consistent state, no action.
        3.  `self.pixel_data` is a NumPy array but `self.has_pixel_data`
            is `False`: Inconsistent. Corrects `self.has_pixel_data` to `True`
            and logs a warning.
        4.  `self.pixel_data` is `None` but `self.has_pixel_data` is `True`:
            Inconsistent. Corrects `self.has_pixel_data` to `False`
            and logs a warning.

        If `self.pixel_data` holds a value that is neither `None` nor
        recognized as a NumPy array by the internal `isinstance` check
        (see Note), this method currently does not perform a specific
        correction or log an error for this particular state beyond what the
        existing conditions cover; it will implicitly return `None`.

        Note
        ----
        The internal type check for the NumPy array is currently performed
        using `isinstance(self.pixel_data, np.array)`. For robust type
        checking of NumPy arrays, `isinstance(self.pixel_data, np.ndarray)`
        is generally the preferred approach.

        Returns
        -------
        None
            This method always returns `None` (implicitly). Its primary purpose
            is to perform consistency checks and corrections as side effects.
            Calling code should not rely on a boolean return value to gate
            further operations based on the outcome of this method.

        Side Effects
        ------------
        - Modifies `self.has_pixel_data` if an inconsistency is found and
          a correction is applied.
        - Logs warning messages to the module's logger when corrections
          are made.
        """
        pixel_data_is_array = isinstance(self.pixel_data, np.array)
        flag_is_true = self.has_pixel_data

        # Case 1 & 2: Consistent states
        if (pixel_data_is_array and flag_is_true) or \
           (not pixel_data_is_array and self.pixel_data is None and not flag_is_true):
            return
        
        # Case 3: Data loaded, flag says not
        if pixel_data_is_array and not flag_is_true:
            logger.warning(
                f"Series No. {self.series_index} - Pixel data is loaded in memory, "
                f"but has_pixel_data was False. Corrected to True."
            )
            self.has_pixel_data = True
            return

        # Case 4: Data not loaded (None), flag says yes
        if not pixel_data_is_array and self.pixel_data is None and flag_is_true:
            logger.warning(
                f"Series No. {self.series_index} - Pixel data is None in memory, "
                f"but has_pixel_data was True. Corrected to False."
            )
            self.has_pixel_data = False
            return


    def _check_consistency_pixeldata_stored(self):
        """Verify and correct consistency of stored pixel data state.

        This method inspects `self.pixel_data_path` and
        `self.pixel_data_stored` against the actual existence of a file
        at `self.pixel_data_path` on the filesystem. It aims to ensure
        that the instance attributes accurately reflect the state of stored
        pixel data.

        The method handles several scenarios:
        1.  If no valid `pixel_data_path` is set:
            - If `pixel_data_stored` is `False` (consistent): No action.
            - If `pixel_data_stored` is `True` (inconsistent): Logs an error
              and corrects `pixel_data_stored` to `False`.
        2.  If a `pixel_data_path` is set:
            - It checks if the file at `pixel_data_path` actually exists.
            - If the file exists:
                - If `pixel_data_stored` is `False` (inconsistent): Logs a
                  warning and corrects `pixel_data_stored` to `True`.
                - If `pixel_data_stored` is `True` (consistent): No action.
            - If the file does NOT exist:
                - (Inconsistent): Logs an error, corrects `pixel_data_stored`
                  to `False`, and sets `pixel_data_path` to `None`.

        Side Effects
        ------------
        - Modifies `self.pixel_data_stored` if an inconsistency is found and
          a correction is applied.
        - Modifies `self.pixel_data_path` (sets to `None`) if the path is set
          but the file does not exist.
        - Logs messages to the module's logger detailing inconsistencies
          and corrections.

        Returns
        -------
        None
            This method does not explicitly return a value; it performs
            consistency checks and corrections in place.
        """
        path_is_actually_set = self.pixel_data_path is not None and self.pixel_data_path != ""
        flag_indicates_stored = self.pixel_data_stored        

        if not path_is_actually_set:
            # Case 1: No valid path is set for stored data.
            if not flag_indicates_stored:
                # Consistent: No path, and flag correctly indicates not stored.
                return
            else:
                # Inconsistent: Flag indicates stored, but no path is set.
                logger.error(
                    f"Series No. {self.series_index} - Pixel data is marked as stored (pixel_data_stored=True), "
                    f"but pixel_data_path is None or empty. Correcting flag."
                )
                self.pixel_data_stored = False
                # self.pixel_data_path is already None or empty, so no change needed for it.
                return
        else:
            # Case 2: A self.pixel_data_path is set. Check if file exists.
            file_actually_exists = os.path.exists(self.pixel_data_path)       

            if file_actually_exists:
                # Path is set and the file exists at that path.
                if not flag_indicates_stored:
                    # Inconsistent: File exists, but flag indicates not stored.
                    logger.warning(
                        f"Series No. {self.series_index} - Pixel data file exists at '{self.pixel_data_path}', "
                        f"but pixel_data_stored was False. Correcting flag to True."
                    )
                    self.pixel_data_stored = True
                # If flag_indicates_stored is True, then it's consistent.
                return        
            else:
                # Path is set, but the file does NOT exist at that path.
                # This implies the stored state information is invalid or stale.
                logger.error(
                    f"Series No. {self.series_index} - pixel_data_path is set to '{self.pixel_data_path}', "
                    f"but this file does not exist. Resetting stored pixel data flags and path."
                )
                self.pixel_data_stored = False
                self.pixel_data_path = None # Clear the invalid path.
                return        


    def read_pixel_data(self):
        """Load pixel data from DICOM files into the instance.

        This method populates the `self.pixel_data` attribute with pixel data
        read from the DICOM files listed in `self.data['FilePath']`.
        It performs several checks before loading:
        1.  Verifies memory consistency using `_check_consistency_pixeldata_memory`.
        2.  If `self.has_pixel_data` is True, it indicates data is already
            loaded, and the method returns early.
        3.  If `self.pixel_data` is None and file paths are available, it
            initializes `self.pixel_data` as a 3D NumPy array. The dimensions
            and data type are determined by reading the first DICOM file in
            the series.
        4.  If no file paths are available, a warning is logged, and the
            method returns.

        After successful pre-checks and initialization, the method iterates
        through each DICOM file path, reads the file, extracts the pixel array,
        and assigns it to the corresponding slice in `self.pixel_data`.

        Upon successful completion of loading all slices, `self.has_pixel_data`
        is set to True, and `self.pixel_data_shape` is updated.

        Side Effects
        ------------
        - Initializes `self.pixel_data` to a NumPy array if it was None and
          DICOM files are present.
        - Populates the `self.pixel_data` NumPy array with image data.
        - Sets `self.has_pixel_data` to `True` upon successful loading.
        - Sets `self.pixel_data_shape` to the shape of the loaded
          `self.pixel_data` array.
        - Logs various informational, warning, or error messages related to
          the loading process, consistency checks, or file access issues.

        Returns
        -------
        None
            This method modifies instance attributes directly and does not
            return a value. It returns early under certain conditions (e.g.,
            data already loaded, consistency check failure, no files,
            initialization failure, or errors during reading).

        Raises
        ------
        This method aims to catch common exceptions during file reading and
        array manipulation, logging them and returning early. However,
        unexpected errors not caught by the try-except blocks could still
        propagate.
        """
        self._check_consistency_pixeldata_memory()
        
        if self.has_pixel_data:
            logger.info(f"Series No. {self.series_index} - Pixel data already loaded.")
            return

        # Initialize self.pixel_data
        if self.pixel_data is None and not self.data['FilePath'].empty:
            first_slice_path = self.data['FilePath'].iloc[0]
            try:
                first_ds = dcm.dcmread(first_slice_path)
                rows, cols = first_ds.pixel_array.shape
                num_slices = len(self.data['FilePath'])
                self.pixel_data = np.empty((rows, cols, num_slices), dtype=first_ds.pixel_array.dtype)
                logger.info(f"Series No. {self.series_index} - Initialized pixel_data array with shape: {self.pixel_data.shape}")
            except Exception as e:
                logger.error(f"Series No. {self.series_index} - Failed to initialize pixel_data array: {e}")
                return
        elif self.data['FilePath'].empty:
            logger.warning(f"Series No. {self.series_index} - No file paths available to load pixel data.")
            return

        logger.info(f"Series No. {self.series_index} - Reading pixel data...")

        try:
            for i, path in enumerate(self.data['FilePath']):
                self.pixel_data[:,:,i] = dcm.dcmread(path).pixel_array
        except TypeError as e:
            if self.pixel_data is None:
                logger.error(
                    f"Series No. {self.series_index} - TypeError during pixel data loading: {e}. "
                    f"self.pixel_data is None. It must be initialized to a NumPy array before population."
                )
            else:
                logger.error(f"Series No. {self.series_index} - TypeError during pixel data loading: {e}.")
            return  # Stop further processing if there's a fundamental error
        except Exception as e:
            logger.error(f"Series No. {self.series_index} - Error reading pixel data for path {path} at index {i}: {e}")
            return
        
        self.has_pixel_data = True
        self.pixel_data_shape = np.shape(self.pixel_data)
        logger.info(f"Series No. {self.series_index} - Pixel data loaded successfully. Shape: {self.pixel_data_shape}")

    def store_pixel_data(self, path='Data', delete_memory=False, move=False):
        """Store pixel data to a file."""
        
        self._check_consistency_pixeldata_memory()
        self._check_consistency_pixeldata_stored()

        if not self.has_pixel_data:
            logger.info(f"Series No. {self.series_index} - Has no pixel data to store, call read_pixel_data() first.")
            return
        
        if self.pixel_data_stored and not move:
            logger.info(f"Series No. {self.series_index} - Pixel data already stored at {self.pixel_data_path}.")
            return

        new_path = os.path.join(path, f"CT_series_{self.series_index}_pixel_data.npy")

        if self.pixel_data_stored and move:
            match = self.pixel_data_path == new_path
            if match:
                logger.info(f"Series No. {self.series_index} - Pixel data already stored at {self.pixel_data_path}, moving not needed.")
                return
            else:
                try:
                    if not os.path.exists(path):
                        os.makedirs(path)
                    shutil.move(self.pixel_data_path, new_path)
                    logger.info(f"Series No. {self.series_index} - Moved pixel data to {new_path}.")
                    self.pixel_data_path = new_path
                    return
                except FileNotFoundError:
                    logger.error(f"Series No. {self.series_index} - Could not move pixel data to {new_path}. Directory does not exist.")
                    return
        
        # If we reach here, we need to save the pixel data to a new file.
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            logger.info(f"Series No. {self.series_index} - Directory {path} does not exist, creating it.")

        np.save("my_array.npy", my_array)


    def del_pixel_data(self, delete_stored=False):
        """Delete in-memory pixel data and optionally its stored file.

        This method first checks for consistency of the in-memory pixel data state.
        If `delete_stored_file` is True, it will also attempt to delete the
        pixel data file stored on disk by calling `self.del_stored_pixel_data()`.

        It then clears the `self.pixel_data` attribute by setting it to None
        and updates `self.has_pixel_data` to False and `self.pixel_data_shape`
        to None, effectively removing the pixel data from memory.

        Parameters
        ----------
        delete_stored_file : bool, optional
            If True, attempts to delete the associated stored pixel data file
            from disk. Defaults to False.

        Side Effects
        ------------
        - Modifies `self.pixel_data` to None.
        - Modifies `self.has_pixel_data` to False.
        - Modifies `self.pixel_data_shape` to None.
        - If `delete_stored_file` is True, may delete a file from disk and
          update `self.pixel_data_stored` and `self.pixel_data_path` via
          `self.del_stored_pixel_data()`.
        - Logs various informational or error messages.

        Returns
        -------
        None
        """       
        if not self._check_consistency_pixeldata_memory():
            # _check_consistency_pixeldata_memory logs errors if not ok
            return
        
        # Delete stored pixeldata if it exists, ignore this first:
        if delete_stored:
            self.del_stored_pixel_data()

        if not self.has_pixel_data:
            # This condition could be met if _check_consistency corrected it,
            # or if data was never loaded.
            logger.info(
                f"Series No. {self.series_index} - No in-memory pixel data to delete."
            )
            return

        self.pixel_data = None
        self.has_pixel_data = False
        self.pixel_data_shape = None
        logger.info(
            f"Series No. {self.series_index} - In-memory pixel data has been deleted."
        )

    def del_stored_pixel_data(self):

        # Delete stored pixel data file if requested:
        
        if delete_stored and self.pixel_data_stored:
            if self.pixel_data_path and os.path.exists(self.pixel_data_path):
                os.remove(self.pixel_data_path)
                logger.info(f"Stored pixel data file {self.pixel_data_path} deleted.")
            self.pixel_data_stored = False
            self.pixel_data_path = None







        
    
   

