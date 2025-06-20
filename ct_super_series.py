from dataclasses import dataclass
import pandas as pd
import numpy as np
import math

import logging
# Set up logger for this module
if __name__ == "__main__":
    logger = logging.getLogger('ct_series')
else:
    logger = logging.getLogger(__name__)


@dataclass
class CTSuperSeries():
    """Manages a collection of similar CT series as a single, coherent entity.

    This class groups multiple `CTSeries` objects that are intended to be
    comparable (e.g., repeated scans of the same phantom or patient). Upon
    initialization, it performs a series of validation and alignment steps to
    ensure the series are geometrically and parametrically consistent.

    The primary purpose is to facilitate aggregate analysis across these scans,
    such as calculating mean and standard deviation images, or summing dose-
    related curves. The class handles complex geometric alignment, allowing for
    scans with minor differences in starting position or length to be processed
    together.

    Attributes
    ----------
    list_of_series : list[CTSeries]
        The list of individual `CTSeries` objects that make up this super-series.
    accept_difference_positioning : bool, optional
        If True, enables flexible alignment to handle series with different
        starting positions or lengths. If False (default), requires all series
        to have identical z-locations.
    z_tolerance : float, optional
        In flexible alignment mode, the maximum allowed deviation (in mm)
        between a slice's z-location and the reference position. Defaults to 0.
        Only used if `accept_difference_positioning` is True.
    SliceLocations : list[float]
        The final, common z-axis coordinates for all slices after alignment and
        cropping. This defines the coordinate system for all aggregate data.
    total_mA_curve : list[float]
        The element-wise sum of the `mA_curve` from all contained series.
    total_ctdi_vol_curve : list[float]
        The element-wise sum of the `ctdi_vol_curve` from all contained series.
    KVP, ConvolutionKernel, etc.
        Key imaging parameters that have been verified to be consistent across
        all series are copied as attributes to the super-series instance.                
    pixel_data_individual : np.ndarray | None
        A 4D NumPy array of shape (height, width, n_slices, n_series) containing
        the stacked pixel data from all aligned series. Populated by calling
        `generate_pixel_data_individual()`.
    pixel_data_super_series : np.ndarray | None
        A 3D NumPy array representing the mean image, calculated pixel-wise
        across all series. Populated by calling `generate_mean_image()`.
    pixel_data_std_series : np.ndarray | None
        A 3D NumPy array representing the standard deviation image. Populated
        by calling `generate_std_image()`.

    Methods
    -------
    generate_pixel_data_individual()
        Loads and stacks the pixel data from all contained series.
    generate_mean_image()
        Calculates the mean image from the stacked individual pixel data.
    generate_std_image()
        Calculates the standard deviation image from the stacked data.
    """
    list_of_series: list
    accept_difference_positioning: bool = False  # Whether to accept different scan positioning and/or length
    z_tolerance: float = 0.0            # Tolerance for SliceLocations consistency check
    SliceLocations: list = None         # List of z-locations for the slices in the super series
    total_mA_curve: list = None         # Total mA curve for the series, if applicable
    total_ctdi_vol_curve: list = None       # Total CTDI curve for the series, if applicable
    KVP: int = None                     # KVP value for the series
    ConvolutionKernel: str = None       # Convolution kernel used
    IterativeAILevel: str = None        # Iterative AI level, if applicable
    MatrixSize: int = None              # Size of the matrix (square matrix asumed)
    SliceThickness: float = None        # Thickness of each slice in mm
    SingleCollimationWidth: float = None# Width of the single collimation
    TotalCollimationWidth: float = None # Total collimation width
    TableSpeed: float = None            # Speed of the table during the scan
    TableFeedPerRotation: float = None  # Table feed per rotation
    SpiralPitchFactor: float = None     # Pitch factor for spiral scans
    ReconstructionDiameter: float = None# Diameter of the reconstruction
    DataCollectionDiameter: float = None# Diameter of the data collection
    FocalSpots: float = None            # Focal spots used in the CT scan
    BodyPartExamined: str = None        # Body part examined in the series
    ProtocolName: str = None            # Name of the protocol used for the series
    StationName: str = None             # Name of the station where the series was acquired
    Manufacturer: str = None            # Manufacturer of the CT scanner
    ManufacturerModelName: str = None   # Model name of the CT scanner
    SoftwareVersions: str = None        # Software version of the CT scanner
    crop_borders: list = None           # List of slices to include in the super series, represents the indexes of the slices to include in the list of series.
    pixel_data_individual: np.ndarray = None       # Pixel data of all the CT series to be included in the super series. 4D array with shape (height, width, n_slices, n_series)
    pixel_data_super_series: np.ndarray = None     # Pixel data of the super series, if applicable. 3D array with shape (height, width, n_slices)

    def __post_init__(self):
        """Performs post-initialization validation and setup for the super-series.

        This special method is automatically called by the `@dataclass` decorator
        immediately after the instance has been created. Its primary role is to
        transform the raw list of `CTSeries` objects into a coherent, validated,
        and fully processed super-series.

        It executes a sequence of critical setup steps:
        1.  **Checks input:** Validates that `list_of_series` is a list and contains
            at least two `CTSeries` objects. If not, it raises a `ValueError`.
            If the input is not a list, it raises a `TypeError`.
            If the list contains less than two series, it raises a `ValueError`.
        2.  **Verifies Attributes:** Calls `_verify_attribute_consistency` to ensure
            that all series share the same attributes, such as `KVP`, `ConvolutionKernel`,
            `IterativeAILevel`, etc. If any attribute is inconsistent, it raises a
            `ValueError`. This is crucial for ensuring that the super-series can be
            meaningfully analyzed as a single entity.
            The attributes to be checked are defined in the `list_of_tags` variable.
            If the attributes are not consistent, it raises a `ValueError`.
        3.  **Aligns Geometry:** Calls `_align_and_check_clicelocations` to ensure
            all series are spatially aligned along the z-axis. If accept_difference_positioning
            is set to `False`, it performs a strict check that all series have identical
            `SliceLocations`. If set to `True`, it allows for flexible alignment, where a common
            overlapping region is determined. The remaining must be aligned within a specified
            tolerance (`z_tolerance`). If the alignment fails, it raises a `ValueError`. Populates
            the crop_borders attribute with the start and end indices of the common overlapping 
            region for each series to facilitate cropping of the pixel data, mA curve, and CTDI 
            olume curve.
        4. **Generates Total mA Curve:** Calls `_generate_total_mA_curve` to compute the
            cumulative mA curve across all series. This method sums the `mA_curve` attributes
            of each `CTSeries` object, ensuring that the resulting curve corresponds to the ovelapping
            slices in the super-series. If the `mA_curve` lists of the series have inconsistent lengths,
            it raises a `ValueError`.
        5. **Generates Total CTDI Volume Curve:** Calls `_generate_total_ctdi_vol_curve` to compute the
            cumulative CTDI volume curve across all series. This method sums the `ctdi_vol_curve` attributes
            of each `CTSeries` object, ensuring that the resulting curve corresponds to the overlapping
            slices in the super-series. If the `ctdi_vol_curve` lists of the series have inconsistent lengths,
            it raises a `ValueError`.

        After this method completes, the instance is considered fully prepared
        for analysis.

        Side Effects
        ------------
        - Populates numerous attributes on the instance (e.g., `SliceLocations`,
          `total_mA_curve` and 'total_ctdi_vol_curve').

        Raises
        ------
        ValueError
            Can be raised indirectly by the various helper methods if validation
            or alignment fails.

        Returns
        -------
        None
            This method does not return a value; it modifies the instance in-place.               
        """
        if not isinstance(self.list_of_series, list):
            raise TypeError("CTSeries must be a list of CT series.")
        logger.debug(f"CTSuperSeries initialized with {len(self.list_of_series)} CT series.")

        # Verify that the list of series contains at least two series.
        if len(self.list_of_series) < 2:
            raise ValueError("CTSuperSeries requires at least two CT series to create a super series.")

        # Verify that all the CT series has the same attributes.
        list_of_tags = ['KVP', 'ConvolutionKernel', 'IterativeAILevel',
                             'MatrixSize', 'SliceThickness', 'SingleCollimationWidth',
                             'TotalCollimationWidth', 'TableSpeed', 'TableFeedPerRotation',
                             'SpiralPitchFactor', 'ReconstructionDiameter',
                             'DataCollectionDiameter', 'FocalSpots', 'BodyPartExamined',
                             'ProtocolName', 'StationName', 'Manufacturer',
                             'ManufacturerModelName', 'SoftwareVersions']
        self._verify_attribute_consistency(list_of_tags)  
        self._align_and_check_slicelocations()
        self._generate_total_mA_curve()
        self._generate_total_ctdi_vol_curve()

    def _verify_attribute_consistency(self, attribute_name: list[str]):
        """Verifies that a given set of attributes are consistent across all CT series.

        This internal helper method iterates through a list of attribute names and
        checks if the value of each attribute is identical for every `CTSeries`
        object contained in `self.list_of_series`.

        It uses the first series in the list as the reference. A special check is
        included to correctly handle `NaN` (Not a Number) values, where two `NaN`s
        are considered consistent, which is not the case with a standard `!=`
        comparison.

        This method is typically called from `__post_init__` to ensure the
        homogeneity of the super-series before further processing.

        Parameters
        ----------
        attribute_names : list[str]
            A list of attribute names (as strings) to check for consistency.

        Raises
        ------
        ValueError
            If the value of any specified attribute is not the same across all
            series in the list.

        Returns
        -------
        None
            This method does not return a value; it raises an exception on failure.
        """
        # Extract the first series to compare against.
        first_series = self.list_of_series[0]
        # Go through each attribute name and check if all series have the same value.
        for attr in attribute_name:
            first_value = getattr(first_series, attr, None)
            
            for series in self.list_of_series[1:]:
                value = getattr(series, attr, None)

                # Handle NaN comparison, as np.nan != np.nan is True.
                # We consider two NaNs to be consistent for this check.
                are_both_nan = pd.isna(first_value) and pd.isna(value)

                if not are_both_nan and value != first_value:
                    raise ValueError(f"Attribute '{attr}' is not consistent across CT series: "
                                     f"{first_value} != {value}")

            setattr(self, attr, first_value)       
        
        logger.debug("All attributes are consistent across CT series.")
    
    def _align_and_check_slicelocations(self):
        """Aligns, verifies, and crops the z-axis slice locations for all series.

        This is a critical internal method that ensures all series in the
        super-series are spatially coherent along the z-axis. It uses the first
        series as the reference coordinate system. Its behavior is controlled by
        the `self.accept_difference_positioning` instance attribute.

        In Strict Mode (`accept_difference_positioning=False`):
        - The method performs a simple, exact check.
        - It verifies that every series has a `z_location` list that is
          identical to the reference series' list.
        - If any list differs, it raises a `ValueError`.

        In Flexible Alignment Mode (`accept_difference_positioning=True`):
        This mode handles complex cases where scans may have different starting
        positions, different lengths, or both, but still share a common
        overlapping region. The process is as follows:
        1.  **Calculate increment:** The slice increment (distance between slices)
            is calculated from the reference series.
        2.  **Calculate Shift:** For each subsequent series, the physical z-offset
            of its first slice is measured against the reference's first slice.
            This offset is then converted into an integer *index shift* by
            dividing by the slice increment.
        3.  **Veriy Overlap:** The method checks that the index shift does not
            exceed the number of slices in the reference or current series (depending on
            the shift direction) to ensure there is sufficient overlap.
            If the shift is too large, a `ValueError` is raised.
        4.  **Extract First Similar Location:** The first slice of each series
            is compared to the corresponding slice in the reference series based
            on the calculated index shift. 
        5.  **Check Tolerance:** The method checks that the first similar slice
            is within `self.z_tolerance` of the reference slice's z-location.
            If not, a `ValueError` is raised.
        6.  **Store Indices:** The method stores the first and last slice index
            in terms of the reference series' z-locations.
        7.  **Find Common Overlap:** The method determines the common overlapping
            region across all series by finding the maximum of the left indices
            and the minimum of the right indices. If no common region exists,
            a `ValueError` is raised.
        8.  **Store SliceLocations:** The `SliceLocations` attribute of the
            super-series is updated to reflect the common overlapping region.
        9.  **Store Crop Borders:** The method stores the start and end indices
            for the common overlapping region for each series in `self.crop_borders`.

        Attributes
        ----------
        accept_difference_positioning : bool
            Flag that controls whether to use strict or flexible alignment mode.
        z_tolerance : float
            The maximum allowed deviation (in mm) between a slice's SliceLocation and the
            reference position in flexible mode.

        Side Effects
        ------------
        - Updates `self.SliceLocations` to a new list containing only the slices
          that are common to all series after alignment.
        - Populates `self.crop_borders` with the start and end indices of the
          common overlapping region for each series.

        Raises
        ------
        ValueError
            - If the series do not have the same SliceLocation (Strict mode).
            - If the SliceLocation increment is not positive (should be unnecessary).
            - If the reference series has no slices (cannot do alignment).
            - If there is no common overlapping region across all series.
            - If the slices are not aligned within the specified `z_tolerance`
            - If there is no common overlaping region after cropping (might be unnecessary).

        Returns
        -------
        None
            This method modifies the instance in-place, updating `self.SliceLocations`
            and `self.crop_borders` attributes.
        """
        # Establish the first series' z-locations as the reference for the super-series.
        reference_z_loc = self.list_of_series[0].SliceLocations


        if not self.accept_difference_positioning:
            # --- STRICT MODE ---
            # Verify that all other series have the exact same z-locations (same scan length and positioning).
            for series in self.list_of_series[1:]:
                if series.SliceLocations != reference_z_loc:
                    raise ValueError(
                        f"Series {series.SeriesIndex} has z-locations that do not exactly "
                        f"match the reference series. To allow for differences in "
                        f"positioning or length, set `accept_difference_positioning=True`."
                    )
            self.SliceLocations = reference_z_loc
            # Make a list of the crop borders from index 0 to the end for each series in the super series.
            self.crop_borders = [[0, len(reference_z_loc)-1]] * len(self.list_of_series)
            logger.debug("All series have identical z-locations as required.")
            return

        # --- FLEXIBLE ALIGNMENT MODE ---
        # Use the first series as a reference for alignment.
        # Calculate the increment between each slice in the reference series.

        # Initialize the lists hold the left and right indices for each series, relative to the reference series.
        left_ind = []
        left_ind.append(0) # The first series has no shift
        right_ind = []
        right_ind.append(len(reference_z_loc)-1) # Holds the last index of the reference series.

        
        # --- 1. Calculate Increment ---
        if len(reference_z_loc) > 1:
            # Calculate the increment between slices in the reference scan (These are sorted from ct_series and import dicom).
            increment = reference_z_loc[1] - reference_z_loc[0]
            # Ensure the increment is positive (Should be unnecessary).
            if increment <= 0:
                raise ValueError("Reference z-location increment must be positive.")
        elif len(reference_z_loc) == 1:
            # If there is only one slice, we cannot calculate an increment.
            logger.warning("Reference series has only one slice. Increment cannot be calculated.")
            increment = None
        else:
            raise ValueError("Reference series has no slices. Cannot align z-locations.")
        

        for series in self.list_of_series[1:]:
            # --- 2. Calculate Z-Offset ---
            # Calculate the offset in the SliceLocation of the first slice of the current series relative to the reference series.
            first_slice_z = series.SliceLocations[0]
            z_offset = first_slice_z - reference_z_loc[0]

            if increment is not None:
                # Calculate the number of slices the series is offset from the reference series.
                # If the index_shift is positive the first slice of the current series is closest to [index_shift] of the reference series.
                # If the index_shift is negative, the first slice of the reference series is closest to the [abs(index_shift)] of the current series.
                index_shift = int(np.round(z_offset / increment))
            else:
                # If no increment, we cannot determine how to shift.
                index_shift = 0
            
            # --- 3. Test of there is sufficient overlap with the reference series ---
            overlap_error = False
            # If the index shift is positive, we need to check if the reference series has enough slices to accommodate the shift.
            if index_shift >= 0:
                if len(reference_z_loc) < index_shift:
                    overlap_error = True
            # If the index shift is negative, we need to check if the current series has enough slices to accommodate the shift.
            else:
                if len(series.SliceLocations) < abs(index_shift):
                    overlap_error = True
            
            if overlap_error:
                raise ValueError(f"Series {series.SeriesIndex} has a z-location that is too far from the reference Series {self.list_of_series[0].SeriesIndex}. "
                                 f"Index shift {index_shift} exceeds the number of slices in the reference series.")

            # --- 4. Extract the first similar location ---
            # If the index shift is positive, we need to compare the first slice of the current series with the [index_shift] slice in the reference series.
            if index_shift >= 0:
                first_similar_location_ref = reference_z_loc[index_shift]
                first_similar_location_series = series.SliceLocations[0]    
            # If the index shift is negative, we need to compare the first slice of the reference series with the [index_shift] slice in the current series.
            else:
                first_similar_location_ref = reference_z_loc[0]
                first_similar_location_series = series.SliceLocations[abs(index_shift)]
            
            # --- 5. Check if the first similar location is within tolerance ---
            EPSILON = 1e-6  # Small value to avoid floating point precision issues.
            if not math.isclose(first_similar_location_ref, first_similar_location_series, abs_tol=self.z_tolerance + EPSILON):
                    raise ValueError(f"Series {series.SeriesIndex} has a z-location is out of tolerance with reference Series {self.list_of_series[0].SeriesIndex}. "
                                     f"First similar location {first_similar_location_series} is not within tolerance of {self.z_tolerance} mm from {first_similar_location_ref}.") 

            # --- 6. Store the index coordinates (using the coordinates of the reference series) ---
            # Store the index of the first slice in terms of the reference series.
            left_ind.append(index_shift)  # Append the index shift for the current series.
            # Store the coordinate of the last slice in terms of the reference series.
            right_ind.append(len(series.SliceLocations)+index_shift-1)

        # --- 7. Find the common overlapping region --- 
        # The common region on the left side, the maximum shifted left index.
        common_left = max(left_ind) 
        # The common region on the right side is defined by the minimum of the right indices.
        common_right = min(right_ind)
        # If the common left index is greater than the common right index, it means there is no overlap.
        if common_left > common_right:
            raise ValueError("No common overlapping region found across all series. "
                             "Check the z-locations and ensure they are within tolerance.")

        # --- 8. Store the SliceLocations for the cropped overlapping region ---
        self.SliceLocations = reference_z_loc[common_left:common_right + 1]
        
        # --- 9. Store the coordinates of the common overlapping region for each series (in terms of the current series) ---
        # These coordinates will be used to crop the slices so that the remainind slices are from the common overlapping region.
        self.crop_borders = [[common_left, common_right]]  # Initialize with the common region for the reference series,
        for i, series in enumerate(self.list_of_series[1:], start=1):
            # Calculate the start and end indices for the current series based on the common overlapping region.
            start_index = common_left - left_ind[i]
            end_index = common_right - left_ind[i]
            self.crop_borders.append([start_index, end_index])
        
        logger.debug(f"Aligned and checked SliceLocations with common region from {common_left} to {common_right}.")
    
    def _generate_total_mA_curve(self):
        """Calculates the cumulative mA curve for the super-series.

        This internal helper method iterates through all the `CTSeries` objects
        in the `list_of_series` and sums their individual `mA_curve`
        attributes on an element-wise basis. The result represents the total
        tube current (mA) at each slice position across all scans.

        This method assumes that the series have already been aligned and cropped
        by `_align_and_check_z_locations`, ensuring that the curves correspond
        to the same set of slices. It is typically called from `__post_init__`.

        Side Effects
        ------------
        - Initializes and populates `self.total_mA_curve` with a list
          of floats or ints representing the summed mA values.

        Raises
        ------
        ValueError
            If the `mA_curve` lists of the series have inconsistent lengths,
            which would indicate an alignment issue.

        Returns
        -------
        None
            This method modifies the instance in-place.
        """
        # Go through each series in the list and add the mA values to the total mA curve:
        # Initialize the total mA curve with zeros, with the same length as the number of slices in the super series.
        self.total_mA_curve = [0] * len(self.SliceLocations)

        for i, series in enumerate(self.list_of_series):
            # Get the cropped mA curve for the current series.
            mA_curve = series.mA_curve[self.crop_borders[i][0]:self.crop_borders[i][1] + 1]
            # Add the mA values to the total mA curve for the slices in the super series.
            self.total_mA_curve = [x + y for x, y in zip(self.total_mA_curve, mA_curve)]
            
    def _generate_total_ctdi_vol_curve(self):
        """Calculates the cumulative CTDIvol curve for the super-series.

        This internal helper method iterates through all the `CTSeries` objects
        in the `list_of_series` and sums their individual `ctdi_vol_curve`
        attributes on an element-wise basis. The result represents the total
        CTDIvol at each slice position across all scans.

        This method assumes that the series have already been aligned and cropped
        by `_align_and_check_z_locations`, ensuring that the curves correspond
        to the same set of slices. It is typically called from `__post_init__`.

        Side Effects
        ------------
        - Initializes and populates `self.total_ctdi_vol_curve` with a list
          of floats representing the summed CTDIvol values.

        Raises
        ------
        ValueError
            If the `ctdi_vol_curve` lists of the series have inconsistent lengths,
            which would indicate an alignment issue.

        Returns
        -------
        None
            This method modifies the instance in-place.
        """
        # Go through each series in the list and add the CTDI volume values to the total CTDI volume curve:
        # Initialize the total CTDI volume curve with zeros, with the same length as the number of slices in the super series.
        self.total_ctdi_vol_curve = [0] * len(self.SliceLocations)

        for i, series in enumerate(self.list_of_series):
            # Get the cropped CTDI volume curve for the current series.
            ctdi_vol_curve = series.ctdi_vol_curve[self.crop_borders[i][0]:self.crop_borders[i][1] + 1]
            # Add the CTDI volume values to the total CTDI volume curve for the slices in the super series.
            self.total_ctdi_vol_curve = [x + y for x, y in zip(self.total_ctdi_vol_curve, ctdi_vol_curve)]
    
    def generate_pixel_data_individual(self, path='Data'):
        """Loads or reads the pixel data for each series and stacks them.

        This method iterates through every `CTSeries` object in the super-series.
        For each series, it ensures the pixel data is loaded into memory. It
        first attempts to find and load a pre-existing .npy file from the
        specified `path`. If no stored data is found, it falls back to reading
        the pixel data from the original DICOM files.

        After loading, it assumes the data for each series has been aligned and
        cropped by the `_align_and_check_z_locations` method. It then stacks
        all the individual 3D pixel data arrays into a single 4D NumPy array.

        The resulting 4D array has the shape (height, width, n_slices, n_series).

        Parameters
        ----------
        path : str, optional
            The base directory to search for pre-stored .npy pixel data files,
            by default 'Data'.

        Side Effects
        ------------
        - Populates `self.pixel_data_individual` with a 4D NumPy array.
        - May trigger file I/O operations as it loads data for each series.
        - Modifies the state of individual `CTSeries` objects by loading their
          pixel data into memory.

        Raises
        ------
        FileNotFoundError
            If a series is expected to have stored data but the file cannot be
            found at the specified path.
        ValueError
            If a series has no pixel data and it cannot be read from DICOM files.

        Returns
        -------
        None
            This method modifies the instance in-place.
        """
        # Make sure all the series have pixel data available
        for series in self.list_of_series:
            # If the series does not have pixel data, try to find it.
            if not (series.has_pixel_data or series.pixel_data_stored):
                series.find_pixel_data(path=path)
            if series.pixel_data_stored and not series.has_pixel_data:
                series.load_stored_pixel_data(path=path)
            # If the pixel data is still not available, log a warning.
            if not series.has_pixel_data:
                series.read_pixel_data()
            if not series.has_pixel_data:
                logger.warning(f"Pixel data for series {series.SeriesIndex} is still not available.")                
                return

        first_series = self.list_of_series[0]
        height, width = first_series.pixel_data.shape[:2]
        
        # Initialize the pixel data array with zeros.
        self.pixel_data_individual = np.zeros((height, width, len(self.SliceLocations), len(self.list_of_series)), dtype=first_series.pixel_data.dtype)

        for i, series in enumerate(self.list_of_series):
            # Get the cropped pixel data for the current series.
            cropped_pixel_data = series.pixel_data[:, :, self.crop_borders[i][0]:self.crop_borders[i][1] + 1]
            # Add the pixel data to the individual pixel data array.
            self.pixel_data_individual[:, :, :, i] = cropped_pixel_data
        logger.debug("Pixel data for individual series generated successfully.")
    
    def generate_mean_image(self, path='Data'):
        """Calculates the mean image from all individual series.

        This method computes the mean on a pixel-by-pixel basis across all the
        aligned CT series contained in `self.pixel_data_individual`. The
        resulting 3D NumPy array represents the average of all the scans.

        This method requires `self.pixel_data_individual` to be populated first,
        typically by calling `generate_pixel_data_individual()`.

        Side Effects
        ------------
        - Populates `self.pixel_data_super_series` with a 3D NumPy array
          containing the mean image.

        Raises
        ------
        ValueError
            If `self.pixel_data_individual` has not been generated yet.

        Returns
        -------
        None
            This method modifies the instance in-place.
        """
        if self.pixel_data_individual is None:
            logger.error("Pixel data for individual series has not been generated. Calling generate_pixel_data_individual() first.")
            self.generate_pixel_data_individual(path=path)
        # Average the pixel data across the series dimension (axis 3).
        self.pixel_data_super_series = np.mean(self.pixel_data_individual, axis=3)
        logger.debug("Pixel data for super series generated successfully.")

    def generate_std_image(self, path='Data'):
        """Calculates the standard deviation image from all individual series.

        This method computes the standard deviation on a pixel-by-pixel basis
        across all the aligned CT series contained in `self.pixel_data_individual`.
        The resulting 3D NumPy array represents the variability between the scans,
        where brighter pixels indicate higher standard deviation.

        This method requires `self.pixel_data_individual` to be populated first,
        typically by calling `generate_pixel_data_individual()`.

        Side Effects
        ------------
        - Populates `self.pixel_data_std_series` with a 3D NumPy array containing
          the standard deviation image.

        Raises
        ------
        ValueError
            If `self.pixel_data_individual` has not been generated yet.

        Returns
        -------
        None
            This method modifies the instance in-place.
        """
        if self.pixel_data_individual is None:
            logger.error("Pixel data for individual series has not been generated. Calling generate_pixel_data_individual() first.")
            self.generate_pixel_data_individual(path=path)
        # Calculate the standard deviation across the series dimension (axis 3).
        self.pixel_data_std_series = np.std(self.pixel_data_individual, axis=3)
        logger.debug("Standard deviation image for super series generated successfully.")