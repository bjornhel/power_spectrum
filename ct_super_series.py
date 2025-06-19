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
    """
    A class to hold series of CT scans with similar characteristics in order
    to make superimages.
    The input is a list of CT series.
    """
    list_of_series: list
    z_tolerance: float = 0.0            # Tolerance for SliceLocations consistency check
    accept_difference_positioning: bool = False  # Whether to accept different scan positioning and/or length
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
        """
        Post-initialization to check if CTSeries is a list.
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
        """
        Verify that all CT series in the list have the same value for a given attribute.

        Parameters
        ----------
        attribute_names : list[str]
            The names of all the attributes to check.
            Every CT series in the list must have the same value for these attributes.

        Raises
        ------
        ValueError
            If the attribute values are not consistent across all series.
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
        """Aligns series z-locations, verifies tolerance, and crops to a common region.

        This method aligns all series within the `list_of_series` based on their
        z-axis positions. It uses the first series as the reference coordinate
        system and has two modes of operation controlled by the
        `accept_difference_positioning` instance attribute.

        In strict mode (`accept_difference_positioning=False`):
        - It verifies that every series has a `z_location` list that is
          identical to the reference series.

        In flexible mode (`accept_difference_positioning=True`):
        1.  Calculates the slice increment (distance between slices) from the
            reference series.
        2.  For each subsequent series, it determines the z-offset of its first
            slice relative to the reference's first slice.
        3.  Calculates an integer index shift based on the offset and increment.
        4.  Checks if the index shift results in a valid overlap with the
            reference series.
        5.  Verifies that the z-locations of all aligned slices are within
            `self.z_tolerance` of their corresponding reference slice.
        6.  Crops the data to the common region where all series have valid,
            overlapping slice data.

        This method is called from `__post_init__` to ensure the super-series
        is spatially coherent before further processing.

        Side Effects
        ------------
        - Updates `self.z_location` to a new list containing only the z-locations
          of the common, overlapping slices after alignment and cropping.
        - Updates `self.crop_borders` to reflect the indices of the slices
          included in the super-series for each CT series.

        Raises
        ------
        ValueError
            - If the reference series has fewer than 2 slices (cannot determine
              slice increment).
            - In strict mode, if any series' z-locations do not match the
              reference exactly.
            - In flexible mode, if any aligned slice's z-position deviates from
              the reference by more than `self.z_tolerance`.
            - In flexible mode, if there is no common overlapping region across
              all series after alignment.
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

        
        # --- 2. Calculate Increment ---
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
            # --- 3. Calculate Z-Offset ---
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
            
            # --- 4. Test of there is sufficient overlap with the reference series ---
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

            # --- 5. Extract the first similar location ---
            # If the index shift is positive, we need to compare the first slice of the current series with the [index_shift] slice in the reference series.
            if index_shift >= 0:
                first_similar_location_ref = reference_z_loc[index_shift]
                first_similar_location_series = series.SliceLocations[0]    
            # If the index shift is negative, we need to compare the first slice of the reference series with the [index_shift] slice in the current series.
            else:
                first_similar_location_ref = reference_z_loc[0]
                first_similar_location_series = series.SliceLocations[abs(index_shift)]
            
            # --- 6. Check if the first similar location is within tolerance ---
            EPSILON = 1e-6  # Small value to avoid floating point precision issues.
            if not math.isclose(first_similar_location_ref, first_similar_location_series, abs_tol=self.z_tolerance + EPSILON):
                    raise ValueError(f"Series {series.SeriesIndex} has a z-location is out of tolerance with reference Series {self.list_of_series[0].SeriesIndex}. "
                                     f"First similar location {first_similar_location_series} is not within tolerance of {self.z_tolerance} mm from {first_similar_location_ref}.") 

            # Store the index of the first slice in terms of the reference series.
            left_ind.append(index_shift)  # Append the index shift for the current series.
            # Store the coordinate of the last slice in terms of the reference series.
            right_ind.append(len(series.SliceLocations)+index_shift-1)

        # Find the common overlapping region across all series.
        # The common region on the left side, the maximum shifted left index.
        common_left = max(left_ind) 
        # The common region on the right side is defined by the minimum of the right indices.
        common_right = min(right_ind)
        # If the common left index is greater than the common right index, it means there is no overlap.
        if common_left > common_right:
            raise ValueError("No common overlapping region found across all series. "
                             "Check the z-locations and ensure they are within tolerance.")

        # Store the slice locations of the cropped reference series:
        self.SliceLocations = reference_z_loc[common_left:common_right + 1]
        
        self.crop_borders = [[common_left, common_right]]  # Initialize with the common region for the reference series,
        for i, series in enumerate(self.list_of_series[1:], start=1):
            # Calculate the start and end indices for the current series based on the common overlapping region.
            start_index = common_left - left_ind[i]
            end_index = common_right - left_ind[i]
            self.crop_borders.append([start_index, end_index])
        
        logger.debug(f"Aligned and checked SliceLocations with common region from {common_left} to {common_right}.")
    
    def _generate_total_mA_curve(self):
        """
        Generate the total mA curve for the super series based on the individual series.
        The total mA curve is a list of the total mA for each slice in the super series.
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
        """
        Generate the total CTDI volume curve for the super series based on the individual series.
        The total CTDI volume curve is a list of the total CTDI volume for each slice in the super series.
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
        """
        Generate the pixel data for each series in the super series.
        The pixel data is a 4D array with shape (height, width, n_slices, n_series).
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
        """
        Generate the pixel data for the super series.
        The pixel data is a 3D array with shape (height, width, n_slices).
        """
        if self.pixel_data_individual is None:
            logger.error("Pixel data for individual series has not been generated. Calling generate_pixel_data_individual() first.")
            self.generate_pixel_data_individual(path=path)
        # Average the pixel data across the series dimension (axis 3).
        self.pixel_data_super_series = np.mean(self.pixel_data_individual, axis=3)
        logger.debug("Pixel data for super series generated successfully.")

    def generate_std_image(self, path='Data'):
        """
        Generate the standard deviation image for the super series.
        The pixel data is a 3D array with shape (height, width, n_slices).
        """
        if self.pixel_data_individual is None:
            logger.error("Pixel data for individual series has not been generated. Calling generate_pixel_data_individual() first.")
            self.generate_pixel_data_individual(path=path)
        # Calculate the standard deviation across the series dimension (axis 3).
        self.pixel_data_std_series = np.std(self.pixel_data_individual, axis=3)
        logger.debug("Standard deviation image for super series generated successfully.")