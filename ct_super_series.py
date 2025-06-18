from dataclasses import dataclass

import logging
# Set up logger for this module
if __name__ == "__main__":
    logger = logging.getLogger('ct_series')
else:
    logger = logging.getLogger(__name__)


@dataclass
class CTSuperSeries(z_tolerance=0):
    """
    A class to hold series of CT scans with similar characteristics in order
    to make superimages.
    The input is a list of CT series.
    """
    list_of_series: list
    z_location: list = None
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

    def __post_init__(self):
        """
        Post-initialization to check if CTSeries is a list.
        """
        if not isinstance(self.CTSeries, list):
            raise TypeError("CTSeries must be a list of CT series.")
        logger.debug(f"CTSuperSeries initialized with {len(self.CTSeries)} CT series.")
        list_of_tags = ['KVP', 'ConvolutionKernel', 'IterativeAILevel',
                             'MatrixSize', 'SliceThickness', 'SingleCollimationWidth',
                             'TotalCollimationWidth', 'TableSpeed', 'TableFeedPerRotation',
                             'SpiralPitchFactor', 'ReconstructionDiameter',
                             'DataCollectionDiameter', 'FocalSpots', 'BodyPartExamined',
                             'ProtocolName', 'StationName', 'Manufacturer',
                             'ManufacturerModelName', 'SoftwareVersions']
        self._verify_attribute_consistency(list_of_tags)    

    def _verify_attribute_consistency(self, attribute_name: list[str]):
        """
        Verify that all CT series in the list have the same value for a given attribute.

        Parameters
        ----------
        attribute_name : list[str]
            The name of the attribute to check on each CTSeries object (e.g., "kvp").


        Raises
        ------
        ValueError
            If the attribute values are not consistent across all series.
        AttributeError
            If any series object does not have the specified `attribute_name`.
        """
        try:
            # Get all unique values for the specified attribute
            value_set = {getattr(series, attribute_name) for series in self.list_of_series}
        except AttributeError:
            logger.error(f"Attribute '{attribute_name}' not found in one or more CT series objects.")
            raise  # Re-raise the AttributeError

        if len(value_set) > 1:
            raise ValueError(
                f"CT series have different {attribute_name} values: {value_set}"
            )
        
        single_value = next(iter(value_set)) if value_set else "N/A"
        logger.debug(f"All CT series have the same: {single_value}")
    
    def _verify_z_location(self, z_tolerance = 0):
        """
        Verify that all CT series in the list have the same z_location within a specified tolerance.

        Parameters
        ----------
        z_tolerance : float, optional
            The tolerance for z_location comparison (default is 0).

        Raises
        ------
        ValueError
            If the z_locations are not consistent within the specified tolerance.
        """
        if self.z_location is None:
            logger.warning("No z_location provided for CT series.")
            return
        
        if len(self.z_location) < 2:
            logger.debug("Not enough z_locations to verify consistency.")
            return
        
        first_z = self.z_location[0]
        for z in self.z_location[1:]:
            if abs(first_z - z) > z_tolerance:
                raise ValueError(f"Z locations are not consistent within tolerance {z_tolerance}: {self.z_location}")
        
        logger.debug(f"All z locations are consistent within tolerance {z_tolerance}.")
        
    

    
