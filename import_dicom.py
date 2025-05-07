"""DICOM metadata extraction and CT image filtering.

This module provides functionality to scan directories for DICOM files, filter for
axial CT images, and extract standardized metadata into structured datasets. It supports
identification of various DICOM types and extraction of both common and vendor-specific tags.

Functions:
    scan_for_axial_ct_dicom_files: Scan directories recursively for axial CT DICOM files
    read_data: Simplified interface to scan for DICOM metadata
    main: Execute standard DICOM processing workflow
    
Private Functions:
    _filter_ct_images: Filter DICOM files by type
    _get_dicom_metadata_tag: Safely retrieve values from DICOM datasets
    _read_metadata: Read DICOM metadata without loading pixel data
    _convert_datatypes: Convert DataFrame columns to appropriate data types
    _extract_axial_ct_metadata: Extract standardized metadata from DICOM datasets
    
Dependencies:
    pydicom: For reading and manipulating DICOM files
    pandas: For data manipulation and storage
    logging: For activity logging
    project_data: For organizing extracted metadata
"""

import os
import pydicom as dcm
import pandas as pd
import numpy as np
from project_data import ProjectData

from logging_config import configure_module_logging
import logging

if __name__ == "__main__":
    logger = logging.getLogger('import_dicom')
else:
    logger = logging.getLogger(__name__)


def _filter_ct_images(ds: dcm.Dataset , fp: str, axial=True, dicomdir=False, rdsr=False, doserep=False, localizer=False) -> bool:
    """Filter DICOM files by type, returning True only for files matching requested criteria.
    
    This function examines DICOM attributes to determine the type of DICOM file and
    filters them based on the provided parameters. It checks for specific attributes
    like FileSetID and SOPClassUID to identify different DICOM file types.
    
    Parameters
    ----------
    ds : dcm.Dataset
        The DICOM dataset object to evaluate.
    fp : str
        File path of the DICOM file (used for logging purposes).
    axial : bool, default=True
        Include axial CT images.
    dicomdir : bool, default=False
        Include DICOMDIR files.
    rdsr : bool, default=False
        Include Radiation Dose Structured Reports.
    doserep : bool, default=False
        Include Dose Report images (Secondary Capture objects).
    localizer : bool, default=False
        Include CT localizer images.
    
    Returns
    -------
    bool
        True if the file should be included based on the specified criteria,
        False otherwise.
        
    Notes
    -----
    - Checks for the following DICOM file types:
      * DICOMDIR: Files with the FileSetID attribute
      * RDSR: Files with SOPClassUID = "1.2.840.10008.5.1.4.1.1.88.67"
      * Dose Reports: Files with SOPClassUID = "1.2.840.10008.5.1.4.1.1.7"
      * CT Localizers: Files with SOPClassUID = "1.2.840.10008.5.1.4.1.1.2" and ImageType[2] = "LOCALIZER"
      * Axial CT Images: Files with SOPClassUID = "1.2.840.10008.5.1.4.1.1.2" and ImageType[2] = "AXIAL"
    - Logs detailed information about excluded files for debugging purposes.
    """
    
    # Check for DICOMDIR files
    if hasattr(ds, "FileSetID"):
        if dicomdir:
            return True
        else:
            logger.info(f"Dicomdir excluded: {fp}. Has the FileSetID tag: {ds.FileSetID}")
            return False

    # Check for Radiation Dose Structured Report (RDSR):
    if hasattr(ds, "SOPClassUID") and (ds.SOPClassUID == "1.2.840.10008.5.1.4.1.1.88.67"):
        if rdsr:
            return True
        else:
            logger.info(f"RDSR excluded: {fp}. Has the SOPClassUID tag: {ds.SOPClassUID}")
            return False
    
    # Check for Dose Report images (Secondary Capture):
    if hasattr(ds, "SOPClassUID") and (ds.SOPClassUID == "1.2.840.10008.5.1.4.1.1.7"):
        if doserep:
            return True
        else:
            logger.info(f"Dose Report (Secondary Capture) excluded: {fp} has the SOPClassUID tag: {ds.SOPClassUID}")
            return False
    
    # Check for CT localizer images:
    if (hasattr(ds, "SOPClassUID") and 
        (ds.SOPClassUID == "1.2.840.10008.5.1.4.1.1.2") and
        hasattr(ds, "ImageType") and
        (ds.ImageType[2] == "LOCALIZER")):
        if localizer:
            return True
        else:
            logger.info(f"Localizer excluded: {fp}. Has the SOPClassUID tag: {ds.SOPClassUID}  and ImageType tag: {ds.ImageType[2]}")
            return False
    
    # Check for axial CT images:
    if (hasattr(ds, "SOPClassUID") and 
        (ds.SOPClassUID == "1.2.840.10008.5.1.4.1.1.2") and
        hasattr(ds, "ImageType") and
        (ds.ImageType[2] == "AXIAL")):
        if axial:
            return True
        else:
            logger.info(f"Axial CT image excluded: {fp} has the SOPClassUID tag: {ds.SOPClassUID}")
            return False
    else:
        logger.info(f"Image {fp} is not cought by any filters")
        if hasattr(ds, "SOPClassUID"):
            logger.info(f"SOPClassUID: {ds.SOPClassUID}")
        if hasattr(ds, "ImageType"):
            logger.info(f"ImageType: {ds.ImageType}")
        return False

def _get_dicom_metadata_tag(ds: dcm.Dataset, tag: str, position=None) -> str:
    """
    Retrieve a value from a DICOM dataset by tag name or hex identifier.
    
    This function safely accesses a DICOM tag by name or hex string representation
    and handles indexed values for sequence-like tags. Type checking ensures valid
    inputs, and appropriate error logging provides visibility into access issues.
    
    Parameters
    ----------
    ds : dcm.Dataset
        The DICOM dataset to extract the tag from.
    tag : str
        The name of the DICOM tag to retrieve (e.g., 'PatientName') or 
        a hex string for private tags (e.g., '0x00531042').
    position : int, optional
        If provided and the tag contains a sequence-like value (list, tuple, array),
        return the element at this position. If the tag value is a string or bytes,
        the entire value is returned regardless of this parameter.
    
    Returns
    -------
    object or None
        The value of the requested tag if found, or a specific element if
        position is provided and applicable. Returns None if the tag doesn't exist,
        position is out of range, or there are any errors in accessing the value.
    
    Notes
    -----
    - Private tags must be specified with a '0x' prefix followed by the tag number
      as an 8-digit hex string (e.g., '0x00531042' for tag (0053,1042)).
    - Strings and bytes are treated as atomic values, not as sequences to be indexed,
      even if position is specified.
    - For sequence-like values (lists, tuples, arrays), the function will return
      the element at the specified position if valid.
    - Type errors and access errors are logged with appropriate severity levels.
    
    Examples
    --------
    >>> # Get entire PatientName
    >>> patient_name = _get_dicom_metadata_tag(ds, 'PatientName')
    >>> 
    >>> # Get first element of ImageType
    >>> image_type_first = _get_dicom_metadata_tag(ds, 'ImageType', 0)
    >>> 
    >>> # Get reconstruction kernel
    >>> kernel = _get_dicom_metadata_tag(ds, 'ConvolutionKernel', 0)
    >>>
    >>> # Get Siemens private tag for iterative strength
    >>> strength = _get_dicom_metadata_tag(ds, '0x00531042')
    """
    # Type checking
    if not isinstance(ds, dcm.Dataset):
        logger.error(f"Invalid dataset type: {type(ds)}. Expected pydicom.Dataset.")
        return None
        
    if not isinstance(tag, str):
        logger.error(f"Invalid tag type: {type(tag)}. Expected string.")
        return None
        
    if position is not None and not isinstance(position, int):
        logger.error(f"Invalid position type: {type(position)}. Expected int or None.")
        return None

    try:
        # If the first two characters in the tag are '0x', treat it as a private tag:
        if tag.startswith("0x"):
            try:
                return ds[tag].value
            except KeyError:
                logger.warning(f"Private tag {tag} not found in dataset.")
                return None
        
        value = getattr(ds, tag, None)
        if value is None:
            logger.warning(f"Tag {tag} not found in dataset.")
            return None

        # If the position is 0 and the value is a string, return the value directly
        if position == 0 and isinstance(value, str):
            return value

        # If position is specified and value is a sequence-like object
        if position is not None and hasattr(value, "__getitem__") and not isinstance(value, (str, bytes)):
            try:
                # Try to access the element at the specified position
                if len(value) > position:
                    return value[position]
                else:
                    logger.warning(f"Position {position} out of range for tag {tag} with length {len(value)}")
                    return None
            except (TypeError, IndexError) as e:
                logger.warning(f"Could not access position {position} in {tag}: {str(e)}")
                return None
        return value

    except Exception as e:
        logger.error(f"Error accessing tag {tag}: {str(e)}")
        return None

def _read_metadata(fp: str) -> dcm.Dataset:
    """Read DICOM metadata from a file without loading pixel data.
    
    This function attempts to read a file as DICOM, using PyDicom's stop_before_pixels
    option to optimize memory usage when only metadata is needed. If the file cannot
    be read as a DICOM file, the function logs a warning and returns None.
    
    Parameters
    ----------
    fp : str
        File path to the potential DICOM file to read.
    
    Returns
    -------
    dcm.Dataset or None
        A PyDicom dataset containing the file's metadata if successfully read,
        or None if the file is not a valid DICOM file.
    
    Notes
    -----
    - Uses stop_before_pixels=True to avoid loading potentially large pixel data
    - Logs warnings for files that cannot be read as DICOM
    - Any exception during reading is caught and results in None being returned
    """
    # Attempt tp read the DICOM file and return the dataset
    try:
        ds = dcm.dcmread(fp, stop_before_pixels=True)
        return ds
    except Exception as e:
        logger.warning(f"Non DICOM file excluded: {fp}, error using dcmread: {str(e)}")
        return None

def _convert_datatypes(df) -> pd.DataFrame:
    """Convert DataFrame columns to appropriate data types based on content.
    
    This function transforms columns in the DICOM metadata DataFrame to their 
    appropriate data types, including numeric values, dates, and times. It handles
    potential missing columns gracefully and uses pandas' type conversion functions
    with error handling.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing DICOM metadata with columns to be converted.
        If empty, the function returns it unchanged.
    
    Returns
    -------
    pd.DataFrame
        The input DataFrame with columns converted to appropriate types.
        
    Notes
    -----
    - Float columns: Includes measurements and physical parameters
    - Integer columns: Includes counters and discrete values
    - Date columns: Converts from YYYYMMDD format to datetime objects
    - Time columns: Converts from HHMMSS.FFFFFF format to time objects
    - Creates a combined 'study_datetime' column when both date and time are available
    - Uses coercion to handle conversion errors gracefully
    - Logs warnings for columns that weren't converted or explicitly ignored
    """
    if df.empty:
        return df

    # Numerical columns that should be converted to float
    float_columns = [
        'slice_location',  'slice_thickness', 'data_collection_diameter', 'reconstruction_diameter', 
        'gantry_tilt', 'table_height', 'distance_source_to_detector', 'distance_source_to_patient',
        'focal_spot', 'exposure_time', 'exposure', 'pixel_spacing', 'detector_element_size', 'total_collimation_width', 
        'table_speed', 'table_feed_per_rotation', 'spiral_pitch_factor', 'tube_current', 'ctdi_vol']
    
    
    int_columns = ['instance_number', 'kvp','generator_power']

    # Date columns that should be converted to datetime
    date_columns = [
        'study_date', 'series_date', 'acquition_date', 'content_date',
        'last_calibration_date']
    
    # Time columns
    time_columns = [
        'study_time', 'series_time', 'acquisition_time', 'content_time',
        'last_calibration_time']
    
    ignored_columns = ['filepath', 'directory', 'filename', 
                       'study_uid','series_uid', 'sop_uid', 
                       'modality', 'station_name', 'manufacturer', 'model', 
                       'device_serial_number', 'software_version', 
                       'study_description', 'series_description', 
                       'body_part', 'protocol_name', 
                       'rotation_direction', 'filter_type', 'convolution_kernel', 
                       'dose_modulation_type', 'image_type', 
                       'ADMIRE_level', 
                       'study_datetime']
    
    # Convert float columns
    for col in float_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert int columns
    for col in int_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

    # Convert date columns (format YYYYMMDD)
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format='%Y%m%d', errors='coerce')
    
    # Convert time columns (format HHMMSS.FFFFFF)
    for col in time_columns:
        if col in df.columns:
            # First convert to string if not already
            df[col] = df[col].astype(str)
            
            # Handle times with decimal fractions
            df[col] = df[col].apply(lambda x: 
                pd.to_datetime(
                    x.split('.')[0].zfill(6), 
                    format='%H%M%S',
                    errors='coerce'
                ).time() if pd.notna(x) and x != 'None' else None
            )
    
    # Convert combined date and time columns
    if 'study_date' in df.columns and 'study_time' in df.columns:
        df['study_datetime'] = pd.to_datetime(
            df['study_date'].dt.strftime('%Y-%m-%d') + ' ' + 
            df['study_time'].apply(lambda x: str(x) if pd.notna(x) else ''),
            errors='coerce'
        )
    
    # Log the columns not converted
    all_columns = df.columns.tolist()
    # remove all columns that are not in any lists
    remaining_columns = [col for col in all_columns if col not in float_columns + int_columns + date_columns + time_columns + ignored_columns]
    if remaining_columns:
        logger.warning(f"Columns not converted or explicitly ignored, please investigate: {remaining_columns}")
    else:
        logger.info("All columns converted or explicitly ignored")
    return df

def _extract_axial_ct_metadata(ds: dcm.Dataset) -> dict:
    """Extract key metadata from a DICOM dataset for axial CT images.
    
    This function uses a tag mapping approach to systematically extract relevant 
    metadata from a DICOM dataset. It supports both standard and vendor-specific
    tags, handling extraction errors gracefully to ensure the process continues
    even if some tags are missing.
    
    Parameters
    ----------
    ds : dcm.Dataset
        The DICOM dataset from which to extract metadata.
    
    Returns
    -------
    dict or None
        A dictionary containing extracted DICOM metadata with standardized keys,
        or None if a critical error occurs during extraction.
        
    Notes
    -----
    - Extracts common identifiers, timing information, series details, and technical parameters
    - Groups tags by category (identifiers, timing, geometric, technical)
    - Handles vendor-specific tags for Siemens (ADMIRE level) and GE (DLIR level)
    - Logs debug information for individual tag extraction failures
    - Handles errors gracefully to preserve as much metadata as possible
    """
    # Define tag mappings (output_key: (dicom_tag, position))
    tag_mappings = {
        # Common identifiers
        'study_uid': ('StudyInstanceUID', None),
        'series_uid': ('SeriesInstanceUID', None),
        'sop_uid': ('SOPInstanceUID', None),
        'modality': ('Modality', None),
        'station_name': ('StationName', None),
        'manufacturer': ('Manufacturer', None),
        'model': ('ManufacturerModelName', None),
        'device_serial_number': ('DeviceSerialNumber', None),
        'software_version': ('SoftwareVersions', None),
        'last_calibration_date': ('DateOfLastCalibration', None),
        'last_calibration_time': ('TimeOfLastCalibration', None),

        # Timing information
        'study_date': ('StudyDate', None),
        'study_time': ('StudyTime', None),
        'series_date': ('SeriesDate', None),
        'series_time': ('SeriesTime', None),
        'acquition_date': ('AcquisitionDate', None),
        'acquisition_time': ('AcquisitionTime', None),
        'content_date': ('ContentDate', None),
        'content_time': ('ContentTime', None),

        # Information relevant to the series
        'study_description': ('StudyDescription', None),
        'series_description': ('SeriesDescription', None),
        'body_part': ('BodyPartExamined', None),
        'protocol_name': ('ProtocolName', None),

        # Geometric information
        'distance_source_to_detector': ('DistanceSourceToDetector', None),
        'distance_source_to_patient': ('DistanceSourceToPatient', None),
        'gantry_tilt': ('GantryDetectorTilt', None),
        'table_height': ('TableHeight', None),
        'rotation_direction': ('RotationDirection', None),

        # Technical parameters relevant for the series
        'kvp': ('KVP', None),
        'filter_type': ('FilterType', None),
        'slice_thickness': ('SliceThickness', None),
        'data_collection_diameter': ('DataCollectionDiameter', None),
        'reconstruction_diameter': ('ReconstructionDiameter', None),
        'pixel_spacing': ('PixelSpacing', 0),
        'generator_power': ('GeneratorPower', None),
        'focal_spot': ('FocalSpots', 0),
        'exposure_time': ('ExposureTime', None),
        'convolution_kernel': ('ConvolutionKernel', 0),
        'detector_element_size': ('SingleCollimationWidth', None),
        'total_collimation_width': ('TotalCollimationWidth', None),
        'table_speed': ('TableSpeed', None),
        'table_feed_per_rotation': ('TableFeedPerRotation', None),
        'spiral_pitch_factor': ('SpiralPitchFactor', None),
        'dose_modulation_type': ('ExposureModulationType', None),

        # Technical parameters relevant for the image
        'tube_current': ('XRayTubeCurrent', None),
        'exposure': ('Exposure', None),
        'ctdi_vol': ('CTDIvol', None),
        
        # Positional information
        'instance_number': ('InstanceNumber', None),
        'slice_location': ('SliceLocation', None),
    }

    # Extract all metadata using the mappings
    file_info = {}
    
    try:
        # Extract basic tags
        for output_key, (tag_name, position) in tag_mappings.items():
            try:
                file_info[output_key] = _get_dicom_metadata_tag(ds, tag_name, position)
            except Exception as e:
                logger.debug(f"Error extracting {output_key} ({tag_name}): {str(e)}")
                # Continue with other tags
        
        # Special case: image_type
        try:
            file_info['image_type'] = tuple(getattr(ds, 'ImageType', []))
        except Exception as e:
            logger.debug(f"Error extracting image_type: {str(e)}")
            file_info['image_type'] = tuple()
        
        # Vendor-specific tags based on manufacturer
        try:
            if file_info.get('manufacturer') == 'SIEMENS':
                file_info['ADMIRE_level'] = _get_dicom_metadata_tag(ds, 'ConvolutionKernel', 1)
            elif file_info.get('manufacturer') == 'GE MEDICAL SYSTEMS':
                file_info['DLIR_level'] = _get_dicom_metadata_tag(ds, '0x00531042')
        except Exception as e:
            logger.debug(f"Error extracting vendor-specific tags: {str(e)}")
    
    except Exception as e:
        logger.warning(f"Error extracting metadata: {str(e)}")
        return None
    
    return file_info   
 
def scan_for_axial_ct_dicom_files(root_dir: str) -> pd.DataFrame:
    """Scan a directory recursively for axial CT DICOM files and extract their metadata.
    
    This function traverses a directory tree to find and process DICOM files
    containing axial CT images. For each valid file, it extracts metadata using
    helper functions, builds a structured dataset, and returns the results in a
    sorted DataFrame.
    
    Parameters
    ----------
    root_dir : str
        Path to the root directory to scan for DICOM files
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing metadata from all valid axial CT DICOM files,
        with one row per file. Empty DataFrame if no valid files are found.
        
    Notes
    -----
    - Uses _read_metadata() to read DICOM files without loading pixel data
    - Uses _filter_ct_images() to identify and select axial CT images
    - Uses _extract_axial_ct_metadata() to extract standardized metadata
    - Uses _convert_datatypes() to convert columns to appropriate data types
    - Adds filepath, directory, and filename columns to track file origins
    - Sorts results by study_date, study_time, series_uid, and slice_location
    - Returns empty DataFrame if no valid DICOM files found
    """
    # List to hold dictionaries of file metadata.
    file_data = []
    
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            ds = _read_metadata(file_path)
            file_info = {'filepath': file_path}
            file_info['directory'] = dirpath
            file_info['filename'] = filename

            if ds is None:
                continue
                
            include = _filter_ct_images(ds, file_path, axial=True)  
            if not include:
                continue

            dicom_info = _extract_axial_ct_metadata(ds)
            if dicom_info is not None:
                dicom_info = file_info | dicom_info
                file_data.append(dicom_info)
    
    # Create DataFrame
    if not file_data:
        return pd.DataFrame()
    logger.info(f"Found {len(file_data)} DICOM files with valid metadata") 
    df = pd.DataFrame(file_data)

    df = _convert_datatypes(df)

    # Sort the Dataframe by series UID then by time then by z-axis
    df.sort_values(by=['study_date', 'study_time', 'series_uid', 'slice_location'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df

def read_data(root_dir: str) -> pd.DataFrame:
    """Read DICOM metadata from the specified directory.
    
    This function scans a directory tree for axial CT DICOM files and extracts
    their metadata into a structured DataFrame. It serves as a convenient wrapper
    around the more detailed scan_for_axial_ct_dicom_files function.
    
    Parameters
    ----------
    root_dir : str
        Path to the root directory to scan for DICOM files
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing metadata from all valid axial CT DICOM files.
        Returns an empty DataFrame if no valid DICOM files are found.
        
    Notes
    -----
    - Logs information about the scan process
    - Delegates to scan_for_axial_ct_dicom_files for the actual scanning
    - Provides appropriate warning messages when no files are found
    """
    logger.info(f"Reading metadata from DICOM files in {root_dir}")
    file_df = scan_for_axial_ct_dicom_files(root_dir)
    if file_df.empty:
        logger.warning(f"No DICOM files found in {root_dir}")
        return pd.DataFrame()
    return file_df

def main():
    """Execute the main workflow for DICOM processing.
    
    This function demonstrates the standard workflow for DICOM data processing:
    1. Reads DICOM metadata from a specified directory
    2. Creates a ProjectData object to organize the extracted metadata
    3. Returns the DataFrame for further analysis or inspection
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing metadata from all valid axial CT DICOM files.
        
    Notes
    -----
    - Uses a hardcoded directory path for DICOM file scanning
    - Creates a ProjectData object named 'Fantomscan'
    - Contains commented-out code for saving to CSV and processing by series_uid
    - Primarily intended for demonstration and debugging purposes
    """
    root_dir = r"/home/bhosteras/Kode/power_spectrum/Fantomscan/"  # Replace with your root directory
    # Step 1: Scan the files into a dataframe
    file_df = read_data(root_dir)

    # Save the dataframe to a CSV file
    # file_df.to_csv('dicom_metadata.csv', index=False)

    # Step 2: Create a ProjectData object
    project_data = ProjectData('Fantomscan', file_df)

    # Step 3: Add each series with a unique series_instance_uid
    # for _, group in file_df.groupby('series_uid'):
    #     project_data.add_series(group)

    # Add a stop to investigate the objects:
    return file_df
     
if __name__ == "__main__":
    """Execute the DICOM processing workflow when run as a script.
    
    This block configures logging for all related modules and runs the main 
    function to process DICOM files. The resulting DataFrame is stored in 
    the 'df' variable for interactive inspection or further processing.
    
    Configuration includes:
    - import_dicom: DEBUG level logging to both file and console
    - project_data: DEBUG level logging to both file and console
    - ct_series: DEBUG level logging to both file and console
    
    Notes
    -----
    - Each module gets its own log file with the module name
    - Console output is enabled for interactive monitoring
    - The main() function's return value is stored in the df variable
    - This setup is intended for direct script execution, not for imports
    """
    # Setup logging
    configure_module_logging({
        'import_dicom': {'file': 'import_dicom.log', 'level': logging.DEBUG, 'console': True},
        'project_data': {'file': 'project_data.log', 'level': logging.DEBUG, 'console': True},
        'ct_series':    {'file': 'ct_series.log',    'level': logging.DEBUG, 'console': True}}
    )
    df = main()