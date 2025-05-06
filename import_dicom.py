# a module to read DICOM metadata and filter axial CT images
# -*- coding: utf-8 -*-
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


# TODO: Make this module a little more general by allowing the user to specify whether to import localizer images
#       RDSR files and dose report images.
#       Also add functionality to import other modalities such as Interventional Radiology, etc.



def _filter_ct_images(ds: dcm.Dataset , fp: str, axial=True, dicomdir=False, rdsr=False, doserep=False, localizer=False) -> bool:
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
    # Attempt tp read the DICOM file and return the dataset
    try:
        ds = dcm.dcmread(fp, stop_before_pixels=True)
        return ds
    except Exception as e:
        logger.warning(f"Non DICOM file excluded: {fp}, error using dcmread: {str(e)}")
        return None

def _convert_datatypes(df) -> pd.DataFrame:
    """Convert DataFrame columns to appropriate data types."""
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
    
    return df

def _extract_axial_ct_metadata(ds: dcm.Dataset) -> dict:
    # Extract key metadata for sorting and grouping
    try:
        
        file_info = {}
        # Common identifiers
        file_info['study_uid'] = _get_dicom_metadata_tag(ds, 'StudyInstanceUID')
        file_info['series_uid'] = _get_dicom_metadata_tag(ds, 'SeriesInstanceUID')
        file_info['sop_uid'] = _get_dicom_metadata_tag(ds, 'SOPInstanceUID')
        file_info['modality'] = _get_dicom_metadata_tag(ds, 'Modality')
        file_info['station_name'] = _get_dicom_metadata_tag(ds, 'StationName')
        file_info['manufacturer'] = _get_dicom_metadata_tag(ds, 'Manufacturer')
        file_info['model'] = _get_dicom_metadata_tag(ds, 'ManufacturerModelName')
        file_info['device_serial_number'] = _get_dicom_metadata_tag(ds, 'DeviceSerialNumber')
        file_info['software_version'] = _get_dicom_metadata_tag(ds, 'SoftwareVersions')
        file_info['last_calibration_date'] = _get_dicom_metadata_tag(ds, 'DateOfLastCalibration')
        file_info['last_calibration_time'] = _get_dicom_metadata_tag(ds, 'TimeOfLastCalibration')

        # Timing information
        file_info['study_date'] = _get_dicom_metadata_tag(ds, 'StudyDate')
        file_info['study_time'] = _get_dicom_metadata_tag(ds, 'StudyTime')
        file_info['series_date'] = _get_dicom_metadata_tag(ds, 'SeriesDate')
        file_info['series_time'] = _get_dicom_metadata_tag(ds, 'SeriesTime')
        file_info['acquition_date'] = _get_dicom_metadata_tag(ds, 'AcquisitionDate')
        file_info['acquisition_time'] = _get_dicom_metadata_tag(ds, 'AcquisitionTime')
        file_info['content_date'] = _get_dicom_metadata_tag(ds, 'ContentDate')
        file_info['content_time'] = _get_dicom_metadata_tag(ds, 'ContentTime')

        # Information relevant to the series
        file_info['study_description'] = _get_dicom_metadata_tag(ds, 'StudyDescription')
        file_info['series_description'] = _get_dicom_metadata_tag(ds, 'SeriesDescription')
        file_info['body_part'] = _get_dicom_metadata_tag(ds, 'BodyPartExamined')
        file_info['protocol_name'] = _get_dicom_metadata_tag(ds, 'ProtocolName')

        # Geometric information
        file_info['distance_source_to_detector'] = _get_dicom_metadata_tag(ds, 'DistanceSourceToDetector')
        file_info['distance_source_to_patient'] = _get_dicom_metadata_tag(ds, 'DistanceSourceToPatient')
        file_info['gantry_tilt'] = _get_dicom_metadata_tag(ds, 'GantryDetectorTilt')
        file_info['table_height'] = _get_dicom_metadata_tag(ds, 'TableHeight')
        file_info['rotation_direction'] = _get_dicom_metadata_tag(ds, 'RotationDirection')

        # Add key technical parameters relevant for the series
        file_info['kvp'] = _get_dicom_metadata_tag(ds, 'KVP')
        file_info['filter_type'] = _get_dicom_metadata_tag(ds, 'FilterType')
        file_info['slice_thickness'] = _get_dicom_metadata_tag(ds, 'SliceThickness')
        file_info['data_collection_diameter'] =_get_dicom_metadata_tag(ds, 'DataCollectionDiameter')
        file_info['reconstruction_diameter'] = _get_dicom_metadata_tag(ds, 'ReconstructionDiameter')
        file_info['pixel_spacing'] = _get_dicom_metadata_tag(ds, 'PixelSpacing', position=0)
        file_info['generator_power'] = _get_dicom_metadata_tag(ds, 'GeneratorPower')
        file_info['focal_spot'] = _get_dicom_metadata_tag(ds, 'FocalSpots', position=0)
        file_info['exposure_time'] = _get_dicom_metadata_tag(ds, 'ExposureTime')
        file_info['convolution_kernel'] = _get_dicom_metadata_tag(ds, 'ConvolutionKernel', position=0)
        if file_info['manufacturer'] == 'SIEMENS':
            file_info['ADMIRE_level'] = _get_dicom_metadata_tag(ds, 'ConvolutionKernel', position=1)
        elif file_info['manufacturer'] == 'GE MEDICAL SYSTEMS':
            file_info['DLIR_level'] = _get_dicom_metadata_tag(ds, '0x00531042')
        file_info['detector_element_size'] = _get_dicom_metadata_tag(ds, 'SingleCollimationWidth')
        file_info['total_collimation_width'] = _get_dicom_metadata_tag(ds, 'TotalCollimationWidth')
        file_info['table_speed'] = _get_dicom_metadata_tag(ds, 'TableSpeed')
        file_info['table_feed_per_rotation'] = _get_dicom_metadata_tag(ds, 'TableFeedPerRotation')
        file_info['spiral_pitch_factor'] = _get_dicom_metadata_tag(ds, 'SpiralPitchFactor')
        file_info['dose_modulation_type'] = _get_dicom_metadata_tag(ds, 'ExposureModulationType')
        
        # add key technical parameters relevant for the image
        file_info['tube_current'] = _get_dicom_metadata_tag(ds, 'XRayTubeCurrent')
        file_info['exposure'] = _get_dicom_metadata_tag(ds, 'Exposure')
        file_info['ctdi_vol'] = _get_dicom_metadata_tag(ds, 'CTDIvol')

        # Add image-specific information
        file_info['image_type'] = tuple(getattr(ds, 'ImageType', []))

        # Positional information
        file_info['instance_number'] = _get_dicom_metadata_tag(ds, 'InstanceNumber')
        file_info['slice_location'] = _get_dicom_metadata_tag(ds, 'SliceLocation')

    except Exception as e:
        logger.warning(f"Error extracting metadata from {file_path}: {str(e)}")
        return None
    
    return file_info

def scan_for_axial_ct_dicom_files(root_dir: str) -> pd.DataFrame:
    """
    Scan all DICOM files in a directory tree and extract key metadata into a DataFrame.
    
    Returns a DataFrame with one row per valid DICOM file.
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
    """
    Short funciton to read the data from a root directory.
    """
    logger.info(f"Reading metadata from DICOM files in {root_dir}")
    file_df = scan_for_axial_ct_dicom_files(root_dir)
    if file_df.empty:
        logger.warning(f"No DICOM files found in {root_dir}")
        return pd.DataFrame()
    return file_df


def main():
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
    # Setup logging
    configure_module_logging({
        'import_dicom': {'file': 'import_dicom.log', 'level': logging.DEBUG, 'console': True},
        'project_data': {'file': 'project_data.log', 'level': logging.DEBUG, 'console': True},
        'ct_series':    {'file': 'ct_series.log',    'level': logging.DEBUG, 'console': True}}
    )
    df = main()