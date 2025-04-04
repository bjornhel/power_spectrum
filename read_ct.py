# a module to read DICOM metadata and filter axial CT images
# -*- coding: utf-8 -*-
import os
import pydicom as dcm
import pandas as pd
import numpy as np
from logging_config import configure_module_logging

import logging
# Set up logger for this module
logger = logging.getLogger(__name__)

# from study_data import StudyData
# from ct_series import CTSeries

def _filter_axial_ct_images(ds: dcm.Dataset , fp: str) -> bool:
    # Check for DICOMDIR files
    if hasattr(ds, "FileSetID"):
        logger.info(f"Dicomdir excluded: {fp} has the FileSetID tag: {ds.FileSetID}")
        return False

    # Check for Radiation Dose Structured Report (RDSR):
    if hasattr(ds, "SOPClassUID") and (ds.SOPClassUID == "1.2.840.10008.5.1.4.1.1.88.67"):
        logger.info(f"RDSR excluded: {fp} has the SOPClassUID tag: {ds.SOPClassUID}")
        return False
    
    # Check for Dose Report images (Secondary Capture):
    if hasattr(ds, "SOPClassUID") and (ds.SOPClassUID == "1.2.840.10008.5.1.4.1.1.7"):
        logger.info(f"Dose Report (Secondary Capture) excluded: {fp} has the SOPClassUID tag: {ds.SOPClassUID}")
        return False
    
    # Check for CT localizer images:
    if (hasattr(ds, "SOPClassUID") and 
        (ds.SOPClassUID == "1.2.840.10008.5.1.4.1.1.2") and
        hasattr(ds, "ImageType") and
        (ds.ImageType[2] == "LOCALIZER")):
        logger.info(f"Localizer excluded: {fp} has the SOPClassUID tag: {ds.SOPClassUID} and ImageType tag: {ds.ImageType[2]}")
        return False
    
    # Check for axial CT images:
    if (hasattr(ds, "SOPClassUID") and 
        (ds.SOPClassUID == "1.2.840.10008.5.1.4.1.1.2") and
        hasattr(ds, "ImageType") and
        (ds.ImageType[2] == "AXIAL")):
        return True
    else:
        logger.info(f"Image {fp} is not cought by any filters")
        if hasattr(ds, "SOPClassUID"):
            logger.info(f"SOPClassUID: {ds.SOPClassUID}")
        if hasattr(ds, "ImageType"):
            logger.info(f"ImageType: {ds.ImageType}")
        return False

def _get_dicom_metadata_tag(ds: dcm.Dataset, tag: str, position=None) -> str:
    """
    Retrieve a value from a DICOM dataset by tag name.
    
    This function safely accesses a DICOM tag by name and handles indexed values
    for sequence-like tags. Type checking ensures valid inputs, and appropriate
    error logging provides visibility into access issues.
    
    Parameters
    ----------
    ds : dcm.Dataset
        The DICOM dataset to extract the tag from.
    tag : str
        The name of the DICOM tag to retrieve (e.g., 'PatientName', 'StudyInstanceUID').
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



def scan_dicom_files(root_dir: str) -> pd.DataFrame:
    """
    Scan all DICOM files in a directory tree and extract key metadata into a DataFrame.
    
    Returns a DataFrame with one row per valid DICOM file.
    """
    # List to hold dictionaries of file metadata
    file_data = []
    
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            ds = _read_metadata(file_path)

            if ds is None:
                continue
                
            include = _filter_axial_ct_images(ds, file_path)
            if not include:
                continue
            
            # Log the metadata for debugging:
            logger.info(f"ds: {ds}")

            # Extract key metadata for sorting and grouping
            try:
                
                file_info = {'file_path': file_path}
                file_info['convolution_kernel'] = _get_dicom_metadata_tag(ds, 'ConvolutionKernel', position=0)
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
                file_info['reconstruction_kernel'] = getattr(ds, 'ConvolutionKernel', None)
                file_info['pixel_spacing'] = getattr(ds, 'PixelSpacing', None)
                file_info['generator_power'] = _get_dicom_metadata_tag(ds, 'GeneratorPower')
                file_info['focal_spot'] = _get_dicom_metadata_tag(ds, 'FocalSpot')
                file_info['exposure_time'] = _get_dicom_metadata_tag(ds, 'ExposureTime')

                
                # add key technical parameters relevant for the image
                file_info['tube_current'] = _get_dicom_metadata_tag(ds, 'XRayTubeCurrent')
                file_info['exposure'] = _get_dicom_metadata_tag(ds, 'Exposure')

                



                # Add image-specific information
                file_info['image_type'] = tuple(getattr(ds, 'ImageType', []))

                # Positional information
                file_info['instance_number'] = _get_dicom_metadata_tag(ds, 'InstanceNumber')
                file_info['slice_location'] = _get_dicom_metadata_tag(ds, 'SliceLocation')

                file_data.append(file_info)
                
# (0018,1210) Convolution Kernel                  SH: ['Br40d', '1']
# (0018,5100) Patient Position                    CS: 'FFS'
# (0018,9306) Single Collimation Width            FD: 0.6
# (0018,9307) Total Collimation Width             FD: 57.599999999999994
# (0018,9309) Table Speed                         FD: 69.0
# (0018,9310) Table Feed per Rotation             FD: 34.5
# (0018,9311) Spiral Pitch Factor                 FD: 0.6
# (0018,9313) Data Collection Center (Patient)    FD: [0.0, -202.5, 381.0]
# (0018,9318) Reconstruction Target Center (Patie FD: [-2.0, -202.5, 381.0]
# (0018,9323) Exposure Modulation Type            CS: 'XYZ_EC'
# (0018,9324) Estimated Dose Saving               FD: 40.6109
# (0018,9345) CTDIvol                             FD: 3.7187267895652174

            except Exception as e:
                logger.warning(f"Error extracting metadata from {file_path}: {str(e)}")
                continue
    
    # Create DataFrame
    if not file_data:
        return pd.DataFrame()
        
    df = pd.DataFrame(file_data)
    
    # Convert numerical columns
    for col in ['slice_location', 'instance_number', 'kv', 'slice_thickness']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    return df

def read_metadata(root_dir: str) -> pd.DataFrame:
    """
    Recursively read metadata from DICOM files in a root directory and series.
    
    This function traverses the directory tree starting from a root directory.
    It reads the metadata from the DICOM files.
    If it finds a new series it creates a new 

    """
    logger.info(f"Reading metadata from DICOM files in {root_dir}")    
    
    # Step 1: Scan the files into a dataframe
    file_df = scan_dicom_files(root_dir)
    
    # Initialize a dataframe to store information about the DICOM files
    print('debug_stop')
    return file_df

    
    # for dirpath, _, filenames in os.walk(root_dir):
    #     for filename in filenames:
    #         file_path = os.path.join(dirpath, filename)
    #         ds = _read_metadata(file_path)
    #         if ds is None:
    #             continue
                
    #         include = _filter_axial_ct_images(ds, file_path)
    #         if not include:
    #             continue
            
    #         # Extract UIDs
    #         study_instance_uid = ds.StudyInstanceUID
    #         series_instance_uid = ds.SeriesInstanceUID
            
    #         # Get or create study object
    #         if study_instance_uid not in studies:
    #             study_description = getattr(ds, 'StudyDescription', None)
    #             studies[study_instance_uid] = StudyData(study_instance_uid, study_description)
    #             logger.info(f"Created new study: {study_instance_uid}")
            
    #         study = studies[study_instance_uid]
            
    #         # Get or create series object
    #         series_description = getattr(ds, 'SeriesDescription', None)
    #         series = study.get_or_create_series(series_instance_uid, series_description)
            
    #         # Add the image to the series
    #         series.add_image(file_path, ds)
    
    # # Extract metadata for all studies and series
    # for study in studies.values():
    #     study.extract_metadata()
    #     for series in study.series.values():
    #         series.extract_metadata()
    
    # logger.info(f"Found {len(studies)} studies with a total of {sum(len(s.series) for s in studies.values())} series")
    # return studies
            

def main():
    root_directory = r"/home/bhosteras/Kode/power_spectrum/Fantomscan/"  # Replace with your root directory
    read_metadata(root_directory)

           
if __name__ == "__main__":
    configure_module_logging({
        'read_ct': {'file': 'read_ct.log', 'level': logging.INFO, 'console': True},
        'study_data': {'file': 'study_data.log', 'level': logging.INFO, 'console': True},
        'ct_series': {'file': 'ct_series.log', 'level': logging.INFO, 'console': True},
    })
    main()