# a module to read DICOM metadata and filter axial CT images
# -*- coding: utf-8 -*-
import os
import pydicom as dcm
import logging

# Set up logging configuration
def setup_logging(logfile = "dicom_filtering.log", file=True, console=False, overwrite=True):
    file = True if logfile else False

    handlers = []
    if file:
        handlers.append(logging.FileHandler(logfile))
        if overwrite:
            handlers.append(logging.FileHandler(logfile, mode='w'))
        else:
            handlers.append(logging.FileHandler(logfile, mode='a'))
    if console:
        handlers.append(logging.StreamHandler())

    logging.basicConfig(
        level=logging.INFO,              # Log level
        format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
        datefmt='%Y-%m-%d %H:%M:%S',      # Date format
        handlers=handlers                # Handlers for logging
    )

def _filter_axial_ct_images(ds: dcm.Dataset , fp: str, verbose=True) -> bool:
    # Check for DICOMDIR files
    if hasattr(ds, "FileSetID"):
        if verbose:
            logging.info(f"Dicomdir excluded: {fp} has the FileSetID tag: {ds.FileSetID}")
        return False

    # Check for Radiation Dose Structured Report (RDSR):
    if hasattr(ds, "SOPClassUID") and (ds.SOPClassUID == "1.2.840.10008.5.1.4.1.1.88.67"):
        if verbose:
            logging.info(f"RDSR excluded: {fp} has the SOPClassUID tag: {ds.SOPClassUID}")
        return False
    
    # Check for Dose Report images (Secondary Capture):
    if hasattr(ds, "SOPClassUID") and (ds.SOPClassUID == "1.2.840.10008.5.1.4.1.1.7"):
        if verbose:
            logging.info(f"Dose Report (Secondary Capture) excluded: {fp} has the SOPClassUID tag: {ds.SOPClassUID}")
        return False
    
    # Check for CT localizer images:
    if (hasattr(ds, "SOPClassUID") and 
        (ds.SOPClassUID == "1.2.840.10008.5.1.4.1.1.2") and
        hasattr(ds, "ImageType") and
        (ds.ImageType[2] == "LOCALIZER")):
        if verbose:
            logging.info(f"Localizer excluded: {fp} has the SOPClassUID tag: {ds.SOPClassUID} and ImageType tag: {ds.ImageType[2]}")
        return False
    
    # Check for axial CT images:
    if (hasattr(ds, "SOPClassUID") and 
        (ds.SOPClassUID == "1.2.840.10008.5.1.4.1.1.2") and
        hasattr(ds, "ImageType") and
        (ds.ImageType[2] == "AXIAL")):
        return True
    else:
        if verbose:
            logging.info(f"Image {fp} is not cought by any filters")
            if hasattr(ds, "SOPClassUID"):
                logging.info(f"SOPClassUID: {ds.SOPClassUID}")
            if hasattr(ds, "ImageType"):
                logging.info(f"ImageType: {ds.ImageType}")
        return False

def _read_metadata(fp: str, verbose=True) -> dcm.Dataset:
    # Attempt tp read the DICOM file and return the dataset
    try:
        ds = dcm.dcmread(fp, stop_before_pixels=True)
        return ds
    except Exception as e:
        if verbose:
            logging.warning(f"Non DICOM file excluded: {fp}, error using dcmread")
        return None


def read_metadata(root_dir: str, verbose=True) -> None:
    """
    Recursively read metadata from DICOM files in a root directory and print the metadata for each SOP Class UID.
    Args:
        root_dir (str): The root directory containing DICOM files.
    """
    # Make one CT series per Series Instance UID
    # Put references to the each series into the study data object.
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            ds = _read_metadata(file_path, verbose=verbose)
            if ds is None:
                continue
            include = _filter_axial_ct_images(ds, file_path, verbose=verbose)
            if not include:
                continue
            
            # Check if the SeriesInstanceUID is already in the dictionary
            # If not, create a new entry and initialize the study data object with a new series (i.e. give it a coordinate in the 4D arrays 4th dimension)
            # If not make a new ct_series object and add it 
            

            

if __name__ == "__main__":
    setup_logging()
    # Example usage
    root_directory = r"/home/bhosteras/Kode/power_spectrum/Fantomscan/"  # Replace with your root directory
    read_metadata(root_directory, verbose=False)
    
    
    
    #ds =dcm.dcmread(r"/home/bhosteras/Kode/power_spectrum/Fantomscan/DICOMDIR")
    # Print the first 10 tages of the dataset
    #i=0
    #for elem in ds.iterall():
    #    if i<50:
    #        i+=1
    #        print(elem)

    #write ds to textfile:
    #print(f"Current working directory: {os.getcwd()}")
    #with open("metadata.txt", "w") as f:
    #    f.write(str(ds))
    


    #Localizer is identified using tha tag Imagetype maybe 'LOCALIZER'