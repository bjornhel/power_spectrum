# Small scripts to explore the data.

# Recursively read metadata from a root directory and stop before pixel data.
# If threr is a new SOP Class UID, print the metadata for this SOP Class UID. and a line to separate the SOP Class UIDs.

import os
import pydicom


def read_metadata(root_dir: str) -> None:
    """
    Recursively read metadata from DICOM files in a root directory and print the metadata for each SOP Class UID.
    Args:
        root_dir (str): The root directory containing DICOM files.
    """
    # initialize a list to keep track of SOP Class UIDs
    sop_class_uids = []
    sop_class_uids_study_description = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            try:
                ds = pydicom.dcmread(file_path, stop_before_pixels=True)
                sop_class_uid = ds.SOPClassUID
                # Check if the SOP Class UID is already in the list
                if sop_class_uid not in sop_class_uids:
                    sop_class_uids.append(sop_class_uid)
                    sop_class_uids_study_description.append(ds.StudyDescription)
                    print(f"New SOP Class UID: {sop_class_uid}")
                    print(f"Metadata for {sop_class_uid}:")
                    print(ds)
                    print("-" * 40)
                else:
                    print(f"Already seen SOP Class UID: {sop_class_uid}")
                
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
            
    print("Finished reading metadata from all DICOM files.")
    print("SOP Class UIDs found:")
    # for loop using zip top print the SOP Class UID and Study Description together
    for uid, description in zip(sop_class_uids, sop_class_uids_study_description):
        print(f"SOP Class UID: {uid}, Study Description: {description}")



if __name__ == "__main__":
    # Example usage
    #root_directory = r"C:\Users\bjorn\BH_Kode\power_spectrum\Fantomscan"  # Replace with your root directory
    #read_metadata(root_directory)
    ds =pydicom.dcmread(r"C:\Users\bjorn\BH_Kode\power_spectrum\Fantomscan\DICOMDIR")
    print(ds)
    print(ds.SOPClassUID)

    #Localizer is identified using tha tag Imagetype maybe 'LOCALIZER'