import os
import re


def remove_zero_from_run(filepath):
    # Extract the directory path and filename from the filepath
    directory, filename = os.path.split(filepath)

    # Define the pattern to match 'run-' followed by one or two digits
    pattern = r'(run-0)(\d{1,2})(_)'

    # Use re.sub to replace the matched pattern in the filename
    modified_filename = re.sub(pattern, lambda x: x.group(1) + str(int(x.group(2))) + x.group(3), filename)

    # Construct the new full path with the modified filename
    new_filepath = os.path.join(directory, modified_filename)

    # Rename the file if the new path is different
    if filepath != new_filepath:
        os.rename(filepath, new_filepath)
        print(f"Renamed: {filepath} -> {new_filepath}")


def remove_zero_from_session(filepath):
    # Extract the directory path and filename from the filepath
    directory, filename = os.path.split(filepath)

    # Define the pattern to match 'ses-' followed by one or two digits before the file extension
    pattern = r'(ses-0)(\d{1,2})(_)$'

    # Use re.sub to replace the matched session number pattern in the filename
    modified_filename = re.sub(pattern, lambda x: x.group(1) + str(int(x.group(2))) + x.group(3), filename)

    # Construct the new full path with the modified filename
    new_filepath = os.path.join(directory, modified_filename)

    # Rename the file if the new path is different
    if filepath != new_filepath:
        os.rename(filepath, new_filepath)
        print(f"Renamed: {filepath} -> {new_filepath}")


def process_directory(directory):
    # Traverse through all files and directories in the specified directory
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith('.nii.gz') or filename.endswith('.tsv'):
                # Construct the full path of the file
                filepath = os.path.join(root, filename)

                # Process the file to remove leading zeros from run and session numbers
                remove_zero_from_run(filepath)


if __name__ == "__main__":
    # Specify the top-level directory to process
    top_directory = "/Users/mihnea/_workspace_/_uni/workshop/fmri_data/input_images"

    # Process all subdirectories within the top-level directory
    for root, dirs, files in os.walk(top_directory):
        for directory in dirs:
            dir_path = os.path.join(root, directory)
            process_directory(dir_path)

    # Also process files directly under the top-level directory
    process_directory(top_directory)
