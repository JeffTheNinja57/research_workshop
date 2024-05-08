import os
import re


def add_run_number(filepath):
    directory, filename = os.path.split(filepath)

    # Separate the filename and extension
    name, ext = os.path.splitext(filename)

    # Check if any 'run' number exists (assuming they always end with an underscore)
    if not re.search(r'run-\d+_', name):
        name = name + '_run-10'  # Insert 'run-10' before the extension

    modified_filename = name + ext
    new_filepath = os.path.join(directory, modified_filename)

    if filepath != new_filepath:
        os.rename(filepath, new_filepath)
        print(f"Renamed: {filepath} -> {new_filepath}")


def remove_leading_zeros(filepath):
    directory, filename = os.path.split(filepath)

    modified_parts = []
    for part in filename.split('_'):
        if part.startswith('ses-0'):
            modified_parts.append('ses-' + part[5:])  # Keep 'ses-' and remove only leading zero
        elif part.startswith('run-0'):
            modified_parts.append('run-' + part[5:])  # Keep 'run-' and remove only leading zero
        else:
            modified_parts.append(part)

    modified_filename = '_'.join(modified_parts)
    new_filepath = os.path.join(directory, modified_filename)

    if filepath != new_filepath:
        os.rename(filepath, new_filepath)
        print(f"Renamed: {filepath} -> {new_filepath}")


def process_session_directory(session_directory):
    for filename in os.listdir(session_directory):
        if filename.endswith('.nii.gz') or filename.endswith('.tsv'):
            filepath = os.path.join(session_directory, filename)
            remove_leading_zeros(filepath)
            add_run_number(filepath)


if __name__ == "__main__":
    top_directory = "/Users/mihnea/_workspace_/_uni/workshop/fmri_data/input_images"

    for root, dirs, files in os.walk(top_directory):
        for subject_directory in dirs:
            dir_path = os.path.join(root, subject_directory)

            for session_directory in os.walk(dir_path):  # Iterate through sessions
                for session in session_directory[1]:
                    session_path = os.path.join(dir_path, session)
                    process_session_directory(session_path)  # Process session files
