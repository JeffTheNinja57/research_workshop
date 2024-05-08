import os
import shutil

def process_session_directory(session_directory, output_directory):
    for filename in os.listdir(session_directory):
        if filename.endswith('.nii.gz') or filename.endswith('.tsv'):
            filepath = os.path.join(session_directory, filename)
            shutil.copy(filepath, output_directory) # Copy the file to the output directory


if __name__ == "__main__":
    top_directory = "/Users/mihnea/_workspace_/_uni/workshop/fmri_data/input_images"
    output_directory = "/Users/mihnea/_workspace_/_uni/workshop/output_files"  # Where to gather files

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for root, dirs, files in os.walk(top_directory):
        for subject_directory in dirs:
            dir_path = os.path.join(root, subject_directory)

            for session_directory in os.walk(dir_path):  # Iterate through sessions
                for session in session_directory[1]:
                    session_path = os.path.join(dir_path, session)
                    process_session_directory(session_path, output_directory)  # Process session files