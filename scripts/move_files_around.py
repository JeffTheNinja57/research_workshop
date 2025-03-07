import os
import shutil

def move_fmri_images(source_dir, destination_dir):
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    for root, dirs, files in os.walk(source_dir):
        for subject_directory in dirs:
            dir_path = os.path.join(root, subject_directory)
            for session_directory in os.walk(dir_path):  # Iterate through sessions
                for session in session_directory[1]:
                    session_path = os.path.join(dir_path, session)
                    for run_directory in os.walk(session_path):
                        for run in run_directory[1]:
                            run_path = os.path.join(session_path, run)
                            for file in os.listdir(run_path):
                                if file.endswith('.png'):
                                    source_file = os.path.join(run_path, file)
                                    destination_file = os.path.join(destination_dir, file)
                                    shutil.move(source_file, destination_file)
                                    print(f"Moved: {source_file} -> {destination_file}")


# Example usage:
source_directory = "/Users/mihnea/_workspace_/_uni/workshop/fmri_data/output_images"
destination_directory = "/Users/mihnea/_workspace_/_uni/workshop/training_images"
move_fmri_images(source_directory, destination_directory)
