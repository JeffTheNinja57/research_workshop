import os
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn import plotting
import numpy as np
from multiprocessing import Pool, cpu_count

def process_run(input_args):
    """
    Process a single run and generate images for each time step.
    """
    # Unpack input arguments
    input_path, output_base_path, subject_id, session_id, run_id = input_args

    # Load the fMRI data from the input path
    img = nib.load(input_path)

    # Define the output folder for this run's images
    run_output_folder = os.path.join(output_base_path, subject_id, session_id, f"run-{run_id}")
    # Create the output folder if it doesn't exist
    os.makedirs(run_output_folder, exist_ok=True)

    # Iterate over each time step in the fMRI data
    for time_index in range(img.shape[3]):
        # Define the output filename for this time step's image
        output_filename = f"fMRI_{subject_id}_{session_id}_run-{run_id}_timestep-{time_index + 1}.png"
        output_path = os.path.join(run_output_folder, output_filename)

        # Extract the activation image data for the current time step
        activation_img = img.slicer[..., time_index]
        activation_img_data = activation_img.get_fdata().astype(np.float32)
        activation_img = nib.Nifti1Image(activation_img_data, img.affine)

        # Generate and save the activation image using nilearn plotting
        plotting.plot_stat_map(activation_img, bg_img=None, threshold=1e2,
                               annotate=False, draw_cross=False, colorbar=False,
                               output_file=output_path)

        # Close the matplotlib figure to prevent memory leaks
        plt.close()
        # Print the path of the generated image for logging
        print(f"Generated image: {output_filename}")

def process_fMRI_data(input_base_path, output_base_path):
    """
    Process fMRI data for all subjects, sessions, and runs in parallel.
    """
    try:
        # List to store input arguments for each run
        input_args_list = []

        # Iterate over subject folders in the input base path
        for subject_folder in os.listdir(input_base_path):
            subject_path = os.path.join(input_base_path, subject_folder)
            if not os.path.isdir(subject_path):
                continue

            # Iterate over session folders within each subject folder
            for session_folder in os.listdir(subject_path):
                session_path = os.path.join(subject_path, session_folder)
                if not os.path.isdir(session_path):
                    continue

                # Iterate over files within each session folder
                for file_name in os.listdir(session_path):
                    # Process only .nii.gz files
                    if file_name.endswith('.nii.gz'):
                        input_path = os.path.join(session_path, file_name)
                        subject_id = os.path.basename(os.path.dirname(session_path))
                        session_id = os.path.basename(session_path)

                        # Extract run ID from filename
                        for part in file_name.split('_'):
                            if part.startswith('run-'):
                                run_id = part[4:]
                                break

                        # Append input arguments for this run to the list
                        input_args_list.append((input_path, output_base_path, subject_id, session_id, run_id))

        # Determine number of processes to use based on CPU count and input arguments count
        num_processes = min(cpu_count(), len(input_args_list))
        # Use multiprocessing Pool to execute process_run in parallel for each input argument
        with Pool(processes=num_processes) as pool:
            pool.map(process_run, input_args_list)

    except Exception as e:
        # Handle any exceptions that occur during processing
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Define input and output base paths
    base_input_path = "/fmri_data/input_images"
    base_output_path = "/fmri_data/output_images"
    # Process the fMRI data with parallel processing
    process_fMRI_data(base_input_path, base_output_path)
