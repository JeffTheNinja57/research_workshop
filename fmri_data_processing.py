# import os
# import matplotlib.pyplot as plt
# import nibabel as nib
# import numpy as np
# import pandas as pd
# from nilearn import plotting
#
#
# # Function to read TSV data into a DataFrame
# def read_tsv_file(file_path):
#     return pd.read_csv(file_path, sep='\t')
#
#
# def process_fMRI_data(input_base_path, output_base_path):
#     # Iterate over subjects
#     for subject_folder in os.listdir(input_base_path):
#         subject_path = os.path.join(input_base_path, subject_folder)
#
#         # Check if the item in the directory is indeed a folder
#         if not os.path.isdir(subject_path):
#             continue
#
#         # Create subject output folder
#         subject_output_folder = os.path.join(output_base_path, subject_folder)
#         os.makedirs(subject_output_folder, exist_ok=True)
#
#         # Iterate over sessions
#         for session_folder in os.listdir(subject_path):
#             session_path = os.path.join(subject_path, session_folder)
#
#             # Check if the item in the directory is indeed a folder
#             if not os.path.isdir(session_path):
#                 continue
#
#             # Create session output folder
#             session_output_folder = os.path.join(subject_output_folder, session_folder)
#             os.makedirs(session_output_folder, exist_ok=True)
#
#             # Iterate over files in the session folder
#             for file_name in os.listdir(session_path):
#                 if file_name.endswith('.nii.gz'):
#
#                     # Construct path to corresponding TSV file
#                     tsv_file_name = file_name.replace('.nii.gz', '_events.tsv')
#                     tsv_file_path = os.path.join(session_path, tsv_file_name)
#
#                     # Load the NIfTI fMRI data
#                     img = nib.load(os.path.join(session_path, file_name))
#
#                     # Read the TSV file to extract event information
#                     tsv_data = read_tsv_file(tsv_file_path)
#                     onset_times = tsv_data['StimOn(s)']
#                     stim_off_times = tsv_data['StimOff(s)']
#                     image_names = tsv_data['ImgName']
#                     trial_numbers = tsv_data['Trial']
#                     run_numbers = tsv_data['Run']
#
#                     # Group events by run number
#                     events_by_run = {}
#                     for i in range(len(onset_times)):
#                         run_num = run_numbers[i]
#                         if run_num not in events_by_run:
#                             events_by_run[run_num] = []
#                         events_by_run[run_num].append(
#                             (onset_times[i], stim_off_times[i], image_names[i], trial_numbers[i]))
#
#                     # Process each run's events
#                     for run_num, events in events_by_run.items():
#
#                         # Create run output folder
#                         run_output_folder = os.path.join(session_output_folder, f"run-{run_num}")
#                         os.makedirs(run_output_folder, exist_ok=True)
#
#                         # Loop over each event and save corresponding activation image
#                         for event_idx, (onset, stim_off, image_name, trial_number) in enumerate(events):
#                             # Calculate time index based on stim_off time and TR
#                             time_index = int(stim_off / img.header['pixdim'][4])  # Assuming TR is in pixdim[4]
#
#                             # Extract the activation image at the specified time index
#                             activation_img = img.slicer[..., time_index]
#
#                             # Convert activation image data to float32
#                             activation_img_data = activation_img.get_fdata().astype(np.float32)
#                             activation_img = nib.Nifti1Image(activation_img_data, img.affine)
#
#                             # Construct the output filename
#                             output_filename = f"fMRI_{subject_folder}_{session_folder}_run-{run_num}_trial-{trial_number}_{image_name.split('.')[0]}.png"
#                             output_path = os.path.join(run_output_folder, output_filename)
#
#                             # Save the activation image with the constructed filename
#                             plotting.plot_stat_map(activation_img).savefig(output_path)
#                             plt.close()
#
#                             print(
#                                 f"Saved image for event '{image_name}' at StimOff time {stim_off} seconds as {output_filename}.")
#
#
# if __name__ == "__main__":
#     # Define paths to the input images and output folder
#     base_input_path = "/Users/mihnea/_workspace_/_uni/workshop/fmri_data/images"
#     base_output_path = "/Users/mihnea/_workspace_/_uni/workshop/fmri_data/output_images"
#
#     # Call the processing function
#     process_fMRI_data(base_input_path, base_output_path)

# import os
# import nibabel as nib
# import matplotlib.pyplot as plt
# from nilearn import plotting
#
#
# def process_session_directory(session_directory, output_directory):
#     for filename in os.listdir(session_directory):
#         if filename.endswith('.nii.gz'):
#             filepath = os.path.join(session_directory, filename)
#
#             # Load the NIfTI fMRI data
#             img = nib.load(filepath)
#
#             # Extract subject, session, and run information
#             subject_id = os.path.basename(os.path.dirname(session_directory))
#             session_id = os.path.basename(session_directory)
#             run_id = filename.split('_')[1]  # Adjust if your run info isn't in this format
#
#             # Create run output folder
#             run_output_folder = os.path.join(output_directory, subject_id, session_id, f"run-{run_id}")
#             os.makedirs(run_output_folder, exist_ok=True)
#
#             # Iterate through each time point and potentially events
#             for time_index in range(img.shape[3]):
#                 # Construct the output filename for this time point
#                 output_filename = f"fMRI_{subject_id}_{session_id}_run-{run_id}_timestep-{time_index + 1}.png"
#                 output_path = os.path.join(run_output_folder, output_filename)
#
#                 # Extract, convert, plot, and save the image (same as before)
#                 # ...
#
# # --- Main Execution ---
# if __name__ == "__main__":
#     top_directory = "/Users/mihnea/_workspace_/_uni/workshop/fmri_data/input_images"
#     output_directory = "/Users/mihnea/_workspace_/_uni/workshop/output_images"
#
#     if not os.path.exists(output_directory):
#         os.makedirs(output_directory)
#
#     for root, dirs, files in os.walk(top_directory):
#         for subject_directory in dirs:
#             dir_path = os.path.join(root, subject_directory)
#
#             for session_directory in os.walk(dir_path):
#                 for session in session_directory[1]:
#                     session_path = os.path.join(dir_path, session)
#                     process_session_directory(session_path, output_directory)

import os
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import plotting


# Function to read TSV data into a DataFrame
def read_tsv_file(file_path):
    return pd.read_csv(file_path, sep='\t')


def process_fMRI_data(input_base_path, output_base_path):
    # Iterate over subjects
    for subject_folder in os.listdir(input_base_path):
        subject_path = os.path.join(input_base_path, subject_folder)

        # Check if the item in the directory is indeed a folder
        if not os.path.isdir(subject_path):
            continue

        # Create subject output folder
        subject_output_folder = os.path.join(output_base_path, subject_folder)
        os.makedirs(subject_output_folder, exist_ok=True)

        # Iterate over sessions
        for session_folder in os.listdir(subject_path):
            session_path = os.path.join(subject_path, session_folder)

            # Check if the item in the directory is indeed a folder
            if not os.path.isdir(session_path):
                continue

            # Create session output folder
            session_output_folder = os.path.join(subject_output_folder, session_folder)
            os.makedirs(session_output_folder, exist_ok=True)

            # Iterate over files in the session folder
            for file_name in os.listdir(session_path):
                if file_name.endswith('.nii.gz'):

                    # Construct path to corresponding TSV file
                    tsv_file_name = file_name.replace('.nii.gz', '_events.tsv')
                    tsv_file_path = os.path.join(session_path, tsv_file_name)

                    # Load the NIfTI fMRI data
                    img = nib.load(os.path.join(session_path, file_name))

                    # Read the TSV file to extract event information
                    tsv_data = read_tsv_file(tsv_file_path)
                    onset_times = tsv_data['StimOn(s)']
                    stim_off_times = tsv_data['StimOff(s)']
                    image_names = tsv_data['ImgName']
                    trial_numbers = tsv_data['Trial']
                    run_numbers = tsv_data['Run']

                    # Group events by run number
                    events_by_run = {}
                    for i in range(len(onset_times)):
                        run_num = run_numbers[i]
                        if run_num not in events_by_run:
                            events_by_run[run_num] = []
                        events_by_run[run_num].append(
                            (onset_times[i], stim_off_times[i], image_names[i], trial_numbers[i]))

                    # Process each run's events
                    for run_num, events in events_by_run.items():
                        run_output_folder = os.path.join(session_output_folder, f"run-{run_num}")
                        os.makedirs(run_output_folder, exist_ok=True)

                        # Iterate through each time point of the fMRI data
                        for time_index in range(img.shape[3]):
                            # Construct the output filename for this time point
                            output_filename = f"fMRI_{subject_folder}_{session_folder}_run-{run_num}_timestep-{time_index + 1}.png"
                            output_path = os.path.join(run_output_folder, output_filename)

                            # Extract the activation image at the specified time index
                            activation_img = img.slicer[..., time_index]

                            # Convert activation image data to float32
                            activation_img_data = activation_img.get_fdata().astype(np.float32)
                            activation_img = nib.Nifti1Image(activation_img_data, img.affine)

                            # Plot and save the activation image
                            plotting.plot_stat_map(activation_img, bg_img=None, threshold=1e2,
                                                   annotate=False, draw_cross=False, colorbar=False,
                                                   output_file=output_path)

                            # Close the current figure to avoid memory issues
                            plt.close()

if __name__ == "__main__":
    # Define paths to the input images and output folder
    base_input_path = "/Users/mihnea/_workspace_/_uni/workshop/fmri_data/input_images"
    base_output_path = "/Users/mihnea/_workspace_/_uni/workshop/fmri_data/output_images"

    # Call the processing function
    process_fMRI_data(base_input_path, base_output_path)


