import os
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn import plotting
import numpy as np


def process_fMRI_data(input_base_path, output_base_path):
    # Iterate over subjects, sessions, and files
    for subject_folder in os.listdir(input_base_path):
        subject_path = os.path.join(input_base_path, subject_folder)
        if not os.path.isdir(subject_path):
            continue

        for session_folder in os.listdir(subject_path):
            session_path = os.path.join(subject_path, session_folder)
            if not os.path.isdir(session_path):
                continue

            for file_name in os.listdir(session_path):
                if file_name.endswith('.nii.gz'):
                    img = nib.load(os.path.join(session_path, file_name))

                    subject_id = os.path.basename(os.path.dirname(session_path))
                    session_id = os.path.basename(session_path)

                    # Extract run ID from filename
                    for part in file_name.split('_'):
                        if part.startswith('run-'):
                            run_id = part[4:]
                            break

                    run_output_folder = os.path.join(output_base_path, subject_id, session_id, f"run-{run_id}")
                    os.makedirs(run_output_folder, exist_ok=True)

                    # Generate images for each time step
                    for time_index in range(img.shape[3]):
                        output_filename = f"fMRI_{subject_id}_{session_id}_run-{run_id}_timestep-{time_index + 1}.png"
                        output_path = os.path.join(run_output_folder, output_filename)

                        activation_img = img.slicer[..., time_index]
                        activation_img_data = activation_img.get_fdata().astype(np.float32)
                        activation_img = nib.Nifti1Image(activation_img_data, img.affine)

                        plotting.plot_stat_map(activation_img, bg_img=None, threshold=1e2,
                                               annotate=False, draw_cross=False, colorbar=False,
                                               output_file=output_path)

                        plt.close()

if __name__ == "__main__":
    base_input_path = "/Users/mihnea/_workspace_/_uni/workshop/fmri_data/input_images"
    base_output_path = "/Users/mihnea/_workspace_/_uni/workshop/fmri_data/output_images"
    process_fMRI_data(base_input_path, base_output_path)
