import os
import pandas as pd
import nibabel as nib
import numpy as np
from nilearn import plotting
import matplotlib.pyplot as plt

data_dir = "../fmri/images/"

def read_tsv_file(file_path):
    return pd.read_csv(file_path, sep='\t')

def process_fmri_data(data_dir, output_dir):
    for subject_folder in os.listdir(data_dir):
        subject_path = os.path.join(data_dir, subject_folder)

        for session_folder in os.listdir(subject_path):
                session_path = os.path.join(subject_path, session_folder)

                for file in os.listdir(session_path):
                        if file.endswith(".nii.gz"):
                            nii_file_path = os.path.join(session_path, file)
                            run_name = file.split("_")[3]
                            tsv_file_path = os.path.join(file.split("_")[:3], f"{run_name}_events.tsv")

                            img = nib.load(nii_file_path)

                            # Read the TSV file to extract event information
                            tsv_data = read_tsv_file(tsv_file_path)
                            onset_times = tsv_data['StimOn(s)']
                            stim_off_times = tsv_data['StimOff(s)']
                            image_names = tsv_data['ImgName']
                            trial_numbers = tsv_data['Trial']

                            # Construct the output folder path based on subject, session, and run
                            output_folder = os.path.join(output_dir, f"subject_{subject_folder}",
                                                            f"session_{session_folder}", f"run_{run_name}")
                            os.makedirs(output_folder,
                                        exist_ok=True) 
                            for i in range(img.shape[-1]):
                                activation_img = img.slicer[:, :, :, i]
                                # Convert activation image data to float32
                                activation_img_data = activation_img.get_fdata().astype(np.float32)
                                activation_img = nib.Nifti1Image(activation_img_data, img.affine)
                                output_filename = f"fMRI_subject_{subject_folder}_session_{session_folder}_run_{run_name}.png"
                                output_path = os.path.join(output_folder, output_filename)

                                # Plot and save the activation image
                                plotting.plot_stat_map(activation_img, bg_img=None, threshold=1e2, annotate=False, draw_cross=False, colorbar=False, output_file=output_path)

                                # Close the current figure to avoid memory issues
                                plt.close()
if __name__ == "__main__":
    data_directory = "/Users/mihnea/_workspace_/_uni/workshop/fmri_data/bold_data"
    output_directory = "/Users/mihnea/_workspace_/_uni/workshop/fmri_data/output_images"

    # Call the main processing function
    process_fmri_data(data_directory, output_directory)