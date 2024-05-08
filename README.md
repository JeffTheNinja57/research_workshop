## Prerequisites

* Docker installed on the cluster.

# Project Setup

## Check for Updates (Before Each Run)

   * **Using Command Line:**

     ```bash
     git fetch 
     git pull origin main # Replace 'main' with your main branch name if different 
     ```

   * **Using GitHub Desktop:**

     Click "Fetch origin" and then "Pull origin" if updates are available.


### Initial Transfer to Cluster

   * Transfer the following files from the project directory to the university cluster:
      * `environment.yml`
      * `fmri_data_processing.py`
      * `file_modifications.py`
      * `Dockerfile`
      * `input_images` -- all the `.nii.gz` and `.tsv` files
   * **Create necessary ```output_images``` output directory on the cluster.**
   * Important Reminder
     * ⁠Ensure the paths you use in the **docker run command** align with the filesystem of the university cluster.

## Running the fMRI Processing

1. **Build the Docker Image**

   ```bash
   docker build -t my-fmri-processing:latest . 
   ```

2. **Run the Docker Container**

   ```bash
   docker run --gpus all -v /path/to/input/data:/input_data -v /path/to/output/images:/output_images my-fmri-processing:latest
   ```

   * Replace `/path/to/input/data`  with the actual path to your input fMRI data.
   * Replace `/path/to/output/images`  with the desired path to store the generated images.

**Retrieving Results**

* Transfer the processed images from the cluster's output directory back to your local machine.
