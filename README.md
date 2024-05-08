# Prerequisites

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
   * **Important Reminder**
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

## Own Environment Route 
### Follow these steps if you'll be running the code on your local machine with

## pip:

### Install required packages:

```bash
pip install -r requirements.txt
```

### Running the fMRI Processing

```bash
python file_modifications.py
python fmri_data_processing.py /path/to/input/data /path/to/output/images
```

* Replace `/path/to/input/data`  with the actual path to your input fMRI data.
* Replace `/path/to/output/images`  with the desired path to store the generated images.

Please note that this assumes the user has Python and pip installed on their system, and that they are in the correct directory to run the commands.