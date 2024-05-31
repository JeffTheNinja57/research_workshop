# Research Workshop

## Overview
This repository contains the code for the fMRI data processing described in our research paper. The project focuses on preprocessing fMRI data to generate embeddings suitable for machine learning models, particularly autoencoders.

## Prerequisites
Install the required packages using the provided `env_rw24.yml` file:
```sh
conda env create -f env_rw24.yml
conda activate research_workshop
```
## Setup
### Check for Updates
```sh
git fetch
git pull origin develop
```
## Running fMRI Preprocessing
Run the scripts in the following order:
1. `file_modifications.py`: Handles necessary file modifications.
   ```sh
   python file_modifications.py
   ```
2. `fmri_data_processing.py`: Main script for processing fMRI data.
   ```sh
   python fmri_data_processing.py
    ```
   Run `multiprocessor_image_processing.py` if you use a Mac, for faster processing speeds
The preprocessed fMRI images will look like this:
![fMRI_sub-CSI1_ses-1_run-1_timestep-1](https://github.com/JeffTheNinja57/research_workshop/assets/118731656/3f0ab9eb-6950-48f0-9697-99a7d5220eee)
3. `final_ae.py`: train the model on the images
  ```sh
  python final_ae.py
  ```

