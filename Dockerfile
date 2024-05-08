FROM nvidia/cuda:12.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app

COPY environment.yml /app/environment.yml
RUN conda env create -f /app/environment.yml

# Activate Conda environment for all subsequent commands
ENV PATH /miniconda/envs/my_fmri_env/bin:$PATH

COPY fmri_data_processing.py /app/
COPY file_modifications.py /app/

# Execute file modifications script first, then processing script
CMD ["python", "/app/file_modifications.py", "&&", "python", "/app/fmri_data_processing.py", "/input_data", "/output_images"]