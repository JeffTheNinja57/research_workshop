FROM nvidia/cuda:12.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app

COPY environment.yml /app/environment.yml
RUN conda env create -f /app/environment.yml
RUN echo "conda activate my_fmri_env" >> ~/.bashrc

# Copy both scripts
COPY fmri_data_processing.py /app/
COPY file_modifications.py /app/

# Execute file modifications script first
CMD ["python", "/app/file_modifications.py", "&&", "python", "/app/fmri_data_processing.py", "/input_data", "/output_images"]