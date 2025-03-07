# GET EMBEDDINGS FOR IMAGES
from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import pickle
import cv2 as cv
import os
import torch  # Importing PyTorch to manage GPU operations

images = {}

# Directories containing images
COCO = "/home/u933585/Presented_Stimuli/COCO/"
ImageNet = "/home/u933585/Presented_Stimuli/ImageNet"
Scene = "/home/u933585/Presented_Stimuli/Scene"

paths = [COCO, ImageNet, Scene]

# Load all images from each directory
for path in paths:
    for filename in sorted(os.listdir(path)):  # Sort files alphabetically before processing
        img_path = os.path.join(path, filename)
        img = cv.imread(img_path)
        if img is not None:
            images[filename] = img

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

# Check if GPU is available and configure the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # Move the model to GPU if available
if torch.cuda.is_available():
    print("Using GPU")

one_dim_embeddings = {}
max_length = 5000

# Process and store embeddings for each image, limited to max_length
for filename in sorted(images.keys()):  # Iterate over sorted filenames
    if len(one_dim_embeddings) >= max_length:
        break
    # Process image and prepare inputs
    inputs = processor(images=Image.fromarray(images[filename]), return_tensors="pt")
    inputs = inputs.to(device)  # Move input tensors to GPU

    # Compute model outputs
    outputs = model(**inputs)
    # Store embeddings, making sure to detach and move them to cpu for storage
    one_dim_embeddings[filename] = outputs.last_hidden_state.detach().cpu().numpy().reshape(-1)

# Save embeddings to a pickle file
with open(f'ordered_one_dim_embeddings{max_length}.pkl', 'wb') as emb_file:
    pickle.dump(one_dim_embeddings, emb_file)