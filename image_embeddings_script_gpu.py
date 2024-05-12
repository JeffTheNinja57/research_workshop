from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import pickle
import os
import torch

# Define paths to your image directories
COCO = "/Users/mihnea/_workspace_/_uni/workshop/BOLD5000/Scene_Stimuli/Presented_Stimuli/COCO"
ImageNet = "/Users/mihnea/_workspace_/_uni/workshop/BOLD5000/Scene_Stimuli/Presented_Stimuli/ImageNet"
Scene = "/Users/mihnea/_workspace_/_uni/workshop/BOLD5000/Scene_Stimuli/Presented_Stimuli/Scene"

paths = [COCO, ImageNet, Scene]

# Create a dictionary to store images
images = {}

# Loop over each directory path
for path in paths:
    for filename in os.listdir(path):
        img_path = os.path.join(path, filename)

        # Use PIL to open and load the image
        img = Image.open(img_path)

        # Optionally convert the image to RGB if it's not already
        img = img.convert("RGB")

        # Store the image in the dictionary with its filename as key
        images[filename] = img

# Initialize ViT image processor and model
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k', force_download=True)
model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k', force_download=True)

# Check if GPU is available and move the model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("Using GPU")
else:
    print("Using CPU")
model.to(device)  # Move the model to GPU if available

# Dictionary to store one-dimensional embeddings
one_dim_embeddings = {}
max_length = 500

# Loop over each image in the dictionary
for image_filename in images:
    if len(one_dim_embeddings) == max_length:
        break
    else:
        # Process images and prepare inputs using the ViT image processor
        inputs = processor(images=images[image_filename], return_tensors="pt")
        inputs = inputs.to(device)  # Move input tensors to GPU if available

        # Compute model outputs
        outputs = model(**inputs)

        # Store embeddings, making sure to detach and move them to CPU for storage
        one_dim_embeddings[image_filename] = outputs.last_hidden_state.detach().cpu().reshape(-1)

# Save the embeddings dictionary to a pickle file
with open(f'one_dim_embeddings_{max_length}.pkl', 'wb') as emb_file:
    pickle.dump(one_dim_embeddings, emb_file)
