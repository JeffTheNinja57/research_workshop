## GETTING ORDERED EMBEDDINGS FOR FMRI IMAGE DATA

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import os
import PIL.Image
import pickle

from final_ae import Autoencoder, ImageDataset
    
IMG_FOLDER = r"C:\Users\cesar\Downloads\stimulus_images\stimulus_images"
MAX_IMAGES = 10 # Maximum number of images you want to load for embeddings

if __name__ == "__main__":
    data = ImageDataset(IMG_FOLDER, max_images=MAX_IMAGES)
    data_loader = DataLoader(data, batch_size=128, shuffle=False)  # Note: No shuffle to maintain order

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)

    model = Autoencoder().to(device)
    model.load_state_dict(torch.load(r"C:\Users\cesar\Downloads\model.pth", map_location=device))
    model.eval()

    one_dim_embeddings = {}

    with torch.no_grad():
        for imgs, labels in data_loader:
            if len(one_dim_embeddings) >= MAX_IMAGES:
                break
            imgs = imgs.to(device)
            outputs = model.encoder(imgs).view(imgs.size(0), -1).cpu().numpy()  # Flatten the output
            for label, output in zip(labels, outputs):
                if len(one_dim_embeddings) >= MAX_IMAGES:
                    break
                one_dim_embeddings[label] = output

    # Sort embeddings by labels before saving
    sorted_embeddings = {label: one_dim_embeddings[label] for label in sorted(one_dim_embeddings.keys())}

    with open('test_embeddings.pkl', 'wb') as emb:
        pickle.dump(sorted_embeddings, emb)