import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import os
from torchvision import transforms
import PIL.Image

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1), #3x672x672 -> 16x336x336
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), #16x336x336 -> 32x168x168
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), #32x168x168 -> 64x84x84
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), #64x84x84 -> 128x42x42
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), #128x42x42 -> 256x21x21
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1), #256x21x21 -> 128x42x42
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), #128x42x42 -> 64x84x84
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), #64x84x84 -> 32x168x168
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), #32x168x168 -> 16x336x336
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1), #16x336x336 -> 3x672x672
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Define the custom dataset class
class ImageDataset(Dataset):
    def __init__(self, folder, max_images=None, is_test=False):
        self.files = []
        self.transform = transforms.Compose([
            transforms.Resize((672, 672)),
            transforms.ToTensor()
        ])
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.endswith(".png"):
                    img = os.path.join(root, file)
                    if is_test and "sub-CSI4" in root:
                        self.files.append(img)
                    elif not is_test and "sub-CSI4" not in root:
                        self.files.append(img)
                    if max_images is not None and len(self.files) >= max_images:
                        break
            if max_images is not None and len(self.files) >= max_images:
                break

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image = PIL.Image.open(self.files[idx]).convert('RGB')
        transformed_image = self.transform(image)
        return transformed_image

def train(model, train_loader, num_epochs=100, learning_rate=1e-3, patience=5):
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    best_loss = float('inf')
    epochs_no_improve = 0
    train_losses = []

    for epoch in range(num_epochs):
        model.train()
        for data in train_loader:
            data = data.to(device)
            recon = model(data)
            loss = criterion(recon, data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_losses.append(loss.item())
        print(f'Epoch: {epoch+1}, Training Loss: {loss:.4f}')

        # Early stopping
        if loss < best_loss:
            best_loss = loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f'Early stopping after {epoch+1} epochs.')
                break

    with open('train_losses.txt', 'w') as f:
        for loss in train_losses:
            f.write(f'{loss}\n')

def test(model, test_loader):
    model.eval()
    criterion = nn.MSELoss()
    test_loss = 0

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            recon = model(data)
            loss = criterion(recon, data)
            test_loss += loss.item()
    test_loss /= len(test_loader.dataset)
    print(f'Test Loss:{test_loss:.4f}')

    # Save test loss
    with open('test_loss.txt', 'w') as f:
        f.write(f'{test_loss}\n')

if __name__ == "__main__":
    DATASET_PATH = r"/home/u933585/Research_workshop/output_images/"
    train_dataset = ImageDataset(DATASET_PATH)
    test_dataset = ImageDataset(DATASET_PATH, is_test=True)

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    # Define and run the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)
    model = Autoencoder().to(device)
    train(model, train_loader, num_epochs=100, patience=10)  # Adjust patience and epochs as necessary
    torch.save(model.state_dict(), 'model.pth')
    print('Model saved')