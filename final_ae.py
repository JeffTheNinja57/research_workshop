 import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, random_split
import os
from torchvision import transforms
import PIL.Image

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class ImageDataset(Dataset):
    def __init__(self, folder, is_test=False):
        self.files = []
        self.transform = transforms.Compose([
            #transforms.ToPILImage(),
            transforms.Resize((672, 672)),
            transforms.ToTensor()
        ])
        
        # Walk through the folder, and separate files based on subdir
        for root, dirs, files in os.walk(folder):
            for file in files:
                if is_test:
                    if file.endswith(".png") and 'sub-CSI4' in root:
                        full_path = os.path.join(root, file)
                        self.files.append(full_path)
                else:
                    if file.endswith(".png") and 'sub-CSI4' not in root:
                        full_path = os.path.join(root, file)
                        self.files.append(full_path)
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image = PIL.Image.open(self.files[idx])
        image = image.convert('RGB')
        image = self.transform(image)
        return image

def train(model, train_loader, test_loader, val_loader, num_epochs=10, learning_rate=1e-3, patience=5):
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        for data in train_loader:
            data = data.to(device)
            recon = model(data)
            loss = criterion(recon, data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                recon = model(data)
                loss = criterion(recon, data)
                val_loss += loss.item()
        val_loss /= len(val_loader.dataset)
        print(f'Epoch:{epoch+1}, Training Loss:{loss.item():.4f}, Validation Loss:{val_loss:.4f}')
        test(model, test_loader)

        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f'Early stopping after {epoch+1} epochs.')
                break

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

if __name__ == "__main__":
    DATASET_PATH = r"/home/u933585/Research_workshop/output_images/"
    train_dataset = ImageDataset(DATASET_PATH)
    test_dataset = ImageDataset(DATASET_PATH, is_test=True)

    # Splitting the training dataset into training and validation sets
    total_train_samples = len(train_dataset)
    val_samples = int(0.1 * total_train_samples)  # 10% for validation
    train_samples = total_train_samples - val_samples
    train_data, val_data = random_split(train_dataset, [train_samples, val_samples])

    train_loader = DataLoader(train_data, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=512, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    # Define and run the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder().to(device)
    train(model, train_loader, test_loader, val_loader, num_epochs=100, patience=10)  # Adjust patience as necessary
    torch.save(model.state_dict(), 'model.pth')
    print('Model saved')