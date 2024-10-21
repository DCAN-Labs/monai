# Setup imports
import logging
import os
import sys
import shutil
import tempfile

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import DataLoader, ImageDataset
from monai.transforms import Compose, EnsureChannelFirst, Resize, ScaleIntensity
from monai.networks.nets import ResNet 

# Setup constants and device
BATCH_SIZE = 2
IMAGE_SHAPE = (96, 96, 96)
VAL_INTERVAL = 2
MAX_EPOCHS = 5
LEARNING_RATE = 1e-4
pin_memory = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
print_config()

# Setup data directory
directory = os.environ.get("MONAI_DATA_DIRECTORY", tempfile.mkdtemp())
os.makedirs(directory, exist_ok=True)
root_dir = directory
print(root_dir)

# Define images and labels
images = [os.path.join(root_dir, "ixi", f"IXI{i}-T1.nii.gz") for i in range(314, 586, 20)]
ages = np.array([45.86, 68.27, 29.0, 29.57, 39.47, 48.68, 47.35, 64.19, 46.17, 38.77, 83.81, 72.27, 64.65, 62.09, 70.95])

if not os.path.isfile(images[0]):
    resource = "https://developer.download.nvidia.com/assets/Clara/monai/tutorials/IXI-T1.tar"
    dataset_dir = os.path.join(root_dir, "ixi")
    download_and_extract(resource, f"{dataset_dir}.tar", dataset_dir, "34901a0593b41dd19c1a1f746eac2d58")

# Utility function for creating dataset and loader
def create_dataloader(image_files, labels, transforms, batch_size, shuffle=False):
    dataset = ImageDataset(image_files=image_files, labels=labels, transform=transforms)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=pin_memory)

# Transforms
transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize(IMAGE_SHAPE)])

# Create training and validation loaders
train_loader = create_dataloader(images[:10], ages[:10], transforms, BATCH_SIZE, shuffle=True)
val_loader = create_dataloader(images[10:], ages[10:], transforms, BATCH_SIZE)

# Model setup
model = ResNet(in_shape=[1, *IMAGE_SHAPE], out_shape=1, channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2))
model.to(device)
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Helper function for evaluating the model
def evaluate_model(model, val_loader):
    model.eval()
    all_labels, all_outputs = [], []
    with torch.no_grad():
        for val_images, val_labels in val_loader:
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            val_outputs = model(val_images).cpu().numpy()
            all_labels.extend(val_labels.cpu().numpy())
            all_outputs.extend(val_outputs.flatten())
    mse = np.mean(np.square(np.subtract(all_labels, all_outputs)))
    return np.sqrt(mse)

# Training loop
best_rmse, best_epoch = float('inf'), -1
writer = SummaryWriter()

for epoch in range(MAX_EPOCHS):
    print(f"Epoch {epoch + 1}/{MAX_EPOCHS}")
    model.train()
    epoch_loss = 0

    for step, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels.float())
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        writer.add_scalar("train_loss", loss.item(), epoch * len(train_loader) + step)
        print(f"Step {step + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")

    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")

    # Validation phase
    if (epoch + 1) % VAL_INTERVAL == 0:
        rmse = evaluate_model(model, val_loader)
        writer.add_scalar("val_rmse", rmse, epoch + 1)
        print(f"Validation RMSE: {rmse:.4f}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_epoch = epoch + 1
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved new best model")

print(f"Training complete. Best RMSE: {best_rmse:.4f} at epoch {best_epoch}")
writer.close()
