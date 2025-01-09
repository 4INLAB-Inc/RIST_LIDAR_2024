#%%
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class PointNetDataset(torch.utils.data.Dataset):
    def __init__(self, csv_files, oversample=False):
        self.data = []
        self.labels = []
        self.files = csv_files

        for file in csv_files:
            df = pd.read_csv(file)
            df.columns = df.columns.str.strip()
            if 'Classification' in df.columns:
                points = df[['//X', 'Y', 'Z']].values
                labels = df['Classification'].values

                self.data.append(points)
                self.labels.append(labels)
        
        self.data = np.concatenate(self.data, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)
        
        if oversample:
            self.oversample()

    def oversample(self):
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        class_counts = dict(zip(unique_labels, counts))
        print(f"Class counts before oversampling: {class_counts}")

        if len(unique_labels) < 3:
            print("No oversampling applied. Ensure that there are at least two minority classes to oversample.")
            return

        # Find the maximum count of the majority class
        max_count = max(counts)

        # Separate the data for each class
        data_by_class = {label: self.data[self.labels == label] for label in unique_labels}

        # Upsample each class to match the majority class
        data_upsampled = [resample(data, replace=True, n_samples=max_count, random_state=42) if count < max_count else data
                          for label, data, count in zip(unique_labels, data_by_class.values(), counts)]
        
        self.data = np.vstack(data_upsampled)
        self.labels = np.hstack([[label] * max_count for label in unique_labels])

        class_counts_after = {label: max_count for label in unique_labels}
        print(f"Class counts after oversampling: {class_counts_after}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], idx

# Directory containing CSV files with point cloud data
csv_directory = "dataset/train"
csv_files = [os.path.join(csv_directory, f) for f in os.listdir(csv_directory) if f.endswith('.csv')]

# Split the dataset into training+validation and testing (90% train+validation, 10% test)
train_val_files, test_files = train_test_split(csv_files, test_size=0.1, random_state=42) 

# Split the training+validation set into training and validation (90% train, 10% validation)
train_files, val_files = train_test_split(train_val_files, test_size=0.1, random_state=42)

train_dataset = PointNetDataset(train_files, oversample=True)   # Improving performance with oversample=True
val_dataset = PointNetDataset(val_files, oversample=False)
test_dataset = PointNetDataset(test_files, oversample=False)

def collate_fn(batch):
    data = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    indices = [item[2] for item in batch]

    max_points = max(pts.shape[0] for pts in data)
    padded_data = np.zeros((len(data), max_points, 3), dtype=np.float32)

    for i, pts in enumerate(data):
        padded_data[i, :pts.shape[0], :] = pts

    return torch.from_numpy(padded_data), torch.tensor(labels, dtype=torch.int64), indices

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, collate_fn=collate_fn)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

class PointNet(nn.Module):
    def __init__(self, k=3):  # Change output classes to 3
        super(PointNet, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.relu = nn.ReLU()
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = x.view(-1, 1024)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = PointNet(k=3)  # Update number of classes to 3
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Calculate class weights
class_counts = np.bincount(train_dataset.labels.astype(int))
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum()
class_weights = torch.tensor(class_weights.astype(np.float32)).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Create lists to store loss and accuracy for each epoch
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# Training the model
num_epochs = 2
best_val_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    # Training loop
    for i, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", ncols=100), 0):
        inputs, labels, batch_indices = data
        inputs, labels = inputs.to(device), labels.to(device)

        # Reset gradients, forward pass, calculate loss, and backpropagate
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Accumulate the training loss
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        all_preds.append(predicted.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    # Calculate metrics for the training set
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    train_accuracy = accuracy_score(all_labels, all_preds)
    train_accuracies.append(train_accuracy)  # Save training accuracy
    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)  # Save training loss
    print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {epoch_loss:.4f}, Accuracy: {train_accuracy:.4f}')

    # Validation phase
    model.eval()
    running_val_loss = 0.0
    all_val_preds = []
    all_val_labels = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(val_loader, desc="Validation", ncols=100)):
            inputs, labels, _ = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            all_val_preds.append(predicted.cpu().numpy())
            all_val_labels.append(labels.cpu().numpy())

    # Calculate metrics for the validation set
    all_val_preds = np.concatenate(all_val_preds)
    all_val_labels = np.concatenate(all_val_labels)
    val_accuracy = accuracy_score(all_val_labels, all_val_preds)
    val_accuracies.append(val_accuracy)  # Save validation accuracy
    epoch_val_loss = running_val_loss / len(val_loader)
    val_losses.append(epoch_val_loss)  # Save validation loss
    print(f'Epoch {epoch + 1}/{num_epochs}, Validation Loss: {epoch_val_loss:.4f}, Accuracy: {val_accuracy:.4f}')

    # Save the model if it has the best validation loss so far
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        torch.save(model.state_dict(), 'model/lidar_poinnet_model_3class_new.pth')

print('Training Finished')
# %%
# Plot training and validation loss over epochs
plt.figure(figsize=(7, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label="Training Loss", color="blue")
plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss", color="orange")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss over Epochs")
plt.legend()
plt.grid(True)
plt.ylim(0.02, 0.3)  # Adjust y-axis range dynamically
plt.savefig("results/training_validation_loss.png")
plt.show()
plt.close()

# Plot training and validation accuracy over epochs
plt.figure(figsize=(7, 5))
plt.plot(range(1, num_epochs + 1), train_accuracies, label="Training Accuracy", color="green")
plt.plot(range(1, num_epochs + 1), val_accuracies, label="Validation Accuracy", color="red")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy over Epochs")
plt.legend()
plt.grid(True)
plt.ylim(0.95, 0.99)  # Adjust y-axis range dynamically
plt.savefig("results/training_validation_accuracy.png")
plt.show()
plt.close()
# %%
