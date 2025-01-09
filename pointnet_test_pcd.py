#%%
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
import time
# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Dataset Class for Point Cloud Data (PCD)
class PointNetPCDDataset(torch.utils.data.Dataset):
    def __init__(self, pcd_file):
        # Read the PCD file using Open3D
        pcd = o3d.io.read_point_cloud(pcd_file)
        points = np.asarray(pcd.points)
        
        # Store data and file names
        self.data = points
        self.file_names = ["merged-20240909-011348.pcd"] * len(points)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], idx, self.file_names[idx]
    
start_time = time.time()
# Load PCD file
pcd_file = "dataset/test/merged-20241002-000022.pcd"
test_dataset = PointNetPCDDataset(pcd_file)

# Custom collate function for batching
def collate_fn(batch):
    data = [item[0] for item in batch]
    indices = [item[1] for item in batch]
    file_names = [item[2] for item in batch]

    max_points = max(len(pts) for pts in data)
    padded_data = np.zeros((len(data), max_points, 3), dtype=np.float32)

    for i, pts in enumerate(data):
        padded_data[i, :len(pts), :] = pts

    return torch.from_numpy(padded_data), indices, file_names

# Initialize DataLoader
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# Define PointNet Model
class PointNet(nn.Module):
    def __init__(self, k=3):  # Update to k=3 for three classes
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

# Initialize and load trained model
model = PointNet(k=3)
model.to(device)
model.load_state_dict(torch.load('model/lidar_poinnet_model_3class_oversample.pth'))
model.eval()

# Testing without labels
all_test_preds = []
all_test_indices = []
all_test_file_names = []

with torch.no_grad():
    for data in test_loader:
        inputs, indices, file_names = data
        inputs = inputs.to(device)
        outputs = model(inputs)

        _, predicted = torch.max(outputs, 1)
        all_test_preds.append(predicted.cpu().numpy())
        all_test_indices.extend(indices)
        all_test_file_names.extend(file_names)

# Concatenate predictions after the testing loop
all_test_preds = np.concatenate(all_test_preds)

# Save results
test_results = []
for idx, file_name in zip(all_test_indices, all_test_file_names):
    original_data = test_dataset.data[idx]
    predicted_label = all_test_preds[idx]
    test_results.append([original_data[0], original_data[1], original_data[2], predicted_label, file_name])

test_results_df = pd.DataFrame(test_results, columns=['X', 'Y', 'Z', 'Predicted_Label', 'File_Name'])
# test_results_df.to_csv('test_results_pcd.csv', index=False)
# print('Test results saved to test_results_pcd.csv')

# Custom colormap
cmap = ListedColormap(['blue', 'red', 'green'])

# Plotting 3D scatter plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(test_results_df['X'], test_results_df['Y'], test_results_df['Z'], c=test_results_df['Predicted_Label'], cmap=cmap, marker='^')
ax.set_title('Predicted Labels for PCD File')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Add color bar
legend = ax.legend(*scatter.legend_elements(), title="Classes")
ax.add_artist(legend)

plt.tight_layout()
plt.show()
# End timing
end_time = time.time()
print(f"Total time from loading PCD file to saving results: {end_time - start_time:.2f} seconds")


# %%
# Filter results for class "1" and "2"
class_1_results_df = test_results_df[test_results_df['Predicted_Label'] == 1]
class_2_results_df = test_results_df[test_results_df['Predicted_Label'] == 2]


############ Plotting 3D scatter plot with rainbow color gradient based on height (Z-coordinate) for classes "1" and "2"
# Normalize Z values for color mapping
norm_1 = plt.Normalize(class_1_results_df['Z'].min(), class_1_results_df['Z'].max())
norm_2 = plt.Normalize(class_2_results_df['Z'].min(), class_2_results_df['Z'].max())

# Create rainbow colormaps based on height for each class
cmap_rainbow_1 = plt.cm.ScalarMappable(cmap="rainbow", norm=norm_1)  # Rainbow gradient for class 1
cmap_rainbow_2 = plt.cm.ScalarMappable(cmap="rainbow", norm=norm_2)  # Rainbow gradient for class 2

# Plotting 3D scatter plot for class "1"
fig = plt.figure(figsize=(14, 7))

# Plot for Predicted Label 1 with height-based rainbow gradient
ax1 = fig.add_subplot(121, projection='3d')
scatter1 = ax1.scatter(class_1_results_df['X'], class_1_results_df['Y'], class_1_results_df['Z'],
                       c=class_1_results_df['Z'], cmap="rainbow", marker='o', norm=norm_1)
ax1.set_title(f'Predicted Labels - Class 1 with Rainbow Height Gradient\nFile: {all_test_file_names[0]}')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
fig.colorbar(cmap_rainbow_1, ax=ax1, label='Height (Z)')

# Plot for Predicted Label 2 with height-based rainbow gradient
ax2 = fig.add_subplot(122, projection='3d')
scatter2 = ax2.scatter(class_2_results_df['X'], class_2_results_df['Y'], class_2_results_df['Z'],
                       c=class_2_results_df['Z'], cmap="rainbow", marker='^', norm=norm_2)
ax2.set_title(f'Predicted Labels - Class 2 with Rainbow Height Gradient\nFile: {all_test_file_names[0]}')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
fig.colorbar(cmap_rainbow_2, ax=ax2, label='Height (Z)')

plt.tight_layout()
plt.show()


# %%
