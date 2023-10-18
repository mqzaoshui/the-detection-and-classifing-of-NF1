import os
import shutil
import random

# Specify the directories for the dataset and labels
images_dir = r"E:\NF1-main\NF1_data\images\total"
labels_dir = r"E:\NF1-main\NF1_data\labels\total"

# Specify the output directories for the training and validation sets
train_images_out_dir = r"E:\NF1-main\NF1_data\images\train"
train_labels_out_dir = r"E:\NF1-main\NF1_data\labels\train"
val_images_out_dir = r"E:\NF1-main\NF1_data\images\val"
val_labels_out_dir = r"E:\NF1-main\NF1_data\labels\val"

# Get all image files
image_files = os.listdir(images_dir)
random.shuffle(image_files)  # Shuffle the order of files

# Calculate the sizes of the training and validation sets
num_train = int(len(image_files) * 0.8)
train_files = image_files[:num_train]
val_files = image_files[num_train:]

# Create output directories if they don't exist
os.makedirs(train_images_out_dir, exist_ok=True)
os.makedirs(train_labels_out_dir, exist_ok=True)
os.makedirs(val_images_out_dir, exist_ok=True)
os.makedirs(val_labels_out_dir, exist_ok=True)

# Copy files to the corresponding output directories
for file in train_files:
    shutil.copy(os.path.join(images_dir, file), train_images_out_dir)
    label_file = os.path.splitext(file)[0] + '.json'
    shutil.copy(os.path.join(labels_dir, label_file), train_labels_out_dir)

for file in val_files:
    shutil.copy(os.path.join(images_dir, file), val_images_out_dir)
    label_file = os.path.splitext(file)[0] + '.json'
    shutil.copy(os.path.join(labels_dir, label_file), val_labels_out_dir)

print("it is completed")
