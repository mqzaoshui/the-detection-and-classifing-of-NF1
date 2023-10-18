import os
import shutil
import random

# 指定数据集和标签的目录
images_dir = r"E:\NF1-main\NF1_data\images\total"
labels_dir = r"E:\NF1-main\NF1_data\labels\total"

# 指定训练集和验证集的输出目录
train_images_out_dir = r"E:\NF1-main\NF1_data\images\train"
train_labels_out_dir = r"E:\NF1-main\NF1_data\labels\train"
val_images_out_dir = r"E:\NF1-main\NF1_data\images\val"
val_labels_out_dir = r"E:\NF1-main\NF1_data\labels\val"

# 获取所有图像文件
image_files = os.listdir(images_dir)
random.shuffle(image_files)  # 打乱文件顺序

# 计算训练集和验证集的大小
num_train = int(len(image_files) * 0.8)
train_files = image_files[:num_train]
val_files = image_files[num_train:]


# 创建输出目录（如果不存在）
os.makedirs(train_images_out_dir, exist_ok=True)
os.makedirs(train_labels_out_dir, exist_ok=True)
os.makedirs(val_images_out_dir, exist_ok=True)
os.makedirs(val_labels_out_dir, exist_ok=True)

# 复制文件到对应的输出目录
for file in train_files:
    shutil.copy(os.path.join(images_dir, file), train_images_out_dir)
    label_file = os.path.splitext(file)[0] + '.json'
    shutil.copy(os.path.join(labels_dir, label_file), train_labels_out_dir)

for file in val_files:
    shutil.copy(os.path.join(images_dir, file), val_images_out_dir)
    label_file = os.path.splitext(file)[0] + '.json'
    shutil.copy(os.path.join(labels_dir, label_file), val_labels_out_dir)

print("it is completed")
