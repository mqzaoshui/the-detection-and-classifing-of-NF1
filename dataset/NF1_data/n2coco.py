import json
import os
import glob

# 类别映射
category_mapping = {
    "NF1": 1,
    "healthy skin": 2
}

# 源文件夹和目标文件夹
source_folder = r"C:\Users\15690\Desktop\500train\数据集\NF1_data\labels\train" # 请替换为你的 JSON 文件所在的文件夹
target_folder = r"C:\Users\15690\Desktop\500train\数据集\NF1_data\coco"  # 请替换为你想要保存新 JSON 文件的文件夹

# 如果目标文件夹不存在，则创建它
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# 遍历源文件夹中的所有 JSON 文件
for json_file in glob.glob(os.path.join(source_folder, '*.json')):
    with open(json_file, 'r') as file:
        data = json.load(file)

    # 初始化 COCO 格式的数据结构
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "NF1"},
            {"id": 2, "name": "healthy skin"}
        ]
    }

    # 添加图像信息
    image_id = os.path.splitext(os.path.basename(data['imagePath']))[0]
    coco_data['images'].append({
        "file_name": data['imagePath'],
        "height": data['imageHeight'],
        "width": data['imageWidth'],
        "id": image_id
    })

    # 添加标注信息
    for i, shape in enumerate(data['shapes']):
        points = shape['points']
        x, y = zip(*points)
        bbox = [min(x), min(y), max(x) - min(x), max(y) - min(y)]

        # 根据标注的类别设置 category_id
        category_id = category_mapping.get(shape['label'], None)
        if category_id is None:
            print(f"Warning: Unknown category {shape['label']}")
            continue

        # 展平 points 列表
        segmentation = [item for sublist in points for item in sublist]

        coco_data['annotations'].append({
            "bbox": bbox,
            "category_id": category_id,
            "image_id": image_id,
            "id": i + 1,
            "segmentation": [segmentation],  # 更新 segmentation 信息
            "area": (bbox[2] * bbox[3]),  # 添加 area 信息
            "iscrowd": 0  # 添加 iscrowd 信息
        })

    # 获取 JSON 文件的文件名，用于创建新的文件名
    base_name = os.path.splitext(os.path.basename(json_file))[0] + '.json'
    new_file_name = os.path.join(target_folder, base_name)

    # 保存为 COCO 格式的 JSON 文件
    with open(new_file_name, 'w') as file:
        json.dump(coco_data, file)

    print(f"Converted {json_file} to COCO format and saved to {new_file_name}")
