import json
import os

def convert_json_to_txt(json_folder):
    # 类别映射
    category_mapping = {
        "NF1": 0,
        "healthy skin": 1
    }

    # 确保文件夹存在
    if not os.path.exists(json_folder):
        print(f"Folder {json_folder} does not exist.")
        return

    # 遍历文件夹中的所有json文件
    for file_name in os.listdir(json_folder):
        if file_name.endswith(".json"):
            file_path = os.path.join(json_folder, file_name)
            with open(file_path, "r") as json_file:
                data = json.load(json_file)

                # 创建基于categories字段的查找表
                id_to_name_mapping = {category["id"]: category["name"] for category in data["categories"]}

                # 获取图像的宽度和高度
                for image in data["images"]:
                    width = image["width"]
                    height = image["height"]

                txt_content = []
                for annotation in data["annotations"]:
                    x_min, y_min, w, h = annotation["bbox"]
                    # 修正bbox信息以确保它们的值是正确的
                    x_max = x_min + w
                    y_max = y_min + h
                    x_min = max(0, x_min)
                    y_min = max(0, y_min)
                    x_max = min(width, x_max)
                    y_max = min(height, y_max)

                    # 计算中心点坐标
                    cx = x_min + w / 2
                    cy = y_min + h / 2
                    # 归一化坐标
                    cx /= width
                    cy /= height
                    w /= width
                    h /= height

                    # 从名称中获取类别ID
                    category_name = id_to_name_mapping.get(annotation['category_id'])
                    category_id = category_mapping.get(category_name, -1)
                    if category_id == -1:
                        print(f"Warning: Unknown category {category_name}")
                        continue

                    txt_content.append(f"{category_id} {cx} {cy} {w} {h}")

                # 将数据保存到txt文件中
                txt_path = os.path.join(json_folder, file_name.replace(".json", ".txt"))
                with open(txt_path, "w") as txt_file:
                    txt_file.write("\n".join(txt_content))


convert_json_to_txt("labels/val")
