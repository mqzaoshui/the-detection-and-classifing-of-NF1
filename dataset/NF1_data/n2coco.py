import json
import os
import glob

# Category mapping
category_mapping = {
    "NF1": 1,
    "healthy skin": 2
}

# Source and target folders
source_folder = r"C:\Users\15690\Desktop\500train\数据集\NF1_data\labels\train" # Replace with the folder where your JSON files are located
target_folder = r"C:\Users\15690\Desktop\500train\数据集\NF1_data\coco"  # Replace with the folder where you want to save the new JSON files

# Create the target folder if it doesn't exist
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# Iterate through all JSON files in the source folder
for json_file in glob.glob(os.path.join(source_folder, '*.json')):
    with open(json_file, 'r') as file:
        data = json.load(file)

    # Initialize the data structure for COCO format
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "NF1"},
            {"id": 2, "name": "healthy skin"}
        ]
    }

    # Add image information
    image_id = os.path.splitext(os.path.basename(data['imagePath']))[0]
    coco_data['images'].append({
        "file_name": data['imagePath'],
        "height": data['imageHeight'],
        "width": data['imageWidth'],
        "id": image_id
    })

    # Add annotation information
    for i, shape in enumerate(data['shapes']):
        points = shape['points']
        x, y = zip(*points)
        bbox = [min(x), min(y), max(x) - min(x), max(y) - min(y)]

        # Set category_id based on the annotation's category
        category_id = category_mapping.get(shape['label'], None)
        if category_id is None:
            print(f"Warning: Unknown category {shape['label']}")
            continue

        # Flatten the points list
        segmentation = [item for sublist in points for item in sublist]

        coco_data['annotations'].append({
            "bbox": bbox,
            "category_id": category_id,
            "image_id": image_id,
            "id": i + 1,
            "segmentation": [segmentation],  # Update segmentation information
            "area": (bbox[2] * bbox[3]),  # Add area information
            "iscrowd": 0  # Add iscrowd information
        })

    # Get the JSON file's name to create a new file name
    base_name = os.path.splitext(os.path.basename(json_file))[0] + '.json'
    new_file_name = os.path.join(target_folder, base_name)

    # Save as a COCO format JSON file
    with open(new_file_name, 'w') as file:
        json.dump(coco_data, file)

    print(f"Converted {json_file} to COCO format and saved to {new_file_name}")

