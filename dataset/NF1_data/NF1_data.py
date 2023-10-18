import json
import os

def convert_json_to_txt(json_folder):
    # Category mapping
    category_mapping = {
        "NF1": 0,
        "healthy skin": 1
    }

    # Ensure the folder exists
    if not os.path.exists(json_folder):
        print(f"Folder {json_folder} does not exist.")
        return

    # Iterate through all json files in the folder
    for file_name in os.listdir(json_folder):
        if file_name.endswith(".json"):
            file_path = os.path.join(json_folder, file_name)
            with open(file_path, "r") as json_file:
                data = json.load(json_file)

                # Create a lookup table based on the categories field
                id_to_name_mapping = {category["id"]: category["name"] for category in data["categories"]}

                # Get the width and height of the image
                for image in data["images"]:
                    width = image["width"]
                    height = image["height"]

                txt_content = []
                for annotation in data["annotations"]:
                    x_min, y_min, w, h = annotation["bbox"]
                    # Correct bbox information to ensure their values are valid
                    x_max = x_min + w
                    y_max = y_min + h
                    x_min = max(0, x_min)
                    y_min = max(0, y_min)
                    x_max = min(width, x_max)
                    y_max = min(height, y_max)

                    # Calculate the center coordinates
                    cx = x_min + w / 2
                    cy = y_min + h / 2
                    # Normalize the coordinates
                    cx /= width
                    cy /= height
                    w /= width
                    h /= height

                    # Get the category ID from the name
                    category_name = id_to_name_mapping.get(annotation['category_id'])
                    category_id = category_mapping.get(category_name, -1)
                    if category_id == -1:
                        print(f"Warning: Unknown category {category_name}")
                        continue

                    txt_content.append(f"{category_id} {cx} {cy} {w} {h}")

                # Save the data to a txt file
                txt_path = os.path.join(json_folder, file_name.replace(".json", ".txt"))
                with open(txt_path, "w") as txt_file:
                    txt_file.write("\n".join(txt_content))


convert_json_to_txt("labels/val")

