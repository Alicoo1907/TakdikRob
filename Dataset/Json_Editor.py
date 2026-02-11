"""

When saving Kinect files, there were some parenthesis errors in the json files.
And we used this code to fix them.
K00N's are the files that have been fixed.
T00N's are the files that no need to be fixed.
"""
import os
import json

# Path Directory
base_directory = "a"
new_base_directory = "Data_Fixed145"

# Create new base directory (if it does not exist)
if not os.path.exists(new_base_directory):
    os.makedirs(new_base_directory)

# Browse all subjects' folders
for subject_folder in os.listdir(base_directory):
    subject_path = os.path.join(base_directory, subject_folder)

    # Check if there is a subject folder
    if os.path.isdir(subject_path) and subject_folder.startswith("Subject "):

        # Yeni denek klasörünü oluştur
        new_subject_path = os.path.join(new_base_directory, subject_folder)
        if not os.path.exists(new_subject_path):
            os.makedirs(new_subject_path)

        # Create new subject folder
        for movement_file in os.listdir(subject_path):
            movement_path = os.path.join(subject_path, movement_file)

            # Check if there is a JSON file
            if movement_file.endswith(".json") and movement_file.startswith("K"):
                target_file = movement_path.replace(".json", ".json")

                # Read and edit the file
                with open(movement_path, 'r') as file:
                    content = file.read()
                    frames = []
                    #Parse every JSON object in the file and add it to the list
                    for json_object in content.split("}\n{"):
                        if not json_object.startswith("{"):
                            json_object = "{" + json_object
                        if not json_object.endswith("}"):
                            json_object += "}"
                        try:
                            frames.append(json.loads(json_object))
                        except json.JSONDecodeError as e:
                            print(f"Error: {e}, JSON: {json_object}")

                # Save edited JSON file in new folder
                new_target_file = os.path.join(new_subject_path, movement_file.replace(".json", ".json"))
                with open(new_target_file, 'w') as outfile:
                    json.dump(frames, outfile, indent=4)

                print(f"Edited JSON file saved: {new_target_file}")
