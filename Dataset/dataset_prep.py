import os
import json
import h5py
import numpy as np

# Define the joints (7 joints as before)
joints = [
    "Center", "ShoulderLeft", "ElbowLeft", "WristLeft",
    "ShoulderRight", "ElbowRight", "WristRight"
]

# Function to calculate the Center joint as the average of ShoulderLeft and ShoulderRight
def calculate_center(shoulder_left, shoulder_right):
    center = {
        "X": (shoulder_left["X"] + shoulder_right["X"]) / 2,
        "Y": (shoulder_left["Y"] + shoulder_right["Y"]) / 2,
        "Z": (shoulder_left["Z"] + shoulder_right["Z"]) / 2
    }
    return center

# Function to process the JSON files and save the data as HDF5
def process_and_save_data(subject_dir, output_dir):
    # Create a single HDF5 file for all subjects
    hdf5_file_path = os.path.join(output_dir, 'motions_data60frame.h5')

    with h5py.File(hdf5_file_path, 'w') as h5f:
        # Iterate over the subjects (Subject01 to Subject19)
        for subject in range(1, 20):
            subject_folder = os.path.join(subject_dir, f"Subject {subject:02d}")

            # Check if subject folder exists
            if not os.path.exists(subject_folder):
                continue

            # Iterate over the JSON files in the subject folder
            for json_file in os.listdir(subject_folder):
                if json_file.endswith(".json"):
                    json_file_path = os.path.join(subject_folder, json_file)

                    # Load the JSON data
                    with open(json_file_path, 'r') as f:
                        data = json.load(f)

                    # Create a group for this JSON file
                    group_name = json_file.replace('.json', '')
                    group = h5f.create_group(group_name)

                    # Iterate through the frames in the JSON file
                    # Total number of frames
                    n_frames = len(data)

                    # Select 60 evenly spaced frame indices (including first and last)
                    selected_indices = np.linspace(0, n_frames - 1, num=60, dtype=int)

                    # Iterate only over selected frames
                    for count, frame_index in enumerate(selected_indices):
                        frame = data[frame_index]
                        frame_group = group.create_group(str(count + 1))

                        joints_data = {}

                        # Extract joint data
                        shoulder_left = frame['ShoulderLeft']
                        elbow_left = frame['ElbowLeft']
                        wrist_left = frame['WristLeft']
                        shoulder_right = frame['ShoulderRight']
                        elbow_right = frame['ElbowRight']
                        wrist_right = frame['WristRight']

                        # Compute center
                        center = calculate_center(shoulder_left, shoulder_right)
                        joints_data["Center"] = [0.0, 0.0, 0.0]

                        for name, joint in [
                            ("ShoulderLeft", shoulder_left),
                            ("ElbowLeft", elbow_left),
                            ("WristLeft", wrist_left),
                            ("ShoulderRight", shoulder_right),
                            ("ElbowRight", elbow_right),
                            ("WristRight", wrist_right)
                        ]:
                            joints_data[name] = [
                                joint["X"] - center["X"],
                                joint["Y"] - center["Y"],
                                joint["Z"] - center["Z"]
                            ]

                        for joint_name in joints:
                            joint_data = joints_data[joint_name]
                            frame_group.create_dataset(f"{joint_name}/X", data=np.array(joint_data[0]), dtype='f4')
                            frame_group.create_dataset(f"{joint_name}/Y", data=np.array(joint_data[1]), dtype='f4')
                            frame_group.create_dataset(f"{joint_name}/Z", data=np.array(joint_data[2]), dtype='f4')

                    print(f"Processed {json_file} and saved to {hdf5_file_path}")

# Example usage
subject_dir = 'Data_fixed'  # The directory containing Subject01 to Subject14
output_dir = 'HDF5_Dataset_60frame'   # Directory where HDF5 files will be saved

# Make sure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Process and save data from Subject01 to Subject14
process_and_save_data(subject_dir, output_dir)
