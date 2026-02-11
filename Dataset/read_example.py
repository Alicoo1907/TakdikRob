"""import h5py

# Function to read and print the first 3 joints (Center, ShoulderLeft, ElbowLeft)
def print_first_three_joints(hdf5_file_path, group_name, frame_num):
    with h5py.File(hdf5_file_path, 'r') as h5f:
        # Access the group (e.g., K01_A001_R01)
        group = h5f[group_name]

        # Access the specified frame (e.g., K01_A001_R01/1)
        frame_group = group[str(frame_num)]

        # Define the first 3 joints
        first_three_joints = ['Center', 'ShoulderLeft', 'ShoulderRight']

        # Print the X, Y, Z values for the first 3 joints
        for joint_name in first_three_joints:
            if joint_name in frame_group:
                # Access the scalar data directly without [:] for scalar types
                x_data = frame_group[f"{joint_name}/X"][()]
                y_data = frame_group[f"{joint_name}/Y"][()]
                z_data = frame_group[f"{joint_name}/Z"][()]
                print(f"{joint_name}: X = {x_data}, Y = {y_data}, Z = {z_data}")
            else:
                print(f"{joint_name} not found in frame {frame_num}")

# Example usage
hdf5_file_path = 'HDF5_Dataset/motions_data.h5'  # Path to your HDF5 file
group_name = 'K01_A001_R01'  # Example group name (e.g., K01_A001_R01)
frame_num = 1 # Example frame number to access

# Print the first 3 joints (Center, ShoulderLeft, ElbowLeft)
#print_first_three_joints(hdf5_file_path, group_name, frame_num)

def count_frames_for_all_groups(hdf5_file_path):
    frame_counts = {}

    with h5py.File(hdf5_file_path, 'r') as h5f:
        for group_name in h5f.keys():
            group = h5f[group_name]
            frame_keys = [key for key in group.keys() if key.isdigit()]
            frame_counts[group_name] = len(frame_keys)

    # Sıralı yazdır (küçükten büyüğe)
    for group_name, count in sorted(frame_counts.items(), key=lambda x: x[1]):
        print(f"{group_name}: {count} frames")

count_frames_for_all_groups(hdf5_file_path)"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# === Ortak ayarlar ===
joint_colors = {
    0: 'red',
    1: 'green',
    2: 'blue',
    3: 'cyan',
    4: 'magenta',
    5: 'yellow',
    6: 'black'
}

joint_names = {
    0: 'Center',
    1: 'ShoulderLeft',
    2: 'ElbowLeft',
    3: 'WristLeft',
    4: 'ShoulderRight',
    5: 'ElbowRight',
    6: 'WristRight'
}

connections = [
    (0, 1), (1, 2), (2, 3),       # Sol kol
    (0, 4), (4, 5), (5, 6)        # Sağ kol
]

# === Animasyon fonksiyonu ===
def animate_real_motion(real_data, save_path):
    T = real_data.shape[2]
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')

    def update(frame):
        ax.cla()
        ax.set_title(f"Real Motion - Frame {frame + 1}/{T}")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)

        for i in range(7):
            x, y, z = real_data[i, 0, frame], real_data[i, 1, frame], real_data[i, 2, frame]
            ax.scatter(x, y, z, color=joint_colors[i])

        for i, j in connections:
            xline = [real_data[i, 0, frame], real_data[j, 0, frame]]
            yline = [real_data[i, 1, frame], real_data[j, 1, frame]]
            zline = [real_data[i, 2, frame], real_data[j, 2, frame]]
            ax.plot(xline, yline, zline, color='gray')

    ani = animation.FuncAnimation(fig, update, frames=T, interval=100)
    ani.save(save_path, writer='pillow')
    print(f"GIF saved: {save_path}")

# === HDF5'ten veriyi oku ve görselleştir ===
def create_gif_from_hdf5(hdf5_path, group_name, save_path, max_frames=None):
    with h5py.File(hdf5_path, 'r') as h5f:
        group = h5f[group_name]
        frame_keys = sorted([k for k in group.keys() if k.isdigit()], key=lambda x: int(x))

        if max_frames:
            frame_keys = frame_keys[:max_frames]

        T = len(frame_keys)
        real_data = np.zeros((7, 3, T))

        for t, frame_key in enumerate(frame_keys):
            frame = group[frame_key]
            for j in range(7):
                joint = joint_names[j]
                try:
                    real_data[j, 0, t] = frame[f"{joint}/X"][()]
                    real_data[j, 1, t] = frame[f"{joint}/Y"][()]
                    real_data[j, 2, t] = frame[f"{joint}/Z"][()]
                except KeyError:
                    print(f"Joint {joint} not found in frame {frame_key}")

    animate_real_motion(real_data, save_path)

# === Kullanım ===
hdf5_path = "HDF5_Dataset60frame/motions_data60frame.h5"
group_name = "K08_A001_R01"
save_path = f"real_motion_{group_name}_60frmae.gif"

os.makedirs("Results/animations", exist_ok=True)
create_gif_from_hdf5(hdf5_path, group_name, save_path, max_frames=995)