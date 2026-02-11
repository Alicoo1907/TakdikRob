import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import h5py
from tqdm import tqdm

# === Settings ===
H5_PATH = "Dataset/HDF5_Dataset_60frame/motions_data60frame.h5"
FAKE_DIR = "Results/K18_Generation"
SAVE_DIR = "Results/k18_comparisons"
NUM_SAMPLES = 60  # Number of comparisons to create
SUBJECT_ID = "K18"

joint_colors = {0: 'red', 1: 'green', 2: 'blue', 3: 'cyan', 4: 'magenta', 5: 'yellow', 6: 'black'}
connections = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6)]

def get_real_seq(h5_path, group_name):
    with h5py.File(h5_path, 'r') as f:
        group = f[group_name]
        frame_keys = sorted(group.keys(), key=lambda x: int(x))
        T = min(len(frame_keys), 60)
        seq = np.zeros((3, 7, T), dtype=np.float32)
        for t, frame_num in enumerate(frame_keys[:T]):
            for j, joint_name in enumerate([
                'Center', 'ShoulderLeft', 'ElbowLeft', 'WristLeft',
                'ShoulderRight', 'ElbowRight', 'WristRight'
            ]):
                seq[0, j, t] = group[frame_num][f'{joint_name}/X'][()]
                seq[1, j, t] = group[frame_num][f'{joint_name}/Y'][()]
                seq[2, j, t] = group[frame_num][f'{joint_name}/Z'][()]
        return seq

def animate_comparison(real_data, fake_data, save_path, title):
    # data: (3, 7, T)
    T = min(real_data.shape[2], fake_data.shape[2])
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    def update(frame):
        for ax, data, subtitle in zip([ax1, ax2], [real_data, fake_data], ["Real (Ground Truth)", "Generated (ActFormer)"]):
            ax.cla()
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_zlim([-1, 1])
            
            # Draw joints
            for i in range(7):
                ax.scatter(data[0, i, frame], data[1, i, frame], data[2, i, frame], color=joint_colors[i])
            
            # Draw connections
            for i, j in connections:
                xline = [data[0, i, frame], data[0, j, frame]]
                yline = [data[1, i, frame], data[1, j, frame]]
                zline = [data[2, i, frame], data[2, j, frame]]
                ax.plot(xline, yline, zline, color='gray')
            
            ax.set_title(f"{subtitle}\nFrame {frame+1}/{T}")

    plt.suptitle(title)
    ani = animation.FuncAnimation(fig, update, frames=T, interval=50)
    ani.save(save_path, writer='pillow')
    plt.close()

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Get K18 sequences from H5
    with h5py.File(H5_PATH, 'r') as f:
        k18_keys = [k for k in f.keys() if k.startswith(SUBJECT_ID)]
    
    found_count = 0
    for group_name in k18_keys:
        if found_count >= NUM_SAMPLES:
            break
            
        # Find matching fake file
        fake_file = None
        for file in os.listdir(FAKE_DIR):
            if group_name in file:
                fake_file = file
                break
        
        if fake_file:
            print(f"Comparing: {group_name}...")
            real_seq = get_real_seq(H5_PATH, group_name)
            fake_seq = np.load(os.path.join(FAKE_DIR, fake_file))
            
            save_path = os.path.join(SAVE_DIR, f"compare_{group_name}.gif")
            animate_comparison(real_seq, fake_seq, save_path, group_name)
            found_count += 1
            
    print(f"\n✅ Comparison GIFs saved to: {SAVE_DIR}")

if __name__ == "__main__":
    main()
