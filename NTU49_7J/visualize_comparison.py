"""
NTU49_7J icin gercek vs sahte hareket karsilastirma visualizasyonu.
Gercek ve uretilen hareketleri yan yana GIF olarak kaydeder.

Joint mapping:
    0: ShoulderLeft, 1: ElbowLeft, 2: WristLeft
    3: Center
    4: ShoulderRight, 5: ElbowRight, 6: WristRight
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import sys
import pickle
from tqdm import tqdm

# ============================================================
#  Settings
# ============================================================
SPLIT = "xsub"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, SPLIT)
REAL_DATA_PATH = os.path.join(DATA_DIR, "val_data_joint.npy")
REAL_LABEL_PATH = os.path.join(DATA_DIR, "val_label.pkl")
FAKE_DIR = os.path.join(BASE_DIR, "Results", "Full_Val_Generation")
SAVE_DIR = os.path.join(BASE_DIR, "Results", "comparisons")
NUM_SAMPLES = 20  # Kac adet karsilastirma GIF olusturulacak

# NTU 7J joint renkleri
joint_names = {
    0: 'ShoulderL', 1: 'ElbowL', 2: 'WristL',
    3: 'Center',
    4: 'ShoulderR', 5: 'ElbowR', 6: 'WristR'
}
joint_colors = {
    0: 'green', 1: 'blue', 2: 'cyan',
    3: 'red',
    4: 'magenta', 5: 'orange', 6: 'purple'
}
# NTU kemik baglantilari
connections = [(0, 1), (1, 2), (0, 3), (3, 4), (4, 5), (5, 6)]


def animate_comparison(real_data, fake_data, save_path, title):
    """
    real_data, fake_data: (3, 7, T)
    Yan yana 3D animasyon olusturur ve GIF olarak kaydeder.
    """
    T = min(real_data.shape[2], fake_data.shape[2])
    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    # Axis limitleri icin data range hesapla
    all_data = np.concatenate([real_data, fake_data], axis=-1)
    x_range = [all_data[0].min() - 0.1, all_data[0].max() + 0.1]
    y_range = [all_data[1].min() - 0.1, all_data[1].max() + 0.1]
    z_range = [all_data[2].min() - 0.1, all_data[2].max() + 0.1]

    def update(frame):
        for ax, data, subtitle in zip(
            [ax1, ax2], [real_data, fake_data],
            ["Real (Ground Truth)", "Generated (ActFormer)"]
        ):
            ax.cla()
            ax.set_xlim(x_range)
            ax.set_ylim(y_range)
            ax.set_zlim(z_range)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            # Draw joints
            for i in range(7):
                ax.scatter(
                    data[0, i, frame], data[1, i, frame], data[2, i, frame],
                    color=joint_colors[i], s=40, zorder=5
                )

            # Draw bones
            for i, j in connections:
                xline = [data[0, i, frame], data[0, j, frame]]
                yline = [data[1, i, frame], data[1, j, frame]]
                zline = [data[2, i, frame], data[2, j, frame]]
                ax.plot(xline, yline, zline, color='gray', linewidth=2)

            ax.set_title(f"{subtitle}\nFrame {frame+1}/{T}")

    plt.suptitle(title, fontsize=12, fontweight='bold')
    ani = animation.FuncAnimation(fig, update, frames=T, interval=50)
    ani.save(save_path, writer='pillow')
    plt.close()


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Gercek veriyi yukle
    print(f"Loading real data from {REAL_DATA_PATH}...")
    real_data = np.load(REAL_DATA_PATH)  # (N, 3, 64, 7, 1)
    with open(REAL_LABEL_PATH, 'rb') as f:
        real_names, real_labels = pickle.load(f)

    # Sahte dosyalari indexle
    if not os.path.exists(FAKE_DIR):
        print(f"Fake directory not found: {FAKE_DIR}")
        print("Run test_final.py first!")
        return

    fake_files = os.listdir(FAKE_DIR)
    fake_name_map = {}
    for ff in fake_files:
        if ff.endswith('.npy'):
            name_part = ff.split('_label_')[0] if '_label_' in ff else ff.replace('.npy', '')
            fake_name_map[name_part] = os.path.join(FAKE_DIR, ff)

    # Karsilastirma yap
    found = 0
    for i in tqdm(range(len(real_names)), desc="Creating comparisons"):
        if found >= NUM_SAMPLES:
            break

        name = real_names[i]
        name_key = name.replace('.skeleton', '') if '.skeleton' in name else name

        fake_path = fake_name_map.get(name) or fake_name_map.get(name_key)
        if fake_path is None:
            continue

        # Gercek veri: (3, 64, 7, 1) -> (3, 7, 64)
        real_seq = real_data[i].squeeze(-1)         # (3, 64, 7)
        real_seq = np.transpose(real_seq, (0, 2, 1))  # (3, 7, 64)

        # Sahte veri
        fake_seq = np.load(fake_path)  # (3, 7, 64)

        label = real_labels[i]
        title = f"{name} | Action: {label}"
        safe_name = name.replace('.skeleton', '').replace('/', '_')
        save_path = os.path.join(SAVE_DIR, f"compare_{safe_name}.gif")

        print(f"  [{found+1}/{NUM_SAMPLES}] {name} (Action {label})")
        animate_comparison(real_seq, fake_seq, save_path, title)
        found += 1

    print(f"\n✅ {found} comparison GIFs saved to: {SAVE_DIR}")


if __name__ == "__main__":
    main()
