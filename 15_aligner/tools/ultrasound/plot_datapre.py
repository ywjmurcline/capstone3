import numpy as np
import matplotlib.pyplot as plt
import sys

def load_datapre(path):
    d = np.load(path)
    arr = np.squeeze(d["datapre"])  # (100, 10740)
    return arr

def plot_heatmap(ax, arr):
    im = ax.imshow(arr, aspect="auto", origin="lower", cmap="RdBu_r",
                   vmin=-np.percentile(np.abs(arr), 99),
                   vmax= np.percentile(np.abs(arr), 99))
    ax.set_xlabel("Time sample")
    ax.set_ylabel("Channel")
    ax.set_title("datapre — heatmap")
    plt.colorbar(im, ax=ax, label="Amplitude")

def plot_channels(ax, arr, step=10):
    n_channels = arr.shape[0]
    x = np.arange(arr.shape[1])
    scale = np.std(arr) * 3 or 1.0
    for i in range(0, n_channels, step):
        ax.plot(x, arr[i] / scale + i, color="steelblue", linewidth=0.4, alpha=0.8)
    ax.set_xlabel("Time sample")
    ax.set_ylabel("Channel (offset)")
    ax.set_title(f"datapre — channels (every {step}th)")
    ax.set_ylim(-1, n_channels)

def main(path="0-0-0.npz"):
    arr = load_datapre(path)
    print(f"Loaded: shape={arr.shape}, dtype={arr.dtype}")
    print(f"  min={arr.min():.4f}  max={arr.max():.4f}  std={arr.std():.6f}")

    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    plot_heatmap(axes[0], arr)
    plot_channels(axes[1], arr, step=5)

    plt.tight_layout()
    out = path.replace(".npz", "_datapre.png")
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.show()

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "0-0-0.npz"
    main(path)
