import numpy as np
import matplotlib.pyplot as plt
import glob


def load_training_data():
    files = sorted(glob.glob("training_returns_seed*.npy"))
    all_seeds = []

    for f in files:
        data = np.load(f)
        all_seeds.append(data)

    max_len = max(len(x) for x in all_seeds)
    aligned = []
    for x in all_seeds:
        pad = np.pad(x, (0, max_len - len(x)), mode='edge')
        aligned.append(pad)

    return np.array(aligned)


def load_evaluation_data():
    files = sorted(glob.glob("*_eval_seed10.npy"))
    all_seeds = []

    for f in files:
        data = np.load(f)
        all_seeds.append(data)

    max_len = max(len(x) for x in all_seeds)
    aligned = []
    for x in all_seeds:
        pad = np.pad(x, (0, max_len - len(x)), mode='edge')
        aligned.append(pad)

    return np.array(aligned)


def plot_with_mean_std(data, title, xlabel, ylabel, filename):
    mean_curve = data.mean(axis=0)
    std_curve = data.std(axis=0)

    length = len(mean_curve)

    if length <= 25:
        x = np.arange(1, length + 1)
    else:
        x = np.arange(length)

    plt.figure(figsize=(8,5))

    if length <= 25:
        plt.xticks(x)

    plt.plot(x, mean_curve, label="Mean", color="blue")
    plt.fill_between(
        x,
        mean_curve - std_curve,
        mean_curve + std_curve,
        color="blue",
        alpha=0.2,
        label="Std"
    )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()

    plt.savefig(filename, dpi=300)
    plt.show()

def main():
    # ===== Training plot =====
    train_data = load_training_data()
    print(f"Loaded training data shape: {train_data.shape}")

    plot_with_mean_std(
        train_data,
        title="Training Plot (Mean & Std)",
        xlabel="Episode/Epochs",
        ylabel="Reward/Returns",
        filename="training_mean_std.png"
    )

    # ===== Evaluation plot =====
    eval_data = load_evaluation_data()
    print(f"Loaded evaluation data shape: {eval_data.shape}")

    plot_with_mean_std(
        eval_data,
        title="Evaluation Plot (Mean & Std)",
        xlabel="Episode/Epochs",
        ylabel="Reward/Returns",
        filename="evaluation_mean_std.png"
    )


if __name__ == "__main__":
    main()
