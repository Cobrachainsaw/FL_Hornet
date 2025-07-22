import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_curves(metrics_dir="metrics", metric_name="loss"):
    """
    Load all client curves for a given metric ('loss' or 'accuracy')
    Returns: {client_id: [round1, round2, ...]}
    """
    curves = {}
    metrics_path = Path(metrics_dir)

    for client_file in metrics_path.glob(f"client_*_{metric_name}_curve.jsonl"):
        client_id = client_file.stem.split("_")[1]
        with open(client_file, "r") as f:
            rounds = []
            for line in f:
                curve = json.loads(line)
                rounds.append(curve)
            curves[client_id] = rounds

    return curves


def average_curves(curves):
    """
    Average multiple rounds per client to get a single mean curve.
    """
    avg_curves = {}
    for client_id, rounds in curves.items():
        # Pad to same length if needed
        max_len = max(len(r) for r in rounds)
        padded = []
        for r in rounds:
            if len(r) < max_len:
                r = r + [r[-1]] * (max_len - len(r))
            padded.append(r)
        avg_curve = np.mean(padded, axis=0)
        avg_curves[client_id] = avg_curve
    return avg_curves


def plot_all(loss_curves, accuracy_curves):
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # 1️⃣ Loss per round
    for client_id, rounds in loss_curves.items():
        for r, curve in enumerate(rounds):
            axes[0, 0].plot(
                curve,
                label=f"Client {client_id} Round {r + 1}",
                alpha=0.5,
            )
    axes[0, 0].set_title("Loss per Round")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True)
    axes[0, 0].legend()

    # 2️⃣ Accuracy per round
    for client_id, rounds in accuracy_curves.items():
        for r, curve in enumerate(rounds):
            axes[0, 1].plot(
                curve,
                label=f"Client {client_id} Round {r + 1}",
                alpha=0.5,
            )
    axes[0, 1].set_title("Accuracy per Round")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].grid(True)
    axes[0, 1].legend()

    # 3️⃣ Avg Loss per client
    avg_loss_curves = average_curves(loss_curves)
    for client_id, avg_curve in avg_loss_curves.items():
        axes[1, 0].plot(
            avg_curve,
            label=f"Client {client_id}",
            linewidth=2,
        )
    axes[1, 0].set_title("Avg Loss per Client")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].grid(True)
    axes[1, 0].legend()

    # 4️⃣ Avg Accuracy per client
    avg_accuracy_curves = average_curves(accuracy_curves)
    for client_id, avg_curve in avg_accuracy_curves.items():
        axes[1, 1].plot(
            avg_curve,
            label=f"Client {client_id}",
            linewidth=2,
        )
    axes[1, 1].set_title("Avg Accuracy per Client")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Accuracy")
    axes[1, 1].grid(True)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    loss_curves = load_curves(metric_name="loss")
    accuracy_curves = load_curves(metric_name="accuracy")

    plot_all(loss_curves, accuracy_curves)
