"""TODO: Add docstring."""
import matplotlib.pyplot as plt
from dialect.utils.postprocessing import compute_epsilon_threshold


def main() -> None:
    """TODO: Add docstring."""
    num_sample_vals = range(30, 1000, 10)
    epsilons = []
    for num_samples in num_sample_vals:
        epsilon = compute_epsilon_threshold(num_samples)
        epsilons.append(epsilon)

    plt.figure(figsize=(12, 8))
    plt.plot(num_sample_vals, epsilons, marker="o", markersize=5, linewidth=2)
    plt.title("Epsilon Threshold Across Subtypes", fontsize=20)
    plt.xlabel("Number of Samples", fontsize=16)
    plt.ylabel("Epsilon Threshold", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(visible=True)
    plt.savefig("figures/epsilon_threshold_across_subtypes.png")

if __name__ == "__main__":
    main()
