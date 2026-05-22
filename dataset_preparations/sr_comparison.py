import numpy as np


def main():
    data_dir = "../data/human_shuffling/shuffle_embeddings.npz"
    human_dir = "../data/multiz100/embeddings.npz"
    data = np.load(data_dir)
    per_sequence_average_sr = []

    for key in data.files:
        sr = data[key]
        per_sequence_average_sr.append(float(sr.mean()))

    overall_average_sr = float(np.mean(per_sequence_average_sr))
    median_average_sr = float(np.median(per_sequence_average_sr))
    std_average_sr = float(np.std(per_sequence_average_sr, ddof=1))

    human_data = np.load(human_dir)
    human_sr = human_data["human_sr"]

    human_average_sr = float(human_sr.mean())

    z_score = (human_average_sr - overall_average_sr) / std_average_sr

    print(
        f"An average sr balance of 1000 randomly scrambleded human sequences : {overall_average_sr}\n",
        f" A median sr balance of 1000 randomly scrambleded human sequences : {median_average_sr}\n",
        f"                               vs actual human average sr balance : {human_average_sr}\n",
        f"      human average sr balance is {z_score:.4f} standard deviations from the shuffled mean",
    )


if __name__ == "__main__":
    main()
