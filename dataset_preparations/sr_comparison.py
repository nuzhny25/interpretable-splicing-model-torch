import numpy as np


def main():
    data_dir = "../data/human_shuffling/shuffle_embeddings.npz"
    human_dir = "../data/multiz100/embeddings.npz"
    data = np.load(data_dir)
    overall_average_sr = 0

    for key in data.files:
        sr = data[key]
        local_average_sr = float(sr.mean())

        overall_average_sr += local_average_sr

    overall_average_sr = overall_average_sr / len(data.files)

    human_data = np.load(human_dir)
    human_sr = human_data["human_sr"]

    human_average_sr = float(human_sr.mean())

    print(
        f"An average sr balance of 1000 randomly scrambleded human sequences : {overall_average_sr}\n",
        f"                               vs actual human average sr balance : {human_average_sr}",
    )


if __name__ == "__main__":
    main()
