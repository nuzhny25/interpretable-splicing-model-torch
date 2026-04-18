paths = ["data/human_malat1.txt", "data/mouse_malat1.txt"]
names = ["data/human_malat1_chunks.csv", "data/mouse_malat1_chunks.csv"]

for index, path in enumerate(paths):

    with open(path, "r") as file:
        malat1_seq = file.read()

    malat1_seq = malat1_seq.replace("\n", "")

    window_size = 70
    step_size = 10

    chunks = []
    positions = []

    for i in range(0, len(malat1_seq) - window_size + 1, step_size):
        chunk = malat1_seq[i : i + window_size]
        chunks.append(chunk)
        positions.append(i)

    import pandas as pd

    df = pd.DataFrame({"exon": chunks})
    df.to_csv(names[index], index=False)
