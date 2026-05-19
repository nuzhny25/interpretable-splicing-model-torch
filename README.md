PyTorch implementation of pre-trained splicing model from "Deciphering RNA splicing logic with interpretable machine learning" (Liao et al., 2023). The manuscript is available [here](https://www.pnas.org/doi/abs/10.1073/pnas.2221165120). Train, test sequences are provided in the `data` directory.

## Setup
### Requirements

Python-side requirements:

- Python 3.9+
- `pandas`
- `numpy`
- `torch`
- `tqdm`

Install them with:

```bash
pip install pandas numpy torch tqdm
```

System requirement:

- ViennaRNA `RNAfold`

### RNAfold Installation

See the [ViennaRNA GitHub](https://github.com/ViennaRNA/ViennaRNA) for instructions for installing RNAFold. The preprocessing code expects `RNAfold` to be available on `PATH`, unless you pass an explicit `rnafold_bin` path.


## Usage
### Expected Input Shapes

The model expects channel-first inputs throughout:

- sequence one-hot: `(N, 4, L)`
- wobble: `(N, 1, L)`
- structure: `(N, 3, L)`

In this repository, preprocessing utilities already return arrays in those shapes.

### From CSV to Full Inputs

The canonical input sequence is an unflanked exon in a column named `exon`. Other columns are allowed and are preserved as metadata in the output dataset. If your sequence column has a different name, pass it explicitly.

By default, preprocessing adds the fixed model flanks:

- left flank: `CATCCAGGTT`
- right flank: `CAGGTCTGAC`

That means a 70 nt exon becomes a 90 nt model input.

If you do not want flanks added, use `add_flanks=False` in Python or `--no-flanks` in the CLI. In that case, `L` is just the input sequence length.

### Dataset Preparation

You can prepare a dataset directly from a CSV file containing an 'exon' column.

```bash
python prepare_dataset.py \
  --input-csv input.csv \
  --output-path dataset.npz
```

If your sequence column is not `exon`:

```bash
python prepare_dataset.py \
  --input-csv input.csv \
  --output-path dataset.npz \
  --sequence-column sequence
```

Optional RNAfold-related arguments: `--rnafold-bin`, `--temperature`, `--max-bp-span`, `--num-threads` and `--commands-file`.

The output is a compressed `.npz` archive containing the following keys (stored as numpy arrays). Specifically, it contains the fields `seq_oh` for one-hot-encoded sequences, `struct_oh` for one-hot-encoded structures, `wobble` for wobble-pair arrays. All additional dataframe columns are stored with the `"metadata_"` prefix as (e.g. `dataset["metadata_PSI"]`).

### Load dataset

```python
import numpy as np

dataset = np.load("your_dataset_path.npz")

print(dataset["seq_oh"].shape)      # (2, 4, 90)
print(dataset["struct_oh"].shape)   # (2, 3, 90)
print(dataset["wobbles"].shape)     # (2, 1, 90)
print(dataset["structure"].shape)   # (2,)
print(dataset["mfe"].shape)         # (2,)
```

### Running the Model

After preprocessing, convert the NumPy arrays to torch tensors and pass them into `PNASModel.forward()`:

Initialize `PNASModel` with an `input_length` that matches the prepared sequence length `L`. If flanks are added, this length includes the flanking nucleotides.

```python
import torch
from model import PNASModel

x_seq = torch.tensor(dataset["seq_oh"], dtype=torch.float32)
x_struct = torch.tensor(dataset["struct_oh"], dtype=torch.float32)
x_wobble = torch.tensor(dataset["wobbles"], dtype=torch.float32)

model = PNASModel(input_length=x_seq.shape[-1])
state_dict = torch.load("model_weights.pt", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

with torch.no_grad():
    prediction = model(x_seq, x_struct, x_wobble)
```

### Sequence-Only Analysis

To inspect sequence properties such as **SR Balance** and **latent sequence activation**, you can use the model methods `compute_sr_balance`, `compute_sequence_activations()`.

```python
# Get one-hot sequences
x_seq = torch.tensor(dataset["seq_oh"], dtype=torch.float32)

# Compute inclusion, skipping sequence activations
a_incl, a_skip = model.compute_sequence_activations(x_seq, agg="mean")

# Compute SR balance
sr_balance = model.compute_sr_balance(x_seq, agg="mean")
```

## Training

`train.py` trains `PNASModel` from a prepared `.npz` dataset. It expects the dataset to contain a `metadata_PSI` column as the regression target (produced automatically by `prepare_dataset.py` if a `PSI` column is present in the source CSV).

#### Quickstart

```bash
python train.py \
    --train-npz data/train_data.npz \
    --epochs 1 \
    --batch-size 64 \
    --patience 1 \
    --checkpoint-dir /tmp/pnas_test \
    --seed 0
```

This runs a single epoch on the provided training data from a randomly initialized model and saves the best checkpoint to `/tmp/pnas_test/`.

#### Full argument reference

| Group | Argument | Default | Description |
|---|---|---|---|
| data | `--train-npz` | *(required)* | Training `.npz` produced by `prepare_dataset.py` |
| data | `--test-npz` | — | Optional held-out set; evaluated once with the best checkpoint after training |
| data | `--val-split` | `0.1` | Fraction of training data used for validation |
| model | `--input-length` | `90` | Input sequence length passed to `PNASModel` |
| model | `--no-batchnorm` | off | Replace `BatchNorm1d` layers in `ResidualTuner` with `nn.Identity` |
| model | `--checkpoint` | — | Warm-start from a checkpoint; partial checkpoints (seq filters only, seq+struct, or full model) are supported — missing parameters stay at random init |
| optimization | `--batch-size` | `64` | |
| optimization | `--epochs` | `100` | Maximum training epochs |
| optimization | `--lr` | `1e-3` | Adam learning rate |
| optimization | `--weight-decay` | `0.0` | Adam weight decay |
| optimization | `--patience` | `10` | Early stopping patience (epochs without val loss improvement) |
| runtime | `--checkpoint-dir` | `./checkpoints` | Directory for saving best-model checkpoints |
| runtime | `--device` | *(auto)* | Torch device string, e.g. `cpu`, `cuda`, `cuda:1`; auto-detects CUDA if omitted |
| runtime | `--seed` | `42` | Seeds `random`, `numpy`, and `torch` |

#### Staged / warm-start training

`--checkpoint` accepts both full and partial checkpoints. A partial checkpoint contains only a subset of parameter names — for example, just the sequence convolution filters and their position biases — and leaves all other parameters at random initialization. This supports workflows where sequence filters are learned first and a new tuner head is trained on top.

```bash
# Stage 1 — train sequence filters only (e.g. after stripping the checkpoint to seq keys)
python train.py --train-npz data/train_data.npz --checkpoint seq_filters.pt ...

# Stage 2 — continue with seq+struct filters loaded, tuner re-initialised
python train.py --train-npz data/train_data.npz --checkpoint stage1_best.pt ...
```

The script logs every parameter as `[LOAD]`, `[INIT]` (kept at random init), or `[SKIP]` (present in the checkpoint but not in the model) at startup.

#### Notes

- BatchNorm is not working as expected on validation data, so **it is recommended to run with the `no-batchnorm`** flag.
- The default public preprocessing path assumes unflanked exon input and adds model flanks automatically.
- `load_state_dict()` in `PNASModel` resamples position-bias tensors when checkpoint and runtime input lengths differ.
- `load_weights_from_dict()` is available for loading weights converted from an external TensorFlow/Keras export format.

## Citation

Please cite: Liao, Susan E., Mukund Sudarshan, and Oded Regev. "Deciphering RNA splicing logic with interpretable machine learning." Proceedings of the National Academy of Sciences 120.41 (2023): e2221165120.
