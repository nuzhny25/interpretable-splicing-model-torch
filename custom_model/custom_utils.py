"""Utility helpers for preparing sequence, structure, and wobble datasets."""

from __future__ import annotations

from pathlib import Path
import random
import re
import shutil
import subprocess
from typing import Any, Sequence

import numpy as np
import pandas as pd

random.seed(42)

LEFT_FLANK = "CATCCAGGTT"
RIGHT_FLANK = "CAGGTCTGAC"
DEFAULT_SEQUENCE_COLUMN = "exon"
RNA_ALPHABET = "ACGU"
DNA_ALPHABET = "ACGT"
STRUCTURE_ALPHABET = ".()"
# Purine/pyrimidine alphabet: the 4 bases collapsed to 2 classes, so a model trained
# on it can only see the class at each position, never the base identity within it.
RY_ALPHABET = "RY"
RY_TRANSLATION = str.maketrans("ACGT", "RYRY")  # A,G -> R (purine); C,T -> Y (pyrimidine)
# Base identity with the C and G channels swapped. Not a different encoding of the
# sequence, only a different channel order, and it exists solely as a control
# (pipeline/run_pipeline_agct.py): a filter-bank property that follows the letters moves
# to different channel slots under this order, one produced by the slots does not.
AGCT_ALPHABET = "AGCT"
# The sequence alphabets a model can be trained on, keyed by the name that travels in a
# checkpoint's hparams and in a run manifest.
ALPHABETS = {"acgt": DNA_ALPHABET, "ry": RY_ALPHABET, "agct": AGCT_ALPHABET}
# Reverse lookup for a checkpoint that predates the hparams key: the conv's in_channels
# is enough to name the alphabet, because the two differ in width. Built from the
# canonical alphabets only — "agct" is as wide as "acgt", so a comprehension over every
# entry would let it claim width 4 and relabel every unstamped A/C/G/T checkpoint.
# Every agct checkpoint is stamped, so it never needs this lookup.
CHANNELS_TO_ALPHABET = {len(ALPHABETS[name]): name for name in ("acgt", "ry")}
RNAFOLD_ENERGY_RE = re.compile(r"^([().]+)\s+\(\s*([^)]+?)\s*\)$")


def add_flanking(seq_nts: Sequence[str]) -> list[str]:
    """Add the fixed flanking context expected by the model.

    Args:
        seq_nts: Core exon sequences.

    Returns:
        A list where each sequence is wrapped with the canonical 10 nt left and
        right flanks.
    """
    return [LEFT_FLANK + seq + RIGHT_FLANK for seq in seq_nts]


def str_to_vector(sequence: str, alphabet: str = DNA_ALPHABET) -> np.ndarray:
    """Convert a string into a one-hot encoded matrix.

    Args:
        sequence: Input string to encode.
        alphabet: Ordered alphabet used to define the one-hot basis.

    Returns:
        A NumPy array with shape ``(len(alphabet), len(sequence))``.

    Raises:
        ValueError: If the sequence contains characters outside ``alphabet``.
    """
    sequence = sequence.upper()
    invalid = sorted(set(sequence) - set(alphabet))
    if invalid:
        raise ValueError(
            f"Sequence contains unsupported characters {invalid} for alphabet "
            f"{alphabet!r}."
        )

    index = {char: idx for idx, char in enumerate(alphabet)}
    basis = np.eye(len(alphabet), dtype=np.float32)
    return basis[[index[char] for char in sequence]].T


def folding_to_vector(structure: str) -> np.ndarray:
    """Convert a dot-bracket structure string into a one-hot matrix.

    Args:
        structure: Dot-bracket structure string containing only ``.``, ``(``,
            and ``)`` characters.

    Returns:
        A NumPy array with shape ``(3, len(structure))`` whose rows encode
        ``.``, ``(``, and ``)`` in that order.
    """
    return str_to_vector(structure, STRUCTURE_ALPHABET)


def encode_sequence(sequence: str, alphabet_name: str = "acgt") -> np.ndarray:
    """One-hot encode a nucleotide sequence in the named sequence alphabet.

    ``"acgt"`` is the 4-channel base identity encoding every model used before the
    purine/pyrimidine experiment. ``"ry"`` collapses the bases to 2 channels by
    translating A/G to R and C/T to Y first, so the encoding retains only which class
    occupies each position. ``"agct"`` is base identity with the C and G channels
    swapped; it needs no translation, since the alphabet's letter order alone sets it.

    Args:
        sequence: Nucleotide sequence, gaps already removed. Case-insensitive.
        alphabet_name: Key into :data:`ALPHABETS` (``"acgt"``, ``"ry"`` or ``"agct"``).

    Returns:
        A NumPy array with shape ``(len(ALPHABETS[alphabet_name]), len(sequence))``:
        ``(4, L)`` for ``"acgt"`` and ``"agct"``, ``(2, L)`` for ``"ry"``.

    Raises:
        KeyError: If ``alphabet_name`` is not a known alphabet.
        ValueError: If the sequence holds characters outside A/C/G/T.
    """
    alphabet = ALPHABETS[alphabet_name]
    sequence = sequence.upper()
    if alphabet_name == "ry":
        # Translate before encoding rather than after, so str_to_vector's character
        # validation still rejects anything that is not a base (an unstripped gap or
        # IUPAC code passes through translate() untouched and raises there).
        sequence = sequence.translate(RY_TRANSLATION)
    return str_to_vector(sequence, alphabet)


def resolve_alphabet(n_channels: int, alphabet_name: str | None = None) -> tuple[str, str]:
    """Name and letters of the alphabet a checkpoint was trained on.

    A plotter knows a checkpoint's channel count from ``conv_incl.weight`` but needs the
    letters to label an axis or build a logo. Checkpoints written since the R/Y work
    stamp ``hparams["alphabet"]``; the ones before it do not, and for those the channel
    count is unambiguous on its own.

    Args:
        n_channels: ``conv_incl.weight.shape[1]`` — the model's sequence in-channels.
        alphabet_name: The checkpoint's stamped alphabet name, when it has one. Takes
            precedence over the channel count; must agree with it.

    Returns:
        ``(name, letters)``, e.g. ``("acgt", "ACGT")`` or ``("ry", "RY")``.

    Raises:
        SystemExit: If the name is unknown, disagrees with the channel count, or the
            channel count matches no known alphabet.
    """
    if alphabet_name is not None:
        if alphabet_name not in ALPHABETS:
            raise SystemExit(
                f"Checkpoint names alphabet {alphabet_name!r}, which is not one of "
                f"{', '.join(sorted(ALPHABETS))}."
            )
        alphabet = ALPHABETS[alphabet_name]
        if len(alphabet) != n_channels:
            raise SystemExit(
                f"Checkpoint names alphabet {alphabet_name!r} ({len(alphabet)} letters) "
                f"but its filters have {n_channels} input channels."
            )
        return alphabet_name, alphabet
    if n_channels not in CHANNELS_TO_ALPHABET:
        raise SystemExit(
            f"Checkpoint has {n_channels} sequence input channels, which matches no "
            f"known alphabet ("
            + ", ".join(f"{name}={len(a)}" for name, a in sorted(ALPHABETS.items()))
            + "), and records no 'alphabet' in its metadata hparams."
        )
    name = CHANNELS_TO_ALPHABET[n_channels]
    return name, ALPHABETS[name]


def base_code_map(alphabet_name: str = "acgt") -> dict[str, int]:
    """Map each nucleotide to its channel index under the named alphabet.

    The one-hot encoding of :func:`encode_sequence` as a lookup table, for consumers
    that index the channel directly instead of building the matrix — the interactive
    heatmap's JavaScript convolution does exactly that.

    Args:
        alphabet_name: Key into :data:`ALPHABETS` (``"acgt"``, ``"ry"`` or ``"agct"``).

    Returns:
        ``{"A": 0, "C": 1, "G": 2, "T": 3}`` for ``"acgt"``;
        ``{"A": 0, "G": 0, "C": 1, "T": 1}`` for ``"ry"``;
        ``{"A": 0, "G": 1, "C": 2, "T": 3}`` for ``"agct"``.
    """
    alphabet = ALPHABETS[alphabet_name]
    if alphabet_name == "ry":
        return {base: alphabet.index(base.translate(RY_TRANSLATION)) for base in DNA_ALPHABET}
    return {base: idx for idx, base in enumerate(alphabet)}


def one_hot_batch(seqs: Sequence[str], alphabet: str = DNA_ALPHABET) -> np.ndarray:
    """One-hot encode a batch of equal-length sequences.

    For the default DNA alphabet both DNA and RNA inputs are accepted: any ``U``
    characters are normalized to ``T`` first, because the model sequence channels are
    DNA-based. That fold applies only to the DNA alphabet — a batch already translated
    into another alphabet (R/Y, say) holds no ``U`` to begin with.

    Args:
        seqs: Batch of sequences, all over ``alphabet``.
        alphabet: Ordered alphabet used to define the one-hot basis.

    Returns:
        A NumPy array with shape ``(batch_size, len(alphabet), sequence_length)``.

    Raises:
        ValueError: If the sequences do not all share the same length.
    """
    if alphabet == DNA_ALPHABET:
        seqs = [seq.upper().replace("U", "T") for seq in seqs]
    else:
        seqs = [seq.upper() for seq in seqs]
    lengths = {len(seq) for seq in seqs}
    if len(lengths) > 1:
        raise ValueError("All sequences must have the same length for batching.")

    return np.asarray(
        [str_to_vector(seq, alphabet) for seq in seqs], dtype=np.float32
    )


def generate_random_exon(length: int) -> str:
    """Generate a random exon sequence.

    Args:
        length: Number of nucleotides to generate.

    Returns:
        A random DNA sequence of length ``length``.

    Raises:
        ValueError: If ``length`` is negative.
    """
    if length < 0:
        raise ValueError("length must be non-negative.")
    return "".join(random.choice(DNA_ALPHABET) for _ in range(length))


def rnafold_available(rnafold_bin: str = "RNAfold") -> bool:
    """Return whether the RNAfold executable is available.

    Args:
        rnafold_bin: Executable name or path to check.

    Returns:
        ``True`` if the executable can be resolved, otherwise ``False``.
    """
    if Path(rnafold_bin).expanduser().exists():
        return True
    return shutil.which(rnafold_bin) is not None


def RNAfold(
    seqs: Sequence[str],
    RNAfold_bin: str = "RNAfold",
    temperature: float = 37.0,
    maxBPspan: int = 0,
    commands_file: str = "",
    num_threads: int = 8,
) -> list[tuple[str, float]]:
    """Fold sequences with ViennaRNA ``RNAfold`` and return structures and MFEs.

    DNA sequences are converted to RNA before folding by replacing ``T`` with
    ``U``. The returned structures are dot-bracket strings aligned to the
    original input length.

    Args:
        seqs: Sequences to fold in a single batched RNAfold call.
        RNAfold_bin: Executable name or path for ViennaRNA ``RNAfold``.
        temperature: Folding temperature in Celsius.
        maxBPspan: Optional maximum base-pair span. ``0`` disables the flag.
        commands_file: Optional ViennaRNA commands file.

    Returns:
        A list of ``(structure, mfe)`` tuples in the same order as ``seqs``.

    Raises:
        FileNotFoundError: If ``RNAfold`` cannot be located.
        RuntimeError: If RNAfold execution fails or the output cannot be parsed.
        ValueError: If the input sequences contain unsupported characters.
    """
    if not seqs:
        return []

    if not rnafold_available(RNAfold_bin):
        raise FileNotFoundError(
            f"Could not find RNAfold executable {RNAfold_bin!r}. Install ViennaRNA "
            "and ensure `RNAfold` is on PATH, or pass `RNAfold_bin` explicitly."
        )

    normalized_seqs = [seq.upper().replace("T", "U") for seq in seqs]
    invalid_chars = sorted(set("".join(normalized_seqs)) - set(RNA_ALPHABET))
    if invalid_chars:
        raise ValueError(
            f"RNAfold only supports A/C/G/U input; found unsupported characters "
            f"{invalid_chars}."
        )

    command = [RNAfold_bin, "--noPS", "-T", str(temperature)]
    if commands_file:
        command.append(f"--commands={commands_file}")
    if maxBPspan:
        command.append(f"--maxBPspan={maxBPspan}")
    if num_threads > 1:
        command.append(f"--jobs={num_threads}")

    try:
        result = subprocess.run(
            command,
            input="\n".join(normalized_seqs),
            text=True,
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else ""
        raise RuntimeError(
            f"RNAfold failed with exit code {exc.returncode}. {stderr}".strip()
        ) from exc

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    expected_lines = len(seqs) * 2
    if len(lines) < expected_lines:
        raise RuntimeError(
            f"RNAfold returned {len(lines)} non-empty lines for {len(seqs)} "
            "sequences; expected at least two lines per sequence."
        )

    folded: list[tuple[str, float]] = []
    for seq_index in range(len(seqs)):
        structure_line = lines[(2 * seq_index) + 1]
        match = RNAFOLD_ENERGY_RE.match(structure_line)
        if match is None:
            raise RuntimeError(
                f"Could not parse RNAfold output line: {structure_line!r}"
            )

        structure, mfe_text = match.groups()
        folded.append((structure, float(mfe_text)))

    return folded


def find_parentheses(structure: str) -> dict[int, int]:
    """Return the index mapping of matched parentheses in a dot-bracket string.

    Args:
        structure: Dot-bracket structure string.

    Returns:
        A dictionary mapping each opening parenthesis index to the matching
        closing parenthesis index.

    Raises:
        ValueError: If the structure contains unbalanced parentheses.
    """
    stack: list[int] = []
    parentheses_locs: dict[int, int] = {}
    for idx, char in enumerate(structure):
        if char == "(":
            stack.append(idx)
        elif char == ")":
            if not stack:
                raise ValueError(f"Too many closing parentheses at index {idx}.")
            parentheses_locs[stack.pop()] = idx

    if stack:
        raise ValueError(
            "No matching closing parenthesis for opening parenthesis at index "
            f"{stack.pop()}."
        )

    return parentheses_locs


def rna_fold_structs(
    seq_nts: Sequence[str],
    maxBPspan: int = 0,
    rnafold_bin: str = "RNAfold",
    temperature: float = 37.0,
    commands_file: str = "",
    num_threads: int = 8,
) -> tuple[list[str], np.ndarray]:
    """Fold a batch of sequences and return structures plus MFEs.

    Args:
        seq_nts: Sequences to fold.
        maxBPspan: Optional maximum base-pair span for RNAfold.
        rnafold_bin: Executable name or path for ``RNAfold``.
        temperature: Folding temperature in Celsius.
        commands_file: Optional ViennaRNA commands file.

    Returns:
        A tuple ``(structures, mfes)`` where ``structures`` is a list of
        dot-bracket strings and ``mfes`` is a NumPy array of floats.
    """
    struct_mfes = RNAfold(
        seq_nts,
        RNAfold_bin=rnafold_bin,
        temperature=temperature,
        maxBPspan=maxBPspan,
        commands_file=commands_file,
        num_threads=num_threads,
    )
    structures = [structure for structure, _ in struct_mfes]
    mfes = np.asarray([mfe for _, mfe in struct_mfes], dtype=np.float32)
    return structures, mfes


def compute_structure(
    seq_nts: Sequence[str],
    maxBPspan: int = 0,
    rnafold_bin: str = "RNAfold",
    temperature: float = 37.0,
    commands_file: str = "",
    num_threads: int = 8,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Compute one-hot structure features and MFEs for input sequences.

    Args:
        seq_nts: Sequences to fold.
        maxBPspan: Optional maximum base-pair span for RNAfold.
        rnafold_bin: Executable name or path for ``RNAfold``.
        temperature: Folding temperature in Celsius.
        commands_file: Optional ViennaRNA commands file.

    Returns:
        A tuple ``(struct_oh, structures, mfes)`` where ``struct_oh`` has shape
        ``(batch_size, 3, sequence_length)``.
    """
    structures, mfes = rna_fold_structs(
        seq_nts,
        maxBPspan=maxBPspan,
        rnafold_bin=rnafold_bin,
        temperature=temperature,
        commands_file=commands_file,
        num_threads=num_threads,
    )
    struct_oh = np.asarray(
        [folding_to_vector(structure) for structure in structures],
        dtype=np.float32,
    )
    return struct_oh, structures, mfes


def compute_bijection(structure: str) -> np.ndarray:
    """Convert a dot-bracket structure into a paired-index lookup array.

    Unpaired positions map to themselves. Paired positions map to their partner
    index.

    Args:
        structure: Dot-bracket structure string.

    Returns:
        A NumPy integer array of length ``len(structure)``.
    """
    parentheses = find_parentheses(structure)
    bijection = np.arange(len(structure), dtype=np.int64)
    for left_idx, right_idx in parentheses.items():
        bijection[left_idx] = right_idx
        bijection[right_idx] = left_idx
    return bijection


def compute_wobble_indicator(sequence: str, structure: str) -> np.ndarray:
    """Mark wobble base-pair positions for a sequence/structure pair.

    A wobble is defined as a paired ``G-U`` or ``U-G`` base pair. DNA ``T`` is
    treated equivalently to RNA ``U``.

    Args:
        sequence: Nucleotide sequence.
        structure: Dot-bracket structure string for the same sequence.

    Returns:
        A NumPy array of shape ``(sequence_length,)`` containing ``0`` or ``1``.

    Raises:
        ValueError: If the sequence length, structure length, or alphabet is
            invalid.
    """
    sequence = sequence.upper().replace("U", "T")
    if len(sequence) != len(structure):
        raise ValueError("sequence and structure must have the same length.")

    invalid_chars = sorted(set(sequence) - set(DNA_ALPHABET))
    if invalid_chars:
        raise ValueError(
            f"Sequence contains unsupported characters {invalid_chars}; only "
            "A/C/G/T/U are supported."
        )

    bijection = compute_bijection(structure)
    wobble = [
        1 if {sequence[idx], sequence[bijection[idx]]} == {"G", "T"} else 0
        for idx in range(len(sequence))
    ]
    return np.asarray(wobble, dtype=np.float32)


def compute_wobbles(seq_nts: Sequence[str], structs: Sequence[str]) -> np.ndarray:
    """Compute wobble indicator channels for a batch of sequences.

    Args:
        seq_nts: Batch of nucleotide sequences.
        structs: Batch of dot-bracket structures aligned to ``seq_nts``.

    Returns:
        A NumPy array with shape ``(batch_size, 1, sequence_length)``.

    Raises:
        ValueError: If the sequence and structure batches differ in length.
    """
    if len(seq_nts) != len(structs):
        raise ValueError("seq_nts and structs must contain the same number of items.")

    wobble_batch = [
        np.expand_dims(compute_wobble_indicator(seq, structure), axis=0)
        for seq, structure in zip(seq_nts, structs)
    ]
    return np.asarray(wobble_batch, dtype=np.float32)


def _get_sequence_values(df: pd.DataFrame, sequence_column: str) -> list[str]:
    """Extract and normalize the selected sequence column from a dataframe.

    Args:
        df: Input dataframe.
        sequence_column: Name of the unflanked sequence column.

    Returns:
        A list of uppercased sequence strings.

    Raises:
        ValueError: If ``sequence_column`` is missing.
    """
    if sequence_column not in df.columns:
        raise ValueError(
            f"Column {sequence_column!r} was not found. Available columns: "
            f"{list(df.columns)!r}."
        )
    return df[sequence_column].astype(str).str.upper().tolist()


def _normalize_metadata_value(series: pd.Series) -> np.ndarray:
    """Convert a dataframe column into an NPZ-safe NumPy array.

    Args:
        series: Metadata column to convert.

    Returns:
        A NumPy array suitable for inclusion in an ``.npz`` archive.
    """
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return series.to_numpy()
    return series.astype(str).to_numpy(dtype=str)


def make_dataset_dict(
    seq_nts: Sequence[str],
    *,
    add_flanks: bool = True,
    rnafold_bin: str = "RNAfold",
    temperature: float = 37.0,
    maxBPspan: int = 0,
    commands_file: str = "",
    num_threads: int = 8,
) -> dict[str, Any]:
    """Create a model-ready dataset dictionary from unflanked exon sequences.

    The returned arrays use channel-first ordering ``(N, C, L)`` throughout.
    They are NumPy arrays; callers should convert them to PyTorch tensors before
    model inference.

    Args:
        seq_nts: Unflanked exon sequences.
        add_flanks: Whether to add the fixed model flanks before feature
            computation.
        rnafold_bin: Executable name or path for ``RNAfold``.
        temperature: Folding temperature in Celsius.
        maxBPspan: Optional maximum base-pair span for RNAfold.
        commands_file: Optional ViennaRNA commands file.

    Returns:
        A dictionary containing the canonical preprocessing outputs:
        ``exon``, ``model_sequence``, ``seq_oh``, ``struct_oh``, ``wobbles``,
        ``structure``, and ``mfe``.
    """
    exons = [seq.upper() for seq in seq_nts]
    model_sequences = add_flanking(exons) if add_flanks else exons
    seq_oh = one_hot_batch(model_sequences)
    struct_oh, structures, mfes = compute_structure(
        model_sequences,
        maxBPspan=maxBPspan,
        rnafold_bin=rnafold_bin,
        temperature=temperature,
        commands_file=commands_file,
        num_threads=num_threads,
    )
    wobbles = compute_wobbles(model_sequences, structures)

    return {
        "exon": np.asarray(exons, dtype=str),
        "model_sequence": np.asarray(model_sequences, dtype=str),
        "seq_oh": seq_oh,
        "struct_oh": struct_oh,
        "wobbles": wobbles,
        "structure": np.asarray(structures, dtype=str),
        "mfe": mfes,
        "added_flanks": np.asarray(add_flanks),
    }


def dataframe_to_dataset(
    df: pd.DataFrame,
    *,
    sequence_column: str = DEFAULT_SEQUENCE_COLUMN,
    add_flanks: bool = True,
    rnafold_bin: str = "RNAfold",
    temperature: float = 37.0,
    maxBPspan: int = 0,
    commands_file: str = "",
    num_threads: int = 8,
) -> dict[str, Any]:
    """Convert a sequence dataframe into a model-ready dataset dictionary.

    The selected sequence column is interpreted as the unflanked exon input.
    All arrays in the returned dataset are NumPy arrays. To run the model, the
    user still needs to convert feature arrays such as ``seq_oh``,
    ``struct_oh``, and ``wobbles`` to PyTorch tensors.

    Args:
        df: Input dataframe containing at least one sequence column.
        sequence_column: Name of the unflanked input sequence column. Defaults
            to ``"exon"``.
        add_flanks: Whether to add the fixed model flanks.
        rnafold_bin: Executable name or path for ``RNAfold``.
        temperature: Folding temperature in Celsius.
        maxBPspan: Optional maximum base-pair span for RNAfold.
        commands_file: Optional ViennaRNA commands file.

    Returns:
        A simple dictionary of NumPy arrays containing canonical preprocessing
        outputs plus pass-through metadata columns from ``df``.
    """
    exons = _get_sequence_values(df, sequence_column)
    dataset = make_dataset_dict(
        exons,
        add_flanks=add_flanks,
        rnafold_bin=rnafold_bin,
        temperature=temperature,
        maxBPspan=maxBPspan,
        commands_file=commands_file,
        num_threads=num_threads,
    )
    dataset["sequence_column"] = np.asarray(sequence_column, dtype=str)

    for column in df.columns:
        if column == sequence_column:
            continue
        dataset[f"metadata_{column}"] = _normalize_metadata_value(df[column])

    return dataset


def save_dataset_npz(dataset: dict[str, Any], output_path: str | Path) -> Path:
    """Write a dataset dictionary to a compressed ``.npz`` archive.

    Args:
        dataset: Dataset dictionary produced by :func:`make_dataset_dict` or
            :func:`dataframe_to_dataset`.
        output_path: Output archive path.

    Returns:
        The resolved output path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **dataset)
    return output_path


def create_input_data(
    seq_nts: Sequence[str],
    return_mfe: bool = False,
    *,
    add_flanks: bool = True,
    rnafold_bin: str = "RNAfold",
    temperature: float = 37.0,
    maxBPspan: int = 0,
    commands_file: str = "",
    num_threads: int = 8,
) -> (
    tuple[np.ndarray, np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
):
    """Create model-ready sequence, structure, and wobble inputs.

    This is a compatibility wrapper around :func:`make_dataset_dict`. New code
    should prefer :func:`make_dataset_dict` or :func:`dataframe_to_dataset`.

    Args:
        seq_nts: Unflanked exon sequences.
        return_mfe: Whether to also return structures and MFEs.
        add_flanks: Whether to add the fixed model flanks before feature
            computation.
        rnafold_bin: Executable name or path for ``RNAfold``.
        temperature: Folding temperature in Celsius.
        maxBPspan: Optional maximum base-pair span for RNAfold.
        commands_file: Optional ViennaRNA commands file.

    Returns:
        If ``return_mfe`` is ``False``, returns ``(seq_oh, struct_oh, wobbles)``.
        If ``return_mfe`` is ``True``, returns
        ``(seq_oh, struct_oh, wobbles, structures, mfes)``.
    """
    dataset = make_dataset_dict(
        seq_nts,
        add_flanks=add_flanks,
        rnafold_bin=rnafold_bin,
        temperature=temperature,
        maxBPspan=maxBPspan,
        commands_file=commands_file,
        num_threads=num_threads,
    )
    if return_mfe:
        return (
            dataset["seq_oh"],
            dataset["struct_oh"],
            dataset["wobbles"],
            dataset["structure"],
            dataset["mfe"],
        )
    return dataset["seq_oh"], dataset["struct_oh"], dataset["wobbles"]


__all__ = [
    "ALPHABETS",
    "CHANNELS_TO_ALPHABET",
    "DEFAULT_SEQUENCE_COLUMN",
    "LEFT_FLANK",
    "RIGHT_FLANK",
    "RNAfold",
    "RY_ALPHABET",
    "RY_TRANSLATION",
    "add_flanking",
    "base_code_map",
    "compute_bijection",
    "compute_structure",
    "compute_wobble_indicator",
    "compute_wobbles",
    "create_input_data",
    "dataframe_to_dataset",
    "encode_sequence",
    "find_parentheses",
    "folding_to_vector",
    "generate_random_exon",
    "make_dataset_dict",
    "one_hot_batch",
    "resolve_alphabet",
    "rna_fold_structs",
    "rnafold_available",
    "save_dataset_npz",
    "str_to_vector",
]
