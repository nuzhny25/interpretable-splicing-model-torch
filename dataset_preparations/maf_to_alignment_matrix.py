"""Turn a multiz MAF into the ``alignment_matrix.npy`` the pipeline trains on.

    python dataset_preparations/maf_to_alignment_matrix.py --gene hotair il6 vegfa xist

This is the gene-agnostic replacement for the hand-maintained
``maf_processing.py`` + ``plots/creating_alignment/alignment_mapping.py`` pair, which
between them carried a hardcoded 62-entry ``SPECIES`` table of full contig ids
(``hg38.chr11``, ``mm10.chr19``, ...). That table is per-locus by construction: every
gene lives on a different chromosome, so none of those ids survive a move from MALAT1 to
HOTAIR, IL6, VEGFA or XIST. Here the species set is discovered from the MAF itself,
keyed by *assembly* (the part of an ``s`` line's id before the first ``.``) rather than
by contig, which also keeps a species whole when its alignment is spread over several
scaffolds — 26 of HOTAIR's 99 assemblies and 30 of VEGFA's 100 are in that situation,
and contig-keying would have split each of them into unrelated fragments.

── What the pipeline requires of the output ──────────────────────────────────

``custom_model/pipeline/run_pipeline.py`` resolves ``--gene <name>`` to
``data/<name>/alignment_matrix.npy`` (unless the name is one of its aliases) and hands
that one file to training, ``plot_filters.py`` and ``filter_coactivation_covariance.py``.
Everything those three need is the array itself, so the contract is narrow but strict:

  1. A plain (non-pickled) 2-D ``.npy`` of single characters, dtype ``<U1``, shaped
     ``(n_aligned, n_species)`` — one row per alignment column, one column per species.
     ``run_pipeline.main`` reads ``shape[1]`` through ``mmap_mode="r"``, so an object or
     pickled array cannot be loaded there at all.
  2. An alphabet of exactly ``-``, ``N`` and ``A/C/G/T``, upper or lower case. Every
     consumer splits nucleotide from gap with ``(row != "-") & (row != "N")`` on the raw
     characters and only *then* calls ``.upper()``, so lowercase ``n`` passes the gap test
     and reaches ``custom_utils.str_to_vector``, which rejects anything outside ACGT.
     Soft-masked ``a/c/g/t`` are fine and are preserved. Any IUPAC ambiguity code
     (R, Y, S, W, K, M, ...) would crash the same way ``n`` does. This script folds
     ``n`` into ``N`` and refuses to write a matrix containing anything else.
  3. Column semantics: a species' sequence is its row read with the gaps taken out, and
     the position of its k-th nucleotide in the array is the aligned column of the k-th
     sliding window. That is only meaningful if MAF blocks concatenate in reference
     order, so the reference must be present, on ``+``, and the blocks must not overlap
     or run backwards along it — all checked below before anything is written.
  4. Length: the shortest gap-removed species must hold at least ``--kernel-size``
     nucleotides, and at least one species must hold ``--pool-length`` (default 3 x
     kernel), or training exits. The coverage filter below keeps this far from binding.
  5. Species count: ``n_species = matrix.shape[1]`` sets the training defaults
     ``--n-sample`` (all of them) and ``--min-species`` (``ceil(n_species / 2)``, the
     number of species that must be present for a column to count toward the loss). So
     the species set is not a neutral choice — it moves the column-validity threshold.

All-gap rows are dropped, as in both predecessor scripts.

── Which species are kept ────────────────────────────────────────────────────

``--min-coverage`` (default 0.10) applies the rule ``neat1_maf_processing.py`` documented:
a species whose aligned nucleotides come to less than 10% of the reference's is dropped
rather than carried as a mostly-gap row. Such a row barely moves the per-column
cross-species deviation, but it does enter the within-species normalizer in the loss,
where a mean absolute deviation taken over a few hundred nucleotides is noise being
averaged in on equal footing with a full-length species.

Coverage is measured against the reference's own nucleotide count and can exceed 100%
for a species carrying insertions relative to it. Every assembly's coverage is written
to ``species.json`` whether it was kept or not, so raising the bar is a re-run with a
different ``--min-coverage`` and no re-derivation of anything.

Species keep their order of first appearance in the MAF, which is multiz's phylogenetic
ordering, so column order is stable and reproducible. The ``.npy`` itself has no names
in it; ``species.json`` is what records which column is which. Where an assembly is one
of the 62 in ``maf_processing.SPECIES`` it also gets that file's friendly name, so
``hg38`` reads as ``human`` here exactly as it does there.

── Outputs, per gene, into ``data/<gene>/`` ──────────────────────────────────

    alignment_matrix.npy    what the pipeline consumes (see the contract above)
    alignment_mapping.json  per species, the aligned column of each of its nucleotides
                            (the sidecar neat1_alignment_mapping.py wrote; nothing in
                            the pipeline reads it, several plots/ scripts do)
    species.json            column order, assembly ids, friendly names, coverage, the
                            reference span, and the settings this run used

``--write-sequences`` additionally writes the ungapped per-species ``<name>_<gene>.txt``
files that ``maf_processing.py`` produced, for the MALAT1-era chunking/reverse/complement
scripts. Off by default: nothing in the pipeline reads them.
"""

import argparse
import collections
import json
import os
import sys

import numpy as np
from Bio import AlignIO

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, ROOT)

from dataset_preparations.maf_processing import SPECIES as MALAT1_SPECIES  # noqa: E402

# Assembly -> friendly name, recovered from the MALAT1 table by dropping each entry's
# contig. Only ever used for labelling in species.json; an assembly the table does not
# know (the fish, birds and reptiles that reach HOTAIR and VEGFA but were never in the
# MALAT1 set) keeps its UCSC assembly id as its name.
ASSEMBLY_NAMES = {
    maf_id.split(".", 1)[0]: name for name, maf_id in MALAT1_SPECIES.items()
}

# The characters the pipeline can read, after lowercase "n" has been folded to "N".
# Anything else -- an IUPAC ambiguity code, most likely -- reaches str_to_vector and
# raises there, so it is caught here instead, where the message can name the gene.
ALLOWED = set("-NACGTacgt")


def build_gene(gene, maf_path, out_dir, reference, min_coverage, write_sequences):
    """Read one MAF, validate it, and write the matrix + sidecars for one gene."""
    print(f"\n=== {gene} ===\n  reading {maf_path}")
    blocks = list(AlignIO.parse(maf_path, "maf"))
    if not blocks:
        raise SystemExit(f"{maf_path} contains no alignment blocks.")

    # ── Pass 1: species inventory, coverage, and the reference-order checks ──
    # Coverage and ordering are both needed before a single column is built: the first
    # decides which rows exist, the second decides whether concatenating blocks means
    # anything at all.
    order = []  # assemblies in order of first appearance = multiz tree order
    nuc_count = collections.Counter()  # assembly -> aligned nucleotides
    contigs = collections.defaultdict(set)  # assembly -> the contigs it aligned through
    duplicate_rows = collections.Counter()  # assembly -> blocks holding it twice
    ref_intervals = []
    ref_contigs = set()
    n_cols = 0

    for block_idx, block in enumerate(blocks):
        block_len = block.get_alignment_length()
        n_cols += block_len
        seen = set()
        ref_record = None
        for record in block:
            assembly = record.id.split(".", 1)[0]
            if assembly in seen:
                # multiz can emit a second row for one assembly (a paralogous or
                # duplicated segment). Concatenation has one row per species, so the
                # first is kept and the rest reported rather than silently merged.
                duplicate_rows[assembly] += 1
                continue
            seen.add(assembly)
            if assembly not in nuc_count:
                order.append(assembly)
            contigs[assembly].add(record.id)
            nuc_count[assembly] += block_len - str(record.seq).count("-")
            if assembly == reference:
                ref_record = record

        if ref_record is None:
            raise SystemExit(
                f"{maf_path}: block {block_idx} has no {reference} row, so its columns "
                f"cannot be placed along the reference. Re-extract the MAF against "
                f"{reference}, or pass --reference for the assembly this MAF is "
                f"anchored on."
            )
        ref_contigs.add(ref_record.id)
        start = ref_record.annotations["start"]
        size = ref_record.annotations["size"]
        if ref_record.annotations["strand"] != 1:
            raise SystemExit(
                f"{maf_path}: block {block_idx} carries {reference} on the minus "
                f"strand. Reference-ordered concatenation assumes a plus-strand "
                f"reference in every block."
            )
        ref_intervals.append((start, start + size))

    if len(ref_contigs) > 1:
        raise SystemExit(
            f"{maf_path}: the {reference} reference spans more than one contig "
            f"({', '.join(sorted(ref_contigs))}). Concatenation would splice unrelated "
            f"coordinates together; extract one locus per MAF."
        )
    for i in range(len(ref_intervals) - 1):
        if ref_intervals[i][1] > ref_intervals[i + 1][0]:
            raise SystemExit(
                f"{maf_path}: blocks {i} and {i + 1} overlap or run backwards on "
                f"{reference} ({ref_intervals[i]} then {ref_intervals[i + 1]}). Blocks "
                f"must be sorted and disjoint along the reference for the concatenated "
                f"column index to mean a position in the transcript."
            )

    ref_nt = sum(end - start for start, end in ref_intervals)
    ref_span = (ref_intervals[0][0], ref_intervals[-1][1])
    gaps_between_blocks = sum(
        ref_intervals[i + 1][0] - ref_intervals[i][1]
        for i in range(len(ref_intervals) - 1)
    )
    print(
        f"  {len(blocks)} blocks, {n_cols:,} alignment columns, "
        f"{len(order)} assemblies\n"
        f"  reference {next(iter(ref_contigs))}:{ref_span[0]}-{ref_span[1]} "
        f"(+ strand in every block), {ref_nt:,} nt"
    )
    if gaps_between_blocks:
        # Not fatal: it only means those reference bases aligned to nothing and are
        # absent from the matrix, so adjacent columns can straddle a jump in coordinate.
        print(
            f"  note: {gaps_between_blocks:,} reference nt fall between blocks and are "
            f"not represented in the matrix"
        )
    if duplicate_rows:
        print(
            f"  note: kept the first row only for {len(duplicate_rows)} assemblies "
            f"appearing twice in a block: {dict(duplicate_rows)}"
        )

    # ── Species selection ────────────────────────────────────────────────────
    coverage = {a: nuc_count[a] / ref_nt for a in order}
    kept = [a for a in order if a == reference or coverage[a] >= min_coverage]
    dropped = [a for a in order if a not in set(kept)]
    print(
        f"  keeping {len(kept)}/{len(order)} species at >= {min_coverage:.0%} coverage "
        f"of the reference ({len(dropped)} dropped)"
    )
    if dropped:
        worst = sorted(dropped, key=lambda a: -coverage[a])
        shown = ", ".join(f"{a} {coverage[a]:.1%}" for a in worst[:6])
        print(f"    dropped, highest coverage first: {shown}" + (" ..." if len(worst) > 6 else ""))
    if len(kept) < 2:
        raise SystemExit(
            f"{gene}: only {len(kept)} species clear --min-coverage {min_coverage}; "
            f"the loss is a cross-species deviation and needs several."
        )

    # ── Pass 2: concatenate the blocks into one gapped string per kept species ──
    # A species missing from a block contributes that block's width in gaps, which is
    # what keeps every string the same length and every column aligned.
    kept_set = set(kept)
    pieces = {a: [] for a in kept}
    for block in blocks:
        block_len = block.get_alignment_length()
        present = {}
        for record in block:
            assembly = record.id.split(".", 1)[0]
            if assembly in kept_set and assembly not in present:
                present[assembly] = str(record.seq)
        for assembly in kept:
            pieces[assembly].append(present.get(assembly, "-" * block_len))

    sequences = {a: "".join(p) for a, p in pieces.items()}
    lengths = {len(s) for s in sequences.values()}
    if len(lengths) != 1:
        raise SystemExit(f"{gene}: concatenated rows differ in length: {sorted(lengths)}")

    matrix = np.array([list(sequences[a]) for a in kept])  # (n_species, n_cols)

    # Fold lowercase soft-masked "n" into the "N" every consumer treats as a gap; see
    # requirement 2 in the module docstring. Other lowercase letters stay as they are.
    n_soft = int((matrix == "n").sum())
    matrix[matrix == "n"] = "N"

    matrix = matrix.T  # (n_cols, n_species), the orientation the pipeline expects

    # Drop columns no kept species occupies. These arise once low-coverage species are
    # removed: a column that existed only to hold a dropped species' insertion.
    keep_rows = ~np.all(matrix == "-", axis=1)
    n_empty = int((~keep_rows).sum())
    matrix = matrix[keep_rows]

    stray = sorted(set(np.unique(matrix)) - ALLOWED)
    if stray:
        raise SystemExit(
            f"{gene}: alignment contains characters the model cannot encode: {stray}. "
            f"custom_utils.str_to_vector accepts only ACGT after the -/N gap test, so "
            f"these would raise during training."
        )

    # Per-species ungapped lengths: requirement 4, checked here so a too-short species
    # is visible now rather than as a conv1d shape error mid-training.
    ungapped = {
        a: int(((matrix[:, i] != "-") & (matrix[:, i] != "N")).sum())
        for i, a in enumerate(kept)
    }
    shortest = min(ungapped.values())
    print(
        f"  matrix {matrix.shape} (aligned positions x species), dropped {n_empty:,} "
        f"all-gap columns"
        + (f", folded {n_soft} soft-masked 'n' to 'N'" if n_soft else "")
    )
    print(
        f"  ungapped lengths: shortest {shortest:,} nt ({min(ungapped, key=ungapped.get)}), "
        f"longest {max(ungapped.values()):,} nt; training default --min-species will be "
        f"{-(-len(kept) // 2)} of {len(kept)}"
    )

    os.makedirs(out_dir, exist_ok=True)
    matrix_path = os.path.join(out_dir, "alignment_matrix.npy")
    np.save(matrix_path, matrix)

    # Sidecar 1: for each species, the aligned column of each of its nucleotides. Same
    # content and gap test as neat1_alignment_mapping.py wrote.
    mapping = [
        np.nonzero((matrix[:, i] != "-") & (matrix[:, i] != "N"))[0].tolist()
        for i in range(matrix.shape[1])
    ]
    mapping_path = os.path.join(out_dir, "alignment_mapping.json")
    with open(mapping_path, "w") as fh:
        json.dump(mapping, fh)

    # Sidecar 2: what each column is, and what this run decided. The .npy carries no
    # names, so without this the column order is only recoverable by re-running.
    species_path = os.path.join(out_dir, "species.json")
    with open(species_path, "w") as fh:
        json.dump(
            {
                "gene": gene,
                "maf": os.path.abspath(maf_path),
                "reference": {
                    "assembly": reference,
                    "contig": next(iter(ref_contigs)),
                    "start": ref_span[0],
                    "end": ref_span[1],
                    "aligned_nt": ref_nt,
                    "nt_between_blocks": gaps_between_blocks,
                },
                "settings": {"min_coverage": min_coverage},
                "n_aligned": int(matrix.shape[0]),
                "n_species": int(matrix.shape[1]),
                "columns": [
                    {
                        "index": i,
                        "assembly": a,
                        "name": ASSEMBLY_NAMES.get(a, a),
                        "contigs": sorted(contigs[a]),
                        "coverage": round(coverage[a], 4),
                        "ungapped_nt": ungapped[a],
                    }
                    for i, a in enumerate(kept)
                ],
                "dropped": [
                    {
                        "assembly": a,
                        "name": ASSEMBLY_NAMES.get(a, a),
                        "coverage": round(coverage[a], 4),
                    }
                    for a in sorted(dropped, key=lambda a: -coverage[a])
                ],
            },
            fh,
            indent=2,
        )

    written = [matrix_path, mapping_path, species_path]

    if write_sequences:
        for i, a in enumerate(kept):
            col = matrix[:, i]
            seq = "".join(col[(col != "-") & (col != "N")]).upper()
            name = ASSEMBLY_NAMES.get(a, a)
            path = os.path.join(out_dir, f"{name}_{gene}.txt")
            with open(path, "w") as fh:
                fh.write(seq)
        print(f"  wrote {len(kept)} ungapped per-species .txt files")

    for path in written:
        print(f"  -> {path}")
    return matrix.shape, len(kept)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--gene",
        nargs="+",
        required=True,
        help="One or more gene directories under data/. Each is expected to hold "
        "<gene>.maf unless --maf is given (which then allows only one gene).",
    )
    parser.add_argument(
        "--maf",
        default=None,
        help="Explicit path to the MAF, for a single --gene whose file is not named "
        "data/<gene>/<gene>.maf.",
    )
    parser.add_argument(
        "--data-dir",
        default=os.path.join(ROOT, "data"),
        help="Root holding the per-gene directories. Default: <repo>/data.",
    )
    parser.add_argument(
        "--reference",
        default="hg38",
        help="Assembly the MAF is anchored on; must be present, on the plus strand, in "
        "every block, and is always kept regardless of --min-coverage. Default: hg38.",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.10,
        help="Drop a species whose aligned nucleotides are less than this fraction of "
        "the reference's. Default: 0.10, the rule neat1_maf_processing.py documents — "
        "a mostly-gap row contributes almost nothing to the cross-species deviation but "
        "still enters the within-species normalizer, where it is noise.",
    )
    parser.add_argument(
        "--write-sequences",
        action="store_true",
        help="Also write the ungapped per-species <name>_<gene>.txt files that "
        "maf_processing.py produced. Nothing in the pipeline reads them.",
    )
    args = parser.parse_args()

    if args.maf and len(args.gene) > 1:
        parser.error("--maf names one file, so it applies to a single --gene.")
    if not 0 <= args.min_coverage <= 1:
        parser.error("--min-coverage is a fraction between 0 and 1.")

    results = []
    for gene in args.gene:
        out_dir = os.path.join(args.data_dir, gene)
        maf_path = args.maf or os.path.join(out_dir, f"{gene}.maf")
        if not os.path.exists(maf_path):
            raise SystemExit(
                f"No MAF for --gene {gene} (looked for {maf_path}). Pass --maf to point "
                f"at it directly."
            )
        shape, n_kept = build_gene(
            gene,
            maf_path,
            out_dir,
            args.reference,
            args.min_coverage,
            args.write_sequences,
        )
        results.append((gene, shape, n_kept))

    print(f"\n=== Done: {len(results)} gene(s) ===")
    for gene, shape, n_kept in results:
        print(
            f"  {gene:8s} {shape[0]:>7,} aligned positions x {shape[1]:>3} species  "
            f"->  python custom_model/pipeline/run_pipeline.py --gene {gene}"
        )


if __name__ == "__main__":
    main()
