from Bio import AlignIO

"""maf 100 multiz allignment has weird names for each species:

human - hg38.chr11
chimp - panTro4.chr11
mouse - mm10.chr19
rat - rn6.chr1
cat - felCat8.chrD1
elephant - loxAfr3.scaffold_71

I can find more via this link: https://genome.ucsc.edu/cgi-bin/hgc?db=hg38&c=chr17&l=43106526&r=43106527&o=43106526&t=43106527&g=multiz100way&i=multiz100way

"""

SPECIES = {
    "human": "hg38.chr11",
    "chimp": "panTro4.chr11",
    "mouse": "mm10.chr19",
    "rat": "rn6.chr1",
    "cat": "felCat8.chrD1",
    "elephant": "loxAfr3.scaffold_71",
}


def main():
    alignments = list(
        AlignIO.parse("../data/multiz100/MALAT1_orthologues_multiz100.maf", "maf")
    )

    sequences = {name: "" for name in SPECIES}

    for alignment in alignments:
        for line in alignment:
            for name, maf_id in SPECIES.items():
                if line.id == maf_id:
                    sequences[name] += str(line.seq)

    for name, seq in sequences.items():
        seq = clean_str(seq)
        if seq:
            with open(f"../data/multiz100/{name}_malat1.txt", "w") as file:
                file.write(seq)


def clean_str(string):
    return string.replace("-", "").upper().strip()


if __name__ == "__main__":
    main()
