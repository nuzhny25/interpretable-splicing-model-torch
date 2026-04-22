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


def main():
    alignments = list(
        AlignIO.parse("../data/multiz100/MALAT1_orthologues_multiz100.maf", "maf")
    )

    human_sequence = ""

    for alignment in alignments:
        for line in alignment:
            if line.id == "hg38.chr11":
                human_sequence += str(line.seq)

    human_sequence = clean_str(human_sequence)

    with open("../data/multiz100/human_malat1.txt", "w") as file:
        file.write(human_sequence)


def clean_str(string):
    return string.replace("-", "").upper().strip()


if __name__ == main():
    main()
