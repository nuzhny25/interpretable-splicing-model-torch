from Bio import AlignIO

"""maf 100 multiz allignment has weird names for each species:

human - hg38
chimp - panTro4
mouse - mm10
rat - rn6
cat - felCat8
elephant - loxAfr3

I can find more via this link: https://genome.ucsc.edu/cgi-bin/hgc?db=hg38&c=chr17&l=43106526&r=43106527&o=43106526&t=43106527&g=multiz100way&i=multiz100way

"""

alignments = list(AlignIO.parse("data/MALAT1_orthologues_multiz100.maf", "maf"))


human_sequence = ""

for alignment in alignments:
    for line in alignment:
        if line.id == "hg38.chr11":
            human_sequence += str(line.seq)

# print(human_sequence)

human_sequence = human_sequence.replace("-", "")

print(human_sequence)
