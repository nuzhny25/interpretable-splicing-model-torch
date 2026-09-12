from Bio import AlignIO

"""NEAT1 counterpart of maf_processing.py (control gene for the cross-species run).

data/neat1/NEAT1.maf is a multiz100 alignment over hg38.chr11:65,422,797-65,445,540
(22,743 nt, the NEAT1_2 long isoform), 899 blocks, human present on the + strand in
every block, blocks contiguous and non-overlapping along the reference.

NEAT1 sits ~50 kb from MALAT1 on chr11, so 60 of the 62 MALAT1 species keep the exact
same contig id. The differences, all verified against the ids actually present in
NEAT1.maf:

  Platypus         ornAna1.Contig10406 -> ornAna1.Contig2592   (different syntenic contig)
  opossum          monDom5.chr8        -> absent               (no monDom5 line at all)
  tasmanian_devil  sarHar1.chr6_GL864985_random  146 nt (0.6%) -> dropped
  wallaby          macEug2.GL112469              120 nt (0.5%) -> dropped

The marsupials are dropped rather than kept-as-mostly-gaps: a row that is >99% gap
contributes almost nothing to the per-column cross-species deviation but does enter the
within-species normalizer in custom_real_train_absolute_deviation.py, where a mean
absolute deviation taken over ~120 nt is pure noise. NEAT1 is far less conserved outside
Eutheria than MALAT1 is, which is why this trimming is needed here and was not for MALAT1
(every MALAT1 species had >= 35% coverage).

The comment after each entry is that species' aligned-nucleotide count as a percentage of
the 22,743 human nucleotides. Species below 10% are excluded; raise that bar by deleting
entries from the bottom of this list (hedgehog 20.6% and Platypus 11.3% are the next two
to go) and re-running neat1_alignment_mapping.py.

Non-mammalian ids present in NEAT1.maf but deliberately not used (they are not in the
MALAT1 species set either): pelSin1.JH212172 (softshell turtle, 469 nt) and
allMis1.JH735170 (alligator, 199 nt).
"""

SPECIES = {
    "human": "hg38.chr11",  # 100.0% of human NEAT1
    "chimp": "panTro4.chr11",  # 99.7%
    "mouse": "mm10.chr19",  # 34.9%
    "rat": "rn6.chr1",  # 31.2%
    "cat": "felCat8.chrD1",  # 71.5%
    "elephant": "loxAfr3.scaffold_71",  # 72.0%
    "gorilla": "gorGor3.chr11",  # 99.6%
    "orangutan": "ponAbe2.chr11",  # 93.5%
    "gibbon": "nomLeu3.chr4",  # 97.9%
    "rhesus": "rheMac3.chr14",  # 97.2%
    "macaque": "macFas5.chr14",  # 97.5%
    "baboon": "papAnu2.chr14",  # 91.6%
    "green_monkey": "chlSab2.chr1",  # 98.1%
    "marmoset": "calJac3.chr11",  # 87.7%
    "squirrel_monkey": "saiBol1.JH378199",  # 84.0%
    "bushbaby": "otoGar3.GL873659",  # 83.3%
    "chinese_tree_shrew": "tupChi1.KB320702",  # 80.9%
    "squirrel": "speTri2.JH393409",  # 54.3%
    "lesser_egyptian_jerboa": "jacJac1.JH725456",  # 53.6%
    "prairie_vole": "micOch1.chr8",  # 33.4%
    "chinese_hamster": "criGri1.KE376951",  # 40.6%
    "golden_hamster": "mesAur1.KB708127",  # 40.5%
    "naked_mole-rat": "hetGla2.JH602080",  # 68.3%
    "guinea_pig": "cavPor3.scaffold_42",  # 69.9%
    "chinchilla": "chiLan1.JH721882",  # 70.8%
    "brush-tailed_rat": "octDeg1.JH651533",  # 61.9%
    "rabbit": "oryCun2.chrUn0724",  # 37.9%
    "pika": "ochPri3.JH802146",  # 54.2%
    "pig": "susScr3.chr2",  # 73.3%
    "alpaca": "vicPac2.KB632698",  # 76.3%
    "bactrian_camel": "camFer1.KB016715",  # 77.5%
    "dolphin": "turTru2.JH484543",  # 45.4%
    "killer_whale": "orcOrc1.KB316848",  # 59.6%
    "tibetan_antelope": "panHod1.KE112582",  # 76.7%
    "cow": "bosTau8.chr29",  # 78.2%
    "sheep": "oviAri3.chr21",  # 75.6%
    "domestic_goat": "capHir1.chr29",  # 76.2%
    "horse": "equCab2.chr12",  # 82.3%
    "white_rhinoceros": "cerSim1.JH767791",  # 84.3%
    "dog": "canFam3.chr18",  # 66.2%
    "ferret": "musFur1.GL897050",  # 51.3%
    "panda": "ailMel1.GL192492.1",  # 60.7%
    "pacific_walrus": "odoRosDiv1.KB228903",  # 67.9%
    "weddell_seal": "lepWed1.KB715540",  # 66.2%
    "black_flying-fox": "pteAle1.KB030625",  # 76.6%
    "megabat": "pteVam1.scaffold_2759",  # 72.0%
    "big_brown_bat": "eptFus1.JH977650",  # 71.2%
    "david's_myotis_bat": "myoDav1.KB112964",  # 73.1%
    "little_brown_bat": "myoLuc2.GL430140",  # 59.7%
    "hedgehog": "eriEur2.JH835378",  # 20.6%
    "shrew": "sorAra2.JH798265",  # 30.4%
    "star-nosed_mole": "conCri1.JH655892",  # 28.7%
    "manatee": "triMan1.JH594719",  # 77.5%
    "cape_golden_mole": "chrAsi1.JH823296",  # 64.2%
    "tenrec": "echTel2.JH980369",  # 33.5%
    "aardvark": "oryAfe1.JH863848",  # 58.0%
    "armadillo": "dasNov3.JH577683",  # 74.5%
    "Platypus": "ornAna1.Contig2592",  # 11.3%
    "cape_elephant_shrew": "eleEdw1.JH947457",  # 62.8%
}


def main():
    alignments = list(AlignIO.parse("../data/neat1/NEAT1.maf", "maf"))

    sequences = {name: "" for name in SPECIES}

    # extracting the aligned sequences
    for alignment in alignments:
        for line in alignment:
            for name, maf_id in SPECIES.items():
                if line.id == maf_id:
                    sequences[name] += str(line.seq)

    # removing alignment from the sequences and saving them in files s.t. I can run the model on it
    for name, seq in sequences.items():
        seq = clean_str(seq)
        if seq:
            with open(f"../data/neat1/{name}_neat1.txt", "w") as file:
                file.write(seq)


def clean_str(string):
    # NEAT1.maf carries lowercase soft-masked n as well as uppercase N, so both cases
    # are stripped here; MALAT1's maf only ever had uppercase N.
    return string.replace("n", "-").replace("N", "-").replace("-", "").upper().strip()


if __name__ == "__main__":
    main()
