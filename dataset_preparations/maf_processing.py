from Bio import AlignIO

"""maf 100 multiz allignment has weird names for each species:

human - hg38.chr11
chimp - panTro4.chr11
mouse - mm10.chr19
rat - rn6.chr1
cat - felCat8.chrD1
elephant - loxAfr3.scaffold_71
gorilla - gorGor3.chr11
orangutan - ponAbe2.chr11
gibbon - nomLeu3.chr4
rhesus - rheMac3.chr14
macaque - macFas5.chr14
baboon - papAnu2.chr14
green_monkey - chlSab2.chr1
marmoset - calJac3.chr11
squirrel_monkey - saiBol1.JH378199
bushbaby - otoGar3.GL873659
chinese_tree_shrew - tupChi1.KB320702
squirrel - speTri2.JH393409
lesser_egyptian_jerboa - jacJac1.JH725456
prairie_vole - micOch1.chr8
chinese_hamster - criGri1.KE376951
golden_hamster - mesAur1.KB708127
naked_mole-rat - hetGla2.JH602080
guinea_pig - cavPor3.scaffold_42
chinchilla - chiLan1.JH721882
brush-tailed_rat - octDeg1.JH651533
rabbit - oryCun2.chrUn0724
pika - ochPri3.JH802146
pig - susScr3.chr2
alpaca - vicPac2.KB632698
bactrian_camel - camFer1.KB016715
dolphin - turTru2.JH484543
killer_whale - orcOrc1.KB316848
tibetan_antelope - panHod1.KE112582
cow - bosTau8.chr29
sheep - oviAri3.chr21
domestic_goat - capHir1.chr29
horse - equCab2.chr12
white_rhinoceros - cerSim1.JH767791
dog - canFam3.chr18
ferret - musFur1.GL897050
panda - ailMel1.GL192492.1
pacific_walrus - odoRosDiv1.KB228903
weddell_seal - lepWed1.KB715540
black_flying-fox - pteAle1.KB030625
megabat - pteVam1.scaffold_2759
big_brown_bat - eptFus1.JH977650
david's_myotis_bat - myoDav1.KB112964
little_brown_bat - myoLuc2.GL430140
hedgehog - eriEur2.JH835378
shrew - sorAra2.JH798265
star-nosed_mole - conCri1.JH655892
manatee - triMan1.JH594719
cape_golden_mole - chrAsi1.JH823296
tenrec - echTel2.JH980369
aardvark - oryAfe1.JH863848
armadillo - dasNov3.JH577683
opossum - monDom5.chr8
tasmanian_devil - sarHar1.chr6_GL864985_random
wallaby - macEug2.GL112469
Platypus - ornAna1.Contig10406
cape_elephant_shrew - eleEdw1.JH947457

I can find more via this link: https://genome.ucsc.edu/cgi-bin/hgc?db=hg38&c=chr17&l=43106526&r=43106527&o=43106526&t=43106527&g=multiz100way&i=multiz100way

"""

# Some species that I decided to analyse
SPECIES = {
    "human": "hg38.chr11",
    "chimp": "panTro4.chr11",
    "mouse": "mm10.chr19",
    "rat": "rn6.chr1",
    "cat": "felCat8.chrD1",
    "elephant": "loxAfr3.scaffold_71",
    "gorilla": "gorGor3.chr11",
    "orangutan": "ponAbe2.chr11",
    "gibbon": "nomLeu3.chr4",
    "rhesus": "rheMac3.chr14",
    "macaque": "macFas5.chr14",
    "baboon": "papAnu2.chr14",
    "green_monkey": "chlSab2.chr1",
    "marmoset": "calJac3.chr11",
    "squirrel_monkey": "saiBol1.JH378199",
    "bushbaby": "otoGar3.GL873659",
    "chinese_tree_shrew": "tupChi1.KB320702",
    "squirrel": "speTri2.JH393409",
    "lesser_egyptian_jerboa": "jacJac1.JH725456",
    "prairie_vole": "micOch1.chr8",
    "chinese_hamster": "criGri1.KE376951",
    "golden_hamster": "mesAur1.KB708127",
    "naked_mole-rat": "hetGla2.JH602080",
    "guinea_pig": "cavPor3.scaffold_42",
    "chinchilla": "chiLan1.JH721882",
    "brush-tailed_rat": "octDeg1.JH651533",
    "rabbit": "oryCun2.chrUn0724",
    "pika": "ochPri3.JH802146",
    "pig": "susScr3.chr2",
    "alpaca": "vicPac2.KB632698",
    "bactrian_camel": "camFer1.KB016715",
    "dolphin": "turTru2.JH484543",
    "killer_whale": "orcOrc1.KB316848",
    "tibetan_antelope": "panHod1.KE112582",
    "cow": "bosTau8.chr29",
    "sheep": "oviAri3.chr21",
    "domestic_goat": "capHir1.chr29",
    "horse": "equCab2.chr12",
    "white_rhinoceros": "cerSim1.JH767791",
    "dog": "canFam3.chr18",
    "ferret": "musFur1.GL897050",
    "panda": "ailMel1.GL192492.1",
    "pacific_walrus": "odoRosDiv1.KB228903",
    "weddell_seal": "lepWed1.KB715540",
    "black_flying-fox": "pteAle1.KB030625",
    "megabat": "pteVam1.scaffold_2759",
    "big_brown_bat": "eptFus1.JH977650",
    "david's_myotis_bat": "myoDav1.KB112964",
    "little_brown_bat": "myoLuc2.GL430140",
    "hedgehog": "eriEur2.JH835378",
    "shrew": "sorAra2.JH798265",
    "star-nosed_mole": "conCri1.JH655892",
    "manatee": "triMan1.JH594719",
    "cape_golden_mole": "chrAsi1.JH823296",
    "tenrec": "echTel2.JH980369",
    "aardvark": "oryAfe1.JH863848",
    "armadillo": "dasNov3.JH577683",
    "opossum": "monDom5.chr8",
    "tasmanian_devil": "sarHar1.chr6_GL864985_random",
    "wallaby": "macEug2.GL112469",
    "Platypus": "ornAna1.Contig10406",
    "cape_elephant_shrew": "eleEdw1.JH947457",
}


def main():
    alignments = list(
        AlignIO.parse("../data/multiz100/MALAT1_orthologues_multiz100.maf", "maf")
    )

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
            with open(f"../data/multiz100/{name}_malat1.txt", "w") as file:
                file.write(seq)


def clean_str(string):
    return string.replace("N", "-").replace("-", "").upper().strip()


if __name__ == "__main__":
    main()
