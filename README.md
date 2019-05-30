# PolyFastA

Run the code with the help `-h` flag and it will pull down some options.

    python PolyFastA.py -h


```usage: PolyFastA.py [-h] [--file FILE] [--dir DIR] [--pops [pop1,pop2,pop3]]
                    [--out OUT] [--pipe] [--cds] [--silent] [--name NAME]
                    [--jc]

    Fast estimator of nucleotide diversity (pi), theta, and Tajimas's D 
    based on the site frequency spectrum.

optional arguments:
  -h, --help            show this help message and exit
  --file FILE, -f FILE  an alignment in FASTA format.
  --dir DIR, -d DIR     directory containing FASTA files only.
  --pops [pop1,pop2,pop3], -p [pop1,pop2,pop3]
                        split alignment by populations. A comma-separated list of strings that are found in the sequence headers.
  --out OUT, -o OUT     name of output file. (default/empty will print to screen
  --pipe, -i            if FASTA file is being piped in from STDIN.
  --cds, -c             the alignment is protein coding. Will split into synonymous and nonsynonymous sites.
  --silent, -s          suppress header and verbose output.
  --name NAME, -n NAME  name of the DNA region to show in output.
  --jc                  Jukes-Cantor correction for Pi.

    Examples:
    python PolyFastA.py -f myAlignment.fas -p pop1,pop2
    "-p pop1,pop2" assumes that the alignment has sequences that are labeled:

    >pop1_ind1_XXX
    ATGC...
    >pop1_ind2_XXX
    ATGC...
    >pop2_ind1_XXX
    ATGC...
    >pop2_ind2_XXX
    
    Alignment is inframe cooding sequence:
    python PolyFastA.py -f myAlignment.fas -p pop1,pop2 --cds
    
    Want Jukes-Cantor corrected estimates for CDS:
    python PolyFastA.py -f myAlignment.fas -p pop1,pop2 --cds --jc

    FASTA files in directory:
    python PolyFastA.py -d myFastaDir/ --out allpoly.csv

    Read from pipe:
    python PolyFastA.py --pipe -p pop1,pop2

    The format is not strict but the identifier (e.g. pop1) needs to be somewhere in the header.```
