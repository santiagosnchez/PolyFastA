#!/usr/bin/env python
# PolyFastA.py
# Santiago Sanchez-Ramirez, University of Toronto, santiago.snchez@gmail.com

import argparse
import math
import os
import sys
import re
import numpy as np
import fileinput

def main():
    # parse arguments
    parser = argparse.ArgumentParser(prog="PolyFastA.py",
    formatter_class=argparse.RawTextHelpFormatter,
    description="""
    Fast estimator of nucleotide diversity (theta_pi),
    Watterson's theta (theta_w), and Tajimas\'s D for coding and
    non-coding sequences\n""",
    epilog="""

    It can also print a vector of the folded site frequency spectrum and
    correct theta_pi for multiple hits (jukes-cantor).

    Examples:
    python PolyFastA.py -f myAlignment.fas -p pop1,pop2
    \"-p pop1,pop2\" assumes that the alignment has sequences that are labeled:

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

    The format is not strict but the identifier (e.g. pop1) needs to be somewhere in the header.\n""")
    parser.add_argument(
    '--file', '-f', nargs="*", type=str, default="",
    help='one or several alignment files in FASTA format.')
    parser.add_argument(
    '--dir', '-d', type=str, default=".",
    help='directory containing FASTA files only.')
    parser.add_argument(
    '--pops', '-p', nargs="?", default=False, metavar='pop1,pop2,pop3', type=str,
    help='split alignment by populations. A comma-separated list of strings that are found in the sequence headers.')
    parser.add_argument(
    '--out', '-o', type=str, default="",
    help='name of output file. (default/empty will print to screen')
    parser.add_argument(
    '--pipe', '-i', action="store_true", default=False,
    help='if FASTA file is being piped in from STDIN.')
    parser.add_argument(
    '--cds', '-c', action="store_true", default=False,
    help='the alignment is protein coding. Will split into synonymous and nonsynonymous sites.')
    parser.add_argument(
    '--silent', '-s', action="store_true", default=False,
    help='suppress header and verbose output.')
    parser.add_argument(
    '--name', '-n', type=str, default=False,
    help='name of the DNA region to show in output.')
    parser.add_argument(
    '--jc', action="store_true", default=False,
    help='Jukes-Cantor correction for Pi.')

    # process arguments
    args = parser.parse_args()

    r = None # empry response variable
    # check file and dir arguments
    if args.file == ''  and args.dir == '.' and not args.pipe:
        print("Both --file/-f and --dir/-d were not found.")
        r = input("Do you wish to run polySFS on all files in the current directory? [y|n]: ")
        if r != 'y':
            parser.error(parser.print_help())
    # stop if both dir and file are provided
    if len(args.file) != 0 and args.dir != '.':
        parser.error("Run with either --file/-f or --dir/-d, but not both")
    # if only dir is provided
    elif len(args.file) == 0 and args.dir != '.':
        args.file = [ args.dir + "/" + i for i in os.listdir(args.dir) ]
    # if only stdin is provided
    elif len(args.file) == 0 and args.dir == '.' and args.pipe:
        if not args.name:
            args.file = ["stdin"]
        else:
            args.file = [args.name]
    # read data and get alignment length
    if not args.silent:
        print_result({}, "", args.cds, args.out, "w", args.file, "NA", args.silent, 1, args.jc)
    for file in args.file:
        if len(args.file) > 0  and args.pipe and not (args.file[0] == "stdin" or args.file[0] == args.name):
            parser.error("A FASTA file or multiple files cannot be used with the --pipe argument.")
        d = readfasta(file, args.pipe)
        file = file.split('/')[-1]
        seqlen = list(map(lambda x: len(d[x]), d.keys()))
        if all_same(seqlen):
            seqlen = seqlen[0]
            if args.cds and (seqlen % 3) != 0:
                if len(args.out) != 0:
                    with open(args.out, "a") as o:
                        o.write(f"# CDS sequence length is not a multiple of 3: {file}\n")
                else:
                    print(f"# CDS sequence length is not a multiple of 3: {file}")
            if not args.pops:
                print_result(d, seqlen, args.cds, args.out, "a", file, "NA", args.silent, 0, args.jc)
            else:
                popkeys = args.pops.split(',')
                for pop in popkeys:
                    grp = filter(lambda x: pop in x, d.keys())
                    if len(grp) == 0:
                        if len(args.out) != 0:
                            with open(args.out, "a") as o:
                                o.write(f"# Pop {pop} string was not found in fasta headers.\n")
                        else:
                            print(f"# Pop {pop} string was not found in fasta headers.")
                        continue
                    dgrp = {k: d[k] for k in grp}
                    print_result(dgrp, seqlen, args.cds, args.out, "a", file, pop, args.silent, 0, args.jc)
        else:
            if len(args.out) != 0:
                with open(args.out, "a") as o:
                    o.write(f"# Sequences do not have the same length: {file}\n")
            else:
                print(f"# Sequences do not have the same length: {file}")



# functions

def print_result(d, seqlen, cds, out, aow, file, pop, silent, header, jc):
    def no_header(d, seqlen, cds, out, aow, file, pop):
        pos,var = getvarsites(d,seqlen)
        if cds:
            if len(var) == 0:
                if len(out) != 0:
                    with open(out,aow) as o:
                        if not silent:
                            sys.stdout.write(f"Writting to {out}, pop: {pop:<10s}, parsing: {file:<15s}\n")
                            sys.stdout.flush()
                        o.write(f"{file},{seqlen},{pop},{len(d)},{no_var()},{no_var()},{0}\n")
                else:
                    print(f"{file},{seqlen},{pop},{len(d)},{no_var()},{no_var()},{0}")
            else:
                # get synonymous and non-synonymous sites
                ssites,s,n,nstops,missingsites = getvarCDSsites(d, seqlen)
                # remainder non-synonymous sites
                nsites = (seqlen-missingsites)-ssites
                # match up all var sites with syn and nonsyn
                var_s = [ var[pos.index(i)] for i in s if i in pos ]
                var_n = [ var[pos.index(i)] for i in n if i in pos ]
                # polymorphism
                ply_s = polymorphism(var_s,ssites,jc)
                ply_n = polymorphism(var_n,nsites,jc)
                if len(out) != 0:
                    with open(out,aow) as o:
                        if not silent:
                            sys.stdout.write(f"Writting to {out}, pop: {pop:<10s}, parsing: {file:<15s}\n" % (out,pop,file)),
                            sys.stdout.flush()
                        o.write(f"{file},{seqlen},{pop},{len(var[0])},{ply_s[0]},{ply_n[0]},{ply_s[1]},{ply_n[1]},{ply_s[2]},{ply_n[2]},{ply_s[3]},{ply_n[3]},{nstops}\n")
                else:
                    print(f"{file},{seqlen},{pop},{len(var[0])},{ply_s[0]},{ply_n[0]},{ply_s[1]},{ply_n[1]},{ply_s[2]},{ply_n[2]},{ply_s[3]},{ply_n[3]},{nstops}")
        else:
            if len(var) == 0:
                if len(out) != 0:
                    with open(out,aow) as o:
                        if not silent:
                            sys.stdout.write(f"Writting to {out}, pop: {pop:<10s}, parsing: {file:<15s}\n")
                            sys.stdout.flush()
                        o.write(f"{file},{seqlen},{pop},{len(d)},{no_var()}\n")
                else:
                    print(f"{file},{seqlen},{pop},{len(d)},{no_var()}")
            else:
                ply = polymorphism(var,seqlen,jc)
                if len(out) != 0:
                    with open(out,aow) as o:
                        if not silent:
                            sys.stdout.write(f"Writting to {out}, pop: {pop:<10s}, parsing: {file:<15s}\n")
                            sys.stdout.flush()
                        o.write(f"{file},{seqlen},{pop},{len(var[0])},{ply[0]},{ply[1]},{ply[2]},{ply[3]}\n")
                else:
                    print(f"{file},{seqlen},{pop},{len(var[0])},{ply[0]},{ply[1]},{ply[2]},{ply[3]}")
    if silent:
        no_header(d, seqlen, cds, out, aow, file, pop)
    else:
        if header == 0:
            no_header(d, seqlen, cds, out, aow, file, pop)
        elif header == 1:
            print_header(header, cds, out, aow)
        elif header == 2:
            print_header(header, cds, out, aow)
            no_header(d, seqlen, cds, out, "a", file, pop)

def print_header(header, cds, out, aow):
    if cds:
        if header == 1 or header == 2:
            if len(out) != 0:
                with open(out,aow) as o:
                    o.write("file,seqlen,pop,N,seg_sites_S,seg_sites_N,pi_S,pi_N,theta_S,theta_N,tajimasD_S,tajimasD_N,nstops\n")
            else:
                print("file,seqlen,pop,N,seg_sites_S,seg_sites_N,pi_S,pi_N,theta_S,theta_N,tajimasD_S,tajimasD_N,nstops")
    else:
        if header == 1 or header == 2:
            if len(out) != 0:
                with open(out,aow) as o:
                    o.write("file,seqlen,pop,N,seg_sites,pi,theta,tajimasD\n")
            else:
                print("file,seqlen,pop,N,seg_sites,pi,theta,tajimasD")

def readfasta(file, stdin):
    data = {}
    if stdin:
        for line in sys.stdin.readlines():
            if line[0] == ">":
                head = line[1:].rstrip()
                data[head] = ''
            else:
                data[head] += line.rstrip().upper()
    else:
        data = {}
        with open(file, 'r') as f:
            for line in f:
                if line[0] == ">":
                    head = line[1:].rstrip()
                    data[head] = ''
                else:
                    data[head] += line.rstrip().upper()
    return data

def getvarsites(d, seqlen):
    var = []
    pos = []
    for p in range(seqlen):
        site = [ d[x][p:p+1] for x in d.keys() ]
        #if not all_same(site):
        if len(set(site)) > 1: # all var sites
            var.append(site)
            pos.append(p)
    return pos,var

def gethaplotypes(d, seqlen):
    var = []
    pos = []
    for p in range(seqlen):
        site = [ d[x][p:p+1] for x in d.keys() ]
        #if not all_same(site):
        if len(set(site)) > 1: # all var sites
            var.append(site)
            pos.append(p)
    return pos,list(map(list, zip(*var)))

def getsfs(var):
    N = len(var[0])
    sfs = [ 0 for i in range(int(N/2)) ]
    for v in var:
        a = list(set(v))
        a = filter(lambda x: 'A' in x or 'C' in x or 'T' in x or 'G' in x, a)
        cm = sorted(list(map(lambda x: v.count(x), a)), reverse=True)
        sfs[cm[1]-1] += 1
    return sfs

def getvarCDSsites(d, seqlen):
    stop_cod = ['TGA','TAA','TAG'] # stop codons, Universal Genetic Code
    # vector of every third position
    everythird = range(0,seqlen,3)
    # start synonymous site counter
    count_syn = 0.0
    # split into codons
    cods = [ list(set(map(lambda x: d[x][cp:cp+3], d.keys()))) for cp in everythird ]
    # count stop codons
    nstops = len([ cod for cod in cods if any([ c in stop_cod for c in cod ]) ])
    # remove stop codons
    codsnpos = [ cod for cod in zip(cods,everythird) if not any([ c in stop_cod for c in cod ]) ]
    # start del cod pos
    missingcodpos = 0
    codsnpos_clean = []
    # check codons
    for codpos in codsnpos:
        tmp = [ cod for cod in codpos[0] if re.match("^[AGTC][AGTC][AGTC]$", cod)]
        if len(tmp) > 0:
            codsnpos_clean += [[tmp, codpos[1]]]
        else:
            missingcodpos += 3
    # count synonymous sites
    for cod in codsnpos_clean: count_syn += sum([ syncodfreq(c) for c in cod[0] ])/len(cod[0])
    # keep only variable codons and positions
    codsnpos_clean = [ cod for cod in codsnpos_clean if len(cod[0]) > 1 ]
    # extract synonymous and non-synonymous positions
    S = sum([ get_syn_nonsyn_cod_sites(cod)[0] for cod in codsnpos ], [])
    N = sum([ get_syn_nonsyn_cod_sites(cod)[1] for cod in codsnpos ], [])
    return count_syn,S,N,nstops,missingcodpos


# new function for S and N codon positions
def get_syn_nonsyn_cod_sites(cod):
    S,N = [],[]
    # universal genetic code
    gc = {
     'AAA': 'K', 'AAC': 'N', 'AAG': 'K', 'AAT': 'N', 'ACA': 'T',  'ACC': 'T',  'ACG': 'T',  'ACT': 'T', 'AGA': 'R2',  'AGC': 'S2', 'AGG': 'R2', 'AGT': 'S2', 'ATA': 'I', 'ATC': 'I', 'ATG': 'M', 'ATT': 'I',
     'CAA': 'Q', 'CAC': 'H', 'CAG': 'Q', 'CAT': 'H', 'CCA': 'P',  'CCC': 'P',  'CCG': 'P',  'CCT': 'P', 'CGA': 'R4',  'CGC': 'R4',  'CGG': 'R4', 'CGT': 'R4',  'CTA': 'L', 'CTC': 'L', 'CTG': 'L', 'CTT': 'L',
     'GAA': 'E', 'GAC': 'D', 'GAG': 'E', 'GAT': 'D', 'GCA': 'A',  'GCC': 'A',  'GCG': 'A',  'GCT': 'A', 'GGA': 'G',  'GGC': 'G',  'GGG': 'G', 'GGT': 'G',  'GTA': 'V', 'GTC': 'V', 'GTG': 'V', 'GTT': 'V',
                 'TAC': 'Y',             'TAT': 'Y', 'TCA': 'S4', 'TCC': 'S4', 'TCG': 'S4', 'TCT': 'S4',             'TGC': 'C',  'TGG': 'W', 'TGT': 'C',  'TTA': 'L', 'TTC': 'F', 'TTG': 'L', 'TTT': 'F'}
    # degenerancy dictionary
    aa_degen = {
     'K': '2fAG', 'N': '2fCT', 'T':  '4f', 'R2': '2fAG',  'S2': '2fCT', 'S2': '2fCT', 'I': '3f', 'M': '0f',
     'Q': '2fAG', 'H': '2fCT', 'P':  '4f', 'R4': '4f',    'L': '4f',
     'E': '2fAG', 'D': '2fCT', 'A':  '4f', 'G':  '4f',    'V': '4f',
                  'Y': '2fCT', 'S4': '4f', 'C':  '2fCT',  'W': '0f',    'F': '2fCT', 'L':  '2fAG'}
    # keep only complete codons
    cod = [[ c for c in cod[0] if gc.get(c) ], cod[1]]
    # check if any codons left
    if len(cod[0]) <= 1:
        return S,N
    else:
        # vector with variable positions
        vcp = []
        vcb = []
        for i in range(3):
            codbase = [ c[i] for c in cod[0] ]
            if not all_same(codbase):
                vcp.append(i)
                vcb.append(list(set(codbase)))
        # test codons
        # get amino acids
        aa = [ gc[i] for i in cod[0] ]
        # first synonymous
        if all_same(aa):
            S += [ i + cod[1] for i in vcp ]
        # everything else
        else:
            # R2 synonymous at position 1
            if any([ a == "R1" for a in aa ]) and any([ a == "R2" for i in aa ]) and not any([i == 2 for i in vcp]):
                if len(cod[0]) == 2:
                    # if only R1 and R2
                    S.append(vcp[0] + cod[1])
                else:
                    if len(vcb[0]) == 2:
                        # if there are only two bases at possition 1
                        S.append(vcp[0] + cod[1])
                    else:
                        # treat all changes in position 1 as non-synonymous
                        # even if there is one R2 polymorphism
                        N += [ i + cod[1] for i in vcp ]
            else:
                # check mixed synonymous and non-synonymous changes
                if len(list(set(aa))) < len(aa):
                    # found synonymous
                    syn = {}
                    for a in aa:
                        if syn.get(a):
                            syn[a] += 1
                        else:
                            syn[a] = 1
                    c = sorted(list(syn.values()))
                    if max(c) >= len(vcb[-1]):
                        S.append(vcp[-1] + cod[1])
                    if len(vcp) > 1:
                        N += [ i + cod[1] for i in vcp[:-1] ]
                else:
                    N += [ i + cod[1] for i in vcp ]
            # vectors of synonymous and nonsynonymous changes
        return S,N

def no_var():
    return "0,0,0,NA"

# standard estimation based on Nei and Li 1979
# based on differences and frequency of haplotypes
def nucleotide_diversity2(haplo, seqlen):
    def nt_diff(i, j, haplo):
        return sum([ haplo[i][d] != haplo[j][d] for d in range(len(haplo[i])) ])

    def haplo_freqs(haplo):
        count = {}
        for h in [ ''.join(x) for x in haplo ]:
            if count.get(h):
                count[h] += 1
            else:
                count[h] = 1
        prop = { h:count[h]/len(haplo) for h in count_haplo.keys() }
        return prop

    N = len(haplo)
    p = []
    freqs = haplo_freqs(haplo)
    for i,j in itertools.combinations(range(N), 2):
        h_i = ''.join(haplo[i])
        h_j = ''.join(haplo[j])
        p.append(freqs[h_i] * freqs[h_j] * (nt_diff(i, j, haplo)/seqlen))
    pi = 2 * sum(p)
    return pi

# based on Tajima 1989
# average number of pairwise haplotype differences
# identical to per site estimation
def nucleotide_diversity3(haplo):
    def nt_diff(i, j, haplo):
        return sum([ haplo[i][d] != haplo[j][d] for d in range(len(haplo[i])) ])

    N = len(haplo)
    p = 0
    pw = list(itertools.combinations(range(N), 2))
    for i,j in pw:
        h_i = ''.join(haplo[i])
        h_j = ''.join(haplo[j])
        p += nt_diff(i, j, haplo)
    pi = p / len(pw)
    return pi

# based on the Tajima 1989 and Cutter 2019
# per site measure of heterozygosity
# scaled by sample size
def nucleotide_diversity(var):
    def sum_P_ij2(x):
        a = list(set(x))
        n = len(x)
        return sum([ (x.count(j)/n) ** 2 for j in a ])
    N = len(var[0])
    pi = N/(N-1) * sum([ 1 - sum_P_ij2(x) for x in var ])
    return pi

def wattersons_theta(var):
    N = len(var[0])
    a1 = sum([ 1/i for i in range(1,N)])
    return len(var)/a1

def jukes_cantor_correction(x):
    return -0.75*math.log(1-(4./3.)*x)

def polymorphism(var, seqlen, jc):
    if len(var) == 0:
        return 0,0,0,"NA"
    else:
        th_pi = nucleotide_diversity(var)
        th_wa = wattersons_theta(var)
        try:
            D = (th_pi - th_wa) / Dvar(var)
        except ZeroDivisionError:
            D = "NA"
        if jc:
            try:
                th_pi_site = jukes_cantor_correction(th_pi/seqlen)
            except:
                th_pi_site = th_pi/seqlen
        else:
            th_pi_site = th_pi/seqlen
        th_wa_site = th_wa/seqlen
        return len(var),th_pi_site,th_wa_site,D

def Dvar(var):
    N = len(var[0])
    ss = len(var)
    a1 = sum(list(map(lambda x: 1.0/x, range(1,N))))
    a2 = sum(list(map(lambda x: 1.0/(x**2), range(1,N))))
    b1 = (N+1.0)/(3.0*(N-1.0))
    b2 = (2.0*((N**2.0)+N+3.0))/(9.0*N*(N-1.0))
    c1 = b1 - (1/a1)
    c2 = b2 - ((N+2)/(a1*N)) + (a2/(a1**2))
    e1 = c1/a1
    e2 = c2/((a1**2)+a2)
    Dv = math.sqrt((e1*ss)+(e2*ss*(ss-1)))
    return Dv

def syncodfreq(cod):
    freq = {
    'TTT' : 1/3, 'TCT' : 1.,'TAT' : 1/3,'TGT' : 1/3,
    'TTC' : 1/3, 'TCC' : 1.,'TAC' : 1/3,'TGC' : 1/3,
    'TTA' : 2/3, 'TCA' : 1.,'TAA' : 0,  'TGA' : 0,
    'TTG' : 2/3, 'TCG' : 1.,'TAG' : 0,  'TGG' : 0,

    'CTT' : 1.,   'CCT' : 1.,'CAT' : 1/3,'CGT' : 1.,
    'CTC' : 1.,   'CCC' : 1.,'CAC' : 1/3,'CGC' : 1.,
    'CTA' : 4/3, 'CCA' : 1.,'CAA' : 1/3,'CGA' : 4/3,
    'CTG' : 4/3, 'CCG' : 1.,'CAG' : 1/3,'CGG' : 4/3,

    'ATT' : 2/3, 'ACT' : 1.,'AAT' : 1/3,'AGT' : 1/3,
    'ATC' : 2/3, 'ACC' : 1.,'AAC' : 1/3,'AGC' : 1/3,
    'ATA' : 2/3, 'ACA' : 1.,'AAA' : 1/3,'AGA' : 2/3,
    'ATG' : 0,   'ACG' : 1.,'AAG' : 1/3,'AGG' : 2/3,

    'GTT' : 1.,  'GCT' : 1.,'GAT' : 1/3,'GGT' : 1.,
    'GTC' : 1.,  'GCC' : 1.,'GAC' : 1/3,'GGC' : 1.,
    'GTA' : 1.,  'GCA' : 1.,'GAA' : 1/3,'GGA' : 1.,
    'GTG' : 1.,  'GCG' : 1.,'GAG' : 1/3,'GGG' : 1.}
    return freq[cod]

def delstop(codons,nstops):
    stop = ['TGA','TAA','TAG']
    for i in stop:
        if i in codons:
            nstops = nstops+1
            codons.remove(i)
    return codons,nstops

def tstv(var):
    ts = []
    tv = []
    for i in var:
        if ''.join(sorted(set(i))) == 'AG' or ''.join(sorted(set(i))) == 'CT':
            ts += [i]
        else:
            tv += [i]
    return ts,tv

def choose_n(k, n=2):
    return int(k*(k-1)/n)

def all_same(items):
    return all([ x == items[0] for x in items ])

def wrapseq(seq):
    chunks = []
    interval = list(map(lambda x: x*100, range((len(seq)/100)+2)))
    for i in interval:
        if i != interval[-1]:
            chunks.append(seq[i:interval[interval.index(i)+1]-1])
    return("\n".join(chunks))

if __name__ == '__main__':
    main()
