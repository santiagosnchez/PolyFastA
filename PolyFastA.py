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
    '--file', '-f', type=str, default="",
    help='an alignment in FASTA format.')
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

    args = parser.parse_args()
    # process arguments
    r = None
    if args.file == ''  and args.dir == '.' and not args.pipe:
        print("Both --file/-f and --dir/-d were not found.")
        r = input("Do you wish to run polySFS on all files in the current directory? [y|n]: ")
        if r != 'y':
            parser.error(parser.print_help())
    if len(args.file) != 0 and args.dir != '.':
        parser.error("Run with either --file/-f or --dir/-d, but not both")
    elif (len(args.file) != 0 and args.dir == '.') or args.pipe:
        if args.pipe:
            if not args.name:
                args.file = "stdin"
            else:
                args.file = args.name
        else:
            if args.name:
                args.file = args.name
        # read data and get alignment length
        d = readfasta(args.file, args.pipe)
        seqlen = list(map(lambda x: len(d[x]), d.keys()))
        if all_same(seqlen):
            seqlen = seqlen[0]
        else:
            parser.error("Sequences do not have the same length.")
        if args.cds and (seqlen % 3) != 0:
            parser.error("CDS sequence length is not a multiple of 3.")
        if not args.pops:
            print_result(d, seqlen, args.cds, args.out, "w", args.file, "NA", args.silent, 2, args.jc)
        else:
            popkeys = args.pops.split(',')
            if not args.silent:
                print_result({}, "", args.cds, args.out, "w", args.file, "NA", args.silent, 1, args.jc)
            for pop in popkeys:
                grp = filter(lambda x: pop in x, d.keys())
                if len(grp) == 0:
                    print(f"Pop {pop} string was not found in fasta headers.")
                    continue
                dgrp = {k:d[k] for k in grp}
                print_result(dgrp, seqlen, args.cds, args.out, "a", args.file, pop, args.silent, 0, args.jc)
    elif len(args.file) == 0 and args.dir != '.' or r:
        files = [ args.dir + "/" + i for i in os.listdir(args.dir) ]
        if not args.silent:
            print_result({}, "", args.cds, args.out, "w", args.file, "NA", args.silent, 1, args.jc)
        for file in files:
            if args.pipe:
                parser.error("Multiple files cannot be used with the --pipe argument.")
            d = readfasta(file, args.pipe)
            file = file.split('/')[-1]
            seqlen = list(map(lambda x: len(d[x]), d.keys()))
            if all_same(seqlen):
                seqlen = seqlen[0]
            else:
                parser.error("Sequences do not have the same length.")
            if args.cds and (seqlen % 3) != 0:
                parser.error(f"CDS sequence length is not a multiple of 3: {file}")
            if not args.pops:
                print_result(d, seqlen, args.cds, args.out, "a", file, "NA", args.silent, 0, args.jc)
            else:
                popkeys = args.pops.split(',')
                for pop in popkeys:
                    grp = filter(lambda x: pop in x, d.keys())
                    if len(grp) == 0:
                        print(f"Pop {pop} string was not found in fasta headers.")
                        continue
                    dgrp = {k: d[k] for k in grp}
                    print_result(dgrp, seqlen, args.cds, args.out, "a", file, pop, args.silent, 0, args.jc)


# functions

def print_result(d, seqlen, cds, out, aow, file, pop, silent, header, jc):
    def no_header(d, seqlen, cds, out, aow, file, pop):
        pos,var = getvarsites(d,seqlen)
        if cds:
            s,n,ssites,nstops = var_site_class(d, seqlen)
            nsites = seqlen-ssites
            var_s = [ var[pos.index(x)] for x in s ]
            var_n = [ var[pos.index(x)] for x in n ]
            # ssfs = getsfs(var_s)
            # nsfs = getsfs(var_n)
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

def getsfs(var):
    N = len(var[0])
    sfs = [ 0 for i in range(int(N/2)) ]
    for v in var:
        a = list(set(v))
        a = filter(lambda x: 'A' in x or 'C' in x or 'T' in x or 'G' in x, a)
        cm = sorted(list(map(lambda x: v.count(x), a)), reverse=True)
        sfs[cm[1]-1] += 1
    return sfs

def var_site_class(d, seqlen):
    stop_cod = ['TGA','TAA','TAG'] # stop codons, Universal Genetic Code
    # vector of every third position
    everythird = range(0,seqlen,3)
    # start synonymous site counter
    count_syn = 0.0
    # for storing N and S positions
    S = []
    N = []
    aamat = syn_nonsyn_matrix()
    # split into codons
    cods = [ list(set(map(lambda x: d[x][cp:cp+3], d.keys()))) for cp in everythird ]
    # count stop codons
    nstops = len([ cod for cod in cods if any([ c in stop_cod for c in cod ]) ])
    # remove stop codons
    cods = [ cod for cod in cods if not any([ c in stop_cod for c in cod ]) ]
    # count synonymous sites
    for cod in cods: count_syn += sum([ syncodfreq(c) for c in cod ])/len(cod)
    # # keep variable codons
    # cods = [ cod for cod in cods if len(cod) > 1 ]
    for i in range(len(cods)):
        cod = cods[i]
        cp = i*3
        if len(cod) == 2:
            if aamat[cod[0]][cod[1]]['S']:
                _ = [ S.append(cp+x) for x in aamat[cod[0]][cod[1]]['S'] ]
            if aamat[cod[0]][cod[1]]['N']:
                _ = [ N.append(cp+x) for x in aamat[cod[0]][cod[1]]['N'] ]
        elif len(cod) > 2:
            p = {'S':[0,0,0],'N':[0,0,0]}
            for i in range(len(cod)):
                for j in range(i+1, len(cod)):
                    if i != len(cod):
                        if aamat[cod[i]][cod[j]]['S']:
                            for k in aamat[cod[i]][cod[j]]['S']:
                                p['S'][k] += 1
                        if aamat[cod[i]][cod[j]]['N']:
                            for k in aamat[cod[i]][cod[j]]['N']:
                                p['N'][k] += 1
            for i in range(3):
                if p['S'][i]+p['N'][i] == 2 or p['S'][i]+p['N'][i] == 1:
                    if p['N'][i] == 0:
                        S.append(cp+i)
                    elif p['S'][i] == 0:
                        N.append(cp+i)
    return S,N,count_syn,nstops

def nucleotide_diversity(var):
    def sum_P_ij2(x):
        a = list(set(x))
        n = len(x)
        return sum([ (x.count(j)/n) ** 2 for j in a ])
    N = len(var[0])
    p = sum([ 1 - sum_P_ij2(x) for x in var ])
    pi = N/(N-1) * p
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
                th_pi_site = "Inf"
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

def syn_nonsyn_matrix(aa=None, degen=None):
    # universal genetic code
    gc = {
     'AAA': 'K', 'AAC': 'N', 'AAG': 'K', 'AAT': 'N', 'ACA': 'T', 'ACC': 'T', 'ACG': 'T', 'ACT': 'T', 'AGA': 'R', 'AGC': 'S', 'AGG': 'R', 'AGT': 'S', 'ATA': 'I', 'ATC': 'I', 'ATG': 'M', 'ATT': 'I',
     'CAA': 'Q', 'CAC': 'H', 'CAG': 'Q', 'CAT': 'H', 'CCA': 'P', 'CCC': 'P', 'CCG': 'P', 'CCT': 'P', 'CGA': 'R', 'CGC': 'R', 'CGG': 'R', 'CGT': 'R', 'CTA': 'L', 'CTC': 'L', 'CTG': 'L', 'CTT': 'L',
     'GAA': 'E', 'GAC': 'D', 'GAG': 'E', 'GAT': 'D', 'GCA': 'A', 'GCC': 'A', 'GCG': 'A', 'GCT': 'A', 'GGA': 'G', 'GGC': 'G', 'GGG': 'G', 'GGT': 'G', 'GTA': 'V', 'GTC': 'V', 'GTG': 'V', 'GTT': 'V',
     'TAA': 'stop', 'TAC': 'Y', 'TAG': 'stop', 'TAT': 'Y', 'TCA': 'S', 'TCC': 'S', 'TCG': 'S', 'TCT': 'S', 'TGA': 'stop', 'TGC': 'C', 'TGG': 'W', 'TGT': 'C', 'TTA': 'L', 'TTC': 'F', 'TTG': 'L', 'TTT': 'F'}
    # degenerancy dictionary
    gc_degen = {
     'AAA': '2fAG', 'AAC': '2fCT', 'AAG': '2fAG', 'AAT': '2fCT', 'ACA': '4f', 'ACC': '4f', 'ACG': '4f', 'ACT': '4f', 'AGA': '2fAG', 'AGC': '2fCT', 'AGG': '2fAG', 'AGT': '2fCT', 'ATA': '3f', 'ATC': '3f', 'ATG': '0f', 'ATT': '3f',
     'CAA': '2fAG', 'CAC': '2fCT', 'CAG': '2fAG', 'CAT': '2fCT', 'CCA': '4f', 'CCC': '4f', 'CCG': '4f', 'CCT': '4f', 'CGA': '4f', 'CGC': '4f', 'CGG': '4f', 'CGT': '4f', 'CTA': '4f', 'CTC': '4f', 'CTG': '4f', 'CTT': '4f',
     'GAA': '2fAG', 'GAC': '2fCT', 'GAG': '2fAG', 'GAT': '2fCT', 'GCA': '4f', 'GCC': '4f', 'GCG': '4f', 'GCT': '4f', 'GGA': '4f', 'GGC': '4f', 'GGG': '4f', 'GGT': '4f', 'GTA': '4f', 'GTC': '4f', 'GTG': '4f', 'GTT': '4f',
     'TAA': 'stop', 'TAC': '2fCT', 'TAG': 'stop', 'TAT': '2fCT', 'TCA': '4f', 'TCC': '4f', 'TCG': '4f', 'TCT': '4f', 'TGA': 'stop', 'TGC': '2fCT', 'TGG': '0f', 'TGT': '2fCT', 'TTA': '2fAG', 'TTC': '2fCT', 'TTG': '2fAG', 'TTT': '2fCT'}
    # build codon 64x64 dictionary
    codons = sorted(gc.keys())
    aamat = {}
    c = 0
    for i in codons:
        aamat[i] = {}
        for j in codons:
            aamat[i][j] = c
            c += 1

    # propagate mutation
    for i in range(64):
        for j in range(64):
            if codons[i] != codons[j]:
                # if synonymous
                if gc[codons[i]] == gc[codons[j]]:
                    # deal with Leucine
                    if gc[codons[i]] == 'L':
                        if gc_degen[codons[i]] == '2fAG' or gc_degen[codons[j]] == '2fAG':
                            if codons[i][2] != codons[j][2] and codons[i][0] != codons[j][0]:
                                aamat[codons[i]][codons[j]] = {'S':[0,2],'N':None}
                            elif codons[i][2] != codons[j][2] and codons[i][0] == codons[j][0]:
                                aamat[codons[i]][codons[j]] = {'S':[2],'N':None}
                            elif codons[i][2] == codons[j][2] and codons[i][0] != codons[j][0]:
                                aamat[codons[i]][codons[j]] = {'S':[0],'N':None}
                        else:
                            aamat[codons[i]][codons[j]] = {'S':[2],'N':None}
                    # deal with Arginine
                    elif gc[codons[i]] == 'R':
                        if gc_degen[codons[i]] == '2fAG' or gc_degen[codons[j]] == '2fAG':
                            if codons[i][2] != codons[j][2] and codons[i][0] != codons[j][0]:
                                aamat[codons[i]][codons[j]] = {'S':[0,2],'N':None}
                            elif codons[i][2] != codons[j][2] and codons[i][0] == codons[j][0]:
                                aamat[codons[i]][codons[j]] = {'S':[2],'N':None}
                            elif codons[i][2] == codons[j][2] and codons[i][0] != codons[j][0]:
                                aamat[codons[i]][codons[j]] = {'S':[0],'N':None}
                        else:
                            aamat[codons[i]][codons[j]] = {'S':[2],'N':None}
                    # deal with Serine
                    elif gc[codons[i]] == 'S':
                        if gc_degen[codons[i]] == '2fCT' or gc_degen[codons[j]] == '2fCT':
                            if codons[i][2] != codons[j][2] and codons[i][0] == codons[j][0]:
                                aamat[codons[i]][codons[j]] = {'S':[2],'N':None}
                            elif codons[i][2] == codons[j][2] and codons[i][0] != codons[j][0]:
                                aamat[codons[i]][codons[j]] = {'S':None,'N':[0,1]}
                            else:
                                aamat[codons[i]][codons[j]] = {'S':[2],'N':[0,1]}
                        else:
                            aamat[codons[i]][codons[j]] = {'S':[2],'N':None}
                    # all others
                    else:
                        aamat[codons[i]][codons[j]] = {'S':[2],'N':None}
                # if non-synonymous
                else:
                    # traverse position 1 for T
                    if codons[i][0] == 'T' and codons[j][0] == 'T':
                        # 4-fold first
                        if gc_degen[codons[i]] == '4f' or gc_degen[codons[j]] == '4f':
                            if codons[i][2] != codons[j][2]:
                                aamat[codons[i]][codons[j]] = {'S':[2],'N':[1]}
                            else:
                                aamat[codons[i]][codons[j]] = {'S':None,'N':[1]}
                        # 2-fold second CT
                        elif gc_degen[codons[i]] == '2fCT' and gc_degen[codons[j]] == '2fCT':
                            if codons[i][2] != codons[j][2]:
                                aamat[codons[i]][codons[j]] = {'S':[2],'N':[1]}
                            else:
                                aamat[codons[i]][codons[j]] = {'S':None,'N':[1]}
                        # 2-fold second CT and AG
                        elif (gc_degen[codons[i]] == '2fAG' or gc_degen[codons[j]] == '2fAG') and (gc_degen[codons[i]] == '2fCT' or gc_degen[codons[j]] == '2fCT'):
                            if codons[i][1] != codons[j][1]:
                                aamat[codons[i]][codons[j]] = {'S':None,'N':[1,2]}
                            else:
                                aamat[codons[i]][codons[j]] = {'S':None,'N':[2]}
                        # Triptophan
                        elif gc_degen[codons[i]] == '0f' or gc_degen[codons[j]] == '0f':
                            if gc_degen[codons[i]] == '2fAG' or gc_degen[codons[j]] == '2fAG':
                                if codons[i][2] != codons[j][2]:
                                    aamat[codons[i]][codons[j]] = {'S':[2],'N':[1]}
                                else:
                                    aamat[codons[i]][codons[j]] = {'S':None,'N':[1]}
                            elif gc_degen[codons[i]] == '2fCT' or gc_degen[codons[j]] == '2fCT':
                                if codons[i][1] != codons[j][1]:
                                    aamat[codons[i]][codons[j]] = {'S':None,'N':[1,2]}
                                else:
                                    aamat[codons[i]][codons[j]] = {'S':None,'N':[2]}
                    # traverse position 1 for C and G
                    if (codons[i][0] == 'C' and codons[j][0] == 'C') or (codons[i][0] == 'G' and codons[j][0] == 'G'):
                        # 4-fold first
                        if gc_degen[codons[i]] == '4f' or gc_degen[codons[j]] == '4f':
                            if codons[i][2] != codons[j][2]:
                                aamat[codons[i]][codons[j]] = {'S':[2],'N':[1]}
                            else:
                                aamat[codons[i]][codons[j]] = {'S':None,'N':[1]}
                        # 2-fold second CT and AG
                        elif (gc_degen[codons[i]] == '2fAG' or gc_degen[codons[j]] == '2fAG') and (gc_degen[codons[i]] == '2fCT' or gc_degen[codons[j]] == '2fCT'):
                            aamat[codons[i]][codons[j]] = {'S':None,'N':[2]}
                    # traverse position 1 for A
                    if codons[i][0] == 'A' and codons[j][0] == 'A':
                        # 4-fold first
                        if gc_degen[codons[i]] == '4f' or gc_degen[codons[j]] == '4f':
                            if codons[i][2] != codons[j][2]:
                                aamat[codons[i]][codons[j]] = {'S':[2],'N':[1]}
                            else:
                                aamat[codons[i]][codons[j]] = {'S':None,'N':[1]}
                        # 2-fold second CT
                        elif gc_degen[codons[i]] == '2fCT' and gc_degen[codons[j]] == '2fCT':
                            if codons[i][2] != codons[j][2]:
                                aamat[codons[i]][codons[j]] = {'S':[2],'N':[1]}
                            else:
                                aamat[codons[i]][codons[j]] = {'S':None,'N':[1]}
                        # 2-fold second AG
                        elif gc_degen[codons[i]] == '2fAG' and gc_degen[codons[j]] == '2fAG':
                            if codons[i][2] != codons[j][2]:
                                aamat[codons[i]][codons[j]] = {'S':[2],'N':[1]}
                            else:
                                aamat[codons[i]][codons[j]] = {'S':None,'N':[1]}
                        # 2-fold second CT and AG
                        elif (gc_degen[codons[i]] == '2fAG' or gc_degen[codons[j]] == '2fAG') and (gc_degen[codons[i]] == '2fCT' or gc_degen[codons[j]] == '2fCT'):
                            if codons[i][1] != codons[j][1]:
                                aamat[codons[i]][codons[j]] = {'S':None,'N':[1,2]}
                            else:
                                aamat[codons[i]][codons[j]] = {'S':None,'N':[2]}
                        # 3-fold
                        elif gc_degen[codons[i]] == '3f' or gc_degen[codons[j]] == '3f':
                            if codons[i][1] != codons[j][1]:
                                if codons[i][2] != codons[j][2]:
                                    aamat[codons[i]][codons[j]] = {'S':[2],'N':[1]}
                                else:
                                    aamat[codons[i]][codons[j]] = {'S':None,'N':[1]}
                            else:
                                aamat[codons[i]][codons[j]] = {'S':None,'N':[2]}
                        # Methionine
                        elif (gc_degen[codons[i]] == '0f' or gc_degen[codons[j]] == '0f') and (gc_degen[codons[i]] == '2fAG' or gc_degen[codons[j]] == '2fAG'):
                            if codons[i][2] != codons[j][2]:
                                aamat[codons[i]][codons[j]] = {'S':[2],'N':[1]}
                            else:
                                aamat[codons[i]][codons[j]] = {'S':None,'N':[1]}
                        elif (gc_degen[codons[i]] == '0f' or gc_degen[codons[j]] == '0f') and (gc_degen[codons[i]] == '2fCT' or gc_degen[codons[j]] == '2fCT'):
                            aamat[codons[i]][codons[j]] = {'S':None,'N':[1,2]}
                    # traverse position 2 for T
                    if (codons[i][1] == 'T' and codons[j][1] == 'T') and (codons[i][0] != codons[j][0]):
                        # 4-fold first
                        if gc_degen[codons[i]] == '4f' or gc_degen[codons[j]] == '4f':
                            if codons[i][2] != codons[j][2]:
                                aamat[codons[i]][codons[j]] = {'S':[2],'N':[0]}
                            else:
                                aamat[codons[i]][codons[j]] = {'S':None,'N':[0]}
                        # 3-fold
                        elif gc_degen[codons[i]] == '3f' or gc_degen[codons[j]] == '3f':
                            if codons[i][2] != codons[j][2]:
                                aamat[codons[i]][codons[j]] = {'S':[2],'N':[0]}
                            else:
                                aamat[codons[i]][codons[j]] = {'S':None,'N':[0]}
                        # Methionine
                        elif gc_degen[codons[i]] == '0f' or gc_degen[codons[j]] == '0f':
                            if gc_degen[codons[i]] == '2fCT' or gc_degen[codons[j]] == '2fCT':
                                aamat[codons[i]][codons[j]] = {'S':None,'N':[0,2]}
                            elif gc_degen[codons[i]] == '2fAG' or gc_degen[codons[j]] == '2fAG':
                                if codons[i][2] == codons[j][2]:
                                    aamat[codons[i]][codons[j]] = {'S':None,'N':[0]}
                                else:
                                    aamat[codons[i]][codons[j]] = {'S':[2],'N':[0]}
                    # traverse position 2 for C
                    if (codons[i][1] == 'C' and codons[j][1] == 'C') and (codons[i][0] != codons[j][0]):
                        # all 4-fold
                        if codons[i][2] != codons[j][2]:
                            aamat[codons[i]][codons[j]] = {'S':[2],'N':[0]}
                        else:
                            aamat[codons[i]][codons[j]] = {'S':None,'N':[0]}
                    # traverse position 2 for A
                    if (codons[i][1] == 'A' and codons[j][1] == 'A') and (codons[i][0] != codons[j][0]):
                        # only 2-fold
                        if (gc_degen[codons[i]] == '2fCT' and gc_degen[codons[j]] == '2fCT') or (gc_degen[codons[i]] == '2fAG' and gc_degen[codons[j]] == '2fAG'):
                            if codons[i][2] != codons[j][2]:
                                aamat[codons[i]][codons[j]] = {'S':[2],'N':[0]}
                            else:
                                aamat[codons[i]][codons[j]] = {'S':None,'N':[0]}
                        else:
                            aamat[codons[i]][codons[j]] = {'S':None,'N':[0,2]}
                    # traverse position 2 for G
                    if (codons[i][1] == 'G' and codons[j][1] == 'G') and (codons[i][0] != codons[j][0]):
                        # 4-fold first
                        if gc_degen[codons[i]] == '4f' or gc_degen[codons[j]] == '4f':
                            if codons[i][2] != codons[j][2]:
                                aamat[codons[i]][codons[j]] = {'S':[2],'N':[0]}
                            else:
                                aamat[codons[i]][codons[j]] = {'S':None,'N':[0]}
                        # 2-fold second CT
                        elif gc_degen[codons[i]] == '2fCT' and gc_degen[codons[j]] == '2fCT':
                            if codons[i][2] != codons[j][2]:
                                aamat[codons[i]][codons[j]] = {'S':[2],'N':[0]}
                            else:
                                aamat[codons[i]][codons[j]] = {'S':None,'N':[0]}
                        # Tryptophan and 2-fold second CT/AG
                        elif (gc_degen[codons[i]] == '2fAG' or gc_degen[codons[j]] == '2fAG') and (gc_degen[codons[i]] == '0f' or gc_degen[codons[j]] == '0f'):
                            if codons[i][2] != codons[j][2]:
                                aamat[codons[i]][codons[j]] = {'S':[2],'N':[0]}
                            else:
                                aamat[codons[i]][codons[j]] = {'S':None,'N':[0]}
                        elif (gc_degen[codons[i]] == '2fCT' or gc_degen[codons[j]] == '2fCT') and (gc_degen[codons[i]] == '0f' or gc_degen[codons[j]] == '0f'):
                            aamat[codons[i]][codons[j]] = {'S':None,'N':[0,2]}
                        # 2-fold second CT and AG
                        elif (gc_degen[codons[i]] == '2fAG' or gc_degen[codons[j]] == '2fAG') and (gc_degen[codons[i]] == '2fCT' or gc_degen[codons[j]] == '2fCT'):
                            aamat[codons[i]][codons[j]] = {'S':None,'N':[0,2]}
                    # mismatch in position 1 and 2
                    if codons[i][0] != codons[j][0] and codons[i][1] != codons[j][1]:
                        if codons[i][2] == codons[j][2]:
                            aamat[codons[i]][codons[j]] = {'S':None,'N':[0,1]}
                        else:
                            # 4-fold
                            if gc_degen[codons[i]] == '4f' or gc_degen[codons[j]] == '4f':
                                aamat[codons[i]][codons[j]] = {'S':[2],'N':[0,1]}
                            # 2-fold
                            elif gc_degen[codons[i]] == '2fCT' and gc_degen[codons[j]] == '2fCT':
                                aamat[codons[i]][codons[j]] = {'S':[2],'N':[0,1]}
                            elif gc_degen[codons[i]] == '2fAG' and gc_degen[codons[j]] == '2fAG':
                                aamat[codons[i]][codons[j]] = {'S':[2],'N':[0,1]}
                            elif (gc_degen[codons[i]] == '2fAG' or gc_degen[codons[j]] == '2fAG') or (gc_degen[codons[i]] == '2fCT' or gc_degen[codons[j]] == '2fCT'):
                                aamat[codons[i]][codons[j]] = {'S':None,'N':[0,1,2]}
                            # 3-fold
                            elif gc_degen[codons[i]] == '3f' or gc_degen[codons[j]] == '3f':
                                if gc_degen[codons[i]] == '0f' or gc_degen[codons[j]] == '0f':
                                    aamat[codons[i]][codons[j]] = {'S':None,'N':[0,1,2]}
                                elif (gc_degen[codons[i]] == '2fAG' or gc_degen[codons[j]] == '2fAG') or (gc_degen[codons[i]] == '2fCT' or gc_degen[codons[j]] == '2fCT'):
                                    aamat[codons[i]][codons[j]] = {'S':[2],'N':[0,1]}
    if aa:
        return gc[aa]
    elif degen:
        return gc_degen[degen]
    else:
        return aamat

if __name__ == '__main__':
    main()
