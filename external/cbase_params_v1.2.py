#!/usr/bin/env python

# ***********************************************************************
# * Cancer Bayesian Selection Estimation (CBaSE):            			*
# * Original code accompanying Weghorn & Sunyaev, Nat. Genet. (2017). 	*
# *                                                          			*
# * Author:      Donate Weghorn                              			*
# *                                                          			*
# * Copyright:   (C) 2017-2022 Donate Weghorn                			*
# *                                                          			*
# * License:     Public Domain                               			*
# *                                                          			*
# * Version:     1.2                                         			*
# ***********************************************************************


import argparse
import glob
import gzip
import itertools as it
import math
import os
import random
import subprocess
import sys

import mpmath as mp
import numpy as np
import pandas as pd
import scipy.special as sp
import scipy.stats as st
from scipy.optimize import minimize
from scipy.stats import multinomial

# ************************************ FUNCTION DEFINITIONS *************************************


def import_one_column_string(filename):
    fin = open(filename)
    lines = fin.readlines()
    fin.close()
    return [line.strip().split()[0] for line in lines]


def import_two_columns(filename):
    fin = open(filename)
    lines = fin.readlines()
    fin.close()
    return [int(line.strip().split("\t")[1]) for line in lines]


def minimize_neg_ln_L(p_start, function, mks_array, aux, bound_array, n_param):
    if n_param == 2:
        p0, p1 = p_start
        res = minimize(
            function,
            (p0, p1),
            args=(mks_array, aux),
            method="L-BFGS-B",
            bounds=bound_array,
            options={
                "disp": None,
                "gtol": 1e-12,
                "eps": 1e-5,
                "maxiter": 15000,
                "ftol": 1e-12,
            },
        )
        return [res.x[0], res.x[1], res.fun]
    elif n_param == 4:
        p0, p1, p2, p3 = p_start
        res = minimize(
            function,
            (p0, p1, p2, p3),
            args=(mks_array, aux),
            method="L-BFGS-B",
            bounds=bound_array,
            options={
                "disp": None,
                "gtol": 1e-12,
                "eps": 1e-5,
                "maxiter": 15000,
                "ftol": 1e-12,
            },
        )
        return [res.x[0], res.x[1], res.x[2], res.x[3], res.fun]
    elif n_param == 5:
        p0, p1, p2, p3, p4 = p_start
        res = minimize(
            function,
            (p0, p1, p2, p3, p4),
            args=(mks_array, aux),
            method="L-BFGS-B",
            bounds=bound_array,
            options={
                "disp": None,
                "gtol": 1e-12,
                "eps": 1e-5,
                "maxiter": 15000,
                "ftol": 1e-12,
            },
        )
        return [res.x[0], res.x[1], res.x[2], res.x[3], res.x[4], res.fun]


def neg_ln_L(p, genes, aux):
    modC = aux
    if [1, 2].count(modC):
        a, b = p
        if a < 0:
            a = 1e-6
        if b < 0:
            b = 1e-6
    elif [3, 4].count(modC):
        a, b, t, w = p
        if a < 0:
            a = 1e-6
        if b < 0:
            b = 1e-6
        if w < 0:
            w = 1e-6
        if t < 0:
            t = 1e-6
    elif [5, 6].count(modC):
        a, b, g, d, w = p
        if a < 0:
            a = 1e-6
        if b < 0:
            b = 1e-6
        if g < 0:
            g = 1e-6
        if d < 0:
            d = 1e-6
        if w < 0:
            w = 1e-6

    genes_by_sobs = [
        [ka, len(list(gr))]
        for ka, gr in it.groupby(
            sorted(genes, key=lambda arg: int(arg["obs"][2])),
            key=lambda arg: int(arg["obs"][2]),
        )
    ]

    summe = 0.0
    if modC == 1:
        for sval in genes_by_sobs:
            s = sval[0]
            # *************** lambda ~ Gamma:
            summe += sval[1] * (
                s * np.log(b)
                + (-s - a) * np.log(1 + b)
                + sp.gammaln(s + a)
                - sp.gammaln(s + 1)
                - sp.gammaln(a)
            )
    elif modC == 2:
        for sval in genes_by_sobs:
            s = sval[0]
            # *************** lambda ~ IG:
            if s > 25:
                summe += sval[1] * (
                    math.log(2.0)
                    + ((s + a) / 2.0) * math.log(b)
                    + float(mp.log(mp.besselk(-s + a, 2 * math.sqrt(b)).real))
                    - sp.gammaln(s + 1)
                    - sp.gammaln(a)
                )
            else:
                try:
                    summe += sval[1] * (
                        math.log(2.0)
                        + ((s + a) / 2.0) * math.log(b)
                        + math.log(sp.kv(-s + a, 2 * math.sqrt(b)))
                        - sp.gammaln(s + 1)
                        - sp.gammaln(a)
                    )
                except:
                    summe += sval[1] * (
                        math.log(2.0)
                        + ((s + a) / 2.0) * math.log(b)
                        + float(mp.log(mp.besselk(-s + a, 2 * math.sqrt(b)).real))
                        - sp.gammaln(s + 1)
                        - sp.gammaln(a)
                    )
    elif modC == 3:
        for sval in genes_by_sobs:
            s = sval[0]
            # *************** lambda ~ w * Exp + (1-w) * Gamma:
            summe += sval[1] * (
                np.log(
                    math.exp(math.log(w * t) + (-1 - s) * math.log(1 + t))
                    + math.exp(
                        math.log(1.0 - w)
                        + s * math.log(b)
                        + (-s - a) * math.log(1 + b)
                        + sp.gammaln(s + a)
                        - sp.gammaln(s + 1)
                        - sp.gammaln(a)
                    )
                )
            )
    elif modC == 4:
        for sval in genes_by_sobs:
            s = sval[0]
            # *************** lambda ~ w * Exp + (1-w) * InvGamma:
            if s > 25:
                summe += sval[1] * (
                    np.log(
                        math.exp(math.log(w * t) + (-1 - s) * math.log(1 + t))
                        + math.exp(
                            math.log(1.0 - w)
                            + math.log(2.0)
                            + ((s + a) / 2.0) * math.log(b)
                            + float(mp.log(mp.besselk(-s + a, 2 * math.sqrt(b)).real))
                            - sp.gammaln(s + 1)
                            - sp.gammaln(a)
                        )
                    )
                )
            else:
                try:
                    summe += sval[1] * (
                        np.log(
                            math.exp(math.log(w * t) + (-1 - s) * math.log(1 + t))
                            + math.exp(
                                math.log(1.0 - w)
                                + math.log(2.0)
                                + ((s + a) / 2.0) * math.log(b)
                                + math.log(sp.kv(-s + a, 2 * math.sqrt(b)))
                                - sp.gammaln(s + 1)
                                - sp.gammaln(a)
                            )
                        )
                    )
                except:
                    summe += sval[1] * (
                        np.log(
                            math.exp(math.log(w * t) + (-1 - s) * math.log(1 + t))
                            + math.exp(
                                math.log(1.0 - w)
                                + math.log(2.0)
                                + ((s + a) / 2.0) * math.log(b)
                                + float(
                                    mp.log(mp.besselk(-s + a, 2 * math.sqrt(b)).real)
                                )
                                - sp.gammaln(s + 1)
                                - sp.gammaln(a)
                            )
                        )
                    )

    elif modC == 5:
        for sval in genes_by_sobs:
            s = sval[0]
            # *************** lambda ~ w * Gamma + (1-w) * Gamma:
            summe += sval[1] * (
                np.log(
                    np.exp(
                        np.log(w)
                        + s * np.log(b)
                        + (-s - a) * np.log(1 + b)
                        + sp.gammaln(s + a)
                        - sp.gammaln(s + 1)
                        - sp.gammaln(a)
                    )
                    + np.exp(
                        np.log(1.0 - w)
                        + s * np.log(d)
                        + (-s - g) * np.log(1 + d)
                        + sp.gammaln(s + g)
                        - sp.gammaln(s + 1)
                        - sp.gammaln(g)
                    )
                )
            )
    elif modC == 6:
        for sval in genes_by_sobs:
            s = sval[0]
            # *************** lambda ~ w * Gamma + (1-w) * InvGamma:
            if s > 25:
                summe += sval[1] * (
                    np.log(
                        (
                            w
                            * math.exp(
                                s * math.log(b)
                                + (-s - a) * math.log(1 + b)
                                + sp.gammaln(s + a)
                                - sp.gammaln(s + 1)
                                - sp.gammaln(a)
                            )
                        )
                        + (
                            (1.0 - w)
                            * math.exp(
                                math.log(2.0)
                                + ((s + g) / 2.0) * math.log(d)
                                + float(
                                    mp.log(mp.besselk(-s + g, 2 * math.sqrt(d)).real)
                                )
                                - sp.gammaln(s + 1)
                                - sp.gammaln(g)
                            )
                        )
                    )
                )
            else:
                try:
                    summe += sval[1] * (
                        np.log(
                            (
                                w
                                * b**s
                                * (1 + b) ** (-s - a)
                                * math.exp(
                                    sp.gammaln(s + a)
                                    - sp.gammaln(s + 1)
                                    - sp.gammaln(a)
                                )
                            )
                            + (
                                (1.0 - w)
                                * math.exp(
                                    math.log(2.0)
                                    + ((s + g) / 2.0) * math.log(d)
                                    + math.log(sp.kv(-s + g, 2 * math.sqrt(d)))
                                    - sp.gammaln(s + 1)
                                    - sp.gammaln(g)
                                )
                            )
                        )
                    )
                except:
                    summe += sval[1] * (
                        np.log(
                            (
                                w
                                * math.exp(
                                    s * math.log(b)
                                    + (-s - a) * math.log(1 + b)
                                    + sp.gammaln(s + a)
                                    - sp.gammaln(s + 1)
                                    - sp.gammaln(a)
                                )
                            )
                            + (
                                (1.0 - w)
                                * math.exp(
                                    math.log(2.0)
                                    + ((s + g) / 2.0) * math.log(d)
                                    + float(
                                        mp.log(
                                            mp.besselk(-s + g, 2 * math.sqrt(d)).real
                                        )
                                    )
                                    - sp.gammaln(s + 1)
                                    - sp.gammaln(g)
                                )
                            )
                        )
                    )

    lastres = -summe
    if lastres > 1e8:
        return 0
    return -summe


# ************************************** NEW FUNCTIONS ******************************************


def import_annotation_chr(filename, reference_gene_set, chr_nr):
    # 	Contains information in direction of transcription.
    fin = gzip.open(filename, "rt")
    lines = fin.readlines()
    fin.close()
    genes = []
    gene_pos = []
    gene_info = {}
    for line in lines:
        proxy = line.strip().split("\t")
        if len(proxy) == 5 or len(proxy) == 4:
            genes.append([gene_info, gene_pos])
            gene_info = {
                "chr": chr_nr,
                "gene": proxy[0],
                "transcript": proxy[1],
                "strand": proxy[2],
                "genebegin": int(proxy[3]),
                "geneend": int(proxy[4]),
            }
            gene_pos = []
        elif proxy[0] != "pos":
            # 	Format: [pos, ref_tx, alt_tx, effect, tri_tx, penta_tx, hepta_tx]
            gene_pos.append(
                [
                    int(proxy[0]),
                    int(proxy[1]),
                    int(proxy[2]),
                    int(proxy[3]),
                    int(proxy[4]),
                    int(proxy[5]),
                    int(proxy[6]),
                ]
            )
    genes.append([gene_info, gene_pos])
    genes = [el for el in genes[1:] if reference_gene_set.count(el[0]["gene"])]
    sys.stderr.write(
        "Annotating mutations on %i genes on chromosome %i.\n" % (len(genes), chr_nr)
    )
    return sorted(genes, key=lambda arg: arg[0]["genebegin"])


def import_maf_data(filename, context_mode):
    fin = open(filename)
    lines = fin.readlines()
    fin.close()
    mut_array = []
    no_sample_info_flag = 0
    for line in lines[1:]:
        field = line.strip().split("\t")
        if len(field) < 4:
            sys.stderr.write(
                "Number of columns in maf file not as expected (>=4): %i.\n"
                % len(field)
            )
            sys.exit()
        elif len(field) < 5:
            sample_name = "jane_doe"
        else:
            sample_name = field[4]
        # context is 0-based; triplets mapped to legacy
        if ["missense", "nonsense", "coding-synon"].count(field[1]) == 0:
            continue
        if context_mode == 0:
            mut_array.append(
                {
                    "gene": field[0],
                    "effect": field[1],
                    "alt": field[2].upper(),
                    "context": triplets.index(triplets_user[int(field[3])]),
                    "sample": sample_name,
                }
            )
        else:
            mut_array.append(
                {
                    "gene": field[0],
                    "effect": field[1],
                    "alt": field[2].upper(),
                    "context": int(field[3]),
                    "sample": sample_name,
                }
            )
    return mut_array


def import_vcf_data(filename):
    if filename.split(".")[-1] == "gz":
        fin = gzip.open(filename, "rt")
    else:
        fin = open(filename)
    lines = fin.readlines()
    fin.close()
    lines = [
        line
        for line in lines
        if (line[:1] != "#" and line[:3] != "CHR") or line[:6] == "#CHROM"
    ]

    # 	Note: Split by any white space, not just tabs.
    N_columns = len(lines[0].strip().split())

    if N_columns < 5:
        sys.stderr.write(
            "Number of columns in vcf file not as expected (>=5): %i.\n" % len(field)
        )
        sys.stderr.write(
            "Expected format: [1. CHROM (integer or X or Y), 2. POS (integer, 1-based), 3. ID (not used), 4. REF (string), 5. ALT (string), 6. SAMPLE (optional)]\n"
        )
        sys.exit()

    mut_array = []
    sample_name = "john_doe"
    for line in lines:
        # 	Note: Split by any white space, not just tabs.
        field = line.strip().split("\t")
        if field[0] == "#CHROM" and N_columns >= 10:
            sample_name = field[10]
        elif field[0] == "X" or field[0] == "chrX":
            chrnr = 23
        elif field[0] == "Y" or field[0] == "chrY":
            chrnr = 24
        elif field[0].isdigit():
            chrnr = int(field[0])
        elif field[0][3:].isdigit():
            chrnr = int(field[0][3:])
        else:
            continue
        if N_columns == 6:
            print(field)
            sample_name = field[5]
        if bases.count(field[3]) and bases.count(field[4]) and field[3] != field[4]:
            mut_array.append(
                {
                    "chr": chrnr,
                    "pos": int(field[1]) - 1,
                    "ref": field[3].upper(),
                    "alt": field[4].upper(),
                    "sample": sample_name,
                }
            )  # annotation file coordinates are 0-based
    return mut_array


def write_output_for_mutation(current_mutation, current_gene_info, position_tracker):
    cm = current_mutation
    # 	Format current_gene_info[1]: [pos, ref_tx, alt_tx, effect, tri_tx, penta_tx, hepta_tx]
    # 	cur_options = [el for el in current_gene_info[1] if el[0]==cm["pos"]]
    cur_options = []
    for ind in range(position_tracker, len(current_gene_info[1])):
        if current_gene_info[1][ind][0] == cm["pos"]:
            cur_options.append(current_gene_info[1][ind])
        if len(cur_options) == 3:
            position_tracker = ind - 2
            break

    if len(cur_options) == 0:  # 	If mutation is not on exonic region.
        return 1, position_tracker
    elif len(cur_options) != 3:
        sys.stderr.write("Not three alternate states at site. --> Mutation omitted.\n")
        return {}, position_tracker
    if current_gene_info[0]["strand"] == "+" and cm["ref"] != bases[cur_options[0][1]]:
        sys.stderr.write("Mismatching reference nucleotide. --> Mutation omitted.\n")
        return {}, position_tracker
    elif (
        current_gene_info[0]["strand"] == "-"
        and cm["ref"] != bases_inv[cur_options[0][1]]
    ):
        sys.stderr.write("Mismatching reference nucleotide. --> Mutation omitted.\n")
        return {}, position_tracker
    else:
        if current_gene_info[0]["strand"] == "+":
            cur_effect_ind = [
                ind
                for ind in range(len(cur_options))
                if bases[cur_options[ind][2]] == cm["alt"]
            ][0]
        else:
            cur_effect_ind = [
                ind
                for ind in range(len(cur_options))
                if cur_options[ind][2] == bases_inv.index(cm["alt"])
            ][0]
        cur_effect = cur_options[cur_effect_ind]
        # 	Annotate mutations already in direction of transcription.
        return {
            "chr": current_gene_info[0]["chr"],
            "gene": current_gene_info[0]["gene"],
            "effect": ["missense", "nonsense", "coding-synon"][cur_effect[3]],
            "alt": bases[cur_effect[2]],
            "context": [cur_effect[4], cur_effect[5], cur_effect[6]][CMODE],
            "pos_in_annotation": position_tracker + cur_effect_ind,
            "sample": cm["sample"],
        }, position_tracker


def generate_CBaSE_ready_input(mutation_array, annotation_infile, reference_gene_set):

    muts_by_chr = [
        list(g)
        for k, g in it.groupby(
            sorted(mutation_array, key=lambda arg: arg["chr"]),
            key=lambda arg: arg["chr"],
        )
    ]

    output_for_CBaSE = []
    between_genes = 0
    tot_mut_cnt = 0

    for chrnr in range(1, 25):
        info_by_gene = import_annotation_chr(
            annotation_infile % (REFERENCE_DIR, BUILD, chrnr), reference_gene_set, chrnr
        )  # 0-based, begin and end inclusive

        cur_chr_muts = [el for el in muts_by_chr if el[0]["chr"] == chrnr]
        if len(cur_chr_muts):
            cur_chr_muts = sorted(cur_chr_muts[0], key=lambda arg: arg["pos"])
            tot_mut_cnt += len(cur_chr_muts)
        else:
            continue

        i = 0
        i_tracker = 20000
        for mut in cur_chr_muts:

            if mut["pos"] < info_by_gene[i][0]["genebegin"]:
                between_genes += 1
                continue

            while (
                i < len(info_by_gene) and mut["pos"] >= info_by_gene[i][0]["genebegin"]
            ):  # Finds the gene with *largest* "genebegin" that could harbor the mutation, hence need to backtrack.
                i += 1
            i -= 1

            if mut["pos"] <= info_by_gene[i][0]["geneend"]:  # 	Found the gene.

                if i == i_tracker:
                    res, pos_tracker = write_output_for_mutation(
                        mut, info_by_gene[i], pos_tracker
                    )
                else:
                    res, pos_tracker = write_output_for_mutation(
                        mut, info_by_gene[i], 0
                    )
                    i_tracker = i

                if res == 1:
                    pass
                elif len(res):
                    output_for_CBaSE.append(res)

                j = i - 1
                while (
                    j >= 0
                    and info_by_gene[j][0]["geneend"]
                    >= mut["pos"]
                    >= info_by_gene[j][0]["genebegin"]
                ):  # 	Mutation on another, overlapping gene.
                    res, dummy = write_output_for_mutation(mut, info_by_gene[j], 0)
                    if res == 1:
                        pass
                    elif len(res):
                        output_for_CBaSE.append(res)
                    j -= 1

            else:  # 	Mutation might be on previous gene, which embeds current gene, or...
                j = i - 1
                flag = 0
                while (
                    j >= 0
                    and info_by_gene[j][0]["geneend"]
                    >= mut["pos"]
                    >= info_by_gene[j][0]["genebegin"]
                ):  # 	Mutation on another, overlapping gene.

                    flag = 1
                    res, dummy = write_output_for_mutation(mut, info_by_gene[j], 0)
                    if res == 1:
                        pass
                    elif len(res):
                        output_for_CBaSE.append(res)
                    j -= 1

                if flag == 0:  # 	...between genes.
                    between_genes += 1

    sys.stderr.write("%i mutations between genes.\n" % between_genes)
    sys.stderr.write(
        "%i out of %i mutations used in analysis.\n"
        % (len(output_for_CBaSE), tot_mut_cnt)
    )
    return output_for_CBaSE


def import_effects_by_gene(filename):
    fin = gzip.open(filename, "rt")
    lines = fin.readlines()
    fin.close()
    genes = []
    gene_target = [
        [[0 for eff in range(3)] for alt in range(4)] for cont in range(len(used_plets))
    ]
    gene_info = {}

    cnt = 0
    for line in lines:
        sys.stderr.write(
            "%i%% of importing effects by gene done.\r" % (100.0 * cnt / len(lines))
        )
        cnt += 1
        proxy = line.strip().split("\t")
        if len(proxy) > 4:
            genes.append([gene_info, gene_target])
            gene_info = {
                "chr": int(proxy[0]),
                "gene": proxy[1],
                "transcript": proxy[2],
                "strand": proxy[3],
                "genebegin": int(proxy[4]),
                "geneend": int(proxy[5]),
            }
            gene_target = [
                [[0 for eff in range(3)] for alt in range(4)]
                for cont in range(len(used_plets))
            ]
        else:
            # 	Format: gene_target[cont][alt][effect] = count
            gene_target[int(proxy[0])][int(proxy[1])][int(proxy[2])] = int(proxy[3])
    sys.stderr.write("100% of importing effects by gene done.\n")
    genes.append([gene_info, gene_target])
    return sorted(genes[1:], key=lambda arg: arg[0]["gene"])


# def import_effects_by_gene(filename):
#     with gzip.open(filename, "rt") as fin:
#         lines = fin.readlines()
#     fin.close()
#     genes = []
#     gene_target = [[[] for alt in range(4)] for cont in range(len(used_plets))]
#     gene_info = {}
#     for line in lines:
#         proxy = line.strip().split("\t")
#         if len(proxy) > 3:
#             genes.append([gene_info, gene_target])
#             gene_info = {
#                 "chr": int(proxy[0]),
#                 "gene": proxy[1],
#                 "transcript": proxy[2],
#                 "strand": proxy[3],
#                 "genebegin": int(proxy[4]),
#                 "geneend": int(proxy[5]),
#             }
#             gene_target = [[[] for alt in range(4)] for cont in range(len(used_plets))]
#             line_cnt = 0
#         else:
#             # 	Format: [context_ind, alternate_ind, effect_ind]
#             gene_target[int(line_cnt / 4)][line_cnt % 4] = [
#                 int(proxy[0]),
#                 int(proxy[1]),
#                 int(proxy[2]),
#             ]
#             line_cnt += 1
#     genes.append([gene_info, gene_target])
#     return sorted(genes[1:], key=lambda arg: arg[0]["gene"])


def export_expected_observed_mks_per_gene(
    mut_array, kmer_abundances, effect_array, outfile
):
    # 	Create the mutation matrix:
    neutral_mutations = [
        mut
        for mut in mut_array
        if ["missense", "nonsense", "coding-synon"].count(mut["effect"])
    ]
    sys.stderr.write(
        "Total number of coding mutations used for generation of mutation matrix: %i.\n"
        % len(neutral_mutations)
    )
    neutral_muts_by_context = [
        [mut["context"], bases.index(mut["alt"])] for mut in neutral_mutations
    ]

    sum_kmers = sum([el for el in kmer_abundances])
    neutral_mut_matrix = [[0.0 for i in range(4)] for j in range(len(kmer_abundances))]

    for el in neutral_muts_by_context:
        neutral_mut_matrix[el[0]][el[1]] += (
            1.0 / kmer_abundances[el[0]] * sum_kmers / (len(neutral_mutations))
        )

    fout_mat = open(outfile, "w")
    for i in range(len(neutral_mut_matrix)):
        fout_mat.write("%s" % (used_plets[i]))
        for j in range(len(neutral_mut_matrix[i])):
            fout_mat.write("\t%f" % (neutral_mut_matrix[i][j]))
        fout_mat.write("\n")
    fout_mat.close()

    # 	Derive expected and observed counts per gene:
    muts_by_gene = [
        list(g)
        for k, g in it.groupby(
            sorted(mut_array, key=lambda arg: arg["gene"]), key=lambda arg: arg["gene"]
        )
    ]
    sys.stderr.write("Mutations distributed across %i genes.\n" % len(muts_by_gene))

    # 	NOTE: effect_array is sorted by "gene".
    exp_obs_per_gene = []
    for gene in effect_array:

        # 	Compute observed count:
        x_obs = [0 for i in range(3)]
        cur_gene_muts = [el for el in muts_by_gene if el[0]["gene"] == gene[0]["gene"]]
        if len(cur_gene_muts):
            for mut in cur_gene_muts[0]:
                x_obs[
                    ["missense", "nonsense", "coding-synon"].index(mut["effect"])
                ] += 1

        # 	Compute neutral expectation (up to a constant factor):
        x_exp = [0.0 for t in range(3)]
        gene_len = 0
        for cont in range(len(used_plets)):
            for alt in range(4):
                for eff in range(3):
                    x_exp[eff] += (
                        gene[1][cont][alt][eff] * neutral_mut_matrix[cont][alt]
                    )
                    gene_len += gene[1][cont][alt][eff]

        exp_obs_per_gene.append(
            [
                gene[0]["gene"],
                x_exp[0],
                x_exp[1],
                x_exp[2],
                x_obs[0],
                x_obs[1],
                x_obs[2],
                gene_len / 3,
            ]
        )
    return exp_obs_per_gene, neutral_mut_matrix


def lambda_hat_given_s(p, s, modC, thr):
    # 	This is the integral int_0^infty lambda Pois(s;lambda) P(lambda;theta) dlambda.
    if [1, 2].count(modC):
        a, b = p
    elif [3, 4].count(modC):
        a, b, t, w = p
    elif [5, 6].count(modC):
        a, b, g, d, w = p

    L = 1
    if modC == 1:
        # *************** lambda ~ Gamma:
        def numerator(s):
            return np.exp(
                (1.0 + s) * np.log(b)
                + (-1.0 - s - a) * np.log(1 + b)
                + sp.gammaln(1.0 + s + a)
                - sp.gammaln(s + 1)
                - sp.gammaln(a)
            )

        def pofs(s, L, thr):
            return np.exp(
                s * np.log(L * b)
                + (-s - a) * np.log(1 + L * b)
                + sp.gammaln(s + a)
                - sp.gammaln(s + 1)
                - sp.gammaln(a)
            )

    elif modC == 2:
        # *************** lambda ~ IG:
        def numerator(s):
            if thr:
                return mp.exp(
                    math.log(2.0)
                    + (0.5 * (1.0 + s + a)) * math.log(b)
                    + mp.log(mp.besselk(1.0 + s - a, 2.0 * math.sqrt(b)))
                    - sp.gammaln(s + 1)
                    - sp.gammaln(a)
                )
            else:
                return np.exp(
                    math.log(2.0)
                    + (0.5 * (1.0 + s + a)) * math.log(b)
                    + np.log(sp.kv(1.0 + s - a, 2.0 * math.sqrt(b)))
                    - sp.gammaln(s + 1)
                    - sp.gammaln(a)
                )

        def pofs(s, L, thr):
            if thr:
                return 2.0 * mp.exp(
                    ((s + a) / 2.0) * math.log(L * b)
                    + mp.log(mp.besselk(-s + a, 2 * np.sqrt(L * b)))
                    - sp.gammaln(s + 1)
                    - sp.gammaln(a)
                )
            else:
                return 2.0 * math.exp(
                    ((s + a) / 2.0) * math.log(L * b)
                    + np.log(sp.kv(-s + a, 2 * math.sqrt(L * b)))
                    - sp.gammaln(s + 1)
                    - sp.gammaln(a)
                )

    elif modC == 3:
        # *************** lambda ~ w * Exp + (1-w) * Gamma:
        def numerator(s):
            return w * ((1.0 + s) * t * (1 + t) ** (-2.0 - s)) + (1 - w) * np.exp(
                (1.0 + s) * np.log(b)
                + (-1.0 - s - a) * np.log(1 + b)
                + sp.gammaln(1.0 + s + a)
                - sp.gammaln(s + 1)
                - sp.gammaln(a)
            )

        def pofs(s, L, thr):
            return np.exp(
                np.log(w) + s * np.log(L) + np.log(t) + (-1 - s) * np.log(L + t)
            ) + np.exp(
                np.log(1.0 - w)
                + s * np.log(L * b)
                + (-s - a) * np.log(1 + L * b)
                + sp.gammaln(s + a)
                - sp.gammaln(s + 1)
                - sp.gammaln(a)
            )

    elif modC == 4:
        # *************** lambda ~ w * Exp + (1-w) * InvGamma:
        def numerator(s):
            if thr:
                return w * ((1.0 + s) * t * (1 + t) ** (-2.0 - s)) + (1 - w) * mp.exp(
                    math.log(2.0)
                    + (0.5 * (1.0 + s + a)) * math.log(b)
                    + mp.log(mp.besselk(1.0 + s - a, 2.0 * math.sqrt(b)))
                    - sp.gammaln(s + 1)
                    - sp.gammaln(a)
                )
            else:
                return w * ((1.0 + s) * t * (1 + t) ** (-2.0 - s)) + (1 - w) * np.exp(
                    math.log(2.0)
                    + (0.5 * (1.0 + s + a)) * math.log(b)
                    + np.log(sp.kv(1.0 + s - a, 2.0 * math.sqrt(b)))
                    - sp.gammaln(s + 1)
                    - sp.gammaln(a)
                )

        def pofs(s, L, thr):
            if thr:
                return (
                    w * t * mp.exp(s * np.log(L) + (-1 - s) * np.log(L + t))
                ) + mp.exp(
                    np.log(1.0 - w)
                    + np.log(2.0)
                    + ((s + a) / 2.0) * np.log(L * b)
                    + mp.log(mp.besselk(-s + a, 2 * math.sqrt(L * b)))
                    - sp.gammaln(s + 1)
                    - sp.gammaln(a)
                )
            else:
                return (w * L**s * t * (L + t) ** (-1 - s)) + np.exp(
                    np.log(1.0 - w)
                    + np.log(2.0)
                    + ((s + a) / 2.0) * np.log(L * b)
                    + np.log(sp.kv(-s + a, 2 * math.sqrt(L * b)))
                    - sp.gammaln(s + 1)
                    - sp.gammaln(a)
                )

    elif modC == 5:
        # *************** lambda ~ w * Gamma + (1-w) * Gamma (Gamma mixture model):
        def numerator(s):
            return w * np.exp(
                (1.0 + s) * np.log(b)
                + (-1.0 - s - a) * np.log(1 + b)
                + sp.gammaln(1.0 + s + a)
                - sp.gammaln(s + 1)
                - sp.gammaln(a)
            ) + (1 - w) * np.exp(
                (1.0 + s) * np.log(d)
                + (-1.0 - s - g) * np.log(1 + d)
                + sp.gammaln(1.0 + s + g)
                - sp.gammaln(s + 1)
                - sp.gammaln(g)
            )

        def pofs(s, L, thr):
            return np.exp(
                np.log(w)
                + s * np.log(L * b)
                + (-s - a) * np.log(1 + L * b)
                + sp.gammaln(s + a)
                - sp.gammaln(s + 1)
                - sp.gammaln(a)
            ) + np.exp(
                np.log(1.0 - w)
                + s * np.log(L * d)
                + (-s - g) * np.log(1 + L * d)
                + sp.gammaln(s + g)
                - sp.gammaln(s + 1)
                - sp.gammaln(g)
            )

    elif modC == 6:
        # *************** lambda ~ w * Gamma + (1-w) * InvGamma (mixture model):
        def numerator(s):
            if thr:
                return w * np.exp(
                    (1.0 + s) * np.log(b)
                    + (-1.0 - s - a) * np.log(1 + b)
                    + sp.gammaln(1.0 + s + a)
                    - sp.gammaln(s + 1)
                    - sp.gammaln(a)
                ) + (1 - w) * mp.exp(
                    math.log(2.0)
                    + (0.5 * (1.0 + s + g)) * math.log(d)
                    + mp.log(mp.besselk(1.0 + s - g, 2.0 * math.sqrt(d)))
                    - sp.gammaln(s + 1)
                    - sp.gammaln(g)
                )
            else:
                return w * np.exp(
                    (1.0 + s) * np.log(b)
                    + (-1.0 - s - a) * np.log(1 + b)
                    + sp.gammaln(1.0 + s + a)
                    - sp.gammaln(s + 1)
                    - sp.gammaln(a)
                ) + (1 - w) * np.exp(
                    math.log(2.0)
                    + (0.5 * (1.0 + s + g)) * math.log(d)
                    + np.log(sp.kv(1.0 + s - g, 2.0 * math.sqrt(d)))
                    - sp.gammaln(s + 1)
                    - sp.gammaln(g)
                )

        def pofs(s, L, thr):
            if thr:
                return np.exp(
                    np.log(w)
                    + s * np.log(L * b)
                    + (-s - a) * np.log(1 + L * b)
                    + sp.gammaln(s + a)
                    - sp.gammaln(s + 1)
                    - sp.gammaln(a)
                ) + mp.exp(
                    np.log(1.0 - w)
                    + np.log(2.0)
                    + ((s + g) / 2.0) * np.log(L * d)
                    + mp.log(mp.besselk(-s + g, 2 * mp.sqrt(L * d)))
                    - sp.gammaln(s + 1)
                    - sp.gammaln(g)
                )
            else:
                return np.exp(
                    np.log(w)
                    + s * np.log(L * b)
                    + (-s - a) * np.log(1 + L * b)
                    + sp.gammaln(s + a)
                    - sp.gammaln(s + 1)
                    - sp.gammaln(a)
                ) + np.exp(
                    np.log(1.0 - w)
                    + np.log(2.0)
                    + ((s + g) / 2.0) * np.log(L * d)
                    + np.log(sp.kv(-s + g, 2 * np.sqrt(L * d)))
                    - sp.gammaln(s + 1)
                    - sp.gammaln(g)
                )

    return numerator(s) / pofs(s, L, thr)


def import_plet_arrays(infile_tri):
    tri_array = [
        "AAA",
        "AAC",
        "AAG",
        "AAT",
        "CAA",
        "CAC",
        "CAG",
        "CAT",
        "GAA",
        "GAC",
        "GAG",
        "GAT",
        "TAA",
        "TAC",
        "TAG",
        "TAT",
        "ACA",
        "ACC",
        "ACG",
        "ACT",
        "CCA",
        "CCC",
        "CCG",
        "CCT",
        "GCA",
        "GCC",
        "GCG",
        "GCT",
        "TCA",
        "TCC",
        "TCG",
        "TCT",
        "AGA",
        "AGC",
        "AGG",
        "AGT",
        "CGA",
        "CGC",
        "CGG",
        "CGT",
        "GGA",
        "GGC",
        "GGG",
        "GGT",
        "TGA",
        "TGC",
        "TGG",
        "TGT",
        "ATA",
        "ATC",
        "ATG",
        "ATT",
        "CTA",
        "CTC",
        "CTG",
        "CTT",
        "GTA",
        "GTC",
        "GTG",
        "GTT",
        "TTA",
        "TTC",
        "TTG",
        "TTT",
    ]
    tri_array_user = import_one_column_string(
        infile_tri
    )  # 	NOTE: "triplets_user" is the only k-mer array that is different between the required user input and what is used internally. Indices provided by the user are internally converted to indices in list "triplets".
    hepta_array = []
    nona_array = []
    quintu_array = []
    for i in bases:
        for j in bases:
            for k in bases:
                for l in bases:
                    for m in bases:
                        quintu_array.append("".join([i, j, k, l, m]))
                        for n in bases:
                            for o in bases:
                                hepta_array.append("".join([i, j, k, l, m, n, o]))
                                for p in bases:
                                    for q in bases:
                                        nona_array.append(
                                            "".join([i, j, k, l, m, n, o, p, q])
                                        )
    return tri_array_user, tri_array, quintu_array, hepta_array, nona_array


def check_user_input(VCF, BUILD, CMODE, MODEL):
    flag = 0
    if VCF != 0 and VCF != 1:
        sys.stderr.write("VCF parameter not allowed (should be 0 or 1).\n")
        flag = 1
    if VCF:
        sys.stderr.write(
            "Use input columns (corresponding to chosen genome build):\nchromosome\tposition (1-based)\tID\treference_allele\talternate_allele\n"
        )
        if BUILD != "hg19" and BUILD != "hg38":
            sys.stderr.write("BUILD parameter not allowed (should be hg19 or hg38).\n")
            flag = 1
    else:
        sys.stderr.write("Use input columns:\ngene\teffect\talt\tcontext\n")
    if [0, 1, 2].count(CMODE) == 0:
        sys.stderr.write("CMODE parameter not allowed (should be one of [3, 5, 7]).\n")
        flag = 1
    if range(7).count(MODEL) == 0:
        sys.stderr.write("MODEL parameter not allowed (should be between 0 and 6).\n")
        flag = 1
    if flag:
        sys.stderr.write("\n***** ABORTING CBaSE *****\n\n")
        sys.exit()


# ************************************************************************************************

# ************************************** COMMAND LINE ARGS ***************************************

INFILE = str(sys.argv[1])  # 	somatic mutation data input file
VCF = int(
    sys.argv[2]
)  # 	1=input format is vcf, 0=input format is CBaSE v1.0 input format
BUILD = str(sys.argv[3])  # 	one of [hg19, hg38] (not used when VCF=0)
CMODE = int(
    (int(sys.argv[4]) - 1) / 2 - 1
)  # 	3=trinucleotides, 5=pentanucleotides, 7=heptanucleotides
MODEL = int(
    sys.argv[5]
)  # 	model choice: 0=all, 1=G, 2=IG, 3=EmixG, 4=EmixIG, 5=GmixG, 6=GmixIG
OUTNAME = str(sys.argv[6])  # 	name for the output file containing the q-values
REFERENCE_DIR = str(sys.argv[7])  #   path to the reference files folder
TEMP_DIR = str(sys.argv[8])  #   path to the temp files folder

# ***************************** GLOBAL DEFINITIONS & AUXILIARY DATA ******************************

check_user_input(VCF, BUILD, CMODE, MODEL)
sys.stderr.write(
    "User input provided:\nINFILE\t%s\nVCF\t%i\nBUILD\t%s\nCMODE\t%i\nMODEL\t%i\nOUTNAME\t%s\n"
    % (INFILE.split("/")[-1], VCF, BUILD, int(sys.argv[4]), MODEL, OUTNAME)
)

bases = ["A", "C", "G", "T"]
bases_inv = ["T", "G", "C", "A"]
triplets_user, triplets, quintuplets, heptaplets, nonaplets = import_plet_arrays(
    "%s/triplets_user.txt" % REFERENCE_DIR
)  # 	NOTE: "triplets_user" is the only k-mer array that is different between the required user input and what is used internally. Indices provided by the user (from file "triplets_user") are internally converted to indices in new list "triplets".
used_plets = [triplets, quintuplets, heptaplets, nonaplets][CMODE]
used_pname = [
    "trinucleotides",
    "pentanucleotides",
    "heptanucleotides",
    "nonanucleotides",
][CMODE]

mod_choice_short = ["", "G", "IG", "EmixG", "EmixIG", "GmixG", "GmixIG"]  # model choice
mod_choice = [
    "",
    "Gamma(a,b): [a,b] =",
    "InverseGamma(a,b): [a,b] =",
    "w Exp(t) + (1-w) Gamma(a,b): [a,b,t,w] =",
    "w Exp(t) + (1-w) InverseGamma(a,b): [a,b,t,w] =",
    "w Gamma(a,b) + (1-w) Gamma(g,d): [a,b,g,d,w] =",
    "w Gamma(a,b) + (1-w) InverseGamma(g,d): [a,b,g,d,w] =",
]  # model choice
modC_map = [2, 2, 4, 4, 5, 5]  # 	map model --> number of params
rep_no = 25  # 	No. of independent runs to estimate model parameters (maximizing log-likelihood)

cancer_genes = import_one_column_string("%s/COSMIC_genes_v80.txt" % REFERENCE_DIR)
used_genes = import_one_column_string(
    "%s/used_genes_new_CBaSE.txt" % REFERENCE_DIR
)  # 	Created in create_aux_info.py; excludes OR genes and artifacts from SI of Martincorena (2017).
abundances = import_two_columns(
    "%s/abundances_%s_tx.txt" % (REFERENCE_DIR, used_pname)
)  # 	Were computed for the set used_genes in create_aux_info.py
# 	Format: [context_ind, alternate_ind, effect_ind]
effects_by_gene = import_effects_by_gene(
    "%s/context_alt_effect_by_gene_%s_%s.txt.gz" % (REFERENCE_DIR, BUILD, used_pname)
)  # 	Were computed for the set used_genes in create_aux_info.py
# 	effects: [0: "missense", 1: "nonsense", 2: "coding-synon"]
sys.stderr.write("Derive selection predictions for %i genes.\n" % len(used_genes))

# ************************************************************************************************************
# ************************************************************************************************************

sys.stderr.write("Running data preparation.\n")

# 	(1) Import mutations.
if (
    VCF
):  # 	Import mutation file (vcf) with format: [1. CHROM, 2. POS (1-based), 3. ID (not used), 4. REF, 5. ALT, optional: 6. or 10. SAMPLE_ID)] for given genome build.
    mutations_vcf = import_vcf_data(INFILE)
    mutations = generate_CBaSE_ready_input(
        mutations_vcf, "%s/gene_annotations_%s/chr%s.txt.gz", used_genes
    )
    mutations_df = pd.DataFrame(mutations)
    mutations_df.to_csv(
        "{}/{}_kept_mutations.csv".format(TEMP_DIR, OUTNAME), index=False, sep="\t"
    )
else:  # 	Import mutation file (original CBaSE) with format: [gene, effect, alt, context]
    mutations = import_maf_data(INFILE, CMODE)
    sys.stderr.write("%i SNVs imported for file %s.\n" % (len(mutations), OUTNAME))

lengths = [
    [k, len(list(g))]
    for k, g in it.groupby(
        sorted(mutations, key=lambda arg: arg["effect"]), key=lambda arg: arg["effect"]
    )
]
for el in lengths:
    sys.stderr.write("%s\t%i\n" % (el[0], el[1]))

muts_by_sample = [
    [k, list(g)]
    for k, g in it.groupby(
        sorted(mutations, key=lambda arg: arg["sample"]), key=lambda arg: arg["sample"]
    )
]
for el in muts_by_sample:
    sys.stderr.write("Sample %s: %i mutations.\n" % (el[0], len(el[1])))
N_samples = len(muts_by_sample)

# 	(2) Derive target size and observed mutation counts for all three categories per gene.
res, mut_matrix = export_expected_observed_mks_per_gene(
    mutations,
    abundances,
    effects_by_gene,
    "%s/mutation_mat_%s.txt" % (TEMP_DIR, OUTNAME),
)
sys.stderr.write("Finished data preparation.\n")

# ************************************************************************************************************

sys.stderr.write("Running parameter estimation.\n")

mks_type = []
for gene in res:
    mks_type.append(
        {
            "gene": gene[0],
            "exp": [float(gene[1]), float(gene[2]), float(gene[3])],
            "obs": [float(gene[4]), float(gene[5]), float(gene[6])],
            "len": int(gene[7]),
        }
    )
mks_type = sorted(mks_type, key=lambda arg: arg["gene"])

sys.stderr.write("Running ML routine %i times.\n" % rep_no)
if MODEL > 6:
    sys.stderr.write("Not a valid model choice.\n")
    sys.exit()
elif MODEL == 0:
    sys.stderr.write("Testing all six models.\n")
else:
    sys.stderr.write("lam_s ~ %s.\n" % mod_choice_short[MODEL])
sys.stderr.write("%i genes in total.\n" % len(mks_type))

if MODEL == 1 or MODEL == 0:
    sys.stderr.write("Fitting model 1...\n")
    low_b = [1e-5 * random.uniform(1.0, 3.0) for i in range(2)]
    up_b = [50.0 * random.uniform(1.0, 2.0) for i in range(2)]
    cur_min_res = [0, 0, 1e20]
    for rep in range(rep_no):
        sys.stderr.write("%.f%% done.\r" % (100.0 * rep / rep_no))
        p_res = minimize_neg_ln_L(
            [random.uniform(0.02, 10.0), random.uniform(0.02, 10.0)],
            neg_ln_L,
            mks_type,
            1,
            [(low_b[0], up_b[0]), (low_b[1], up_b[1])],
            2,
        )
        if p_res[2] > 0 and p_res[2] < cur_min_res[2]:
            cur_min_res = p_res[:]
    if cur_min_res[2] == 1e20:
        sys.stderr.write("Could not find a converging solution for model 1.\n")
    fout = open("%s/param_estimates_%s_1.txt" % (TEMP_DIR, OUTNAME), "w")
    fout.write("%e, %e, %f, %i\n" % (cur_min_res[0], cur_min_res[1], cur_min_res[2], 1))
    fout.close()

if MODEL == 2 or MODEL == 0:
    sys.stderr.write("Fitting model 2...\n")
    low_b = [1e-5 * random.uniform(1.0, 3.0) for i in range(2)]
    up_b = [50.0 * random.uniform(1.0, 2.0) for i in range(2)]
    cur_min_res = [0, 0, 1e20]
    for rep in range(rep_no):
        sys.stderr.write("%.f%% done.\r" % (100.0 * rep / rep_no))
        p_res = minimize_neg_ln_L(
            [random.uniform(0.02, 10.0), random.uniform(0.02, 10.0)],
            neg_ln_L,
            mks_type,
            2,
            [(low_b[0], up_b[0]), (low_b[1], up_b[1])],
            2,
        )
        if p_res[2] > 0 and p_res[2] < cur_min_res[2]:
            cur_min_res = p_res[:]
    if cur_min_res[2] == 1e20:
        sys.stderr.write("Could not find a converging solution for model 2.\n")
    fout = open("%s/param_estimates_%s_2.txt" % (TEMP_DIR, OUTNAME), "w")
    fout.write("%e, %e, %f, %i\n" % (cur_min_res[0], cur_min_res[1], cur_min_res[2], 2))
    fout.close()

if MODEL == 3 or MODEL == 0:
    sys.stderr.write("Fitting model 3...\n")
    low_b = [1e-5 * random.uniform(1.0, 3.0) for i in range(4)]
    up_b = [50.0 * random.uniform(1.0, 2.0) for i in range(4)]
    cur_min_res = [0, 0, 0, 0, 1e20]
    for rep in range(rep_no):
        sys.stderr.write("%.f%% done.\r" % (100.0 * rep / rep_no))
        p_res = minimize_neg_ln_L(
            [
                random.uniform(0.02, 10.0),
                random.uniform(0.02, 10.0),
                random.uniform(0.02, 10.0),
                random.uniform(2e-5, 0.95),
            ],
            neg_ln_L,
            mks_type,
            3,
            [
                (low_b[0], up_b[0]),
                (low_b[1], up_b[1]),
                (low_b[2], up_b[2]),
                (low_b[3], 0.9999),
            ],
            4,
        )
        if p_res[4] > 0 and p_res[4] < cur_min_res[4]:
            cur_min_res = p_res[:]
    if cur_min_res[4] == 1e20:
        sys.stderr.write("Could not find a converging solution for model 3.\n")
    fout = open("%s/param_estimates_%s_3.txt" % (TEMP_DIR, OUTNAME), "w")
    fout.write(
        "%e, %e, %e, %e, %f, %i\n"
        % (
            cur_min_res[0],
            cur_min_res[1],
            cur_min_res[2],
            cur_min_res[3],
            cur_min_res[4],
            3,
        )
    )
    fout.close()

if MODEL == 4 or MODEL == 0:
    sys.stderr.write("Fitting model 4...\n")
    low_b = [1e-5 * random.uniform(1.0, 3.0) for i in range(4)]
    up_b = [50.0 * random.uniform(1.0, 2.0) for i in range(4)]
    cur_min_res = [0, 0, 0, 0, 1e20]
    for rep in range(rep_no):
        sys.stderr.write("%.f%% done.\r" % (100.0 * rep / rep_no))
        p_res = minimize_neg_ln_L(
            [
                random.uniform(0.02, 10.0),
                random.uniform(0.02, 10.0),
                random.uniform(0.02, 10.0),
                random.uniform(2e-5, 0.95),
            ],
            neg_ln_L,
            mks_type,
            4,
            [
                (low_b[0], up_b[0]),
                (low_b[1], up_b[1]),
                (low_b[2], up_b[2]),
                (low_b[3], 0.9999),
            ],
            4,
        )
        if p_res[4] > 0 and p_res[4] < cur_min_res[4]:
            cur_min_res = p_res[:]
    if cur_min_res[4] == 1e20:
        sys.stderr.write("Could not find a converging solution for model 4.\n")
    fout = open("%s/param_estimates_%s_4.txt" % (TEMP_DIR, OUTNAME), "w")
    fout.write(
        "%e, %e, %e, %e, %f, %i\n"
        % (
            cur_min_res[0],
            cur_min_res[1],
            cur_min_res[2],
            cur_min_res[3],
            cur_min_res[4],
            4,
        )
    )
    fout.close()

if MODEL == 5 or MODEL == 0:
    sys.stderr.write("Fitting model 5...\n")
    low_b = [1e-5 * random.uniform(1.0, 3.0) for i in range(5)]
    up_b = [50.0 * random.uniform(1.0, 2.0) for i in range(5)]
    cur_min_res = [0, 0, 0, 0, 0, 1e20]
    for rep in range(int(rep_no)):
        sys.stderr.write("%.f%% done.\r" % (100.0 * rep / rep_no))
        p_res = minimize_neg_ln_L(
            [
                random.uniform(0.02, 10.0),
                random.uniform(0.02, 5.0),
                random.uniform(0.02, 10.0),
                random.uniform(0.02, 10.0),
                random.uniform(2e-5, 0.95),
            ],
            neg_ln_L,
            mks_type,
            5,
            [
                (low_b[0], up_b[0]),
                (low_b[1], up_b[1]),
                (low_b[2], up_b[2]),
                (low_b[3], up_b[3]),
                (low_b[4], 0.9999),
            ],
            5,
        )
        if p_res[5] > 0 and p_res[5] < cur_min_res[5]:
            cur_min_res = p_res[:]
    if cur_min_res[5] == 1e20:
        sys.stderr.write("Could not find a converging solution for model 5.\n")
    fout = open("%s/param_estimates_%s_5.txt" % (TEMP_DIR, OUTNAME), "w")
    fout.write(
        "%e, %e, %e, %e, %e, %f, %i\n"
        % (
            cur_min_res[0],
            cur_min_res[1],
            cur_min_res[2],
            cur_min_res[3],
            cur_min_res[4],
            cur_min_res[5],
            5,
        )
    )
    fout.close()

if MODEL == 6 or MODEL == 0:
    sys.stderr.write("Fitting model 6...\n")
    low_b = [1e-5 * random.uniform(1.0, 3.0) for i in range(5)]
    up_b = [50.0 * random.uniform(1.0, 2.0) for i in range(5)]
    cur_min_res = [0, 0, 0, 0, 0, 1e20]
    for rep in range(int(2 * rep_no)):
        sys.stderr.write("%.f%% done.\r" % (100.0 * rep / (2 * rep_no)))
        p_res = minimize_neg_ln_L(
            [
                random.uniform(0.02, 10.0),
                random.uniform(0.02, 5.0),
                random.uniform(0.02, 10.0),
                random.uniform(0.02, 10.0),
                random.uniform(2e-5, 0.95),
            ],
            neg_ln_L,
            mks_type,
            6,
            [
                (low_b[0], up_b[0]),
                (low_b[1], up_b[1]),
                (low_b[2], up_b[2]),
                (low_b[3], up_b[3]),
                (low_b[4], 0.9999),
            ],
            5,
        )
        if p_res[5] > 0 and p_res[5] < cur_min_res[5]:
            cur_min_res = p_res[:]
    if cur_min_res[5] == 1e20:
        sys.stderr.write("Could not find a converging solution for model 6.\n")
    fout = open("%s/param_estimates_%s_6.txt" % (TEMP_DIR, OUTNAME), "w")
    fout.write(
        "%e, %e, %e, %e, %e, %f, %i\n"
        % (
            cur_min_res[0],
            cur_min_res[1],
            cur_min_res[2],
            cur_min_res[3],
            cur_min_res[4],
            cur_min_res[5],
            6,
        )
    )
    fout.close()

# ************************************************************************************************************

p_files = glob.glob("%s/param_estimates_%s_*.txt" % (TEMP_DIR, OUTNAME))

all_models = []
for p_file in p_files:
    fin = open(p_file)
    lines = fin.readlines()
    fin.close()
    field = lines[0].strip().split(", ")
    all_models.append([float(el) for el in field])

cur_min = 1e20
cur_ind = 10
for m in range(len(all_models)):
    if 2.0 * modC_map[int(all_models[m][-1]) - 1] + 2.0 * all_models[m][-2] < cur_min:
        cur_min = 2.0 * modC_map[int(all_models[m][-1]) - 1] + 2.0 * all_models[m][-2]
        cur_ind = m
if cur_min < 1e20:
    # 	Export parameters and index of chosen model.
    sys.stderr.write("Best model fit: model %i.\n" % int(all_models[cur_ind][-1]))
    fout = open("%s/used_params_and_model_%s.txt" % (TEMP_DIR, OUTNAME), "w")
    fout.write(
        "".join(
            [
                "".join(
                    ["%e, " for i in range(modC_map[int(all_models[cur_ind][-1]) - 1])]
                ),
                "%i\n",
            ]
        )
        % tuple(all_models[cur_ind][:-2] + [int(all_models[cur_ind][-1])])
    )
    fout.close()
    # 	Import parameters and index of chosen model.
    fin = open("%s/used_params_and_model_%s.txt" % (TEMP_DIR, OUTNAME))
    lines = fin.readlines()
    fin.close()
    field = lines[0].strip().split(", ")
    MODEL = int(field[-1])  # overwrite MODEL
    params = [float(el) for el in field[:-1]]
else:
    sys.stderr.write("Could not find a converging solution.\n")
    sys.exit()

if len(params) != modC_map[MODEL - 1]:
    sys.stderr.write(
        "Number of inferred parameters does not match the chosen model: %i vs. %i.\n"
        % (len(params), modC_map[MODEL - 1])
    )
    sys.exit()

# 	Output synonymous count per gene for all samples for Raphael group (October 2022):
syn_per_gene_per_sample = []
for sam in muts_by_sample:
    sam_syn = [mut for mut in sam[1] if mut["effect"] == "coding-synon"]
    syn_per_gene_per_sample.append(
        [
            sam[1][0]["sample"],
            [
                [k, len(list(g))]
                for k, g in it.groupby(
                    sorted(sam_syn, key=lambda arg: arg["gene"]),
                    key=lambda arg: arg["gene"],
                )
            ],
        ]
    )

fout = open("%s/output_data_preparation_%s.txt" % (TEMP_DIR, OUTNAME), "w")
# Output format: [gene, lm, lk, ls, mobs, kobs, sobs, Lgene, lambda_s]
fout.write(
    "gene\tl_m\tl_k\tl_s\tm_obs\tk_obs\ts_obs\tL_gene\tlambda_s\ts_max_per_sample\tN_samples=%i\n"
    % N_samples
)
for gene in mks_type:
    # 	Compute E[lambda_s]:
    if gene["obs"][2] > 25:
        thresh = 1
    else:
        thresh = 0
    lam_s = lambda_hat_given_s(params, gene["obs"][2], MODEL, thresh)
    # 	Compute maximum synonymous count across all samples:
    proxy = [
        el
        for el in [num for le in syn_per_gene_per_sample for num in le[1]]
        if el[0] == gene["gene"]
    ]
    if len(proxy):
        s_max_per_sample = max([el[1] for el in proxy])
    else:
        s_max_per_sample = 0
    fout.write(
        "%s\t%f\t%f\t%f\t%i\t%i\t%i\t%i\t%e\t%i\n"
        % (
            gene["gene"],
            gene["exp"][0],
            gene["exp"][1],
            gene["exp"][2],
            gene["obs"][0],
            gene["obs"][1],
            gene["obs"][2],
            gene["len"],
            lam_s,
            s_max_per_sample,
        )
    )
fout.close()
