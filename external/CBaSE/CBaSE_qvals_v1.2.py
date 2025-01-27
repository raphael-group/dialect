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

import glob
import math
import sys

import mpmath as mp
import numpy as np
import scipy.special as sp
import scipy.stats as st

# ************************************ FUNCTION DEFINITIONS *************************************


def compute_p_values(p, genes, aux):
    [modC, simC, runC] = aux
    if simC == 0:
        runC = 1

    if [1, 2].count(modC):
        a, b = p
    elif [3, 4].count(modC):
        a, b, t, w = p
    elif [5, 6].count(modC):
        a, b, g, d, w = p

    if modC == 1:
        # *************** lambda ~ Gamma:
        def pofs(s, L):
            return np.exp(
                s * np.log(L * b)
                + (-s - a) * np.log(1 + L * b)
                + sp.gammaln(s + a)
                - sp.gammaln(s + 1)
                - sp.gammaln(a),
            )

        def pofx_given_s(x, s, L, r, thr):
            return np.exp(
                x * np.log(r)
                + (s + x) * np.log(L * b)
                + (-s - x - a) * np.log(1 + L * (1 + r) * b)
                + sp.gammaln(s + x + a)
                - sp.gammaln(s + 1)
                - sp.gammaln(x + 1)
                - sp.gammaln(a),
            ) / pofs(s, L)

    elif modC == 2:
        # *************** lambda ~ IG:
        def pofs(s, L, thr):
            if thr:
                return 2.0 * mp.exp(
                    ((s + a) / 2.0) * math.log(L * b)
                    + mp.log(mp.besselk(-s + a, 2 * np.sqrt(L * b)))
                    - sp.gammaln(s + 1)
                    - sp.gammaln(a),
                )
            return 2.0 * math.exp(
                ((s + a) / 2.0) * math.log(L * b)
                + np.log(sp.kv(-s + a, 2 * math.sqrt(L * b)))
                - sp.gammaln(s + 1)
                - sp.gammaln(a),
            )

        def pofx_given_s(x, s, L, r, thr):
            if thr:
                return mp.exp(
                    np.log(2)
                    + (s + x) * np.log(L)
                    + x * np.log(r)
                    + (1 / 2.0 * (-s - x + a)) * np.log((L * (1 + r)) / b)
                    + a * np.log(b)
                    + mp.log(
                        mp.besselk(s + x - a, 2 * math.sqrt(L * (1 + r) * b)),
                    )
                    - sp.gammaln(s + 1)
                    - sp.gammaln(x + 1)
                    - sp.gammaln(a),
                ) / pofs(s, L, thr)
            return np.exp(
                np.log(2)
                + (s + x) * np.log(L)
                + x * np.log(r)
                + (1 / 2.0 * (-s - x + a)) * np.log((L * (1 + r)) / b)
                + a * np.log(b)
                + np.log(sp.kv(s + x - a, 2 * math.sqrt(L * (1 + r) * b)))
                - sp.gammaln(s + 1)
                - sp.gammaln(x + 1)
                - sp.gammaln(a),
            ) / pofs(s, L, thr)

    elif modC == 3:
        # *************** lambda ~ w * Exp + (1-w) * Gamma:
        def pofs(s, L):
            return np.exp(
                np.log(w) + s * np.log(L) + np.log(t) + (-1 - s) * np.log(L + t),
            ) + np.exp(
                np.log(1.0 - w)
                + s * np.log(L * b)
                + (-s - a) * np.log(1 + L * b)
                + sp.gammaln(s + a)
                - sp.gammaln(s + 1)
                - sp.gammaln(a),
            )

        def pofx_given_s(x, s, L, r, thr):
            return (
                np.exp(
                    np.log(w)
                    + (s + x) * np.log(L)
                    + x * np.log(r)
                    + np.log(t)
                    + (-1 - s - x) * np.log(L + L * r + t)
                    + sp.gammaln(1 + s + x)
                    - sp.gammaln(s + 1)
                    - sp.gammaln(x + 1),
                )
                + np.exp(
                    np.log(1 - w)
                    + x * np.log(r)
                    + (s + x) * np.log(L * b)
                    + (-s - x - a) * np.log(1 + L * (1 + r) * b)
                    + sp.gammaln(s + x + a)
                    - sp.gammaln(s + 1)
                    - sp.gammaln(x + 1)
                    - sp.gammaln(a),
                )
            ) / pofs(s, L)

    elif modC == 4:
        # *************** lambda ~ w * Exp + (1-w) * InvGamma:
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
                    - sp.gammaln(a),
                )
            return (w * L**s * t * (L + t) ** (-1 - s)) + np.exp(
                np.log(1.0 - w)
                + np.log(2.0)
                + ((s + a) / 2.0) * np.log(L * b)
                + np.log(sp.kv(-s + a, 2 * math.sqrt(L * b)))
                - sp.gammaln(s + 1)
                - sp.gammaln(a),
            )

        def pofx_given_s(x, s, L, r, thr):
            if thr:
                return (
                    np.exp(
                        np.log(w)
                        + (s + x) * np.log(L)
                        + x * np.log(r)
                        + np.log(t)
                        + (-1 - s - x) * np.log(L + L * r + t)
                        + sp.gammaln(1 + s + x)
                        - sp.gammaln(s + 1)
                        - sp.gammaln(x + 1),
                    )
                    + mp.exp(
                        np.log(1.0 - w)
                        + np.log(2)
                        + (s + x) * np.log(L)
                        + x * np.log(r)
                        + (0.5 * (-s - x + a)) * np.log((L * (1 + r)) / b)
                        + a * np.log(b)
                        + mp.log(
                            mp.besselk(s + x - a, 2 * np.sqrt(L * (1 + r) * b)),
                        )
                        - sp.gammaln(s + 1)
                        - sp.gammaln(x + 1)
                        - sp.gammaln(a),
                    )
                ) / pofs(s, L, thr)
            return (
                np.exp(
                    np.log(w)
                    + (s + x) * np.log(L)
                    + x * np.log(r)
                    + np.log(t)
                    + (-1 - s - x) * np.log(L + L * r + t)
                    + sp.gammaln(1 + s + x)
                    - sp.gammaln(s + 1)
                    - sp.gammaln(x + 1),
                )
                + np.exp(
                    np.log(1.0 - w)
                    + np.log(2)
                    + (s + x) * np.log(L)
                    + x * np.log(r)
                    + (0.5 * (-s - x + a)) * np.log((L * (1 + r)) / b)
                    + a * np.log(b)
                    + np.log(sp.kv(s + x - a, 2 * np.sqrt(L * (1 + r) * b)))
                    - sp.gammaln(s + 1)
                    - sp.gammaln(x + 1)
                    - sp.gammaln(a),
                )
            ) / pofs(s, L, thr)

    elif modC == 5:
        # *************** lambda ~ w * Gamma + (1-w) * Gamma (Gamma mixture model):
        def pofs(s, L):
            return np.exp(
                np.log(w)
                + s * np.log(L * b)
                + (-s - a) * np.log(1 + L * b)
                + sp.gammaln(s + a)
                - sp.gammaln(s + 1)
                - sp.gammaln(a),
            ) + np.exp(
                np.log(1.0 - w)
                + s * np.log(L * d)
                + (-s - g) * np.log(1 + L * d)
                + sp.gammaln(s + g)
                - sp.gammaln(s + 1)
                - sp.gammaln(g),
            )

        def pofx_given_s(x, s, L, r, thr):
            return (
                np.exp(
                    np.log(w)
                    + x * np.log(r)
                    + (s + x) * np.log(L * b)
                    + (-s - x - a) * np.log(1 + L * (1 + r) * b)
                    + sp.gammaln(s + x + a)
                    - sp.gammaln(s + 1)
                    - sp.gammaln(x + 1)
                    - sp.gammaln(a),
                )
                + np.exp(
                    np.log(1 - w)
                    + x * np.log(r)
                    + (s + x) * np.log(L * d)
                    + (-s - x - g) * np.log(1 + L * (1 + r) * d)
                    + sp.gammaln(s + x + g)
                    - sp.gammaln(s + 1)
                    - sp.gammaln(x + 1)
                    - sp.gammaln(g),
                )
            ) / pofs(s, L)

    elif modC == 6:
        # *************** lambda ~ w * Gamma + (1-w) * InvGamma (mixture model):
        def pofs(s, L, thr):
            if thr:
                return np.exp(
                    np.log(w)
                    + s * np.log(L * b)
                    + (-s - a) * np.log(1 + L * b)
                    + sp.gammaln(s + a)
                    - sp.gammaln(s + 1)
                    - sp.gammaln(a),
                ) + mp.exp(
                    np.log(1.0 - w)
                    + np.log(2.0)
                    + ((s + g) / 2.0) * np.log(L * d)
                    + mp.log(mp.besselk(-s + g, 2 * mp.sqrt(L * d)))
                    - sp.gammaln(s + 1)
                    - sp.gammaln(g),
                )
            return np.exp(
                np.log(w)
                + s * np.log(L * b)
                + (-s - a) * np.log(1 + L * b)
                + sp.gammaln(s + a)
                - sp.gammaln(s + 1)
                - sp.gammaln(a),
            ) + np.exp(
                np.log(1.0 - w)
                + np.log(2.0)
                + ((s + g) / 2.0) * np.log(L * d)
                + np.log(sp.kv(-s + g, 2 * np.sqrt(L * d)))
                - sp.gammaln(s + 1)
                - sp.gammaln(g),
            )

        def pofx_given_s(x, s, L, r, thr):
            if thr:
                return (
                    np.exp(
                        np.log(w)
                        + x * np.log(r)
                        + (s + x) * np.log(L * b)
                        + (-s - x - a) * np.log(1 + L * (1 + r) * b)
                        + sp.gammaln(s + x + a)
                        - sp.gammaln(s + 1)
                        - sp.gammaln(x + 1)
                        - sp.gammaln(a),
                    )
                    + mp.exp(
                        np.log(1 - w)
                        + np.log(2)
                        + (s + x) * np.log(L)
                        + x * np.log(r)
                        + (0.5 * (-s - x + g)) * np.log((L * (1 + r)) / d)
                        + g * np.log(d)
                        + mp.log(
                            mp.besselk(s + x - g, 2 * np.sqrt(L * (1 + r) * d)),
                        )
                        - sp.gammaln(s + 1)
                        - sp.gammaln(x + 1)
                        - sp.gammaln(g),
                    )
                ) / pofs(s, L, thr)
            return (
                np.exp(
                    np.log(w)
                    + x * np.log(r)
                    + (s + x) * np.log(L * b)
                    + (-s - x - a) * np.log(1 + L * (1 + r) * b)
                    + sp.gammaln(s + x + a)
                    - sp.gammaln(s + 1)
                    - sp.gammaln(x + 1)
                    - sp.gammaln(a),
                )
                + np.exp(
                    np.log(1 - w)
                    + np.log(2)
                    + (s + x) * np.log(L)
                    + x * np.log(r)
                    + (0.5 * (-s - x + g)) * np.log((L * (1 + r)) / d)
                    + g * np.log(d)
                    + np.log(sp.kv(s + x - g, 2 * np.sqrt(L * (1 + r) * d)))
                    - sp.gammaln(s + 1)
                    - sp.gammaln(x + 1)
                    - sp.gammaln(g),
                )
            ) / pofs(s, L, thr)

    pvals = []
    L = 1.0
    gcnt = 0
    for gene in genes:
        gcnt += 1
        sys.stderr.write("%i%% done.\r" % int(float(gcnt) / len(genes) * 100.0))

        sobs = int(gene["obs"][2])
        mobs = int(gene["obs"][0])
        kobs = int(gene["obs"][1])
        sexp = gene["exp"][2]
        mexp = gene["exp"][0]
        kexp = gene["exp"][1]
        lams = gene["lambda_s"]
        gene["s_max_per_sample"]
        ratm = mexp / sexp
        ratk = kexp / sexp

        large_flag = 0
        last_p = 0.0
        if sobs == 0:
            meant2 = 2  # ~= E[m] * 2
        else:
            meant2 = int(
                ratm * sobs + 3.0 * math.sqrt(ratm * sobs),
            )  # ~= E[m] + 3*sigma_m
        for mtest in range(meant2):
            if math.isnan(pofx_given_s(mtest, sobs, L, ratm, 0)):
                large_flag = 1  # 	Going to large-x mode.

                class P_of_x_given_s(st.rv_continuous):
                    def _pdf(self, x, s, eL, rat):
                        if s == 1e-10:
                            s = 0
                        return pofx_given_s(x, s, eL, rat, 1)

                inst_pofx = P_of_x_given_s(a=0)
                break
            cur_p = pofx_given_s(mtest, sobs, L, ratm, 0)
            diff = cur_p - last_p
            last_p = cur_p
            if pofx_given_s(mtest, sobs, L, ratm, 0) > 1.0:
                if (
                    pofx_given_s(mtest - 1, sobs, L, ratm, 0) > 1.0 / runC
                    or diff > 0
                ):
                    large_flag = 1  # 	Going to large-x mode.

                    class P_of_x_given_s(st.rv_continuous):
                        def _pdf(self, x, s, eL, rat):
                            if s == 1e-10:
                                s = 0
                            return pofx_given_s(x, s, eL, rat, 1)

                    inst_pofx = P_of_x_given_s(a=0)
                else:

                    class P_of_x_given_s(st.rv_discrete):
                        def _pmf(self, x, s, eL, rat):
                            if s == 1e-10:
                                s = 0
                            return pofx_given_s(x, s, eL, rat, 0)

                    inst_pofx = P_of_x_given_s()
                break
        if large_flag == 0:

            class P_of_x_given_s(st.rv_discrete):
                def _pmf(self, x, s, eL, rat):
                    if s == 1e-10:
                        s = 0
                    return pofx_given_s(x, s, eL, rat, 0)

            inst_pofx = P_of_x_given_s()

        pofm = []
        pofk = []

        # The upper limit for mtest should coincide with the upper limit above (meant2).
        sum_p = 0.0
        testm_array = []
        for mtest in range(meant2):
            next_prob = pofx_given_s(mtest, sobs, L, ratm, large_flag)
            pofm.append([mtest, next_prob])
            m_pneg = sum_p + next_prob
            m_ppos = 1.0 - sum_p
            m_ppos = m_ppos.real
            m_pneg = m_pneg.real
            testm_array.append([m_pneg, m_ppos])
            sum_p += next_prob

        sum_p = 0.0
        testk_array = []
        for ktest in range((int(ratk * sobs) + 1) * 2):
            next_prob = pofx_given_s(ktest, sobs, L, ratk, large_flag)
            pofk.append([ktest, next_prob])
            k_pneg = sum_p + next_prob
            k_ppos = 1.0 - sum_p
            k_ppos = k_ppos.real
            k_pneg = k_pneg.real
            testk_array.append([k_pneg, k_ppos])
            sum_p += next_prob

        for _rep in range(runC):
            if simC:
                # 	Simulate expectation under null.
                if sobs == 0:
                    sobs = 1e-10
                try:
                    msim = inst_pofx.rvs(sobs, L, ratm)
                    ksim = inst_pofx.rvs(sobs, L, ratk)
                except:
                    continue  # If this happens *very* frequently (out of runC*18,437 runs in total), try decreasing meant2 above. Otherwise results may overestimate the degree of negative selection.
                if sobs == 1e-10:
                    sobs = 0
                mobs = int(round(msim))
                kobs = int(round(ksim))

            try:
                [m_pneg, m_ppos] = testm_array[mobs]
                [k_pneg, k_ppos] = testk_array[kobs]
            except:
                pofm = []
                pofk = []

                cum_p = 0.0
                for x in range(mobs):
                    next_prob = pofx_given_s(x, sobs, L, ratm, large_flag)
                    cum_p += next_prob
                m_pneg = cum_p + pofx_given_s(mobs, sobs, L, ratm, large_flag)
                m_ppos = 1.0 - cum_p
                m_ppos = float(m_ppos.real)
                m_pneg = float(m_pneg.real)
                if (
                    m_ppos < 0
                    or m_ppos > 1
                    or math.isinf(m_ppos)
                    or math.isnan(m_ppos)
                ):
                    cum_p = 0.0
                    for x in range(mobs):
                        next_prob = pofx_given_s(x, sobs, L, ratm, 1)
                        cum_p += next_prob
                    m_pneg = cum_p + pofx_given_s(mobs, sobs, L, ratm, 1)
                    m_ppos = 1.0 - cum_p
                    m_ppos = float(m_ppos.real)
                    m_pneg = float(m_pneg.real)
                    if (
                        m_ppos < 0
                        or m_ppos > 1
                        or math.isinf(m_ppos)
                        or math.isnan(m_ppos)
                    ):
                        sys.stderr.write(
                            "Setting p_m^pos --> 0 on gene {} (was {:e}).\n".format(
                                gene["gene"], m_ppos
                            ),
                        )
                        m_ppos = 0
                if m_pneg > 1:
                    if m_pneg < 1.01:
                        m_pneg = 1.0
                    elif math.isinf(m_pneg) or math.isnan(m_pneg):
                        if simC == 0:
                            sys.stderr.write(
                                "p_m^neg on gene {}: {:f} --> 1.\n".format(
                                    gene["gene"], m_pneg
                                ),
                            )
                        m_pneg = 1.0

                cum_p = 0.0
                for x in range(kobs):
                    next_prob = pofx_given_s(x, sobs, L, ratk, large_flag)
                    cum_p += next_prob
                k_pneg = cum_p + pofx_given_s(kobs, sobs, L, ratk, large_flag)
                k_ppos = 1.0 - cum_p
                k_ppos = float(k_ppos.real)
                k_pneg = float(k_pneg.real)
                if (
                    k_ppos < 0
                    or k_ppos > 1
                    or math.isinf(k_ppos)
                    or math.isnan(k_ppos)
                ):
                    cum_p = 0.0
                    for x in range(kobs):
                        next_prob = pofx_given_s(x, sobs, L, ratk, 1)
                        cum_p += next_prob
                    k_pneg = cum_p + pofx_given_s(kobs, sobs, L, ratk, 1)
                    k_ppos = 1.0 - cum_p
                    k_ppos = float(k_ppos.real)
                    k_pneg = float(k_pneg.real)
                    if (
                        k_ppos < 0
                        or k_ppos > 1
                        or math.isinf(k_ppos)
                        or math.isnan(k_ppos)
                    ):
                        sys.stderr.write(
                            "Setting p_k^pos --> 0 on gene {} (was {:e}).\n".format(
                                gene["gene"], k_ppos
                            ),
                        )
                        k_ppos = 0
                if k_pneg > 1:
                    if k_pneg < 1.01:
                        k_pneg = 1.0
                    elif math.isinf(k_pneg) or math.isnan(k_pneg):
                        if simC == 0:
                            sys.stderr.write(
                                "p_k^neg on gene {}: {:f} --> 1.\n".format(
                                    gene["gene"], k_pneg
                                ),
                            )
                        k_pneg = 1.0

            # 	Output [x, P(x|s)] for all x for Raphael group (October 2022):
            if simC == 0:
                i = len(pofm)
                test_prob = pofx_given_s(i, sobs, L, ratm, large_flag)
                while abs(test_prob) < 9e-7:
                    pofm.append([i, test_prob])
                    i += 1
                    test_prob = pofx_given_s(i, sobs, L, ratm, large_flag)
                    if test_prob - pofm[-1][1] < 0:
                        break
                while test_prob > 9e-7 and math.isinf(test_prob) == 0:
                    pofm.append([i, test_prob])
                    i += 1
                    test_prob = pofx_given_s(i, sobs, L, ratm, large_flag)

                i = len(pofk)
                test_prob = pofx_given_s(i, sobs, L, ratk, large_flag)
                while abs(test_prob) < 9e-7:
                    pofk.append([i, test_prob])
                    i += 1
                    test_prob = pofx_given_s(i, sobs, L, ratk, large_flag)
                    if test_prob - pofk[-1][1] < 0:
                        break
                while test_prob > 9e-7 and math.isinf(test_prob) == 0:
                    pofk.append([i, test_prob])
                    i += 1
                    test_prob = pofx_given_s(i, sobs, L, ratk, large_flag)
            else:
                pofk = []
                pofm = []

            # 	Output [x_g, P(x_g|s)] for all genes:
            pofm_per_sample = []
            pofk_per_sample = []
            if simC == 0:
                i = 0
                test_prob = pofx_given_s(
                    i,
                    sobs,
                    L,
                    ratm / N_samples,
                    large_flag,
                )
                while abs(test_prob) < THRESHOLD:
                    pofm_per_sample.append([i, test_prob])
                    i += 1
                    test_prob = pofx_given_s(
                        i,
                        sobs,
                        L,
                        ratm / N_samples,
                        large_flag,
                    )
                    if test_prob - pofm_per_sample[-1][1] < 0:
                        break
                while test_prob > THRESHOLD and math.isinf(test_prob) == 0:
                    pofm_per_sample.append([i, test_prob])
                    i += 1
                    test_prob = pofx_given_s(
                        i,
                        sobs,
                        L,
                        ratm / N_samples,
                        large_flag,
                    )
                i = 0
                test_prob = pofx_given_s(
                    i,
                    sobs,
                    L,
                    ratk / N_samples,
                    large_flag,
                )
                while abs(test_prob) < THRESHOLD:
                    pofk_per_sample.append([i, test_prob])
                    i += 1
                    test_prob = pofx_given_s(
                        i,
                        sobs,
                        L,
                        ratk / N_samples,
                        large_flag,
                    )
                    if test_prob - pofk_per_sample[-1][1] < 0:
                        break
                while test_prob > THRESHOLD and math.isinf(test_prob) == 0:
                    pofk_per_sample.append([i, test_prob])
                    i += 1
                    test_prob = pofx_given_s(
                        i,
                        sobs,
                        L,
                        ratk / N_samples,
                        large_flag,
                    )

            pm0 = pofx_given_s(0, sobs, L, ratm, 0)
            pk0 = pofx_given_s(0, sobs, L, ratk, 0)
            dxds = (kobs + mobs) / (lams * (ratk + ratm))
            if ratm < 1e-30:
                m_pneg, m_ppos, pm0, dmds, pofm, pofm_per_sample = (
                    1,
                    1,
                    1,
                    1,
                    [],
                    [],
                )
            else:
                dmds = mobs / (lams * ratm)
            if ratk < 1e-30:
                k_pneg, k_ppos, pk0, dkds, pofk, pofk_per_sample = (
                    1,
                    1,
                    1,
                    1,
                    [],
                    [],
                )
            else:
                dkds = kobs / (lams * ratk)

            pvals.append(
                [
                    gene["gene"],
                    m_pneg,
                    k_pneg,
                    m_ppos,
                    k_ppos,
                    pm0,
                    pk0,
                    mobs,
                    kobs,
                    sobs,
                    [dmds, dkds, dxds],
                    pofm,
                    pofk,
                    pofm_per_sample,
                    pofk_per_sample,
                ],
            )

    sys.stderr.write("100% done.\n")

    return pvals


def export_pofxigivens_table(pvals_array, x_ind) -> None:
    label = ["m", "k"][x_ind - 13]
    fout = open(f"{TEMP_DIR}/pof{label}igivens.txt", "w")
    for gene in pvals_array:
        fout.write(f"{gene[0]}_{label}i\t")
        for i in range(len(gene[x_ind])):
            if gene[x_ind][i][1] > THRESHOLD:
                fout.write("%i\t" % (gene[x_ind][i][0]))
        fout.write("\n")
        fout.write(f"{gene[0]}_pof{label}i\t")
        for i in range(len(gene[x_ind])):
            if gene[x_ind][i][1] > THRESHOLD:
                fout.write(f"{gene[x_ind][i][1]:.6e}\t")
        fout.write("\n")
    fout.close()


def export_pofxgivens_table(pvals_array, x_ind, outfile) -> None:
    fout = open(f"{TEMP_DIR}/{outfile}.txt", "w")
    label = ["m", "k"][x_ind - 11]
    for gene in pvals_array:
        fout.write(f"{gene[0]}_{label}\t")
        for i in range(len(gene[x_ind])):
            if gene[x_ind][i][1] > 9e-7:
                fout.write("%i\t" % (gene[x_ind][i][0]))
        fout.write("\n")
        fout.write(f"{gene[0]}_pof{label}\t")
        for i in range(len(gene[x_ind])):
            if gene[x_ind][i][1] > 9e-7:
                fout.write(f"{gene[x_ind][i][1]:.6f}\t")
        fout.write("\n")
    fout.close()


def construct_histogram(var_array, bin_var):
    noinf = [el for el in var_array if el < 1e50]
    var_max = max(noinf) + bin_var
    hist = [0.0 for i in range(int(var_max / bin_var))]
    for var in noinf:
        hist[int(var / bin_var)] += 1
    return [
        [(i + 0.5) * bin_var, hist[i] / len(noinf)] for i in range(len(hist))
    ]


def compute_phi_sim(pvals_array, ind1, ind2):
    # 	Compute simulated phi, measuring joint effects of missense and nonsense mutations.
    all_phi = []
    for gene in pvals_array:
        if abs(gene[ind1]) < 1e-100 or abs(gene[ind2]) < 1e-100:
            all_phi.append(1e5)
        else:
            all_phi.append(-math.log(gene[ind1]) - math.log(gene[ind2]))
    # 	Compute negative log of individual simulated p-values for the missense and nonsense category.
    all_phi_m = []
    for gene in pvals_array:
        if abs(gene[ind1]) < 1e-100:
            all_phi_m.append(1e5)
        else:
            all_phi_m.append(-math.log(gene[ind1]))
    all_phi_k = []
    for gene in pvals_array:
        if abs(gene[ind2]) < 1e-100:
            all_phi_k.append(1e5)
        else:
            all_phi_k.append(-math.log(gene[ind2]))
    return all_phi, all_phi_m, all_phi_k


def compute_phi_obs(pvals_array, ind1, ind2):
    # 	Compute observed phi, measuring joint effects of missense and nonsense mutations.
    all_phi = []
    for gene in pvals_array:
        if abs(gene[ind1]) < 1e-100 or abs(gene[ind2]) < 1e-100:
            cur_phi = 1e5
        else:
            cur_phi = -math.log(gene[ind1]) - math.log(gene[ind2])
        all_phi.append(
            {
                "gene": gene[0],
                "phi": cur_phi,
                "p0m": gene[5],
                "p0k": gene[6],
                "mks": [gene[7], gene[8], gene[9]],
                "dnds": gene[10],
            },
        )
    # 	Compute negative log of individual observed p-values for the missense and nonsense category.
    all_phi_m = []
    for gene in pvals_array:
        cur_phi = 100000.0 if abs(gene[ind1]) < 1e-100 else -math.log(gene[ind1])
        all_phi_m.append(
            {
                "gene": gene[0],
                "phi": cur_phi,
                "p0m": gene[5],
                "p0k": gene[6],
                "mks": [gene[7], gene[8], gene[9]],
                "dnds": gene[10],
            },
        )
    all_phi_k = []
    for gene in pvals_array:
        cur_phi = 100000.0 if abs(gene[ind2]) < 1e-100 else -math.log(gene[ind2])
        all_phi_k.append(
            {
                "gene": gene[0],
                "phi": cur_phi,
                "p0m": gene[5],
                "p0k": gene[6],
                "mks": [gene[7], gene[8], gene[9]],
                "dnds": gene[10],
            },
        )
    return all_phi, all_phi_m, all_phi_k


def FDR_discrete(phi_sim_array, gene_phi_real, bin_phi, bin_p, noncat):

    # 	Build histogram of simulated neutral phi values.
    phi_sim_hist = construct_histogram(phi_sim_array, bin_phi)
    phi_sim_max = max(phi_sim_array)

    # 	Sort the observed phi values in decreasing order.
    gene_phi_real.sort(key=lambda arg: arg["phi"], reverse=True)
    larger_0 = [el for el in gene_phi_real if el["phi"] >= 1e-30]
    if noncat == 0:
        equal_0 = sorted(
            [el for el in gene_phi_real if abs(el["phi"]) < 1e-30],
            key=lambda arg: arg["p0m"] * arg["p0k"],
            reverse=True,
        )
    elif noncat == 1:
        equal_0 = sorted(
            [el for el in gene_phi_real if abs(el["phi"]) < 1e-30],
            key=lambda arg: arg["p0m"],
            reverse=True,
        )
    elif noncat == 2:
        equal_0 = sorted(
            [el for el in gene_phi_real if abs(el["phi"]) < 1e-30],
            key=lambda arg: arg["p0k"],
            reverse=True,
        )
    sorted_phi = larger_0 + equal_0
    if len(sorted_phi) < len(gene_phi_real):
        excl = [
            el
            for el in gene_phi_real
            if [gen["gene"] for gen in sorted_phi].count(el["gene"]) == 0
        ]
        sys.stderr.write("Note: Not including these gene(s) in output:\n")
        for el in excl:
            sys.stderr.write(f"{el}\n")

    # 	Compute the p-values of the observed phi values.
    phi_pvals_obs = []
    for gene in sorted_phi:
        if gene["phi"] > phi_sim_max + 0.5 * bin_phi:
            phi_pvals_obs.append(
                {
                    "gene": gene["gene"],
                    "p_phi": 0,
                    "phi": gene["phi"],
                    "mks": gene["mks"],
                    "dnds": gene["dnds"],
                },
            )
        elif abs(gene["phi"]) < 1e-30:
            use_phi = [gene["p0m"] * gene["p0k"], gene["p0m"], gene["p0k"]][noncat]
            phi_pvals_obs.append(
                {
                    "gene": gene["gene"],
                    "p_phi": 1.0,
                    "phi": use_phi,
                    "mks": gene["mks"],
                    "dnds": gene["dnds"],
                },
            )
        else:
            cumprob = 0.0
            i = 0
            while i < len(phi_sim_hist) and phi_sim_hist[i][0] + 0.5 * bin_phi <= max(
                0.0, gene["phi"]
            ):
                cumprob += phi_sim_hist[i][1]
                i += 1
            phi_pvals_obs.append(
                {
                    "gene": gene["gene"],
                    "p_phi": 1.0 - cumprob,
                    "phi": gene["phi"],
                    "mks": gene["mks"],
                    "dnds": gene["dnds"],
                },
            )

    # 	Compute the q-values of the observed phi with BH procedure.
    phi_qvals_obs = []
    grank = 0.0
    for gene in phi_pvals_obs:
        grank += 1.0
        if (
            abs(gene["p_phi"]) < 1e-100
        ):  # 	If p-value was numerically 0, q-value is too.
            phi_qvals_obs.append(
                {
                    "gene": gene["gene"],
                    "p_phi": gene["p_phi"],
                    "q_phi": 0,
                    "phi": gene["phi"],
                    "mks": gene["mks"],
                    "dnds": gene["dnds"],
                },
            )
        else:
            q_phi_BH = gene["p_phi"] / (grank / len(phi_pvals_obs))
            phi_qvals_obs.append(
                {
                    "gene": gene["gene"],
                    "p_phi": gene["p_phi"],
                    "q_phi": q_phi_BH,
                    "phi": gene["phi"],
                    "mks": gene["mks"],
                    "dnds": gene["dnds"],
                },
            )

    if len(phi_qvals_obs) != len(gene_phi_real):
        sys.stderr.write(
            "Number of genes in output different from original: %i vs. %i.\n"
            % (len(phi_qvals_obs), len(gene_phi_real)),
        )

    phi_qvals_adj = []
    for g in range(len(phi_qvals_obs)):
        cur_min = min([el["q_phi"] for el in phi_qvals_obs[g:]])
        gene = phi_qvals_obs[g]
        phi_qvals_adj.append(
            {
                "gene": gene["gene"],
                "p_phi": gene["p_phi"],
                "q_adj": min(1.0, cur_min),
                "phi": gene["phi"],
                "mks": gene["mks"],
                "dnds": gene["dnds"],
            },
        )

    return phi_qvals_adj


def combine_qvalues(all_neg, all_pos):
    # 	This function creates an array that has all p-values and q-values, for positive and negative selection, by gene.
    if (
        len(all_neg[0]) != len(all_pos[2])
        or len(all_pos[0]) != len(all_pos[1])
        or len(all_neg[0]) != len(all_neg[2])
    ):
        sys.stderr.write(
            "Warning: Number of genes not identical across q-value arrays, matching in output file compromised.\n",
        )
    large_qval_array = []
    for g in range(len(all_neg[0])):
        large_qval_array.append(
            {
                "gene": all_neg[0][g]["gene"],
                "p_phi_neg": all_neg[0][g]["p_phi"],
                "q_adj_neg": all_neg[0][g]["q_adj"],
                "phi_neg": all_neg[0][g]["phi"],
                "p_phi_m_neg": all_neg[1][g]["p_phi"],
                "q_adj_m_neg": all_neg[1][g]["q_adj"],
                "phi_m_neg": all_neg[1][g]["phi"],
                "p_phi_k_neg": all_neg[2][g]["p_phi"],
                "q_adj_k_neg": all_neg[2][g]["q_adj"],
                "phi_k_neg": all_neg[2][g]["phi"],
                "p_phi_pos": all_pos[0][g]["p_phi"],
                "q_adj_pos": all_pos[0][g]["q_adj"],
                "phi_pos": all_pos[0][g]["phi"],
                "p_phi_m_pos": all_pos[1][g]["p_phi"],
                "q_adj_m_pos": all_pos[1][g]["q_adj"],
                "phi_m_pos": all_pos[1][g]["phi"],
                "p_phi_k_pos": all_pos[2][g]["p_phi"],
                "q_adj_k_pos": all_pos[2][g]["q_adj"],
                "phi_k_pos": all_pos[2][g]["phi"],
                "mks": all_neg[0][g]["mks"],
                "dnds": all_neg[0][g]["dnds"],
            },
        )
    return large_qval_array


# ************************************* GLOBAL DEFINITIONS **************************************

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
run_no = 30  # 	No. of simulation replicates used for computing FDR (default 50)
outname = str(sys.argv[1])  # 	name for the output file containing the q values
TEMP_DIR = str(sys.argv[2])  #   path to the temp files folder
THRESHOLD = float(sys.argv[3])  # 	threshold for categorical bmr pmf generation

# ***********************************************************************************************
# ***********************************************************************************************
#
# 	Collect parameter estimates from all fitted models in working directory.

p_files = glob.glob(f"{TEMP_DIR}/param_estimates_*.txt")

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
    if (
        2.0 * modC_map[int(all_models[m][-1]) - 1] + 2.0 * all_models[m][-2]
        < cur_min
    ):
        cur_min = (
            2.0 * modC_map[int(all_models[m][-1]) - 1] + 2.0 * all_models[m][-2]
        )
        cur_ind = m
if cur_min < 1e20:
    sys.stderr.write(
        "Best model fit: model %i.\n" % int(all_models[cur_ind][-1]),
    )
    fout = open(f"{TEMP_DIR}/used_params_and_model.txt", "w")
    fout.write(
        "".join(
            [
                "".join(
                    [
                        "%e, "
                        for i in range(
                            modC_map[int(all_models[cur_ind][-1]) - 1],
                        )
                    ],
                ),
                "%i\n",
            ],
        )
        % tuple(all_models[cur_ind][:-2] + [int(all_models[cur_ind][-1])]),
    )
    fout.close()
else:
    sys.stderr.write("Could not find a converging solution.\n")

# ***********************************************************************************************
#
# 	Compute q-values for all genes, using best fitting model.

# 	Import parameters and index of chosen model.
fin = open(f"{TEMP_DIR}/used_params_and_model.txt")
lines = fin.readlines()
fin.close()
field = lines[0].strip().split(", ")

mod_C = int(field[-1])
params = [float(el) for el in field[:-1]]

if len(params) != modC_map[mod_C - 1]:
    sys.stderr.write(
        "Number of inferred parameters does not match the chosen model: %i vs. %i.\n"
        % (len(params), modC_map[mod_C - 1]),
    )
    sys.exit()

# 	Import output from data_preparation, containing l_x and (m,k,s)_obs.
fin = open(f"{TEMP_DIR}/output_data_preparation.txt")
lines = fin.readlines()
fin.close()
mks_type = []
N_samples = int(lines[0].strip().split("\t")[-1].split("=")[-1])
sys.stderr.write("%i sample(s) used in total.\n" % N_samples)
for line in lines[1:]:
    field = line.strip().split("\t")
    mks_type.append(
        {
            "gene": field[0],
            "exp": [float(field[1]), float(field[2]), float(field[3])],
            "obs": [float(field[4]), float(field[5]), float(field[6])],
            "len": int(field[7]),
            "lambda_s": float(field[8]),
            "s_max_per_sample": int(field[9]),
        },
    )
mks_type = sorted(mks_type, key=lambda arg: arg["gene"])

sys.stderr.write("Computing real p-values.\n")
pvals_obs = compute_p_values(params, mks_type, [mod_C, 0, 1])
export_pofxgivens_table(pvals_obs, 11, "pofmgivens")
export_pofxgivens_table(pvals_obs, 12, "pofkgivens")
export_pofxigivens_table(pvals_obs, 13)
export_pofxigivens_table(pvals_obs, 14)
sys.stderr.write("Computing simulated p-values.\n")
sys.stderr.write("Simulation runs\t= %i.\n" % (run_no))
pvals_sim = compute_p_values(params, mks_type, [mod_C, 1, run_no])
# Format: [gene, m_pneg, k_pneg, m_ppos, k_ppos, pofx_given_s(0, sobs, L, ratm), pofx_given_s(0, sobs, L, ratk)]

# 	Negative selection
phi_sim, phi_sim_m, phi_sim_k = compute_phi_sim(pvals_sim, 1, 2)
gene_phi_obs, gene_phi_obs_m, gene_phi_obs_k = compute_phi_obs(pvals_obs, 1, 2)
q_neg_adj = sorted(
    FDR_discrete(phi_sim, gene_phi_obs, 0.02, 0.000001, 0),
    key=lambda arg: arg["gene"],
)
q_neg_adj_m = sorted(
    FDR_discrete(phi_sim_m, gene_phi_obs_m, 0.02, 0.000001, 1),
    key=lambda arg: arg["gene"],
)
q_neg_adj_k = sorted(
    FDR_discrete(phi_sim_k, gene_phi_obs_k, 0.02, 0.000001, 2),
    key=lambda arg: arg["gene"],
)

# 	Positive selection
phi_sim, phi_sim_m, phi_sim_k = compute_phi_sim(pvals_sim, 3, 4)
gene_phi_obs, gene_phi_obs_m, gene_phi_obs_k = compute_phi_obs(pvals_obs, 3, 4)
q_pos_adj = sorted(
    FDR_discrete(phi_sim, gene_phi_obs, 0.02, 0.000001, 0),
    key=lambda arg: arg["gene"],
)
q_pos_adj_m = sorted(
    FDR_discrete(phi_sim_m, gene_phi_obs_m, 0.02, 0.000001, 1),
    key=lambda arg: arg["gene"],
)
q_pos_adj_k = sorted(
    FDR_discrete(phi_sim_k, gene_phi_obs_k, 0.02, 0.000001, 2),
    key=lambda arg: arg["gene"],
)

# 	Combine and output gene-specific q-values.
all_qvalues = combine_qvalues(
    [q_neg_adj, q_neg_adj_m, q_neg_adj_k],
    [q_pos_adj, q_pos_adj_m, q_pos_adj_k],
)

# 	Output q-values in file.
fout = open(f"{TEMP_DIR}/q_values.txt", "w")
fout.write(f"{mod_choice[mod_C]}\t{params}\n")
fout.write(
    "gene\tp_phi_m_neg\tq_phi_m_neg\tphi_m_neg\tp_phi_k_neg\tq_phi_k_neg\tphi_k_neg\tp_phi_neg\tq_phi_neg\tphi_neg\tp_phi_m_pos\tq_phi_m_pos\tphi_m_pos_or_p(m=0|s)\tp_phi_k_pos\tq_phi_k_pos\tphi_k_pos_or_p(k=0|s)\tp_phi_pos\tq_phi_pos\tphi_pos_or_p(m=0|s)*p(k=0|s)\tm_obs\tk_obs\ts_obs\tdm/ds\tdk/ds\td(m+k)/ds\n",
)
for g in range(len(all_qvalues)):
    curg = all_qvalues[g]
    fout.write(
        "%s\t%e\t%e\t%f\t%e\t%e\t%f\t%e\t%e\t%f\t%e\t%e\t%f\t%e\t%e\t%f\t%e\t%e\t%f\t%i\t%i\t%i\t%f\t%f\t%f\n"
        % (
            curg["gene"],
            curg["p_phi_m_neg"],
            curg["q_adj_m_neg"],
            curg["phi_m_neg"],
            curg["p_phi_k_neg"],
            curg["q_adj_k_neg"],
            curg["phi_k_neg"],
            curg["p_phi_neg"],
            curg["q_adj_neg"],
            curg["phi_neg"],
            curg["p_phi_m_pos"],
            curg["q_adj_m_pos"],
            curg["phi_m_pos"],
            curg["p_phi_k_pos"],
            curg["q_adj_k_pos"],
            curg["phi_k_pos"],
            curg["p_phi_pos"],
            curg["q_adj_pos"],
            curg["phi_pos"],
            curg["mks"][0],
            curg["mks"][1],
            curg["mks"][2],
            curg["dnds"][0],
            curg["dnds"][1],
            curg["dnds"][2],
        ),
    )
fout.close()
