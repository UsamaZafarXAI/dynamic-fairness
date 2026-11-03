"""JBHI-00470-2026.R3: tables requested by the Associate Editor.

Inputs (export these from the notebook that produced the published results):
  --mimic        MIMIC-IV cohort BEFORE imputation (one row per stay; includes
                 intersectional_group, mortality_in_hospital, and the 17 predictors)
  --audit        one or more CSVs of t=0 test-set predictions with columns
                 model, y, pred, group  (Baseline and Static Reweighing)
  --eicu         eICU cohort BEFORE imputation, with hospitalid and intersectional_group
  --eicu-preds   eICU predictions with columns y, pred, group, hospitalid
Outputs go to --out: CSV tables plus summary.txt with the numbers for the manuscript.
"""
import argparse, glob, os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import betaln
from scipy.stats import chi2_contingency, spearmanr

ID_COLS = {"stay_id", "hadm_id", "subject_id", "patientunitstayid", "hospitalid",
           "mortality_in_hospital", "intersectional_group", "ethnicity", "age_group", "gender"}
Z = 1.959964


def wilson(k, n):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / den
    h = Z * np.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def predictors(df, given):
    if given:
        return [c for c in given.split(",")]
    return [c for c in df.columns if c not in ID_COLS and pd.api.types.is_numeric_dtype(df[c])]


# ---------------------------------------------------------------- missingness
def missingness_tables(df, preds, label, out, hosp_col=None):
    overall = pd.DataFrame({"predictor": preds,
                            "n_missing": [int(df[p].isna().sum()) for p in preds],
                            "pct_missing": [round(100 * df[p].isna().mean(), 2) for p in preds]})
    by_g = df.groupby("intersectional_group")[preds].apply(lambda x: x.isna().mean() * 100).round(2)
    by_g.to_csv(os.path.join(out, f"{label}_missing_by_subgroup.csv"))
    pvals, lo, hi = [], [], []
    for p in preds:
        tab = pd.crosstab(df["intersectional_group"], df[p].isna())
        pvals.append(chi2_contingency(tab)[1] if tab.shape[1] == 2 else np.nan)
        lo.append(by_g[p].min()); hi.append(by_g[p].max())
    overall["subgroup_min_pct"], overall["subgroup_max_pct"] = lo, hi
    overall["chi2_p_subgroup"] = np.round(pvals, 4)
    if hosp_col:
        by_h = df.groupby(hosp_col)[preds].apply(lambda x: x.isna().mean() * 100)
        by_h.round(2).to_csv(os.path.join(out, f"{label}_missing_by_hospital.csv"))
        overall["hosp_median_pct"] = by_h.median().round(2).values
        overall["hosp_q1_pct"] = by_h.quantile(.25).round(2).values
        overall["hosp_q3_pct"] = by_h.quantile(.75).round(2).values
        overall["hosp_all_missing_n"] = (by_h == 100).sum().values
        overall["n_hospitals"] = len(by_h)
    overall.to_csv(os.path.join(out, f"{label}_missing_overall.csv"), index=False)
    return overall


# ------------------------------------------------- subgroup counts and CIs
def subgroup_counts(y, pred, g):
    d = pd.DataFrame({"y": y, "pred": pred, "g": g})
    rows = []
    for k, s in d.groupby("g"):
        tp = int(((s.y == 1) & (s.pred == 1)).sum()); fn = int(((s.y == 1) & (s.pred == 0)).sum())
        fp = int(((s.y == 0) & (s.pred == 1)).sum()); tn = int(((s.y == 0) & (s.pred == 0)).sum())
        r = dict(group=k, n=len(s), events=tp + fn, tp=tp, fn=fn, fp=fp, tn=tn)
        for name, num, den in (("TPR", tp, tp + fn), ("FPR", fp, fp + tn), ("FOR", fn, fn + tn)):
            r[name] = num / den if den else np.nan
            r[name + "_lo"], r[name + "_hi"] = wilson(num, den)
            r[name + "_den"] = den
        rows.append(r)
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def fit_beta_prior(x, n):
    """Beta-binomial marginal likelihood; returns (a, b)."""
    x, n = np.asarray(x, float), np.asarray(n, float)
    keep = n > 0
    x, n = x[keep], n[keep]
    if len(x) < 3:
        p = x.sum() / max(n.sum(), 1)
        return p * 10, (1 - p) * 10

    def nll(t):
        a, b = np.exp(t)
        return -np.sum(betaln(x + a, n - x + b) - betaln(a, b))
    p0 = np.clip(x.sum() / n.sum(), .02, .98)
    res = minimize(nll, np.log([p0 * 10, (1 - p0) * 10]), method="Nelder-Mead")
    return tuple(np.exp(res.x))


def gaps(t, min_cell):
    out = {}
    for m, num, den in (("TPR", "tp", "TPR_den"), ("FPR", "fp", "FPR_den"), ("FOR", "fn", "FOR_den")):
        v = t[t[den] > 0]
        out[m + "_gap_raw"] = v[m].max() - v[m].min()
        vv = v[v[den] >= min_cell]
        out[m + "_gap_mincell"] = (vv[m].max() - vv[m].min()) if len(vv) > 1 else np.nan
        out[m + "_cells_excluded"] = int((v[den] < min_cell).sum())
        a, b = fit_beta_prior(v[num], v[den])
        post = (v[num] + a) / (v[den] + a + b)
        out[m + "_gap_eb"] = post.max() - post.min()
    return out


def bootstrap_gaps(y, pred, g, min_cell, B, seed):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"y": y, "pred": pred, "g": g})
    idx = {k: s.index.values for k, s in d.groupby("g")}
    res = []
    for _ in range(B):
        take = np.concatenate([rng.choice(v, len(v), replace=True) for v in idx.values()])
        s = d.loc[take]
        res.append(gaps(subgroup_counts(s.y.values, s.pred.values, s.g.values), min_cell))
    return pd.DataFrame(res)


def audit_block(files, out, min_cell, B, seed, lines):
    preds = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    for model, s in preds.groupby("model"):
        t = subgroup_counts(s.y.values, s.pred.values, s.group.values)
        tag = str(model).replace(" ", "_")
        t.round(4).to_csv(os.path.join(out, f"audit_subgroup_counts_{tag}.csv"), index=False)
        point = gaps(t, min_cell)
        bs = bootstrap_gaps(s.y.values, s.pred.values, s.group.values, min_cell, B, seed)
        rows = []
        for k, v in point.items():
            if k.endswith("excluded"):
                rows.append(dict(metric=k, estimate=v, ci_lo=np.nan, ci_hi=np.nan)); continue
            col = bs[k].dropna()
            q_lo, q_hi = col.quantile(.025), col.quantile(.975)
            rows.append(dict(metric=k, estimate=round(v, 4),
                             ci_lo=round(q_lo, 4), ci_hi=round(q_hi, 4),
                             boot_bias=round(col.mean() - v, 4),
                             basic_lo=round(max(0.0, 2 * v - q_hi), 4), basic_hi=round(min(1.0, 2 * v - q_lo), 4)))
        pd.DataFrame(rows).to_csv(os.path.join(out, f"audit_gap_ci_{tag}.csv"), index=False)
        lines.append(f"\n[t=0 audit: {model}] test n={len(s)}, events={int(s.y.sum())}")
        lines.append(f"  smallest subgroup events: {t.events.min()} ({t.loc[t.events.idxmin(), 'group']}); "
                     f"subgroups with <{min_cell} events: {int((t.events < min_cell).sum())}/16")
        for r in rows:
            if "excluded" in r["metric"]:
                lines.append(f"  {r['metric']}: {r['estimate']}")
            else:
                lines.append(f"  {r['metric']}: {r['estimate']}  percentile CI [{r['ci_lo']}, {r['ci_hi']}]  "
                             f"basic CI [{r['basic_lo']}, {r['basic_hi']}]  bootstrap bias {r['boot_bias']:+}")


# ------------------------------------------------------------ eICU hospitals
def eicu_hospital_block(path, out, min_cell, min_hosp_n, lines):
    d = pd.read_csv(path)
    a_tpr = b_tpr = None
    cells = []
    for h, s in d.groupby("hospitalid"):
        if len(s) < min_hosp_n:
            continue
        t = subgroup_counts(s.y.values, s.pred.values, s.group.values)
        t["hospitalid"], t["hosp_n"] = h, len(s)
        cells.append(t)
    cells = pd.concat(cells, ignore_index=True)
    a_tpr, b_tpr = fit_beta_prior(cells.tp, cells.TPR_den)
    a_for, b_for = fit_beta_prior(cells.fn, cells.FOR_den)
    rows = []
    for h, t in cells.groupby("hospitalid"):
        v = t[t.TPR_den > 0]; vf = t[t.FOR_den > 0]
        r = dict(hospitalid=h, n=int(t.hosp_n.iloc[0]), events=int(t.events.sum()),
                 cells_tpr_ge_min=int((v.TPR_den >= min_cell).sum()),
                 tpr_gap_raw=v.TPR.max() - v.TPR.min() if len(v) > 1 else np.nan,
                 for_gap_raw=vf.FOR.max() - vf.FOR.min() if len(vf) > 1 else np.nan)
        vv = v[v.TPR_den >= min_cell]
        r["tpr_gap_mincell"] = vv.TPR.max() - vv.TPR.min() if len(vv) > 1 else np.nan
        pt = (v.tp + a_tpr) / (v.TPR_den + a_tpr + b_tpr)
        pf = (vf.fn + a_for) / (vf.FOR_den + a_for + b_for)
        r["tpr_gap_eb"] = pt.max() - pt.min() if len(pt) > 1 else np.nan
        r["for_gap_eb"] = pf.max() - pf.min() if len(pf) > 1 else np.nan
        rows.append(r)
    H = pd.DataFrame(rows)
    H["size_quartile"] = pd.qcut(H.n, 4, labels=["Q1", "Q2", "Q3", "Q4"])
    H.round(4).to_csv(os.path.join(out, "eicu_hospital_gaps_smallsample.csv"), index=False)
    Q = H.groupby("size_quartile", observed=True).agg(
        hospitals=("n", "size"), median_n=("n", "median"), median_events=("events", "median"),
        tpr_raw=("tpr_gap_raw", "median"), tpr_eb=("tpr_gap_eb", "median"),
        tpr_mincell=("tpr_gap_mincell", "median"), hosp_with_2plus_eligible_cells=("cells_tpr_ge_min", lambda x: int((x >= 2).sum())),
        for_raw=("for_gap_raw", "median"), for_eb=("for_gap_eb", "median")).round(3)
    Q.to_csv(os.path.join(out, "eicu_quartiles_smallsample.csv"))
    lines.append(f"\n[eICU per-hospital] hospitals={len(H)}, stays={H.n.sum()}, events={H.events.sum()}")
    lines.append(f"  hospitals with >=2 subgroups having >={min_cell} events: {int((H.cells_tpr_ge_min >= 2).sum())}")
    for c in ("tpr_gap_raw", "tpr_gap_eb", "for_gap_raw", "for_gap_eb"):
        s = H[[c, "n"]].dropna()
        rho, p = spearmanr(s.n, s[c])
        lines.append(f"  Spearman(n, {c}) rho={rho:.3f} p={p:.4f}")
    lines.append(Q.to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic"); ap.add_argument("--audit", nargs="*", default=[])
    ap.add_argument("--eicu"); ap.add_argument("--eicu-preds")
    ap.add_argument("--predictors", default=None)
    ap.add_argument("--min-cell", type=int, default=10)
    ap.add_argument("--min-hosp-n", type=int, default=30)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="r3_out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    lines = []
    if a.mimic:
        m = pd.read_csv(a.mimic); p = predictors(m, a.predictors)
        o = missingness_tables(m, p, "mimic", a.out)
        lines.append(f"[MIMIC-IV missingness] n={len(m)}\n" + o.to_string(index=False))
        g = m.groupby("intersectional_group").mortality_in_hospital.agg(n="size", deaths="sum")
        g[["lo", "hi"]] = [wilson(k, n) for k, n in zip(g.deaths, g.n)]
        g.round(4).to_csv(os.path.join(a.out, "mimic_subgroup_mortality_ci.csv"))
    files = [f for pat in a.audit for f in glob.glob(pat)]
    if files:
        audit_block(files, a.out, a.min_cell, a.boot, a.seed, lines)
    if a.eicu:
        e = pd.read_csv(a.eicu); p = predictors(e, a.predictors)
        o = missingness_tables(e, p, "eicu", a.out, hosp_col="hospitalid")
        lines.append(f"\n[eICU missingness] n={len(e)}, hospitals={e.hospitalid.nunique()}\n" + o.to_string(index=False))
    if a.eicu_preds:
        eicu_hospital_block(a.eicu_preds, a.out, a.min_cell, a.min_hosp_n, lines)
    open(os.path.join(a.out, "summary.txt"), "w").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
