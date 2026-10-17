"""Check a filled labeling sheet and compare it with the automatic labels (key file).

Usage: .venv/bin/python scripts/pilot_qc.py data/ground_truth/pilot/BHA_labels_filled.csv \
           data/ground_truth/pilot/BHA_key.csv
Writes <sheet>_clean.csv (whitespace/case normalised) and prints QC issues and pilot statistics.
"""

import argparse
from pathlib import Path

import pandas as pd

CLASSES = {"Water", "Forest", "Bare", "Other", "NA"}
SOURCES = {"GE", "S2", "LS"}
WINDOW_DAYS = 15


def clean(df):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip().replace({"nan": ""})
    for t in ("t1", "t2"):
        df[f"label_{t}"] = df[f"label_{t}"].str.capitalize().replace({"Na": "NA"})
        df[f"ref_source_{t}"] = df[f"ref_source_{t}"].str.upper()
        df[f"mixed_{t}"] = df[f"mixed_{t}"].str.upper()
    return df


def qc(df):
    issues = []
    for _, r in df.iterrows():
        pid = r["point_id"]
        for t in ("t1", "t2"):
            lab, conf, mix = r[f"label_{t}"], r[f"confidence_{t}"], r[f"mixed_{t}"]
            src, rdate = r[f"ref_source_{t}"], r[f"ref_date_{t}"]
            if lab not in CLASSES:
                issues.append((pid, t, f"label '{lab}' not one of {sorted(CLASSES)}"))
            if src not in SOURCES and lab != "NA":
                issues.append((pid, t, f"ref_source '{src}' not GE/S2/LS"))
            if mix not in {"TRUE", "FALSE"}:
                issues.append((pid, t, f"mixed '{mix}' not TRUE/FALSE"))
            try:
                conf = int(float(conf))
            except ValueError:
                issues.append((pid, t, f"confidence '{conf}' not a number"))
                continue
            if mix == "TRUE" and conf > 2:
                issues.append((pid, t, f"mixed=TRUE but confidence {conf} (rule: ≤ 2)"))
            if src in {"S2", "LS"} and conf > 2:
                issues.append((pid, t, f"{src} source but confidence {conf} (rule: ≤ 2)"))
            if rdate:
                off = (pd.Timestamp(rdate) - pd.Timestamp(r[f"date_{t}"])).days
                if abs(off) > WINDOW_DAYS:
                    issues.append((pid, t, f"ref_date {rdate} is {off:+d} days from {r[f'date_{t}']} (window ±{WINDOW_DAYS})"))
        if not r["interpreter"]:
            issues.append((pid, "-", "interpreter blank"))
    return issues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sheet", type=Path)
    ap.add_argument("key", type=Path)
    args = ap.parse_args()

    df = clean(pd.read_csv(args.sheet, dtype=str, keep_default_na=False))
    out = args.sheet.with_name(args.sheet.stem + "_clean.csv")
    df.to_csv(out, index=False)
    print(f"Clean sheet: {out}\n")

    issues = qc(df)
    print(f"== QC: {len(issues)} issue(s)")
    for i in issues:
        print("  ", *i)

    key = pd.read_csv(args.key, dtype=str)
    m = df.merge(key, on="point_id", how="inner")
    assert len(m) == len(df), "point_id mismatch between sheet and key"
    for t in ("t1", "t2"):
        m[f"confidence_{t}"] = m[f"confidence_{t}"].astype(int)
    m["minutes"] = pd.to_numeric(m["minutes"], errors="coerce")

    print("\n== Effort and imagery")
    print(f"  minutes/point: mean {m.minutes.mean():.1f}, median {m.minutes.median():.0f}, "
          f"total {m.minutes.sum():.0f} (first 10: {m.minutes.head(10).mean():.1f}, last 10: {m.minutes.tail(10).mean():.1f})")
    src = pd.concat([m.ref_source_t1.rename("s"), m.ref_source_t2.rename("s")])
    print("  reference source:", src.value_counts().to_dict())
    conf = pd.concat([m.confidence_t1, m.confidence_t2])
    print("  confidence:", conf.value_counts().sort_index().to_dict())
    mix = pd.concat([m.mixed_t1, m.mixed_t2])
    print("  mixed TRUE share:", f"{(mix == 'TRUE').mean():.0%}")

    print("\n== Agreement of automatic labels with the interpreter (per date)")
    for kind in ("rule", "consensus"):
        rows = []
        for t in ("t1", "t2"):
            sub = m[[f"label_{t}", f"{kind}_label_{t}", f"mixed_{t}", f"confidence_{t}"]].copy()
            sub.columns = ["human", "auto", "mixed", "conf"]
            rows.append(sub)
        a = pd.concat(rows)
        a = a[a.human != "NA"]
        lab = a[a.auto != "Unlabeled"]
        print(f"  {kind}: {len(lab)}/{len(a)} labelled, agreement {(lab.human == lab.auto).mean():.0%}; "
              f"pure pixels {(lab[lab.mixed == 'FALSE'].human == lab[lab.mixed == 'FALSE'].auto).mean():.0%}, "
              f"mixed pixels {(lab[lab.mixed == 'TRUE'].human == lab[lab.mixed == 'TRUE'].auto).mean():.0%}")
        print(pd.crosstab(lab.auto, lab.human, rownames=[f"{kind} (auto)"], colnames=["human"]).to_string(), "\n")

    print("== Per stratum (n, human-confirmed transition, mean confidence, mixed share, minutes)")
    def human_stratum(r):
        a, b = r.label_t1, r.label_t2
        if a == "Water" and b != "Water":
            return "loss"
        if a != "Water" and b == "Water":
            return "gain"
        if a != b:
            return "land change"
        return "stable"
    m["human_change"] = m.apply(human_stratum, axis=1)
    expect = {"S6": "loss", "S7": "gain", "S8": "land change"}
    for s, g in m.groupby("stratum"):
        conf = (g.confidence_t1 + g.confidence_t2).mean() / 2
        mixs = ((g.mixed_t1 == "TRUE") | (g.mixed_t2 == "TRUE")).mean()
        if s in expect:
            ok = (g.human_change == expect[s]).mean()
            note = f"confirmed {expect[s]}: {ok:.0%}"
        else:
            ok = (g.human_change == "stable").mean()
            note = f"human stable: {ok:.0%}"
        print(f"  {s}: n={len(g):2d}  {note:28s} conf {conf:.1f}  mixed {mixs:.0%}  min {g.minutes.mean():.1f}  "
              f"human transitions {g.human_change.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
