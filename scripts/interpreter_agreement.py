"""Inter-interpreter agreement on the second-interpreter subset (validation_protocol.md §4 step 4).

Compares the second interpreter's `second_interpreter/<SITE>_labels_2nd_filled.csv` with the primary
labels of a frozen version (`data/ground_truth/frozen/<version>/validation_points.csv`). Reports Cohen's κ
for the 4 classes and for water vs non-water (the primary endpoint), per date and per site, plus
agreement on the water transition (gain / loss / stable). Point-dates where either interpreter said NA
are left out.

Writes `second_interpreter/adjudication.csv`: every point with a disagreement at either date, both
labels, confidences, sources and notes, and empty `agreed_t1/agreed_t2/resolution` columns to fill in
during the discussion. It holds no strata or automatic labels (adjudication stays blind).

Usage: .venv/bin/python scripts/interpreter_agreement.py --version v1
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parent.parent
SECOND = ROOT / "data/ground_truth/full/second_interpreter"
FIELDS = ["label", "confidence", "mixed", "ref_source", "ref_date"]


def water(s):
    return s.eq("Water").map({True: "Water", False: "Non-water"})


def change(a, b):
    wa, wb = a.eq("Water"), b.eq("Water")
    return pd.Series("stable", index=a.index).mask(wa & ~wb, "loss").mask(~wa & wb, "gain")


def kappa(a, b):
    return cohen_kappa_score(a, b) if len(set(a) | set(b)) > 1 else float("nan")


def report(name, a, b):
    agree = (a == b).mean()
    print(f"  {name:28s} n={len(a):3d}  agreement {agree:6.1%}  κ {kappa(a, b):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1")
    args = ap.parse_args()

    v = pd.read_csv(ROOT / "data/ground_truth/frozen" / args.version / "validation_points.csv",
                    dtype=str, keep_default_na=False)
    second = pd.concat([pd.read_csv(p, dtype=str, keep_default_na=False)
                        for p in sorted(SECOND.glob("*_labels_2nd_filled.csv"))], ignore_index=True)
    m = second.merge(v, on="point_id", suffixes=("_B", "_A"))
    assert len(m) == len(second), "second-interpreter point_id missing from the frozen version"
    ia, ib = m.interpreter_A.iloc[0], m.interpreter_B.iloc[0]
    print(f"{len(m)} points, interpreter A = {ia} ({args.version}), B = {ib}\n")

    rows = []
    for t in ("t1", "t2"):
        rows.append(pd.DataFrame({"site": m.site_A, "date": t,
                                  "A": m[f"label_{t}_A"], "B": m[f"label_{t}_B"]}))
    long = pd.concat(rows, ignore_index=True)
    na = (long.A == "NA") | (long.B == "NA")
    print(f"Point-dates: {len(long)}, left out for NA: {na.sum()} (A {(long.A == 'NA').sum()}, "
          f"B {(long.B == 'NA').sum()})\n")
    long = long[~na]

    print("== Water vs non-water (primary endpoint)")
    report("all", water(long.A), water(long.B))
    for t, g in long.groupby("date"):
        report(f"date {t}", water(g.A), water(g.B))
    for s, g in long.groupby("site"):
        report(s, water(g.A), water(g.B))

    print("\n== 4 classes")
    report("all", long.A, long.B)
    for s, g in long.groupby("site"):
        report(s, g.A, g.B)
    print(pd.crosstab(long.A, long.B, rownames=[f"A ({ia})"], colnames=[f"B ({ib})"]).to_string(), "\n")

    ok = ~((m.label_t1_A == "NA") | (m.label_t2_A == "NA") | (m.label_t1_B == "NA") | (m.label_t2_B == "NA"))
    ca = change(m.label_t1_A[ok], m.label_t2_A[ok])
    cb = change(m.label_t1_B[ok], m.label_t2_B[ok])
    print("== Water transition per point (gain / loss / stable)")
    report("all", ca, cb)
    print(pd.crosstab(ca, cb, rownames=[f"A ({ia})"], colnames=[f"B ({ib})"]).to_string(), "\n")

    dis = m[(m.label_t1_A != m.label_t1_B) | (m.label_t2_A != m.label_t2_B)]
    cols = {"point_id": dis.point_id, "site": dis.site_A, "lon": dis.lon_A, "lat": dis.lat_A}
    for t in ("t1", "t2"):
        cols[f"date_{t}"] = dis[f"date_{t}_A"]
        for who in ("A", "B"):
            for f in FIELDS:
                cols[f"{f}_{t}_{who}"] = dis[f"{f}_{t}_{who}"]
        cols[f"water_disagree_{t}"] = (water(dis[f"label_{t}_A"]) != water(dis[f"label_{t}_B"])) & \
            (dis[f"label_{t}_A"] != "NA") & (dis[f"label_{t}_B"] != "NA")
    cols["notes_A"], cols["notes_B"] = dis.notes_A, dis.notes_B
    out = pd.DataFrame(cols)
    out["agreed_t1"] = out["agreed_t2"] = out["resolution"] = ""
    out = out.sort_values(["water_disagree_t1", "water_disagree_t2", "point_id"],
                          ascending=[False, False, True])
    path = SECOND / "adjudication.csv"
    out.to_csv(path, index=False)
    nw = (out.water_disagree_t1 | out.water_disagree_t2).sum()
    print(f"Disagreements: {len(out)} points ({nw} on water vs non-water) → {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
