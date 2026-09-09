#!/usr/bin/env python3
"""Read a kagome_ed output folder (analysis side, not needed on the target machine).

    python3 read_results.py output/smoke_YYYYMMDD_HHMMSS

Loads smoke.json / estimate.json / sysinfo.json and every results_*/ folder
(per-(sector, r) JSON + the W_*.npy Lehmann matrices) and prints a summary.
"""
import json, sys, glob, os
import numpy as np


def load_run(folder):
    run = {"dir": folder}
    for name in ("sysinfo", "smoke", "estimate"):
        p = os.path.join(folder, name + ".json")
        if os.path.exists(p):
            run[name] = json.load(open(p))
    run["results"] = {}
    for rd in sorted(glob.glob(os.path.join(folder, "results_*"))):
        entries = []
        for jp in sorted(glob.glob(os.path.join(rd, "*.json"))):
            e = json.load(open(jp))
            tag = os.path.basename(jp)[:-5]
            for kname, Gk in e["G"].items():
                for a in e["aset"]:
                    for b in e["bset"]:
                        wp = os.path.join(rd, f"{tag}_{kname}_W_a{a}_b{b}.npy")
                        if os.path.exists(wp):
                            Gk[f"W_a{a}_b{b}"] = np.load(wp)
            entries.append(e)
        run["results"][os.path.basename(rd)] = entries
    return run


def combine_canonical(entries, beta, N):
    """Ensemble average over sectors with the given N, jackknife over r. Returns dict k -> G(tau) (3,3,ntau)."""
    ents = [e for e in entries if e["N"] == N]
    rs = sorted({e["r"] for e in ents})
    shift = max(e["lnZ_full"] for e in ents)
    def est(sel):
        Z = 0.0; G = {}
        for e in ents:
            if e["r"] not in sel:
                continue
            w = e["weight"] * np.exp(e["lnZ_full"] - shift)
            Z += w
            for k, Gk in e["G"].items():
                for a in e["aset"]:
                    for b in e["bset"]:
                        key = f"g_tau_a{a}_b{b}"
                        if key in Gk:
                            G.setdefault(k, np.zeros((3, 3, len(e["tau"]))))[a-1, b-1] += w * np.array(Gk[key])
        return {k: v / Z for k, v in G.items()}
    full = est(rs)
    jk = [est([r for r in rs if r != r0]) for r0 in rs]
    R = len(rs)
    err = {k: np.sqrt((R - 1) / R * sum((j[k] - full[k]) ** 2 for j in jk)) for k in full} if R > 1 else None
    return full, err


if __name__ == "__main__":
    run = load_run(sys.argv[1])
    for name in ("sysinfo", "smoke", "estimate"):
        if name in run:
            print(f"== {name}.json")
            for k, v in run[name].items():
                if isinstance(v, (int, float, str, bool)) or v is None:
                    print(f"   {k}: {v}")
                else:
                    print(f"   {k}: {json.dumps(v)[:120]}")
    for rd, entries in run["results"].items():
        print(f"== {rd}: {len(entries)} entries; sectors {sorted({(e['nup'], e['ndn']) for e in entries})}")
        Ns = sorted({e["N"] for e in entries})
        beta = entries[0]["omega_n"][0] and np.pi / entries[0]["omega_n"][0]
        for N in Ns:
            G, err = combine_canonical(entries, beta, N)
            for k, g in G.items():
                anti = [g[a, a, 0] + g[a, a, -1] for a in range(3)]
                print(f"   N={N} k={k}: G_aa(0)+G_aa(beta) = {np.round(anti, 4)}")
