# ED on the roommate's 256 GB machine — plan (2026-09-08)

**Protocol** (roommate's terms): I publish a git repo, he clones it, runs it, sends
the results back. No SSH, no interactive access, no iteration loop, unknown patience.
**Hardware**: 8 cores / 16 threads, 4×64 GB = 256 GB RAM, RTX 5070 (12 GB VRAM), Windows.
**Scope (revised 2026-09-08 evening)**: **upper (p-type) vHs only, U = 1, 2, 3** — the
physical CsV₃Sb₅ range (U/t ≈ 1.5–3). The lower vHs is dropped from the ED programme.
**Execution policy**: **single core by default** (`-t 1`); threads are opt-in
(`run.ps1 <mode> <threads>`). Reason: the user prefers single-core processes on a
shared machine (no CPU/memory contention, no thread-spawn system time); the kernels
are memory-bound so the loss is ~2–4×, not 8×. Batch 0 measures it.

The protocol, not the hardware, sets the engineering rules:

1. Everything that can be tested on the laptop (18 GB) *is* tested there first; the
   roommate runs only what does not fit in 18 GB.
2. One-shot robustness: `smoke` (2 min) and `estimate` (10 min) modes before any long
   run; resume-safe checkpointing at the granularity of one random vector; every
   finished (U, sector, r) is a usable result on its own; partial batches are useful.
3. The repo is standalone, contains no DQMC data, no credentials and **no binaries**:
   `setup.ps1` downloads the official portable Julia zip from julialang.org at setup
   time, verifies its SHA256 against the official checksum list, and unpacks it inside
   the folder (nothing is installed system-wide; deleting the folder removes everything).
   Results are a few MB of G(k,τ) and come back as a zip of `output/`.
4. The GPU is not used: the workload is memory-bandwidth-bound, a single 18-site
   state vector (11–19 GB) does not fit in 12 GB VRAM, and CUDA cannot be tested on
   the Mac — an untested kernel in a one-shot protocol is a guaranteed failure.

---

## 1. Scientific targets

| id | deliverable | feeds | needs the big machine? |
|---|---|---|---|
| D1 | Same-cluster ED-vs-DQMC at the upper vHs on 2×2 (grand canonical, laptop) and 3×2 (canonical in batch 1, grand canonical in batch 2) at β=3, U = 1, 2, 3: G_ab(k,τ), Σ(k,iωₙ), Z_M. Gives the Δτ=0.1 Trotter bias on Z_M as a measured number (today inferred from ⟨n↑n↓⟩ only) and the ensemble effect. | headline Z table caveats; `build_sigma_iwn.py` chain | 3×2 only |
| D2 | FT-bias check with a real kagome G(τ): ED on Δτ=0.05 vs the spline-FT chain fed the Δτ=0.1 subset. | `validate_ft.py`, `ft_bias.json` | no (12-site) / yes (18-site) |
| D3 | Exact real-axis Σ_ab(M,ω) and Σ(Γ,ω) at β=3 on the 12-site cluster from the Lanczos cross matrix (no continuation): the MaxEnt-on-Σ round trip on a real kagome cluster at the real filling and U. Extends `Sigma_AC_validation` test D (6-site ring). | `Sigma_AC_validation/` | no |
| D4 | Cluster-range series 12 → 18 sites for Z_M(U) at M, U = 1, 2, 3, between DMFT (range 0) and DQMC (L=4–6). | nonlocality claim (`dmft/`) | yes |

Cluster geometry (fixed by filling and symmetry), upper vHs n = 5/6:
- **2×2 (12 sites)**: Γ + all three M, full C6v. N = 10 e (5,5 + other Sz).
- **3×2 (18 sites)**: Γ + one M (M2 = (0,½)), no C6v. N = 15 e (8,7 + all other Sz).
  No 6-cell cluster contains all three M (needs 4 | N_cells).
- **4×2 (24 sites)**: N = 20 e, 3.9×10¹² states → out of reach anywhere. **Not in this plan.**
- Wrap bonds: with L = 1 in a direction a wrap lands on an existing pair and the
  amplitude doubles (stage-2.4 finding, the 2×1 case); with L ≥ 2 all 6·N_cells pairs are
  distinct — *no* doubling on 2×2 or 3×2 (corrected 2026-09-08; the earlier draft said
  otherwise). Either way K is the Bloch-folded ε(k) on the cluster k-grid, which the
  code asserts (`check_bloch`, smoke step S1) before anything else.

Hamiltonian exactly as DQMC: H = −t Σ⟨ij⟩σ (c†c + h.c.) + U Σ (n↑−½)(n↓−½) − μN, t=1,
SmoQy site ordering (orb + 3·cell).

---

## 2. Method

**Basis**: product basis (N↑, N↓) sectors, index = i↑ + D↑·i↓, Lin tables (2¹⁸ lookup).
**No translation symmetry.** Reason: with the product basis H·v is two sparse×dense
products, A↑V + (A↓Vᵀ)ᵀ + diag(U n↑n↓)∘V, streamed columnwise with a blocked transpose,
≈ 5–8 vector-sized memory passes per H·v. A symmetry-reduced basis needs a hash /
representative lookup per matrix element and is 30–100× slower per state. The 256 GB
is what makes the *unreduced* vectors affordable; that is the whole point of the machine.

**Real arithmetic** for untwisted runs: at Γ and M the Bloch phases are ±1, so H, c_k
and all vectors are real. Twists (needed only for the mid-FS class) make everything
complex and double memory — batch 3 at the earliest.

**Finite T by typicality** (canonical per sector, R random Gaussian vectors, ‖r‖=1):

    Z_sec ≈ D_sec ⟨r|e^{−βH}|r⟩
    G_ab(k,τ) ≈ ⟨r| e^{−(β−τ)H} c_{ka} e^{−τH} c†_{kb} |r⟩ · D_sec / Z      (0 ≤ τ ≤ β)

Grand canonical = Σ_N e^{βμN} × canonical traces. μ enters only in post-processing, so
it is tuned *after* the run (⟨N⟩(μ) from the stored Z_N); the run must simply cover
enough N sectors (N = 11…19 around ⟨N⟩ = 15; σ_N ≈ 1.5–2 at β=3).
Spin-averaged G needs c_↑ **and** c_↓ in every Sz ≥ 0 sector (Sz < 0 by spin flip).
Sublattice structure: at M the little group splits {a₀} ⊕ {a₁,a₂} — propagations b = a₀
and b = a₁ suffice; at Γ one propagation (C6v). Also the exact 3×3 matrix at every τ
gives the crossing-band projection the production chain uses.

**Imaginary-time propagation**: Lanczos without reorthogonalisation on r and on each
φ_b = c†_{kb} r, adaptive step count M (stop when β_M |(e^{−τT})_{M1}| / (e^{−τT})_{11} < 10⁻⁹
for τ ∈ {β/4, β/2, 3β/4, β}). The recurrence is deterministic (static partitions, fixed
summation order), so Krylov vectors are **regenerated by replay** instead of stored.
The cross matrix C_ij = ⟨q_i| c_{ka} |p_j⟩ is built in q-batches: n_q Krylov vectors of r
are regenerated and kept in Float32 (n_q from the memory budget), then for every (k, b)
the p-recurrence is replayed and each p_j is dotted (after c_{ka}) against the batch.
With the two Ritz decompositions, W = nφ · diag(U1_q)(U_qᵀ C U_p) diag(U1_p) gives
G(τ) at any τ, exact 𝒢(iωₙ) by the Lehmann formula, and the real axis — all from M×M
algebra. μ enters only here: g_GC(τ) = g(τ)e^{μτ}, and the Lehmann numerators/denominators
carry e^{βμ} and iω + μ (both implemented; smoke S2b checks them against free fermions).
Per (sector, r) the cost is M·(1 + 6) Lanczos steps for pass 1 plus ⌈M/n_q⌉ replays of
the same, plus the block dot products; memory ≈ 4 n-vectors + 4 (n+1)-vectors + 3 work
vectors + n_q·dim·4 B.

**τ grid**: Δτ = 0.05, 61 points, superset of the DQMC grid (31). Sanity identities per
sample: G_aa(0⁺) + G_aa(β⁻) = 1 (anticommutator), Hermiticity of G_ab, Σ_k … .

**Random vectors**: adaptive R; stop when the jackknife error of Z_M at M < 1×10⁻³ or
R = R_max (8). Each r is a checkpoint unit (result file written atomically; a killed
run loses at most one r ≈ hours). At T = t/3 the typicality variance is ~e^{−S}, S ≈
10–16 on 18 sites, so R = 3–5 should suffice; the 12-site runs use R = 200.

---

## 3. Cost on the roommate's machine (to be measured by batch 0)

Vector sizes (real, 8 B): (8,7) = 1.39×10⁹ → 11.1 GB; (9,7) 12.4; (8,8) 15.2; (9,9) 18.9.
Per (r, U) at N=15 the work is: pass 1 on r (M steps) + 6 pass-1 runs on c†_{kb}r (M each,
2 k × 3 b) + the cross matrices, which cost ⌈M/n_q⌉ replays of everything with n_q the
number of Float32 Krylov vectors that fit the budget (≈ 13 at 200 GB for (8,7)).
Every term scales with the sector dimension, so `estimate` measures the per-state cost
of each kernel on the real (8,7)/(9,7) pair and evaluates this model for the N=15 sector
list; the smoke test on the laptop supplies the Lanczos step count M actually needed at
β=3 (6-site: 40; 12-site: see `smoke.json`).

**Measured on the laptop (Apple M3 Pro, one core, `estimate --avail 9`, sector (5,4)/(6,4)):**
H·v 14.5 ns per state, dot/update 0.2 ns per element, fused c-application + block dot
3.5 ns per (n+1)-state + 0.31 ns per n-state per q-vector. ETA model with M = 60, R = 4,
200 GB budget (n_q ≈ 19 Float32 q-vectors at (8,7)), all nine G_ab components:

| k-points | (8,7) sector per r | canonical N=15 per U | batch 1 (U = 1, 2, 3) |
|---|---|---|---|
| M2 only | 8.1 h | 134 h | **16.8 days** single core |
| M2 + Γ | 15.1 h | 253 h | 31.7 days single core |

The (8,7)+(7,8)-type sectors dominate; the cross-matrix stage is ~80 % of the time.
Threads would cut this by roughly 4–5× (the kernels are bandwidth/gather bound; the
threaded path is deterministic and validated by the same smoke test) — that is the
decision to take with the batch-0 numbers from *his* machine in hand. Other levers, in
order: R = 3 (−25 %), Lanczos tol 10⁻⁷ instead of 10⁻⁹ (M ≈ 50, −30 %), Float16 storage
of the q-vectors (n_q ×2, −35 % on the cross stage; needs a scaling factor because
normalised-vector components sit in Float16's subnormal range).

| run | sectors (states) | peak RAM | batch |
|---|---|---|---|
| 12-site, anything | ≤ 8.5×10⁵ | < 1 GB | laptop, minutes |
| 18-site canonical N=15, U = 1, 2, 3 | Σ ≈ 2.8×10⁹, max (8,7) | ~130–180 GB | **batch 1** |
| 18-site grand canonical N=11…19, U = 1, 2, 3 | Σ ≈ 2.5×10¹⁰, max (9,9) | ~200 GB | batch 2 |
| 18-site, one twist (complex) | ×2 memory | ~250 GB | batch 3 only with a reduced τ grid |

If batch 0 says batch 1 exceeds ~1 week single-core, the levers are, in order: R = 3,
only k = M (drop Γ), and threads.

---

## 4. Code

**Julia 1.12.6, standard library only** (no packages, no registry, no network after
setup): `setup.ps1` downloads the official portable zip from julialang.org into
`./julia`, checksum-verified. Output is `.json` + `.npy` (own writers), readable from
numpy. `-t 1` by default; the kernels have a threaded path (static partitions,
deterministic reductions, so replays are bit-identical at any thread count) used only
when a thread count is passed explicitly.

Repo `kagome_ed/` (this folder becomes its own git repo, private GitHub, roommate as
collaborator):

    run.jl              julia -t 1 run.jl <mode> [--config f] [--mem GB]   (stdlib only)
    src/lattice.jl      kagome torus, bond list = SmoQy order, wrap accumulation, ε(k) check
    src/basis.jl        one-species bases, lookup tables, hopping matrices (CSC), c†_{ka} maps
    src/kernels.jl      H·v (two sparse×dense passes + diagonal), dots, block dots, RNG fill
    src/lanczos.jl      adaptive Lanczos for e^{−τH}v, deterministic replay
    src/typicality.jl   per-(sector, r) run: Ritz data, cross matrices, W, g(τ), 𝒢(iωₙ)
    src/exact.jl        dense references (same r / trace), free-fermion G
    src/combine.jl      ensemble combination (canonical/GC, μ at analysis time), jackknife, Z_M
    src/io.jl           .npy/.json writers, output folder + tee log
    src/driver.jl       modes smoke / estimate (batch1 to be added), CLI, config
    configs/{smoke,estimate}.toml
    setup.ps1 run.ps1 batch0.bat README.md .gitignore
    output/<mode>_<timestamp>/   log.txt, config.toml, sysinfo.json, smoke.json | estimate.json, results_*/

Modes now: `smoke` (S1 Bloch check; S2 6-site pipeline vs dense exponentials on the same
random vector; S2b U=0 grand-canonical exact trace vs free-fermion formula; S2c 6-site
typicality vs exact trace; S3 12-site small sectors vs dense at all M points; S4 12-site
N=10, U=3, k = M1 and Γ through the production path) · `estimate` (times every kernel on
(8,7)/(9,7), touches memtest_GB, evaluates the ETA model) · `batch1` (next).

---

## 5. Validation ladder on the laptop (before anything is shipped)

- T1 6-site 2×1 torus: G(τ), n, docc vs `U_V_DQMC/benchmarks/stage2/ed_cluster.py` full diag, to 1×10⁻¹⁰ (typicality with R → large, and the cross-matrix route).
- T2 U=0 at 12 and 18 sites: G(k,τ) analytic from ε(k) on the cluster grid — checks bond list, doubled wraps, phases, sublattice labels, particle/hole parts.
- T3 12-site convergence: R (20/50/200), Lanczos M (25/30/40), τ batch size (1/4/all) all agree to < 1×10⁻⁴; SOPT at U = 0.5 agrees to O(U³).
- T4 12-site, GC vs canonical, twist = 0: full D1 pipeline end-to-end against a SmoQy 2×2 run (`hubbard_kagome_UV.jl … Lx=2 Ly=2`) including `collect.py` → `build_sigma_iwn.py` → Z_M. This proves the comparison machinery before any 18-site number exists.
- T5 **dress rehearsal**: 18-site canonical N=15 restricted to the small-Sz sectors that fit 18 GB (e.g. (11,4), (12,3)) at U = 2 on the laptop. Same code path as the roommate run; calibrates the H·v model per state.
- T6 memory-path test at (8,7) dimension with Float32 or a single τ-pair to exercise Int64 indexing and the batch sizing at scale (cannot hold 3 × 11 GB on the laptop; verify one H·v against the same operator on a smaller sector via a consistency identity).

---

## 6. Batches for the roommate

- **Batch 0** (≈ 15 min): `batch0.bat` = setup + `smoke` + `estimate`; he zips `output/`.
  I size batch 1 from `estimate.json` (ETA model) and `smoke.json` (M actually needed).
- **Batch 1**: 18-site canonical N=15, U = 1, 2, 3, k = M2 and Γ, all 3×3 sublattice
  components, R = 4 (adaptive). Order: U=2 first, then 1, then 3 — a partial batch is
  still one complete U.
- **Batch 2** (after batch 1 validates): 18-site grand canonical N = 11…19, U = 1, 2, 3
  → the apples-to-apples comparison with a 3×2 SmoQy run.
- **Batch 3** (optional): one twist for the mid-FS class with a 16-point τ grid;
  real-axis Σ at 18 sites straight from the stored (ε, ε′, W).

Same-cluster SmoQy runs (2×2 and 3×2, β=3, Δτ=0.1 and 0.05, U = 1, 2, 3, μ tuned to
⟨n⟩ = 5/6) run on the laptop in parallel; minutes each.

---

## 7. Roommate instructions

See `README.md` (Chinese quick-start + English). Batch 0 is one double-click:
`batch0.bat` → downloads Julia (official zip, checksum-verified) into the folder, runs
`smoke` (~3 min) and `estimate` (~10 min), then asks him to zip `output/`.
Manual equivalent:

    powershell -NoProfile -ExecutionPolicy Bypass -File setup.ps1
    powershell -NoProfile -ExecutionPolicy Bypass -File run.ps1 smoke 1
    powershell -NoProfile -ExecutionPolicy Bypass -File run.ps1 estimate 1

Single core, BelowNormal priority; nothing else on his machine is touched. The batch-1
run will be the same `run.ps1 batch1 1`, resumable after a reboot.

---

## 8. Timeline

- 2026-09-08: core + smoke + estimate written and passing locally (single core); batch-0 kit ready.
- Next: `batch1` mode (job list, per-(U, sector, r) result files, resume), `collect.py`
  into the `build_sigma_iwn.py` layout, T4 (2×2 vs SmoQy on the laptop), T5 dress
  rehearsal on the laptop (18-site N=15 small-Sz sectors or a 12-site full run).
- Then hand over batch 1; SmoQy 3×2 runs + analysis scaffolding while it runs.

---

## 9. Risks

- **ETA off by 3×** (bandwidth, Windows allocation behaviour): mitigated by batch 0 and the U-subset ordering (U=4,7 first).
- **Partial results**: every (U, sector, r) file stands alone; GC needs *all* N sectors of a U, so the job order is sector-complete per U before moving to the next U.
- **Windows**: antivirus scanning large scratch files (keep scratch in RAM, no big checkpoints); console closed by accident (resume); path handling (Julia is fine).
- **Canonical vs GC at 18 sites** in batch 1: in a fixed-N run the c† side lives in the
  N+1 sector, so G_N(τ) carries no chemical potential and the anticommutator identity
  G_aa(0)+G_aa(β)=1 does not hold as is; μ is fixed at analysis time from that identity
  (μ_c = ln[(1−G(0))/G(β)]/β, i.e. e^{βμ_c} ≈ Z_N/Z_{N+1}) and the residual is the
  ⟨n⟩_{N+1} − ⟨n⟩_N ≈ 1/N_site ensemble effect (~5 % on G at 18 sites, much less on Z_M
  since it is a ratio at iω₀). Quantified at 12 sites where GC is cheap (T4); removed by
  batch 2 (GC over N = 11…19, ~3.5× the cost of N=15 alone).
- **Typicality variance** larger than expected at U=7: adaptive R, R_max=8, and the error is reported, not assumed.
- **Roommate patience**: batch 1 must deliver something early (U=2 first; one complete U is a result).
- **Single core**: if the ETA is unacceptable, the ordered levers are R=3, k=M only, threads.
