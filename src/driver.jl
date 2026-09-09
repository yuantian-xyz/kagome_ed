# Modes: smoke | estimate | (batch modes added later). Every mode writes into
# output/<mode>_<timestamp>/ (log.txt, config copy, sysinfo.json, results).

function parse_cli(args)
    mode = isempty(args) || startswith(args[1], "--") ? "smoke" : args[1]
    opts = Dict{String,String}()
    i = startswith(get(args, 1, "--"), "--") ? 1 : 2
    while i ≤ length(args)
        a = args[i]
        startswith(a, "--") || error("unexpected argument $a")
        if i < length(args) && !startswith(args[i+1], "--")
            opts[a[3:end]] = args[i+1]; i += 2
        else
            opts[a[3:end]] = "true"; i += 1
        end
    end
    mode, opts
end

function load_config(mode, opts)
    path = get(opts, "config", joinpath(ROOT, "configs", mode * ".toml"))
    isfile(path) || error("config not found: $path")
    cfg = TOML.parsefile(path)
    for key in ("model", "grid", "lanczos")
        haskey(cfg, key) || (cfg[key] = Dict{String,Any}())
    end
    cfg["_path"] = path
    cfg
end

first_M(cl::Cluster) = for name in ("M1", "M2", "M3")
    (κ1, κ2) = KPOINT_NAMES[name]
    isinteger(κ1 * cl.L1) && isinteger(κ2 * cl.L2) && return name
end

canonical_sectors(nsite, N) = [(nup, N - nup) for nup in max(0, N - nsite):min(N, nsite)]
gc_sectors(nsite, Nmin, Nmax) = vcat([canonical_sectors(nsite, N) for N in Nmin:Nmax]...)

function build_pair(cl, bc, nup, ndn, U, knames)
    S = Sector(cl, bc, nup, ndn, U)
    if nup < cl.nsite
        S1 = Sector(cl, bc, nup + 1, ndn, U)
        kops = [KOp(cl, bc, nup, k) for k in knames]
    else
        S1 = nothing; kops = KOp[]
    end
    S, S1, kops
end

function run_ensemble(cl, bc, U, sectors, knames, rc, R; out, seedbase, save_dir = nothing)
    entries = Any[]
    for (nup, ndn) in sectors
        S, S1, kops = build_pair(cl, bc, nup, ndn, U, knames)
        for r in 1:R
            seed = sector_seed(U, nup, ndn, r, seedbase)
            res = run_sector_r(S, S1, kops, rc, r, seed; logger = out)
            push!(entries, res)
            save_dir === nothing || save_result(save_dir, res, @sprintf("sec%d_%d_r%d", nup, ndn, r))
        end
    end
    entries
end

# relative deviation: max |Δg| / max |g_exact| per component (the per-r ratio Num_r/Z_r is
# not bounded by 1 — only the ensemble average is — so absolute deviations are meaningless)
function max_dev(res, ex)
    d = max(abs(res["lnZ_full"] - ex["lnZ_full"]), abs(res["E"] - ex["E"]))
    gmax = 0.0
    for (k, Gk) in res["G"], (key, v) in Gk
        startswith(key, "g_") || continue
        scale = max(maximum(abs.(ex["G"][k][key])), 1e-12)
        d = max(d, maximum(abs.(v .- ex["G"][k][key])) / scale)
        gmax = max(gmax, scale)
    end
    d, gmax
end

# ---------------------------------------------------------------- smoke ----
function mode_smoke(cfg, opts, out)
    sm = cfg["smoke"]
    rcx = RunCfg(cfg; budget_bytes = 8.0 * 2^30, qtype = Float64)      # exactness checks: Float64 Krylov storage
    rc  = RunCfg(cfg; budget_bytes = 8.0 * 2^30)                        # production storage (Float32)
    rc4 = RunCfg(cfg; budget_bytes = 8.0 * 2^30, nq_force = Int(get(sm, "nq", 6)))
    β = rc.β
    report = Dict{String,Any}(); pass = true
    check(name, val, thr) = begin
        ok = val < thr
        report[name] = val; report[name * "_ok"] = ok
        logmsg(out, @sprintf("  %-38s %.3e  (limit %.0e)  %s", name, val, thr, ok ? "ok" : "FAIL"))
        pass &= ok
    end

    logmsg(out, "S1  Bloch consistency of the torus hopping matrix")
    for (L1, L2) in ((2, 1), (2, 2), (3, 2), (4, 2))
        check("bloch_$(L1)x$(L2)", check_bloch(Cluster(L1, L2)), 1e-10)
    end

    logmsg(out, "S2  6-site 2×1 torus, U=4: Lanczos/typicality pipeline vs dense exponentials (same r)")
    cl = Cluster(2, 1); bc = BasisCache(); U = 4.0
    knames = ["G", first_M(cl)]
    worst = 0.0
    for (nup, ndn) in ((3, 3), (4, 2), (2, 3), (1, 5), (5, 1), (6, 2))
        S, S1, kops = build_pair(cl, bc, nup, ndn, U, knames)
        r = newvec(S); fill_randn!(r, 11 + nup); tscal!(r, 1 / tnorm(r))
        res = run_sector_r(S, S1, kops, rcx, 1, 0; rvec = r)
        ex = exact_sector(S, S1, kops, rcx; rvec = r)
        d, gmax = max_dev(res, ex); worst = max(worst, d)
        logmsg(out, @sprintf("    sector (%d,%d) dim %4d  Mq=%d  max rel.dev = %.2e  (max|g| = %.1e)", nup, ndn, S.dim, res["Mq"], d, gmax))
    end
    check("exact6_per_r_float64", worst, 1e-8)

    logmsg(out, "S2b 6-site, U=0, grand canonical μ=−0.5: exact trace vs free-fermion formula")
    μ0 = -0.5
    ents0 = Any[]
    for (nup, ndn) in gc_sectors(cl.nsite, 0, 2cl.nsite)
        S, S1, kops = build_pair(cl, bc, nup, ndn, 0.0, knames)
        push!(ents0, exact_sector(S, S1, kops, rc))
    end
    ens0 = combine(ents0; β = β, μ = μ0)
    d0 = 0.0
    for k in knames
        Gτ, Giw = free_G(cl, kpoint(cl, k), β, μ0, rc.τs, rc.ωn)
        d0 = max(d0, maximum(abs.(ens0.Gτ[k] .- Gτ)), maximum(abs.(ens0.Giw[k] .- Giw)))
    end
    check("free_fermion_gc", d0, 1e-9)
    report["free_fermion_N"] = ens0.N

    logmsg(out, "S2c 6-site, U=4, grand canonical μ=−0.7: typicality (R=$(sm["R_gc"])) vs exact trace")
    μ1 = -0.7
    entsE = Any[]; secs = gc_sectors(cl.nsite, 0, 2cl.nsite)
    for (nup, ndn) in secs
        S, S1, kops = build_pair(cl, bc, nup, ndn, U, knames)
        push!(entsE, exact_sector(S, S1, kops, rc))
    end
    ensE = combine(entsE; β = β, μ = μ1)
    entsT = run_ensemble(cl, bc, U, secs, knames, rc, Int(sm["R_gc"]); out = nothing, seedbase = 100)
    ensT = combine(entsT; β = β, μ = μ1)
    dev = 0.0; devσ = 0.0
    for k in knames
        Δ = abs.(ensT.Gτ[k] .- ensE.Gτ[k])
        dev = max(dev, maximum(Δ))
        devσ = max(devσ, maximum(Δ ./ max.(ensT.Gτ_err[k], 1e-12)))
    end
    logmsg(out, @sprintf("    ⟨N⟩ exact %.4f  typicality %.4f ± %.4f ;  E exact %.4f  typ %.4f ± %.4f",
        ensE.N, ensT.N, ensT.N_err, ensE.E, ensT.E, ensT.E_err))
    report["typicality6_max_abs_dev"] = dev
    logmsg(out, @sprintf("    max |ΔG(τ)| = %.3e (6-site sectors are tiny, so single-vector typicality noise is O(0.1–1))", dev))
    check("typicality6_max_dev_over_sigma", devσ, 5.0)

    logmsg(out, "S3  12-site 2×2 torus, U=4: small sectors, pipeline vs dense (same r), all M points")
    cl2 = Cluster(2, 2); bc2 = BasisCache()
    kn2 = ["G", "M1", "M2", "M3"]
    worst2 = 0.0
    for (nup, ndn) in ((1, 1), (1, 2), (2, 0), (0, 2), (2, 1))
        S, S1, kops = build_pair(cl2, bc2, nup, ndn, U, kn2)
        r = newvec(S); fill_randn!(r, 31 + nup); tscal!(r, 1 / tnorm(r))
        res = run_sector_r(S, S1, kops, rc, 1, 0; rvec = r)
        ex = exact_sector(S, S1, kops, rc; rvec = r)
        d, gmax = max_dev(res, ex); worst2 = max(worst2, d)
        logmsg(out, @sprintf("    sector (%d,%d) dim %5d  Mq=%d  max rel.dev = %.2e  (max|g| = %.1e)", nup, ndn, S.dim, res["Mq"], d, gmax))
    end
    check("exact12_per_r_float32", worst2, 1e-5)

    Uh = Float64(sm["U"])
    logmsg(out, "S4a batching consistency: 12-site sector (3,7)→(4,7), U=$(Uh), q-batches of $(rc4.nq_force) vs all-in-memory (must be bitwise-close)")
    S37, S47, kops37 = build_pair(cl2, bc2, 3, 7, Uh, ["M1"])
    r37 = newvec(S37); fill_randn!(r37, 77); tscal!(r37, 1 / tnorm(r37))
    resA = run_sector_r(S37, S47, kops37, rc, 1, 0; rvec = r37)
    resB = run_sector_r(S37, S47, kops37, rc4, 1, 0; rvec = r37)
    dAB = 0.0
    for (k, Gk) in resA["G"], (key, v) in Gk
        startswith(key, "g_") || continue
        dAB = max(dAB, maximum(abs.(v .- resB["G"][k][key])) / max(maximum(abs.(v)), 1e-12))
    end
    logmsg(out, @sprintf("    Mq=%d: batches %d vs %d", resA["Mq"], resA["nbatch"], resB["nbatch"]))
    check("batching_consistency", dAB, 1e-10)

    R = Int(sm["R"])
    logmsg(out, "S4  12-site upper vHs, canonical N=10, U=$(Uh), k=M1,G, R=$R (the production code path)")
    secs10 = canonical_sectors(cl2.nsite, 10)
    t4 = time()
    save_dir = joinpath(out.dir, "results_hi12_U$(Uh)")
    ents4 = run_ensemble(cl2, bc2, Uh, secs10, ["M1", "G"], rc, R; out = out, seedbase = 1000, save_dir = save_dir)
    ens0 = combine(ents4; β = β, Nsel = [10])            # μ = 0: raw canonical numerators
    logmsg(out, @sprintf("    %d sectors × R=%d in %.1f s;  E = %.5f ± %.5f", length(secs10), R, time() - t4, ens0.E, ens0.E_err))
    # canonical ensemble: c† lives in the N+1 sector, so the anticommutator identity
    # G_aa(0) + e^{βμ} G_aa(β) = 1 defines the chemical potential to use (one per sublattice; report spread)
    μs = Float64[]
    for k in ("M1", "G"), a in 1:3
        g0 = ens0.Gτ[k][a, a, 1]; gβ = ens0.Gτ[k][a, a, end]
        (g0 < 1 && gβ > 0) && push!(μs, log((1 - g0) / gβ) / β)
    end
    μc = sum(μs) / length(μs)
    logmsg(out, @sprintf("    canonical μ from G_aa(0) + e^{βμ}G_aa(β) = 1: mean %.4f, spread %.4f (per k and sublattice)", μc, maximum(μs) - minimum(μs)))
    ens4 = combine(ents4; β = β, μ = μc, Nsel = [10])
    for k in ("M1", "G")
        zf = zfactors(cl2, ens4, k)
        g0 = [ens4.Gτ[k][a, a, 1] for a in 1:3]
        e0 = [ens4.Gτ_err[k][a, a, 1] for a in 1:3]
        anti = [ens4.Gτ[k][a, a, 1] + ens4.Gτ[k][a, a, end] for a in 1:3]
        logmsg(out, @sprintf("    k=%s  G_aa(0) = %s   G_aa(0)+G_aa(β) = %s   (R=%d typicality on 12 sites: noisy by design)", k,
            join([@sprintf("%.3f±%.3f", x, e) for (x, e) in zip(g0, e0)], " "), join([@sprintf("%.3f", x) for x in anti], " "), R))
        logmsg(out, @sprintf("    k=%s  bands %s  Z_M(band) = %s", k, join([@sprintf("%+.2f", x) for x in zf["band_energies"]], " "),
            join([@sprintf("%.4f", x) for x in zf["Z_M_band"]], " ")))
        report["hi12_Z_M_band_" * k] = zf["Z_M_band"]; report["hi12_anticomm_" * k] = anti; report["hi12_G0_" * k] = g0
    end
    report["hi12_mu_c"] = μc; report["hi12_mu_spread"] = maximum(μs) - minimum(μs)
    report["hi12_E"] = ens4.E; report["hi12_E_err"] = ens4.E_err
    report["hi12_seconds"] = time() - t4
    report["pass"] = pass
    json_write(joinpath(out.dir, "smoke.json"), report)
    logmsg(out, pass ? "SMOKE PASS" : "SMOKE FAIL — send me the output folder")
    pass
end

# ------------------------------------------------------------- estimate ----
function timeit(f, n = 3)
    f()                       # JIT warm-up
    best = Inf
    for _ in 1:n
        t = @elapsed f()
        best = min(best, t)
    end
    best
end

function mode_estimate(cfg, opts, out)
    ec = cfg["estimate"]
    L1, L2 = Int(ec["L1"]), Int(ec["L2"])
    nup, ndn = Int(ec["nup"]), Int(ec["ndn"])
    budget = parse(Float64, get(opts, "mem", string(ec["budget_GB"]))) * 2^30
    total = Sys.total_memory(); free = Sys.free_memory()
    # macOS reports only truly free pages; allow an explicit override (--avail GB) for laptop tests
    haskey(opts, "avail") && (free = parse(Float64, opts["avail"]) * 2^30)
    margin = 6.0 * 2^30
    logmsg(out, @sprintf("memory: total %s, available %s%s, budget %s; threads %d", gb(total), gb(free),
        haskey(opts, "avail") ? " (from --avail)" : "", gb(budget), Threads.nthreads()))
    cl = Cluster(L1, L2); bc = BasisCache()
    nsite = cl.nsite
    # working set of the timing block: 2 vectors of (nup,ndn), 2 of (nup+1,ndn), 4 Float32 q, 3 w  → ≈ 7·d + 2·d1 doubles
    need(nu, nd) = 8 * (7 * binomial_dim(nsite, nu, nd) + 2 * binomial_dim(nsite, nu + 1, nd))
    scaled = false
    while need(nup, ndn) > min(free, budget) - margin && nup > 1
        nup -= 1; ndn -= 1; scaled = true
    end
    d = binomial_dim(nsite, nup, ndn); d1 = binomial_dim(nsite, nup + 1, ndn)
    logmsg(out, @sprintf("timing sector (%d,%d) dim %.3e (%s per vector) and (%d,%d) dim %.3e%s",
        nup, ndn, d, gb(8d), nup + 1, ndn, d1, scaled ? "  [SCALED DOWN to fit this machine]" : ""))
    t_build = @elapsed begin
        S = Sector(cl, bc, nup, ndn, 4.0); S1 = Sector(cl, bc, nup + 1, ndn, 4.0)
        K = KOp(cl, bc, nup, first_M(cl))
    end
    logmsg(out, @sprintf("basis/hopping build: %.1f s", t_build))
    t_alloc = @elapsed (V = newvec(S); Y = newvec(S); fill_randn!(V, 1); fill!(Y, 0.0))
    logmsg(out, @sprintf("allocate+fill 2 vectors of %s: %.2f s (%.1f GB/s first touch)", gb(8d), t_alloc, 16d / 2^30 / t_alloc))
    t_hv = timeit(() -> hv!(Y, V, S))
    t_dot = timeit(() -> tdot(V, Y))
    t_upd = timeit(() -> lanczos_update!(Y, V, V, 0.1, 0.1))
    t_scal = timeit(() -> tscal!(Y, 1.0000001))
    step_n = t_hv + t_dot + t_upd + t_scal
    logmsg(out, @sprintf("(%d,%d): H·v %.3f s (%.2f ns/state, %.1f GB/s eff.), dot %.3f, update %.3f, scale %.3f → Lanczos step %.3f s",
        nup, ndn, t_hv, 1e9 * t_hv / d, 5 * 8d / 2^30 / t_hv, t_dot, t_upd, t_scal, step_n))
    V1 = newvec(S1); Y1 = newvec(S1)
    t_cdag = timeit(() -> spmm_gather!(V1, V, K.Bt[1]))
    t_hv1 = timeit(() -> hv!(Y1, V1, S1))
    t_dot1 = timeit(() -> tdot(V1, Y1)); t_upd1 = timeit(() -> lanczos_update!(Y1, V1, V1, 0.1, 0.1)); t_scal1 = timeit(() -> tscal!(Y1, 1.0000001))
    step_n1 = t_hv1 + t_dot1 + t_upd1 + t_scal1
    t_c = timeit(() -> spmm_gather!(Y, V1, K.B[1]))
    logmsg(out, @sprintf("(%d,%d): H·v %.3f s (%.2f ns/state) → Lanczos step %.3f s;  c† %.3f s, c %.3f s", nup + 1, ndn, t_hv1, 1e9 * t_hv1 / d1, step_n1, t_cdag, t_c))
    nqt = 4
    Q = [Matrix{Float32}(undef, size(V)) for _ in 1:nqt]
    for q in Q; tcopy!(q, V); end
    outbd = zeros(nqt, 3)
    t_cd = timeit(() -> cross_dots!(outbd, Q, V1, K.B))
    logmsg(out, @sprintf("cross_dots (3 sublattices × %d q-vectors) on dim %.2e: %.3f s (%.2f ns per (n+1)-state)", nqt, d1, t_cd, 1e9 * t_cd / d1))
    a_cd0 = 0.0      # split: fixed part (c-applications) vs per-q part, estimated from a second size
    Q2 = Q[1:2]; outbd2 = zeros(2, 3)
    t_cd2 = timeit(() -> cross_dots!(outbd2, Q2, V1, K.B))
    a_cdq = max((t_cd - t_cd2) / (nqt - 2), 0.0) / d       # seconds per q-vector per n-state
    a_cd0 = max(t_cd - nqt * a_cdq * d, 0.0) / d1            # seconds per (n+1)-state, q-independent
    logmsg(out, @sprintf("   → c-application part %.2f ns per (n+1)-state, dot part %.2f ns per n-state per q-vector", 1e9 * a_cd0, 1e9 * a_cdq))
    V = Y = V1 = Y1 = Q = Q2 = nothing; GC.gc()

    # first-touch memory test up to memtest_GB (verifies the OS really hands out the RAM)
    mt = min(Float64(ec["memtest_GB"]) * 2^30, Sys.free_memory() - margin, budget)
    chunks = Vector{Vector{Float64}}()
    nchunk = max(1, floor(Int, mt / (4.0 * 2^30)))
    t_mt = @elapsed for i in 1:nchunk
        v = Vector{Float64}(undef, (4 * 2^30) ÷ 8); fill!(v, Float64(i)); push!(chunks, v)
    end
    touched = 4.0 * nchunk
    logmsg(out, @sprintf("memory test: touched %.0f GB in %.1f s (%.1f GB/s); free now %s", touched, t_mt, touched / t_mt, gb(Sys.free_memory())))
    chunks = nothing; GC.gc()

    # ---- ETA model for the batch-1 job list, scaled by dimension ----------
    M = Int(ec["M_assumed"]); Rhi = Int(ec["R_hi"]); nU = length(ec["U_list"])
    na = 3
    per_state_n = step_n / d; per_state_n1 = step_n1 / d1
    function sector_cost(nu, nd, knb)
        dn = binomial_dim(nsite, nu, nd)
        nu + 1 > nsite && return M * per_state_n * dn
        dn1 = binomial_dim(nsite, nu + 1, nd)
        fixed = 8 * (4dn + 4dn1)
        nq = clamp(floor(Int, (budget - fixed) / (4dn)), 1, M)
        nb = cld(M, nq)
        cn = per_state_n * dn; cn1 = per_state_n1 * dn1
        pass1 = M * cn + knb * M * cn1
        qreplay = cn * M * (nb + 1) / 2
        cross = nb * knb * M * (cn1 + a_cd0 * dn1 + a_cdq * dn * nq)
        pass1 + qreplay + cross
    end
    eta = Dict{String,Any}()
    for nk in (1, 2)
        knb = 3nk
        hi15 = sum(sector_cost(nu, nd, knb) for (nu, nd) in canonical_sectors(nsite, 15))
        hi_gc = sum(sector_cost(nu, nd, knb) for (nu, nd) in gc_sectors(nsite, 11, 19))
        eta["nk$(nk)_sector_8_7_hours_per_r"] = sector_cost(8, 7, knb) / 3600
        eta["nk$(nk)_hi_canonical_N15_hours_per_U"] = hi15 * Rhi / 3600
        eta["nk$(nk)_hi_canonical_N15_days_batch1"] = hi15 * Rhi * nU / 86400
        eta["nk$(nk)_hi_gc_N11-19_days_per_U_R3"] = hi_gc * 3 / 86400
        logmsg(out, @sprintf("ETA model, %d k-point(s) (M=%d, R=%d, budget %s, %d thread(s)): (8,7) sector %.1f h per r;  canonical N=15: %.1f h per U → batch 1 (U=%s) %.1f days;  GC N=11..19: %.1f days per U at R=3",
            nk, M, Rhi, gb(budget), Threads.nthreads(), eta["nk$(nk)_sector_8_7_hours_per_r"], eta["nk$(nk)_hi_canonical_N15_hours_per_U"],
            join(string.(ec["U_list"]), ","), eta["nk$(nk)_hi_canonical_N15_days_batch1"], eta["nk$(nk)_hi_gc_N11-19_days_per_U_R3"]))
    end
    rep = Dict{String,Any}("sector" => [nup, ndn], "scaled_down" => scaled, "dim" => d, "dim1" => d1,
        "t_hv" => t_hv, "t_hv1" => t_hv1, "t_dot" => t_dot, "t_update" => t_upd, "t_scale" => t_scal,
        "t_cdag" => t_cdag, "t_c" => t_c, "t_cross_dots_4x3" => t_cd, "ns_capp_per_state" => 1e9 * a_cd0, "ns_dot_per_state_per_q" => 1e9 * a_cdq, "ns_per_state_hv" => 1e9 * t_hv / d,
        "memtest_GB" => touched, "memtest_GBps" => touched / t_mt, "budget_GB" => budget / 2^30,
        "free_GB_at_start" => free / 2^30, "total_GB" => total / 2^30, "eta" => eta)
    json_write(joinpath(out.dir, "estimate.json"), rep)
    logmsg(out, "ESTIMATE DONE — send me the output folder")
    true
end

# ----------------------------------------------------------------- main ----
function main(args)
    mode, opts = parse_cli(args)
    cfg = load_config(mode, opts)
    out = open_output(ROOT, mode)
    cp(cfg["_path"], joinpath(out.dir, "config.toml"); force = true)
    si = sysinfo(); json_write(joinpath(out.dir, "sysinfo.json"), si)
    logmsg(out, "kagome_ed  mode=$mode  julia $(si["julia"])  threads=$(si["julia_threads"])/$(si["cpu_threads"])  RAM $(round(si["total_memory_GB"], digits = 1)) GB  ($(si["cpu"]))")
    logmsg(out, "output → $(out.dir)")
    ok = mode == "smoke" ? mode_smoke(cfg, opts, out) :
         mode == "estimate" ? mode_estimate(cfg, opts, out) :
         error("unknown mode $mode")
    logmsg(out, @sprintf("finished in %.1f s", time() - out.t0))
    close(out.log)
    ok
end
