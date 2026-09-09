# Finite-T typicality for one (n↑,n↓) sector and one random vector r:
#   Z_r      = ⟨r|e^{−βH}|r⟩                              (Lanczos on r)
#   Num_ab(τ) = ⟨r| e^{−(β−τ)H} c_{ka} e^{−τH} c†_{kb} |r⟩   (cross matrix C_ij = ⟨q_i|c_{ka}|p_j⟩
#               between the Krylov bases of r and of φ_b = c†_{kb} r)
# From the two Ritz decompositions and W = nφ·diag(U1q)·(Uqᵀ C Up)·diag(U1p):
#   g(τ) = Num/Z_r on the τ grid, 𝒢(iωₙ)/… by the Lehmann formula, and W, ε, ε′
#   are stored so real-axis quantities can be evaluated later.
# Krylov vectors of r are regenerated in batches (Float32 storage), the
# vectors of φ_b are regenerated on the fly; nothing large is written to disk.

struct KOp
    name::String
    κ::Tuple{Float64,Float64}
    B::Vector{CSC}     # c†_{k,a}: (D_{n+1} × D_n), a = 1..3
    Bt::Vector{CSC}    # transposes (D_n × D_{n+1})
end

function KOp(cl::Cluster, bc::BasisCache, nup::Int, name::AbstractString)
    κ = kpoint(cl, name)
    bn = get_basis!(bc, cl, nup); bn1 = get_basis!(bc, cl, nup + 1)
    B = CSC[]; Bt = CSC[]
    for a in 1:3
        (Ba, Bta) = cdag_matrix(cl, bn, bn1, κ[1], κ[2], a)
        push!(B, Ba); push!(Bt, Bta)
    end
    KOp(String(name), κ, B, Bt)
end

struct RunCfg
    β::Float64
    τs::Vector{Float64}
    ωn::Vector{Float64}
    Mmax::Int
    tol::Float64
    budget_bytes::Float64
    nq_force::Int          # >0 forces the q-batch size (tests); 0 = from budget
    aset::Vector{Int}
    bset::Vector{Int}
    qtype::DataType        # storage type of the regenerated Krylov vectors (Float32 in production)
end

function RunCfg(cfg::Dict; budget_bytes = 0.0, nq_force = 0, qtype = Float32)
    β = Float64(cfg["model"]["beta"])
    dtau = Float64(get(cfg["grid"], "dtau", 0.05))
    nτ = round(Int, β / dtau)
    τs = [β * i / nτ for i in 0:nτ]
    nmats = Int(get(cfg["grid"], "nmats", 20))
    ωn = [(2n + 1) * π / β for n in 0:nmats-1]
    RunCfg(β, τs, ωn, Int(get(cfg["lanczos"], "Mmax", 200)), Float64(get(cfg["lanczos"], "tol", 1e-9)),
           budget_bytes, nq_force, collect(1:3), collect(1:3), qtype)
end

sector_seed(U::Float64, nup::Int, ndn::Int, r::Int, base::Int) =
    base + 1_000_003 * r + 7919 * nup + 104_729 * ndn + round(Int, 1000 * U)

# Number of stored Float32 q-vectors that fit the budget alongside the working set.
function qbatch_size(rc::RunCfg, S::Sector, S1::Sector, na::Int, Mq::Int)
    rc.nq_force > 0 && return min(rc.nq_force, Mq)
    fixed = 4 * vecbytes(S) + 4 * vecbytes(S1)
    free = rc.budget_bytes - fixed
    qb = sizeof(rc.qtype) * S.dim
    free ≤ qb && return 1
    clamp(floor(Int, free / qb), 1, Mq)
end

# g(τ) = Σ_{mn} W_mn e^{−(β−τ)(ε_m−ε0)} e^{−τ(ε_n−ε0)} e^{μτ} / Z   (particle number of n-states = N+1)
# 𝒢(iω) = Σ_{mn} W_mn [e^{−β(ε_n−ε0)} e^{βμ} + e^{−β(ε_m−ε0)}] / (iω + μ + ε_m − ε_n) / Z
function lehmann(W::AbstractMatrix, εm::Vector{Float64}, εn::Vector{Float64}, ε0::Float64, Z::Float64,
                 β::Float64, μ::Float64, τs, ωn)
    em = εm .- ε0; en = εn .- ε0
    gτ = [exp(μ * τ) * dot(exp.(-(β - τ) .* em), W * exp.(-τ .* en)) / Z for τ in τs]
    num = (exp(β * μ) .* exp.(-β .* en))' .+ exp.(-β .* em)
    den0 = μ .+ em .- en'
    giw = [sum(W .* num ./ (im * ω .+ den0)) / Z for ω in ωn]
    gτ, giw
end

"""
    run_sector_r(S, S1, kops, rc, r_idx, seed; log=nothing) -> Dict

S1 === nothing when n↑ = nsite (no room for c†_↑): only Z_r is produced.
"""
function run_sector_r(S::Sector, S1::Union{Sector,Nothing}, kops::Vector{KOp}, rc::RunCfg,
                      r_idx::Int, seed::Int; logger = nothing, rvec::Union{Nothing,Matrix{Float64}} = nothing)
    t0 = time()
    β = rc.β
    stats = Dict{String,Any}("nhv" => 0)
    r = newvec(S)
    if rvec === nothing
        fill_randn!(r, seed); tscal!(r, 1 / tnorm(r))
    else
        tcopy!(r, rvec)
    end
    work = (newvec(S), newvec(S), newvec(S))
    Tq = lanczos_run!(S, r, work; βtemp = β, Mmax = rc.Mmax, tol = rc.tol, stats = stats)
    Rq = ritz(Tq)
    eq = Rq.ε .- Rq.ε0
    wq = Rq.U[1, :] .^ 2 .* exp.(-β .* eq)
    Zr = sum(wq)
    res = Dict{String,Any}(
        "nup" => S.nup, "ndn" => S.ndn, "N" => S.nup + S.ndn, "dim" => S.dim, "weight" => Float64(S.dim),
        "r" => r_idx, "seed" => seed, "Mq" => nsteps(Tq),
        "lnZ_full" => log(Zr) - β * Rq.ε0, "E" => sum(wq .* Rq.ε) / Zr, "eps0" => Rq.ε0, "Zr_shift" => Zr,
        "eps_q" => Rq.ε, "U1_q" => Rq.U[1, :], "tau" => rc.τs, "omega_n" => rc.ωn,
        "kpoints" => [K.name for K in kops], "aset" => rc.aset, "bset" => rc.bset, "G" => Dict{String,Any}())
    if S1 === nothing
        res["nhv"] = stats["nhv"]; res["seconds"] = time() - t0
        return res
    end
    work1 = (newvec(S1), newvec(S1), newvec(S1))
    φ = newvec(S1)
    Tp = Dict{Tuple{Int,Int},Tri}(); nφ = Dict{Tuple{Int,Int},Float64}()
    for (ik, K) in enumerate(kops), b in rc.bset
        spmm_gather!(φ, r, K.Bt[b])
        n = tnorm(φ)
        n < 1e-14 && continue
        tscal!(φ, 1 / n)
        Tp[(ik, b)] = lanczos_run!(S1, φ, work1; βtemp = β, Mmax = rc.Mmax, tol = rc.tol, stats = stats)
        nφ[(ik, b)] = n
    end
    Mq = nsteps(Tq)
    na = length(rc.aset)
    nq = qbatch_size(rc, S, S1, na, Mq)
    C = Dict{Tuple{Int,Int,Int},Matrix{Float64}}()
    for (key, T) in Tp, a in rc.aset
        C[(key[1], key[2], a)] = zeros(Mq, nsteps(T))
    end
    Q = [Matrix{rc.qtype}(undef, size(r)) for _ in 1:nq]
    nbatch = 0
    for i0 in 1:nq:Mq
        i1 = min(i0 + nq - 1, Mq); nb = i1 - i0 + 1; nbatch += 1
        lanczos_run!(S, r, work; βtemp = β, replay = Tq, jmax = i1, stats = stats,
                     onvec = (j, v) -> (j ≥ i0 && tcopy!(Q[j-i0+1], v)))
        Qb = Q[1:nb]
        outbd = zeros(nb, na)
        for (ik, K) in enumerate(kops), b in rc.bset
            haskey(Tp, (ik, b)) || continue
            spmm_gather!(φ, r, K.Bt[b]); tscal!(φ, 1 / nφ[(ik, b)])
            T = Tp[(ik, b)]
            Bsel = [K.B[a] for a in rc.aset]
            lanczos_run!(S1, φ, work1; βtemp = β, replay = T, jmax = nsteps(T), stats = stats,
                         onvec = (j, p) -> begin
                             cross_dots!(outbd, Qb, p, Bsel)
                             for (ia, a) in enumerate(rc.aset)
                                 Cm = C[(ik, b, a)]
                                 @inbounds for i in 1:nb
                                     Cm[i0+i-1, j] = outbd[i, ia]
                                 end
                             end
                         end)
        end
    end
    # assemble W, g(τ) and 𝒢(iωₙ) at μ = 0; (eps_q, eps_p, W) are kept for μ ≠ 0 and the real axis
    for (ik, K) in enumerate(kops)
        Gk = Dict{String,Any}("kappa" => [K.κ[1], K.κ[2]])
        for b in rc.bset
            haskey(Tp, (ik, b)) || continue
            Rp = ritz(Tp[(ik, b)])
            Gk["Mp_b$(b)"] = nsteps(Tp[(ik, b)]); Gk["nphi_b$(b)"] = nφ[(ik, b)]
            Gk["eps_p_b$(b)"] = Rp.ε; Gk["U1_p_b$(b)"] = Rp.U[1, :]
            for a in rc.aset
                W = nφ[(ik, b)] .* (Rq.U' * C[(ik, b, a)] * Rp.U) .* (Rq.U[1, :] * Rp.U[1, :]')
                Gk["W_a$(a)_b$(b)"] = W
                gτ, giw = lehmann(W, Rq.ε, Rp.ε, Rq.ε0, Zr, β, 0.0, rc.τs, rc.ωn)
                Gk["g_tau_a$(a)_b$(b)"] = gτ
                Gk["g_iw_re_a$(a)_b$(b)"] = real.(giw)
                Gk["g_iw_im_a$(a)_b$(b)"] = imag.(giw)
            end
        end
        res["G"][K.name] = Gk
    end
    res["nq"] = nq; res["nbatch"] = nbatch; res["nhv"] = stats["nhv"]; res["seconds"] = time() - t0
    logger !== nothing && logmsg(logger, @sprintf("    sector (%d,%d) dim %d r=%d: Mq=%d nq=%d batches=%d H·v=%d  %.1fs",
        S.nup, S.ndn, S.dim, r_idx, Mq, nq, nbatch, stats["nhv"], res["seconds"]))
    res
end

# Save one result: JSON with scalars/small arrays, W matrices as .npy
function save_result(dir::AbstractString, res::Dict, tag::AbstractString)
    mkpath(dir)
    slim = Dict{String,Any}()
    for (k, v) in res
        k == "G" && continue
        slim[k] = v
    end
    slim["G"] = Dict{String,Any}()
    for (kname, Gk) in res["G"]
        Gs = Dict{String,Any}()
        for (key, v) in Gk
            if startswith(key, "W_")
                npy_write(joinpath(dir, "$(tag)_$(kname)_$(key).npy"), Float32.(v))
            else
                Gs[key] = v
            end
        end
        slim["G"][kname] = Gs
    end
    json_write(joinpath(dir, "$(tag).json"), slim)
end
