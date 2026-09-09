# Lanczos recurrence without reorthogonalisation, used for e^{−τH}v via the
# tridiagonal T. Pass 1 (adaptive): builds T until the a-posteriori estimate
# β_m |(e^{−τT})_{m1}| / (e^{−τT})_{11} < tol for τ ∈ {β/4, β/2, 3β/4, β}.
# Replay: re-runs the identical recurrence (bit-identical, deterministic
# kernels) and hands each Krylov vector to a callback — this is how Krylov
# vectors are regenerated instead of stored.

struct Tri
    α::Vector{Float64}
    β::Vector{Float64}   # length M−1
end
nsteps(T::Tri) = length(T.α)

struct Ritz
    ε::Vector{Float64}       # Ritz values
    U::Matrix{Float64}       # eigenvectors of T (columns)
    ε0::Float64              # shift (min ε)
end
function ritz(T::Tri)
    M = nsteps(T)
    if M == 1
        return Ritz([T.α[1]], ones(1, 1), T.α[1])
    end
    e = eigen(SymTridiagonal(copy(T.α), copy(T.β)))
    Ritz(e.values, e.vectors, minimum(e.values))
end

# shifted coefficients c_j(τ) e^{τ ε0} = Σ_l U[j,l] e^{−τ(ε_l−ε0)} U[1,l]
function expcoef(R::Ritz, τ::Float64)
    w = R.U[1, :] .* exp.(-τ .* (R.ε .- R.ε0))
    R.U * w
end

function _converged(α::Vector{Float64}, β::Vector{Float64}, bnew::Float64, βtemp::Float64, tol::Float64)
    R = ritz(Tri(α, β))
    m = length(α)
    worst = 0.0
    for f in (0.25, 0.5, 0.75, 1.0)
        c = expcoef(R, f * βtemp)
        worst = max(worst, bnew * abs(c[m]) / max(abs(c[1]), 1e-300))
    end
    worst < tol, worst
end

"""
    lanczos_run!(S, v1, work; βtemp, Mmax, tol, replay=nothing, jmax=0, onvec=nothing)

`work = (v, vprev, w)` three sector vectors. `v1` is copied, not modified.
Adaptive mode (replay === nothing): returns `Tri`. Replay mode: replays the
recurrence `replay` for j = 1..jmax calling `onvec(j, v_j)`; returns the number
of H·v applications performed.
"""
function lanczos_run!(S::Sector, v1::Matrix{Float64}, work; βtemp::Float64, Mmax::Int = 200,
                      tol::Float64 = 1e-9, replay::Union{Nothing,Tri} = nothing, jmax::Int = 0,
                      onvec = nothing, stats::Union{Nothing,Dict} = nothing)
    v, vprev, w = work
    tcopy!(v, v1)
    α = Float64[]; β = Float64[]
    bprev = 0.0
    nhv = 0
    jend = replay === nothing ? Mmax : min(jmax, nsteps(replay))
    est = Inf
    for j in 1:jend
        onvec !== nothing && onvec(j, v)
        if replay !== nothing && j == jend
            break
        end
        hv!(w, v, S); nhv += 1
        a = tdot(v, w)
        push!(α, a)
        b = lanczos_update!(w, v, vprev, a, bprev)
        if replay !== nothing
            (abs(a - replay.α[j]) > 1e-9 * (1 + abs(a)) || (j ≤ length(replay.β) && abs(b - replay.β[j]) > 1e-9 * (1 + b))) &&
                error("Lanczos replay is not reproducing pass 1 at step $j (α: $a vs $(replay.α[j]))")
        end
        if b < 1e-12
            break                      # invariant subspace: T is exact
        end
        if replay === nothing && j ≥ 3
            ok, est = _converged(α, β, b, βtemp, tol)
            ok && break
        end
        push!(β, b)
        tscal!(w, 1 / b)
        vprev, v, w = v, w, vprev
        bprev = b
    end
    if stats !== nothing
        stats["nhv"] = get(stats, "nhv", 0) + nhv
    end
    if replay === nothing
        length(α) == Mmax && est ≥ tol && @warn "Lanczos did not converge to tol=$tol in Mmax=$Mmax steps (est=$est)"
        return Tri(α, β)
    else
        return nhv
    end
end
