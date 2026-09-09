# Exact references for small sectors (dense diagonalisation) and the
# noninteracting grand-canonical Green's function. Used by the smoke test.

function dense_H(S::Sector)
    Dup, Ddn = length(S.bup), length(S.bdn)
    H = kron(Matrix{Float64}(I, Ddn, Ddn), Matrix(S.Aup)) + kron(Matrix(S.Adn), Matrix{Float64}(I, Dup, Dup))
    d = [S.U * count_ones(S.bup.states[i] & S.bdn.states[c]) + S.c0 for c in 1:Ddn for i in 1:Dup]
    H + Diagonal(d)
end

# dense c†_{k,a} on the full sector: (I_{D↓} ⊗ B_a)
dense_cdag(S::Sector, B::CSC) = kron(Matrix{Float64}(I, length(S.bdn), length(S.bdn)), Matrix(B))

# Same estimator as run_sector_r but with dense exponentials (per-r exactness test).
# rvec === nothing → exact trace (returns weight = 1 entries).
function exact_sector(S::Sector, S1::Union{Sector,Nothing}, kops::Vector{KOp}, rc::RunCfg;
                      rvec::Union{Nothing,Matrix{Float64}} = nothing)
    β = rc.β
    E, Uev = eigen(Symmetric(dense_H(S)))
    E0 = minimum(E)
    if rvec === nothing
        x2 = ones(length(E))          # trace: Σ_m ⟨m|…|m⟩
    else
        x = Uev' * vec(rvec); x2 = x .^ 2
    end
    wq = x2 .* exp.(-β .* (E .- E0))
    Z = sum(wq)
    res = Dict{String,Any}("nup" => S.nup, "ndn" => S.ndn, "N" => S.nup + S.ndn, "dim" => S.dim,
        "weight" => rvec === nothing ? 1.0 : Float64(S.dim), "r" => 0, "eps0" => E0, "Zr_shift" => Z, "eps_q" => E,
        "lnZ_full" => log(Z) - β * E0, "E" => sum(wq .* E) / Z, "tau" => rc.τs, "omega_n" => rc.ωn,
        "kpoints" => [K.name for K in kops], "aset" => rc.aset, "bset" => rc.bset, "G" => Dict{String,Any}())
    S1 === nothing && return res
    E1, U1ev = eigen(Symmetric(dense_H(S1)))
    keepW = S.dim * S1.dim ≤ 2_000_000
    for K in kops
        Gk = Dict{String,Any}("kappa" => [K.κ[1], K.κ[2]])
        Cmat = Dict(a => Uev' * dense_cdag(S, K.B[a])' * U1ev for a in rc.aset)   # ⟨m| c_{ka} |n⟩
        for b in rc.bset, a in rc.aset
            if rvec === nothing
                Wmn = Cmat[a] .* (Uev' * dense_cdag(S, K.B[b])' * U1ev)     # ⟨m|c_a|n⟩⟨n|c†_b|m⟩ (real)
            else
                φ = vec(dense_cdag(S, K.B[b]) * vec(rvec))
                y = U1ev' * φ
                Wmn = (x * y') .* Cmat[a]
            end
            gτ, giw = lehmann(Wmn, E, E1, E0, Z, β, 0.0, rc.τs, rc.ωn)
            Gk["eps_p_b$(b)"] = E1
            keepW && (Gk["W_a$(a)_b$(b)"] = Wmn)
            Gk["g_tau_a$(a)_b$(b)"] = gτ
            Gk["g_iw_re_a$(a)_b$(b)"] = real.(giw); Gk["g_iw_im_a$(a)_b$(b)"] = imag.(giw)
        end
        res["G"][K.name] = Gk
    end
    res
end

# Noninteracting grand-canonical G_ab(k,τ) = Σ_ν u_aν u*_bν e^{−τ(ε_ν−μ)} (1 − f(ε_ν−μ)) per spin,
# and 𝒢_ab(iωₙ) = Σ_ν u u* / (iωₙ + μ − ε_ν).
function free_G(cl::Cluster, κ::Tuple{Float64,Float64}, β::Float64, μ::Float64, τs, ωn)
    e = eigen(bloch_h(cl, κ[1], κ[2]))
    ε = e.values; u = e.vectors
    Gτ = zeros(3, 3, length(τs)); Giw = zeros(ComplexF64, 3, 3, length(ωn))
    for ν in 1:3
        x = ε[ν] - μ
        occ_h = 1 / (1 + exp(-β * x))            # 1 − f
        for (it, τ) in enumerate(τs), a in 1:3, b in 1:3
            Gτ[a, b, it] += real(u[a, ν] * conj(u[b, ν])) * exp(-τ * x) * occ_h
        end
        for (iw, ω) in enumerate(ωn), a in 1:3, b in 1:3
            Giw[a, b, iw] += u[a, ν] * conj(u[b, ν]) / (im * ω - x)
        end
    end
    Gτ, Giw
end
