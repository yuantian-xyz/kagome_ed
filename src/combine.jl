# Ensemble combination of per-(sector, r) results with jackknife errors.
# Each entry carries weight (dim for typicality, 1 for exact traces), lnZ_full,
# N, E and g's. Ensemble weight of entry = weight · exp(lnZ_full + βμN).
# Canonical: pass μ = 0 and a single N. Jackknife: leave out one r index
# (all sectors with that r at once), which is what a DQMC bin corresponds to.

struct Ensemble
    τs::Vector{Float64}
    ωn::Vector{Float64}
    kpoints::Vector{String}
    Gτ::Dict{String,Array{Float64,3}}        # k → (3,3,nτ) mean
    Gτ_err::Dict{String,Array{Float64,3}}
    Giw::Dict{String,Array{ComplexF64,3}}
    Giw_err::Dict{String,Array{Float64,3}}
    N::Float64; N_err::Float64
    E::Float64; E_err::Float64
    lnZ::Float64
    R::Int
end

function _accumulate(entries, β, μ, shift)
    # returns per-r partial sums: Dict r => (Z, N·Z, E·Z, Gτ sums, Giw sums)
    parts = Dict{Int,Any}()
    for e in entries
        w = e["weight"] * exp(e["lnZ_full"] + β * μ * e["N"] - shift)
        r = e["r"]
        if !haskey(parts, r)
            parts[r] = Dict{String,Any}("Z" => 0.0, "NZ" => 0.0, "EZ" => 0.0,
                "Gt" => Dict{String,Array{Float64,3}}(), "Gw" => Dict{String,Array{ComplexF64,3}}())
        end
        p = parts[r]
        p["Z"] += w; p["NZ"] += w * e["N"]; p["EZ"] += w * e["E"]
        for (kname, Gk) in e["G"]
            nτ = length(e["tau"]); nω = length(e["omega_n"])
            haskey(p["Gt"], kname) || (p["Gt"][kname] = zeros(3, 3, nτ); p["Gw"][kname] = zeros(ComplexF64, 3, 3, nω))
            for a in e["aset"], b in e["bset"]
                key = "g_tau_a$(a)_b$(b)"
                haskey(Gk, key) || continue
                if μ == 0.0
                    gτ = Gk[key]
                    giw = Gk["g_iw_re_a$(a)_b$(b)"] .+ im .* Gk["g_iw_im_a$(a)_b$(b)"]
                else
                    haskey(Gk, "W_a$(a)_b$(b)") || error("combine with μ ≠ 0 needs the stored W matrices")
                    gτ, giw = lehmann(Gk["W_a$(a)_b$(b)"], e["eps_q"], Gk["eps_p_b$(b)"], e["eps0"], e["Zr_shift"],
                                      β, μ, e["tau"], e["omega_n"])
                end
                p["Gt"][kname][a, b, :] .+= w .* gτ
                p["Gw"][kname][a, b, :] .+= w .* giw
            end
        end
    end
    parts
end

function combine(entries::Vector; β::Float64, μ::Float64 = 0.0, Nsel = nothing)
    ents = Nsel === nothing ? entries : [e for e in entries if e["N"] in Nsel]
    isempty(ents) && error("combine: no entries")
    shift = maximum(e["lnZ_full"] + β * μ * e["N"] for e in ents)
    parts = _accumulate(ents, β, μ, shift)
    rs = sort(collect(keys(parts)))
    R = length(rs)
    τs = ents[1]["tau"]; ωn = ents[1]["omega_n"]
    kps = String[]
    for p in values(parts), k in keys(p["Gt"]); k in kps || push!(kps, k); end
    sort!(kps)
    # totals and leave-one-out
    tot(f) = sum(f(parts[r]) for r in rs)
    est(sel) = begin
        Z = sum(parts[r]["Z"] for r in sel)
        N = sum(parts[r]["NZ"] for r in sel) / Z
        E = sum(parts[r]["EZ"] for r in sel) / Z
        Gt = Dict(k => sum(get(parts[r]["Gt"], k, 0.0) for r in sel) ./ Z for k in kps)
        Gw = Dict(k => sum(get(parts[r]["Gw"], k, 0.0) for r in sel) ./ Z for k in kps)
        (Z, N, E, Gt, Gw)
    end
    Z, N, E, Gt, Gw = est(rs)
    Gt_err = Dict(k => zeros(size(Gt[k])) for k in kps)
    Gw_err = Dict(k => zeros(size(Gw[k])) for k in kps)
    N_err = 0.0; E_err = 0.0
    if R > 1
        jk = [est(setdiff(rs, [r])) for r in rs]
        fac = (R - 1) / R
        N_err = sqrt(fac * sum((j[2] - N)^2 for j in jk))
        E_err = sqrt(fac * sum((j[3] - E)^2 for j in jk))
        for k in kps
            Gt_err[k] = sqrt.(fac .* sum(abs2.(j[4][k] .- Gt[k]) for j in jk))
            Gw_err[k] = sqrt.(fac .* sum(abs2.(j[5][k] .- Gw[k]) for j in jk))
        end
    end
    Ensemble(τs, ωn, kps, Gt, Gt_err, Gw, Gw_err, N, N_err, E, E_err, log(Z) + shift, R)
end

# Self-energy at iωₙ from the 3×3 matrix: Σ = (iω + μ0 − h(k)) − 𝒢⁻¹ ; band-projected
# Z_M = [1 − Im Σ_νν(iω₀)/ω₀]⁻¹ on the bare band eigenvectors (Im Σ is μ0-independent).
function zfactors(cl::Cluster, ens::Ensemble, kname::AbstractString)
    κ = kpoint(cl, kname)
    h = bloch_h(cl, κ[1], κ[2])
    e = eigen(h); u = e.vectors
    Gw = ens.Giw[kname]
    out = Dict{String,Any}("band_energies" => e.values)
    Σ = [ (im * ens.ωn[n] * I - Matrix(h)) - inv(Gw[:, :, n]) for n in 1:length(ens.ωn)]
    Σproj = [real(u' * Σ[n] * u) for n in 1:length(Σ)]
    Σproj_im = [imag(u' * Σ[n] * u) for n in 1:length(Σ)]
    out["ImSigma_band_iw0"] = [Σproj_im[1][ν, ν] for ν in 1:3]
    out["Z_M_band"] = [1 / (1 - Σproj_im[1][ν, ν] / ens.ωn[1]) for ν in 1:3]
    out["ImSigma_band_all"] = [[Σproj_im[n][ν, ν] for n in 1:length(Σ)] for ν in 1:3]
    out
end
