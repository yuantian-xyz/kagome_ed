# One-spin-species occupation basis: sorted bitmasks with a fixed electron
# number, full 2^nsite lookup table for O(1) ranking (nsite ≤ 24 → ≤ 64 MB).

struct SpinBasis
    nsite::Int
    nel::Int
    states::Vector{UInt32}
    index::Vector{Int32}      # index[mask+1] = position in `states`, 0 if absent
end

function SpinBasis(nsite::Int, nel::Int)
    (0 ≤ nel ≤ nsite) || error("bad electron number")
    states = UInt32[]
    for m in UInt32(0):UInt32((1 << nsite) - 1)
        count_ones(m) == nel && push!(states, m)
    end
    index = zeros(Int32, 1 << nsite)
    for (p, m) in enumerate(states)
        index[m+1] = p
    end
    SpinBasis(nsite, nel, states, index)
end

Base.length(b::SpinBasis) = length(b.states)

# fermionic sign of c†_j c_i acting on mask m (i occupied, j empty, i≠j):
# (−1)^{# occupied sites strictly between i and j}  (1-based site labels → bit s-1)
@inline function hop_sign(m::UInt32, i::Int, j::Int)
    lo, hi = minmax(i, j)
    between = (m >> lo) & ((UInt32(1) << (hi - lo - 1)) - UInt32(1))   # bits lo .. hi-2 (0-based) = sites lo+1..hi-1
    isodd(count_ones(between)) ? -1.0 : 1.0
end

# sign of c†_s (or c_s) acting on mask m: (−1)^{# occupied sites below s}
@inline function site_sign(m::UInt32, s::Int)
    below = m & ((UInt32(1) << (s - 1)) - UInt32(1))
    isodd(count_ones(below)) ? -1.0 : 1.0
end

# Symmetric one-species hopping matrix A (D×D, CSC) with A[j,i] = ⟨j|Σ K_pq c†_p c_q|i⟩.
function hopping_matrix(b::SpinBasis, bonds::Vector{Tuple{Int,Int,Float64}})
    I = Int32[]; J = Int32[]; Vv = Float64[]
    for (p, m) in enumerate(b.states)
        for (i, j, amp) in bonds
            bi = UInt32(1) << (i - 1); bj = UInt32(1) << (j - 1)
            if (m & bi) != 0 && (m & bj) == 0          # c†_j c_i
                m2 = (m ⊻ bi) | bj
                q = b.index[m2+1]
                push!(I, q); push!(J, p); push!(Vv, amp * hop_sign(m, i, j))
            elseif (m & bj) != 0 && (m & bi) == 0      # c†_i c_j
                m2 = (m ⊻ bj) | bi
                q = b.index[m2+1]
                push!(I, q); push!(J, p); push!(Vv, amp * hop_sign(m, j, i))
            end
        end
    end
    D = length(b)
    sparse(I, J, Vv, D, D)
end

# Momentum creation operator on the ↑ index: B = c†_{k,a} : basis(n) → basis(n+1),
# B[q,p] = Σ_{R} phase(R,a)·sign, real phases only (k with 2κ_i L_i ≡ 0 mod L_i… i.e. κ_i ∈ {0, 1/2}).
# Returns (B, Bt) as CSC so that both c† (gather over columns of Bt… see kernels) are cheap.
function cdag_matrix(cl::Cluster, bn::SpinBasis, bn1::SpinBasis, κ1::Float64, κ2::Float64, orb::Int)
    ph = kphases(cl, κ1, κ2)
    all(abs.(imag.(ph)) .< 1e-12) || error("complex k-phases: k=($κ1,$κ2) is not a real-arithmetic point")
    phr = real.(ph)
    sites = [s for s in 1:cl.nsite if cl.site_orb[s] == orb]
    I = Int32[]; J = Int32[]; Vv = Float64[]
    for (p, m) in enumerate(bn.states)
        for s in sites
            bs = UInt32(1) << (s - 1)
            (m & bs) == 0 || continue
            q = bn1.index[(m | bs)+1]
            push!(I, q); push!(J, p); push!(Vv, phr[s] * site_sign(m, s))
        end
    end
    B = sparse(I, J, Vv, length(bn1), length(bn))
    (B, sparse(transpose(B)))
end

binomial_dim(nsite, nup, ndn) = binomial(nsite, nup) * binomial(nsite, ndn)
