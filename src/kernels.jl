# Threaded, deterministic kernels on state vectors stored as Matrix{Float64}
# of shape (D↑, D↓): column c = one ↓-configuration, rows = ↑-configurations.
# Reductions use static thread partitions and a fixed summation order, so a
# replayed Lanczos recurrence is bit-identical to the first pass. Vectors
# shorter than SMALL run single-threaded (thread spawn overhead dominates).

const CSC = SparseMatrixCSC{Float64,Int32}
const SLAB_BYTES = 2 << 20
const SMALL = 1 << 17

struct Sector
    nup::Int
    ndn::Int
    bup::SpinBasis
    bdn::SpinBasis
    Aup::CSC
    Adn::CSC
    U::Float64
    c0::Float64          # U*(nsite/4 - (nup+ndn)/2)  (μ is applied in post-processing)
    nsite::Int
    dim::Int
    slab::Int
end

struct BasisCache
    bases::Dict{Int,SpinBasis}
    hops::Dict{Int,CSC}
end
BasisCache() = BasisCache(Dict{Int,SpinBasis}(), Dict{Int,CSC}())

get_basis!(bc::BasisCache, cl::Cluster, nel::Int) = get!(bc.bases, nel) do
    SpinBasis(cl.nsite, nel)
end
get_hop!(bc::BasisCache, cl::Cluster, nel::Int) = get!(bc.hops, nel) do
    hopping_matrix(get_basis!(bc, cl, nel), cl.bonds)
end

function Sector(cl::Cluster, bc::BasisCache, nup::Int, ndn::Int, U::Float64)
    bup = get_basis!(bc, cl, nup); bdn = get_basis!(bc, cl, ndn)
    Aup = get_hop!(bc, cl, nup); Adn = get_hop!(bc, cl, ndn)
    c0 = U * (cl.nsite / 4 - (nup + ndn) / 2)
    Ddn = length(bdn)
    slab = clamp(SLAB_BYTES ÷ (8 * Ddn), 8, 512)
    slab = max(8, (slab ÷ 8) * 8)
    Sector(nup, ndn, bup, bdn, Aup, Adn, U, c0, cl.nsite, length(bup) * Ddn, slab)
end

newvec(S::Sector) = Matrix{Float64}(undef, length(S.bup), length(S.bdn))
vecbytes(S::Sector) = 8 * S.dim

nthreads_for(n::Int) = n < SMALL ? 1 : Threads.nthreads()

@inline function _range(n::Int, t::Int, nt::Int)
    lo = (n * (t - 1)) ÷ nt + 1
    hi = (n * t) ÷ nt
    lo:hi
end

# run f(t) for t in 1:nt, threaded when nt > 1
@inline function _foreach_thread(f, nt::Int)
    if nt == 1
        f(1)
    else
        Threads.@threads :static for t in 1:nt
            f(t)
        end
    end
    nothing
end

# W[i, c] = Σ_{p ∈ col i of A} A.nzval[p] * V[A.rowval[p], c]   (gather form)
function spmm_gather!(W::Matrix{Float64}, V::Matrix{Float64}, A::CSC)
    Ddn = size(V, 2); n = size(A, 2)
    size(W, 1) == n || error("spmm_gather!: shape mismatch")
    colptr = A.colptr; rowval = A.rowval; nzval = A.nzval
    nt = nthreads_for(length(W))
    _foreach_thread(nt) do t
        @inbounds for c in _range(Ddn, t, nt), i in 1:n
            s = 0.0
            for p in colptr[i]:colptr[i+1]-1
                s += nzval[p] * V[rowval[p], c]
            end
            W[i, c] = s
        end
    end
    W
end

# Y = H V for the sector: ↑-hopping + diagonal (columnwise), then ↓-hopping (row slabs)
function hv!(Y::Matrix{Float64}, V::Matrix{Float64}, S::Sector)
    Dup, Ddn = size(V)
    colptr = S.Aup.colptr; rowval = S.Aup.rowval; nzval = S.Aup.nzval
    sup = S.bup.states; sdn = S.bdn.states; U = S.U; c0 = S.c0
    nt = nthreads_for(S.dim)
    _foreach_thread(nt) do t
        @inbounds for c in _range(Ddn, t, nt)
            sd = sdn[c]
            for i in 1:Dup
                s = 0.0
                for p in colptr[i]:colptr[i+1]-1
                    s += nzval[p] * V[rowval[p], c]
                end
                Y[i, c] = s + (U * count_ones(sup[i] & sd) + c0) * V[i, c]
            end
        end
    end
    cp = S.Adn.colptr; rv = S.Adn.rowval; nz = S.Adn.nzval
    slab = S.slab
    nslab = cld(Dup, slab)
    _foreach_thread(nt) do t
        @inbounds for sb in _range(nslab, t, nt)
            i0 = (sb - 1) * slab + 1
            i1 = min(sb * slab, Dup)
            for c in 1:Ddn
                for p in cp[c]:cp[c+1]-1
                    cc = rv[p]; a = nz[p]
                    @simd for i in i0:i1
                        Y[i, c] += a * V[i, cc]
                    end
                end
            end
        end
    end
    Y
end

function tdot(x::Matrix{Float64}, y::Matrix{Float64})
    n = length(x); nt = nthreads_for(n)
    parts = zeros(nt)
    _foreach_thread(nt) do t
        acc = 0.0
        @inbounds @simd for i in _range(n, t, nt)
            acc += x[i] * y[i]
        end
        parts[t] = acc
    end
    total = 0.0
    for t in 1:nt; total += parts[t]; end
    total
end

tnorm(x::Matrix{Float64}) = sqrt(tdot(x, x))

function tscal!(x::Matrix{Float64}, a::Float64)
    n = length(x); nt = nthreads_for(n)
    _foreach_thread(nt) do t
        @inbounds @simd for i in _range(n, t, nt)
            x[i] *= a
        end
    end
    x
end

function tcopy!(dst::AbstractMatrix, src::Matrix{Float64})
    n = length(src); nt = nthreads_for(n)
    _foreach_thread(nt) do t
        @inbounds @simd for i in _range(n, t, nt)
            dst[i] = src[i]
        end
    end
    dst
end

# w ← w − a v − b vprev ; returns ‖w‖ (b == 0 skips vprev)
function lanczos_update!(w::Matrix{Float64}, v::Matrix{Float64}, vprev::Matrix{Float64}, a::Float64, b::Float64)
    n = length(w); nt = nthreads_for(n)
    parts = zeros(nt)
    _foreach_thread(nt) do t
        acc = 0.0
        if b == 0.0
            @inbounds @simd for i in _range(n, t, nt)
                x = w[i] - a * v[i]
                w[i] = x
                acc += x * x
            end
        else
            @inbounds @simd for i in _range(n, t, nt)
                x = w[i] - a * v[i] - b * vprev[i]
                w[i] = x
                acc += x * x
            end
        end
        parts[t] = acc
    end
    total = 0.0
    for t in 1:nt; total += parts[t]; end
    sqrt(total)
end

# out[i, a] = ⟨Q[i], W[a]⟩, streamed once (Q may be Float32 or Float64 storage)
function blockdot!(out::Matrix{Float64}, Q::Vector{<:Matrix}, W::Vector{Matrix{Float64}})
    nq = length(Q); nw = length(W)
    n = length(W[1]); nt = nthreads_for(n)
    parts = zeros(nq, nw, nt)
    BLK = 16384
    _foreach_thread(nt) do t
        rg = _range(n, t, nt)
        @inbounds for b0 in first(rg):BLK:last(rg)
            b1 = min(b0 + BLK - 1, last(rg))
            for i in 1:nq
                q = Q[i]
                for a in 1:nw
                    w = W[a]
                    s = 0.0
                    @simd for idx in b0:b1
                        s += Float64(q[idx]) * w[idx]
                    end
                    parts[i, a, t] += s
                end
            end
        end
    end
    fill!(out, 0.0)
    for t in 1:nt, a in 1:nw, i in 1:nq
        out[i, a] += parts[i, a, t]
    end
    out
end

# deterministic Gaussian fill (per-thread streams keyed on seed and chunk)
function fill_randn!(x::Matrix{Float64}, seed::Integer)
    n = length(x); nt = nthreads_for(n)
    _foreach_thread(nt) do t
        rng = Random.Xoshiro(UInt64(seed) * 0x9E3779B97F4A7C15 + UInt64(t))
        @inbounds for i in _range(n, t, nt)
            x[i] = randn(rng)
        end
    end
    x
end

# Fused cross kernel: out[i, a] = ⟨Q[i] | c_a P⟩ with c_a = B_aᵀ applied column-wise on the fly,
# so P and every Q[i] are streamed exactly once. Bs[a] is the CSC of B_a = c†_a (D_{n+1} × D_n);
# (B_aᵀ P)[row, c] = Σ_{p ∈ col row of B_a} nz[p] · P[rv[p], c].
function cross_dots!(out::Matrix{Float64}, Q::Vector{<:Matrix}, P::Matrix{Float64}, Bs::Vector{CSC})
    nq = length(Q); na = length(Bs)
    Dn = size(Bs[1], 2); Ddn = size(P, 2)
    size(Q[1], 1) == Dn || error("cross_dots!: shape mismatch")
    nt = nthreads_for(length(P))
    parts = zeros(nq, na, nt)
    _foreach_thread(nt) do t
        wbuf = [Vector{Float64}(undef, Dn) for _ in 1:na]
        @inbounds for c in _range(Ddn, t, nt)
            for a in 1:na
                B = Bs[a]; colptr = B.colptr; rowval = B.rowval; nzval = B.nzval
                w = wbuf[a]
                for row in 1:Dn
                    acc = 0.0
                    for p in colptr[row]:colptr[row+1]-1
                        acc += nzval[p] * P[rowval[p], c]
                    end
                    w[row] = acc
                end
            end
            for i in 1:nq
                q = Q[i]
                for a in 1:na
                    w = wbuf[a]
                    acc = 0.0
                    @simd for row in 1:Dn
                        acc += Float64(q[row, c]) * w[row]
                    end
                    parts[i, a, t] += acc
                end
            end
        end
    end
    fill!(out, 0.0)
    for t in 1:nt, a in 1:na, i in 1:nq
        out[i, a] += parts[i, a, t]
    end
    out
end
