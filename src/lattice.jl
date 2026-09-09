# Kagome torus geometry — SmoQy conventions (hubbard_kagome_UV.jl):
#   a1 = (1,0), a2 = (1/2, √3/2); orbitals A=(0,0), B=a1/2, C=a2/2
#   6 NN bond templates (orb_from, orb_to, cell displacement):
#   (A,B,0) (A,C,0) (B,C,0) (B,A,+a1) (C,A,+a2) (B,C,+a1−a2)
#   site index = orb + 3*(cell-1), cell = m1 + L1*m2 (+1, 1-based), m1 fastest.
# Bonds are ACCUMULATED: for L=1 in a direction a wrap lands on an existing pair
# and the amplitude doubles (stage-2.4 finding); for L≥2 all 6·Ncell pairs are
# distinct. Either way K equals the Bloch-folded ε(k) on the cluster k-grid,
# which `check_bloch` asserts.

const BOND_TEMPLATES = [(1, 2, 0, 0), (1, 3, 0, 0), (2, 3, 0, 0),
                        (2, 1, 1, 0), (3, 1, 0, 1), (2, 3, 1, -1)]

struct Cluster
    L1::Int
    L2::Int
    ncell::Int
    nsite::Int
    t::Float64
    K::Matrix{Float64}                 # one-body hopping matrix (nsite×nsite), H0 = Σ K_ij c†_i c_j
    bonds::Vector{Tuple{Int,Int,Float64}}   # (i<j, amplitude) after accumulation
    site_cell::Vector{Int}             # 1-based cell of each site
    site_orb::Vector{Int}              # 1..3
    cell_m::Vector{Tuple{Int,Int}}     # (m1, m2) of each cell, 0-based
end

cell_index(L1, L2, m1, m2) = mod(m1, L1) + L1 * mod(m2, L2) + 1
site_index(cell, orb) = orb + 3 * (cell - 1)

function Cluster(L1::Int, L2::Int; t::Float64 = 1.0)
    ncell = L1 * L2
    nsite = 3 * ncell
    K = zeros(nsite, nsite)
    site_cell = zeros(Int, nsite); site_orb = zeros(Int, nsite)
    cell_m = Vector{Tuple{Int,Int}}(undef, ncell)
    for m2 in 0:L2-1, m1 in 0:L1-1
        c = cell_index(L1, L2, m1, m2)
        cell_m[c] = (m1, m2)
        for o in 1:3
            s = site_index(c, o); site_cell[s] = c; site_orb[s] = o
        end
    end
    for c in 1:ncell
        (m1, m2) = cell_m[c]
        for (o1, o2, d1, d2) in BOND_TEMPLATES
            c2 = cell_index(L1, L2, m1 + d1, m2 + d2)
            i = site_index(c, o1); j = site_index(c2, o2)
            K[i, j] += -t; K[j, i] += -t
        end
    end
    bonds = Tuple{Int,Int,Float64}[]
    for i in 1:nsite, j in i+1:nsite
        K[i, j] != 0 && push!(bonds, (i, j, K[i, j]))
    end
    Cluster(L1, L2, ncell, nsite, t, K, bonds, site_cell, site_orb, cell_m)
end

# Bloch Hamiltonian in the CELL gauge (phase e^{ik·R} with R the cell vector,
# no sublattice offset): k given in reciprocal-basis fractions (κ1, κ2),
# k·R = 2π(κ1 m1 + κ2 m2).
function bloch_h(cl::Cluster, κ1::Float64, κ2::Float64)
    h = zeros(ComplexF64, 3, 3)
    for (o1, o2, d1, d2) in BOND_TEMPLATES
        ph = cis(2π * (κ1 * d1 + κ2 * d2))
        h[o1, o2] += -cl.t * ph      # c†_{o1,R} c_{o2,R+d}: in k-space e^{ik·d}
        h[o2, o1] += -cl.t * conj(ph)
    end
    Hermitian(h)
end

kgrid(cl::Cluster) = [(n1 / cl.L1, n2 / cl.L2) for n2 in 0:cl.L2-1 for n1 in 0:cl.L1-1]

# Assert the torus K equals ⊕_k h(k) on the cluster k-grid (catches bond-list,
# wrap and gauge mistakes). Returns the max deviation of the sorted spectra.
function check_bloch(cl::Cluster)
    ek = sort(vcat([eigvals(bloch_h(cl, κ...)) for κ in kgrid(cl)]...))
    eK = sort(eigvals(Symmetric(cl.K)))
    maximum(abs.(ek .- eK))
end

# Phases e^{ik·R} per site for the momentum operator c†_{k,a} = N_c^{-1/2} Σ_R e^{ik·R} c†_{R,a}
function kphases(cl::Cluster, κ1::Float64, κ2::Float64)
    ph = Vector{ComplexF64}(undef, cl.nsite)
    for s in 1:cl.nsite
        (m1, m2) = cl.cell_m[cl.site_cell[s]]
        ph[s] = cis(2π * (κ1 * m1 + κ2 * m2)) / sqrt(cl.ncell)
    end
    ph
end

isreal_k(cl::Cluster, κ1, κ2) = isinteger(2κ1 * cl.L1 / cl.L1 * 1) && isinteger(2κ1) && isinteger(2κ2)

# Named k-points on the grid (reciprocal fractions)
const KPOINT_NAMES = Dict("G" => (0.0, 0.0), "M1" => (0.5, 0.0), "M2" => (0.0, 0.5), "M3" => (0.5, 0.5))
function kpoint(cl::Cluster, name::AbstractString)
    haskey(KPOINT_NAMES, name) || error("unknown k-point name $name (use G, M1, M2, M3 or 'n1/L1,n2/L2')")
    (κ1, κ2) = KPOINT_NAMES[name]
    (isinteger(κ1 * cl.L1) && isinteger(κ2 * cl.L2)) || error("k-point $name is not on the $(cl.L1)×$(cl.L2) grid")
    (κ1, κ2)
end
