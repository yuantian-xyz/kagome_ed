# Entry point:  julia -t <threads> run.jl <mode> [--config file] [--mem GB]
# Modes: smoke, estimate. Uses only the Julia standard library.
using LinearAlgebra, SparseArrays, Random, TOML, Printf, Dates
BLAS.set_num_threads(1)
const ROOT = @__DIR__
for f in ("lattice", "basis", "kernels", "io", "lanczos", "typicality", "exact", "combine", "driver")
    include(joinpath(ROOT, "src", f * ".jl"))
end
main(ARGS)
