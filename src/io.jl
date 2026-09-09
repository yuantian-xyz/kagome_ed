# Minimal .npy writer (numpy v1.0 header, Fortran order = Julia column-major)
# and a JSON writer — stdlib only, no packages on the target machine.

function npy_write(path::AbstractString, A::AbstractArray{T}) where {T}
    dt = T === Float64 ? "<f8" : T === Float32 ? "<f4" : T === Int64 ? "<i8" :
         T === Int32 ? "<i4" : T === Bool ? "|b1" : error("npy_write: unsupported eltype $T")
    shp = size(A)
    shpstr = length(shp) == 1 ? "($(shp[1]),)" : "(" * join(string.(shp), ", ") * ")"
    header = "{'descr': '$dt', 'fortran_order': True, 'shape': $shpstr, }"
    pad = 64 - mod(10 + length(header) + 1, 64)
    pad == 64 && (pad = 0)
    header *= " "^pad * "\n"
    open(path, "w") do io
        write(io, UInt8(0x93)); write(io, "NUMPY"); write(io, UInt8(1)); write(io, UInt8(0))
        write(io, htol(UInt16(length(header))))
        write(io, header)
        write(io, collect(A))
    end
    path
end

_json(io, x::AbstractString) = (print(io, '"'); for ch in x
        ch == '"' ? print(io, "\\\"") : ch == '\\' ? print(io, "\\\\") :
        ch == '\n' ? print(io, "\\n") : print(io, ch)
    end; print(io, '"'))
_json(io, x::Bool) = print(io, x ? "true" : "false")
_json(io, x::Nothing) = print(io, "null")
_json(io, x::Integer) = print(io, x)
_json(io, x::AbstractFloat) = isfinite(x) ? print(io, repr(Float64(x))) : print(io, "null")
_json(io, x::Symbol) = _json(io, string(x))
function _json(io, x::AbstractDict)
    print(io, '{'); first = true
    for (k, v) in x
        first || print(io, ", "); first = false
        _json(io, string(k)); print(io, ": "); _json(io, v)
    end
    print(io, '}')
end
function _json(io, x::Union{AbstractVector,Tuple})
    print(io, '['); first = true
    for v in x
        first || print(io, ", "); first = false
        _json(io, v)
    end
    print(io, ']')
end
_json(io, x::AbstractMatrix) = _json(io, [x[i, :] for i in 1:size(x, 1)])
_json(io, x) = _json(io, string(x))

function json_write(path::AbstractString, x)
    open(path, "w") do io
        _json(io, x); println(io)
    end
    path
end

# ---- run output folder + tee logging ------------------------------------
mutable struct Out
    dir::String
    log::IOStream
    t0::Float64
end

function open_output(root::AbstractString, mode::AbstractString)
    stamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    dir = joinpath(root, "output", "$(mode)_$(stamp)")
    mkpath(dir)
    write(joinpath(root, "output", "LATEST.txt"), dir * "\n")
    Out(dir, open(joinpath(dir, "log.txt"), "w"), time())
end

function logmsg(o::Out, parts...)
    msg = string(parts...)
    stamp = @sprintf("[%8.1fs] ", time() - o.t0)
    println(stamp, msg); flush(stdout)
    println(o.log, stamp, msg); flush(o.log)
end

function sysinfo()
    Dict(
        "julia" => string(VERSION),
        "os" => string(Sys.KERNEL), "arch" => string(Sys.ARCH),
        "cpu_threads" => Sys.CPU_THREADS, "julia_threads" => Threads.nthreads(),
        "total_memory_GB" => Sys.total_memory() / 2^30,
        "free_memory_GB" => Sys.free_memory() / 2^30,
        "cpu" => Sys.cpu_info()[1].model,
        "time" => string(Dates.now()),
    )
end

gb(x) = @sprintf("%.2f GB", x / 2^30)
