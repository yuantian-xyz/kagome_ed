# kagome_ed — exact diagonalization runs for the kagome Hubbard model

给室友的说明 (中文在前，英文在后)。整个程序只用 Julia 标准库，**不在系统里安装任何东西**：
`setup.ps1` 会从 Julia 官网 (julialang-s3.julialang.org) 下载官方的便携版 Julia 1.12.6 压缩包，
校验 SHA256 后解压到本文件夹的 `julia\` 子目录。删掉这个文件夹就等于全部卸载。
程序默认**只用一个 CPU 核心**，优先级 BelowNormal，不会影响你正常使用电脑。

## 第 0 批（约 15 分钟）

1. 把这个仓库 clone（或 Download ZIP 解压）到任意位置，例如 `D:\kagome_ed\`。
2. 双击 **`batch0.bat`**。它会依次：下载并解压 Julia（约 230 MB）→ 跑 `smoke`（约 3 分钟的自检）→
   跑 `estimate`（约 10 分钟，测速并申请约 150 GB 内存做一次分配测试，测完立刻释放）。
3. 结束后窗口会提示。把整个 **`output` 文件夹压缩成 zip 发给我**（只有几 MB）。
   如果中途报错，也请把 `output` 打包发我，里面有日志。

不需要装 git 以外的任何东西；如果没有 git，直接在 GitHub 页面 "Download ZIP"。
Windows 可能弹出 "Windows 已保护你的电脑" — 这是因为脚本未签名，点 "更多信息 → 仍要运行" 即可；
或者在 PowerShell 里手动执行下面的三行（等价于双击 bat）：

    powershell -NoProfile -ExecutionPolicy Bypass -File setup.ps1
    powershell -NoProfile -ExecutionPolicy Bypass -File run.ps1 smoke 1
    powershell -NoProfile -ExecutionPolicy Bypass -File run.ps1 estimate 1

后面的第 1 批（真正的计算，单核大约要跑几天）会是同样的方式：`run.ps1 batch1 1`，
可以随时 Ctrl-C 或重启电脑，重新运行会从断点继续。到时候我再单独说明。

---

## English

Finite-temperature exact diagonalization (Lanczos + typicality) of the nearest-neighbour
kagome Hubbard model on periodic clusters, producing the momentum-resolved single-particle
Green's function G_ab(k,τ) and 𝒢_ab(k,iωₙ) at β = 3. Standard library only; the Julia
runtime is downloaded from the official julialang.org server at setup time (checksum
verified) into `./julia`, nothing is installed system-wide.

### Batch 0 — setup, self-test, timing (≈ 15 min)

Double-click `batch0.bat`, or run the three PowerShell lines above. Then zip the whole
`output/` folder and send it back. Everything a run produces goes into
`output/<mode>_<timestamp>/`:

| file | content |
|---|---|
| `log.txt` | full log with timestamps |
| `config.toml` | the configuration used |
| `sysinfo.json` | Julia version, CPU, RAM, threads |
| `smoke.json` | pass/fail flags and numbers of the self-test |
| `estimate.json` | kernel timings, memory test, ETA model for batch 1 |
| `results_*/` | per-(sector, random vector) results of the 12-site test run (`.json` + `.npy`) |

`output/console_<mode>_<stamp>.txt` and `stderr_*.txt` are copies of the console.

### Modes

    run.ps1 smoke [threads]      # ~3 min: exactness checks vs dense diagonalisation + a 12-site run
    run.ps1 estimate [threads]   # ~10 min: times the kernels on the 18-site sectors, touches ~150 GB, prints ETA
    run.ps1 batch1 [threads]     # (next) the production run; resumable

`threads` defaults to 1. Passing e.g. `8` enables the threaded kernels (deterministic at
any thread count) if the machine's owner agrees.

### Physics summary

H = −t Σ⟨ij⟩σ (c†c + h.c.) + U Σ (n↑−½)(n↓−½), t = 1, kagome 3×2 torus (18 sites) at the
upper van Hove filling n = 5/6 (N = 15 electrons, canonical; grand canonical in a later
batch), U = 1, 2, 3, β = 3. Per (n↑,n↓) sector and random vector r the code computes
Z_r = ⟨r|e^{−βH}|r⟩ and the cross matrices ⟨q_i|c_{ka}|p_j⟩ between the Krylov bases of r
and of c†_{kb} r; from them G(τ) on a Δτ = 0.05 grid, 𝒢(iωₙ), and the Lehmann data
(ε, ε′, W) for later real-axis evaluation. Sectors are combined with their dimensions
(and e^{βμN} for the grand-canonical ensemble) in post-processing; errors by jackknife
over r.

### Layout

    run.jl  src/  configs/  setup.ps1  run.ps1  batch0.bat  PLAN.md  output/ (created)
