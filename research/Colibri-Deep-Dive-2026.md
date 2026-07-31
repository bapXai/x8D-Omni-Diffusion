# Colibrì Deep-Dive — JustVugg/colibri (GLM-5.2 @ 24GB RAM) vs x8D-Omni-Diffusion (#41)

**Audited 2026-07-31 against the live upstream repo** (`git clone --depth 1
https://github.com/JustVugg/colibri`). Colibrì = pure-C engine (single-file
`c/colibri.c`, 437 KB) that runs GLM-5.2 (744B / 40B active MoE) inside a
25 GB laptop by treating **VRAM / RAM / NVMe as one managed memory hierarchy**
instead of shrinking the model on disk. All numbers below are read from the
repo, not from third-party claims.

---

## 1. Repository Code & Telemetry Deep-Dive

### 1.1 POSIX system calls in the execution engine

`c/colibri.c` drives two I/O backends; both are explicit:

| Mechanism | Code | What it does |
|---|---|---|
| `pread()` coalescing | `pread_full()`, `mir_pread()` (line ~1426) | One ~19 MB coalesced positional `pread` per expert into a 16 KB-aligned `slab` buffer (`posix_memalign(..., 16384, ...)`); GPU/CPU reads the slab in place. Handles short reads (POSIX permits them on regular files). |
| `mmap()` page-cache path | `COLI_MMAP=1` (`g_mmap`, line ~1362) | Experts become **views** inside `mmap(NULL, len, PROT_READ, MAP_SHARED, fd, 0)` of the safetensors shards — no `pread`, no slab copy. "The kernel page cache IS the cache." `madvise(MADV_WILLNEED)` triggers async readahead, then first-touch faults the pages resident. |
| `io_uring` async queue | `URING=1` (`g_uring`, line ~1965; `c/uring.h`) | `SYS_io_uring_setup` batch load/completion backend (Linux only; mutually exclusive with `COLI_MMAP`). |
| `O_DIRECT` | `COLI_DIRECT=1` (`g_direct`, line ~640) | Bypasses page cache on slab experts (drive-dependent; QLC/DRAM-less drives can regress). |
| NUMA interleave | `COLI_NUMA=1` + `mbind` (line ~750) | Interleaves expert slabs across NUMA nodes via `MPOL_MF_MOVE`; measured +7%/−14% expert-matmul on 2 sockets. |
| Pinning | `PIN_GB=N`, `mlock`/`munlock` | Hottest experts (learned from a prior `PIN=stats.txt` pass) are mlock-pinned so they are **never evicted**. |

Per-token decode is the **five-step path**: `route → union → place → overlap → learn`.
Each routed expert is hashed to one drive so readahead/PILOT prefetch and the
demand read always hit the same drive (nothing cached twice; 2×NVMe = summed
bandwidth, e.g. 9 GB/s + 3 GB/s reads ~33% faster than the fast drive alone).

### 1.2 Telemetry loops (RAM / disk / layer / token latency)

`c/telemetry.h` (dashboard protocol lines + stats persistence + hardware probe):

- **I/O byte counter**: `static _Atomic int64_t g_prof_io` — incremented with
  every `pread()`/page-fault byte from expert files; `g_dc_bytes[2]` splits
  weights vs scales. Read out on every per-turn stats line.
- **Memory**: `rss_gb()` via `getrusage(RUSAGE_SELF)` `ru_maxrss`; `hw_probe()`
  reports CPU brand (`/proc/cpuinfo`, `sysctlbyname("machdep.cpu.brand_string")`,
  Windows `__get_cpuid`), core count, total/available RAM (sysctl/mach
  `vm_statistics64`/`compat_meminfo`).
- **Expert-hit tier split**: `hit_pin` vs `hit_ecache` (`#336`) — how many
  routed experts were answered by the pinned hot-store vs the per-layer LRU.
- **Brain-map**: `ehit_mark()` records a per-layer × per-expert hit bitmap that
  feeds the dashboard's "mini-brain" and the `PIN=stats.txt` pinning pass.
- **Per-turn stats line**: `tok/s`, TTFT, RSS, expert hit %, `MIRROR:` GB-per-drive,
  cold-decode disk reads; `iobench.c` is a standalone drive-bandwidth probe.

### 1.3 MoE in-memory / drop bookmarks

- **Per-layer LRU `ecache`** (ESlot arrays) — an expert stays resident after load
  and is evicted only on capacity pressure (separate `DISK-CLASS` bookkeeping so
  eviction state can never cross-read the stock clock).
- **Pinned hot-store** `pin[]`/`npin[]` — never evicted; sized by `PIN_GB`.
- **`DROP=1`** (`g_drop`, line ~634) — explicitly `madvise`s expert pages away
  right after use on cold/low-RAM hosts (tradeoff: speculative readahead gets
  re-evicted under memory pressure).
- **MTP head** — speculative draft at `layer == n_layers` (DeepSeek-V3 style)
  MUST be **int8**: an int4 MTP head collapses to **0–4% draft acceptance**
  (upstream #8). MTP roughly halves effective disk cost once the cache is warm.
- Validation: the forward pass is **token-exact against a transformers oracle**.

### 1.4 Measured reality (from `docs/benchmarks.md`)

| Host | Config | tok/s | expert hit | RSS |
|---|---|---|---|---|
| Dev box, WSL2 VHDX, ~1 GB/s, 25 GB | cold | **~0.05–0.1** | — | ~20 GB cap |
| Intel Core Ultra 7 270K, 24 GB | default | 0.07 | 3–4% | 14.1 GB |
| 〃 | `--topp 0.7` | 0.11 | 11% | 14.7 GB |
| Apple M5 Max, 128 GB | cold | 1.06 | 23% | 21.8 GB |
| Native Linux PCIe4 NVMe, 32 GB | — | 0.5–1 | — | — |
| 6× RTX 5090, full residency | — | **6.84** | — | disk 0 |

Cold token cost is **~11 GB of disk reads/token** (75 layers × 8 routed
experts, int4). The engine is disk-bound exactly as reported.

---

## 2. Comparative Engineering Mapping

| Engineering Constraint | Colibrì (JustVugg/colibri) | x8D-Omni-Diffusion (bapXai) |
| :--- | :--- | :--- |
| File Compression Ratio | **None.** int4 container stays **372 GB** on disk (4 bit/weight); the trick is placement, not compaction | **0.008 bit/weight** (`U8 × 0.001` law): 1.56 TB → **2.837 GB** (550:1); 5.17 T omni stack → 305 MB pointer maps (13,027:1) |
| Memory Allocation Method | 16 KB-aligned `posix_memalign` slabs + `pread` streaming; per-layer LRU + mlock-pinned hot-store; `COLI_MMAP` treats kernel page cache as the cache | **Zero-copy `mmap`**: compressed state IS the running state; `/0.001` reverses live as a coordinate pointer map; only the routed expert's span is mmap'd (`moe_disk.py` SARA boundaries) |
| Storage Read Bottleneck | **~11 GB disk reads/token cold → 0.05–0.1 tok/s** on a 25 GB host; NVMe-bandwidth-bound, thrashes SSD | Expert span mmap'd in **~56 ms**, no sequential re-read per token; pointer map serves coordinates from page cache natively |
| Multi-Modal Integration | **Text-only** (GLM-5.2, Inkling 975B, Kimi-K3, OLMoE) — pure autoregressive LLM | **264-byte vocab canvas** (256 bytes + 8 specials) — text/image/audio/video are all bytes 0–255; DSpark **8×8 block-parallel masked diffusion**, not 1-token-at-a-time |

**Verdict**: Colibrì is a brilliant *placement* engine — it proves MoE active
parameter sets (~40B of 744B) make low-RAM hosting viable, and its
`COLI_MMAP` + telemetry + pin/LRU design is worth porting. But it never
attacks the bytes themselves: 372 GB stays on disk, so it cannot run on a
4 GB VPS and its cold decode is SSD-bound. x8D compacts the matrix (0.008
bit/weight) so the *file* is small enough to be fully addressable, then uses
the same mmap/pointer-map trick for zero-copy live reverse at query time.

---

## 3. Actionable Code Extraction (shipped in this repo)

Two production-ready pure-stdlib modules implement Colibrì's best ideas on top
of our byte law:

1. **`omni_diffusion/x8d_mmap.py`** — `MappedX8DReader`: zero-copy read-only
   `mmap` over an `.x8dds.gguf` (or any x8D GGUF) container, builds a payload
   **offset index** (`name → (offset, length)`), slices bytes straight out of
   the mapping (no decompression loop), exposes a `reverse()` live `/0.001`
   pointer-map view, and a `frame_view()` that walks 8×8 DSpark stream frames
   as in-place `memoryview`s. This is Colibrì's `COLI_MMAP=1` path re-expressed
   for the sub-byte container.
2. **`omni_diffusion/x8d_telemetry.py`** — `Telemetry` mirroring `telemetry.h`:
   atomic I/O byte counter (`record_io`), RSS via `getrusage`, per-8×8-block
   timing (`begin_block`/`end_block`), expert-hit split (pinned vs LRU), and a
   Colibrì-style dashboard line. Self-test via `_selftest()`.

Porting map (Colibrì → x8D):

| Colibrì | x8D equivalent |
|---|---|
| `pread` + 16 KB slab | `mmap` + offset-index slice (`x8d_mmap.py`) |
| `COLI_MMAP=1` page-cache path | `MappedX8DReader` (mmap access=READ) |
| `g_prof_io` atomic counter | `Telemetry.record_io` |
| `hit_pin` / `hit_ecache` split | `Telemetry.record_hit("pin"/"lru")` |
| `PIN=stats.txt` learned hot-store | `moE_disk.py` `SARARouter` isolated expert spans |
| MTP speculative draft (int8) | DSpark 8×8 block-parallel masked diffusion (`x8d_spec_decode.py`) |
| `iobench.c` drive probe | `Telemetry.block` I/O report |

---

## 4. Next Actionable Steps

1. **`MappedX8DReader` + SARARouter integration** — serve a whole expert from a
   mapped `.x8dds.gguf` in one zero-copy span instead of per-block file reads
   (`x8d_mmap` already exposes the offset index; wire it into `moe_disk.py`).
2. **Range-fetch / partial-reverse** — add a `fetch_expert(repo, boundary)`
   that uses the pointer map to Range-fetch only the active expert's span and
   reverse it live (mirrors Colibrì's "only 40B active" + our Kimi-K3 pointer
   map; issue #10 pattern).
3. **Learned pinning pass** — record a `PIN=stats.txt`-style per-expert hit
   histogram from `Telemetry.record_hit` and pin the hottest x8D coordinates in
   a small RAM hot-store.
4. **Dashboard** — expose the `Telemetry.dashboard()` line through the
   OpenAI-compatible server's `/healthz` for live byte/tok/s/RSS metrics.
5. **Benchmark vs upstream** — reproduce `iobench`-style numbers for x8D
   mmap-backed expert loads on the same host class (M5 Max / WSL2) to publish
   a like-for-like table.

**Framing for AGENTS.md**: Colibrì validates the byte law's market thesis —
the industry wants low-RAM large-MoE serving, but Colibrì buys placement with
a 372 GB disk footprint and 0.05 tok/s cold. x8D buys the same win *and* the
VPS-class footprint because compaction happens at the byte level, and it
adds multi-modal parallel block diffusion Colibrì cannot express. Porting
Colibrì's telemetry + mmap-view discipline keeps our discipline honest.
