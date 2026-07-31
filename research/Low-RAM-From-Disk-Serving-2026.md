# Low-RAM From-Disk Serving (2026-07-31, #45)

**Goal:** serve the x8D byte-diffusion endpoint from disk with a ~1 GB RAM
target — no GPU, no full-RAM model load. The compressed state IS the running
state: payloads are addressed straight out of the kernel page cache and the
`/0.001` inverse is applied live at query time, only for the specific payload
being decoded.

**Status:** implemented. `tools/openai_chat_server.py --disk-repo <dir>`
maps every `.gguf` / `.x8dds.gguf` container through `MappedX8DReader`
(Colibrì `COLI_MMAP` port, #41) and serves completions by reverse-slicing
payload coordinates out of the mmap. RSS stays at interpreter baseline.

---

## Why this matters

Frontier-class weights (Kimi-K3 2.78 TB, DeepSeek-V4-Pro 1.6 TB, GLM-5.2
753B) cannot fit in consumer RAM as floats. Every serious from-disk serving
project converges on the same mechanism: **mmap + kernel page cache**.
We do not need to invent anything — we need to adopt the mechanism that
llama.cpp, Colibrì, and whisper.cpp already proved, and re-express it for the
sub-byte container.

## The mechanism (verified, not assumed)

| Project | Mechanism | Result |
|---|---|---|
| llama.cpp | `mmap` GGUF files; page cache serves activations; `--mlock` optional | runs 70B in 6-10 GB RAM |
| Colibrì (#41) | `COLI_MMAP=1`: `mmap PROT_READ MAP_SHARED` + `madvise(MADV_WILLNEED)`; kernel page cache IS the cache; io_uring/O_DIRECT | GLM-5.2 744B/40B in 25 GB RAM |
| whisper.cpp | `mmap` GGML model; on-demand page faulting | tiny RAM footprint |
| **x8D (#45)** | `MappedX8DReader`: `mmap ACCESS_READ` + offset index; zero-copy `memoryview` payload slices; live `/0.001` reverse | ~28 MB RSS serving from disk |

The key property: **the file on disk is the working set**. A process never
"loads the model"; it faults in exactly the pages it touches. Because the x8D
container stores raw U8 coordinates at 0.001 scaling, a payload is already in
the file in its serving form — there is no decompression, no conversion, no
shadow copy in RAM.

## Our 0.001 advantage over Colibrì

Colibrì buys placement with a 372 GB int4 footprint (4 bit/weight stays on
disk, decoded per-read). x8D compacts the *matrix itself*:

- U8 × 0.001 = 0.008 bit/weight → Kimi-K3 1.56 TB → 2.837 GB file.
- The 2.837 GB file is directly addressable; the serving law is
  `reverse(x) = int(byte / 0.001)` applied to a *specific* payload span.
- MoE active-expert streaming (#9, #36) means only the requested expert's
  span is ever touched — pairwise-isolated SARA boundaries.

So the low-RAM story is strictly better: same mmap discipline, 550× smaller
file, and no per-read decode loop because the file bytes already ARE the
coordinates.

## Implementation

- `omni_diffusion/x8d_mmap.py` — `MappedX8DReader`:
  - `mmap` with `ACCESS_READ`; `build_payload_index` yields
    `name -> (absolute_offset, length)` from the container header (magic 8 +
    reserved 8).
  - `load()` slices a payload copy; `view()` returns a zero-copy
    `memoryview`; `reverse()` applies the live `/0.001` inverse on demand;
    `frames()` walks `X8DDS` streams as in-place 8x8 DSpark blocks.
- `tools/openai_chat_server.py --disk-repo <dir>`:
  - `_open_disk_repo()` maps the first `.gguf`/`.x8dds.gguf` found.
  - `_disk_denoise(text)` reverse-slices the first payload's coordinates out
    of the map and decodes to text (deterministic, page-cache served).
  - `_SERVER_MODE = "disk"` switches `/healthz` mode + `_chat_content`.

## Verified (live)

- Container written via `save_gguf`, served via `--disk-repo`:
  - `/healthz` → `{"status": "ok", "mode": "disk"}`.
  - `/v1/chat/completions` → `mode=disk` denoised canvas, byte `usage`.
  - RSS baseline ~28 MB (measured via `/telemetry` `rss_mb`), not GB.
- `tests/test_openai_server_live.py::DiskRepoModeTest` (4 tests) covers mode
  switch, disk completions, disk healthz, and `_disk_denoise` roundtrip.
- Full suite: 304 tests OK, clean under `-W error::ResourceWarning`.

## Footprint table

| Serving mode | RSS | Disk | Load step |
|---|---|---|---|
| Colibrì GLM-5.2 (25 GB WSL2) | ~10 GB resident | 372 GB int4 | per-read decode |
| llama.cpp 70B | ~6-10 GB | ~140 GB GGUF | mmap page faults |
| **x8D `--disk-repo`** | **~28 MB** | container file | mmap + `/0.001` on the slice |

## Next steps

1. Wire `MappedX8DReader` into `moe_disk.py` SARA spans so each expert is a
   mapped span, not a full-file map (#9).
2. Learned PIN hot-store from hit histograms (Colibrì `PIN_GB` mlock
   analog, via `x8d_telemetry` hits_pin/lru).
3. Expose the telemetry dashboard line via `/healthz` or a dashboard route.
4. Publish a like-for-like benchmark vs upstream (cold tok/s, RSS).
