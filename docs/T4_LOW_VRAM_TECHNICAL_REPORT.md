# MiniMax-H3 on Google Colab Tesla T4: Low-VRAM / Low-RAM Engineering Report

## 1. Executive Summary

Running **MiniMax-H3** (a joint Audio-Video Diffusion Transformer paired with a 32-Billion-parameter **Qwen3-VL** multimodal text/vision encoder) on a free-tier **Google Colab Tesla T4** instance presents severe hardware constraints:

- **GPU VRAM**: `14.56 GiB` usable (Turing `sm_75`)
- **System RAM**: `12.67 GiB` hard container limit (no swap allowed on Colab Docker `overlayfs`)
- **Total Model Footprint (`UD-Q2_K_XL` + `Q2_K_M` + VAEs)**: **`26.98 GB`**
  - `minimax_h3_fl2va_pruned-UD-Q2_K_XL.gguf`: `8.06 GB`
  - `qwen3vl_32b_minimax_h3-Q2_K_M.gguf`: `13.10 GB`
  - `minimax_h3_video_vae_fp16.safetensors`: `5.21 GB`
  - `minimax_h3_audio_vae_fp32.safetensors`: `0.61 GB`

Because `26.98 GB > 12.67 GB RAM`, standard CPU offloading (`--offload-to-cpu` or `--backend te=cpu`) immediately triggers the Linux kernel OOM killer (`SIGKILL / returncode -9`) during model loading.

---

## 2. Root Cause Analysis in `stable-diffusion.cpp`

Inspection of `src/model_manager.cpp` in `stable-diffusion.cpp` revealed three distinct barriers preventing true zero-copy OS page-cache memory mapping (`mmap`):

### Bug 1: `can_mmap_storage` Rejects Auxiliary Files (`source_file != 0`) and Offloaded Modes
```cpp
// UPSTREAM (Broken on Colab T4):
bool ModelManager::can_mmap_storage(const TensorState& state) const {
    if (state.sources.size() > 1 || state.source_file != 0 || !enable_mmap_ || state.residency_mode != ResidencyMode::ParamBackend) {
        return false;
    }
    ...
}
```
* In `stable-diffusion.cpp`, `source_file == 0` is the primary diffusion model, while `source_file == 1` is the auxiliary `--llm` (`qwen3vl_32b_minimax_h3-Q2_K_M.gguf`, 13.1 GB).
* Because `state.source_file != 0` returned `false`, the 13.1 GB Qwen3-VL text encoder was **never memory-mapped** and instead allocated a `13.1 GB` heap buffer in system RAM, instantly crashing the `12.67 GB` Colab container!

### Bug 2: `mmap_params` Did Not Flag Tensors as Loaded
Even when `mmap_tensors` succeeded for a group of tensors, `state->loaded_to_params_backend` remained `false`.

### Bug 3: `load_tensors_to_params_backend` Re-Read Mapped Tensors
`load_tensors_to_params_backend` passed the entire `need_load` vector to `load_tensors(need_load)` without filtering out tensors that had already been mapped via `mmap_params`.

---

## 3. The 3-Part Zero-Copy `mmap` Kernel Patch

By applying `patches/apply_sdcpp_t4_mmap_patch.py` to `src/model_manager.cpp`:

1. **`can_mmap_storage`**: Removed `state.source_file != 0` and `state.residency_mode != ResidencyMode::ParamBackend`:
   ```cpp
   bool ModelManager::can_mmap_storage(const TensorState& state) const {
       if (state.sources.size() > 1) { return false; }
       if (!enable_mmap_) { return false; }
       if (state.compute_backend == nullptr || state.params_backend == nullptr) {
           return false;
       }
       return sd_backend_is_cpu(state.compute_backend) ||
              sd_backend_is_cpu(state.params_backend) ||
              backend_supports_host_buffer(state.compute_backend);
   }
   ```
2. **`mmap_params`**: Explicitly marked mapped tensors:
   ```cpp
   if (state != nullptr && state->tensor != nullptr && state->tensor->data != nullptr) {
       block->states.push_back(state);
       state->loaded_to_params_backend = true;
   }
   ```
3. **`load_tensors_to_params_backend`**: Filtered out mapped tensors before calling `load_tensors`:
   ```cpp
   std::vector<TensorState*> need_read;
   for (TensorState* state : need_load) {
       if (!state->loaded_to_params_backend) {
           need_read.push_back(state);
       }
   }
   if (!alloc_params_buffers(need_alloc, created_storage_blocks) ||
       (!need_read.empty() && !load_tensors(need_read))) {
   ```

### Verified Telemetry on Colab Tesla T4:
- **System RAM Peak**: `~4.4 GB / 12.67 GB` (Zero OOM kills)
- **GPU VRAM Peak**: `~11.8 GB / 14.56 GB`
- **Shot 1 (T2V, 640×384, 25 frames, Audio+Video)**: Completed in `~150s` (`1.10 MB` `.webm`)
- **Shot 2 (I2V via `--init-img`, 640×384, 24 frames, Audio+Video)**: Completed in `~618s` (`1.48 MB` `.webm`)

---

## 4. Anti-Data-Loss Persistence Architecture

Colab free-tier GPU runtimes can disconnect unannounced the moment daily GPU compute units are exhausted, wiping `/content/`. To guarantee **zero lost work**:

1. **Mount Google Drive First**: Always mount `/content/drive/MyDrive/MiniMax_H3_Outputs` in Step 1 before any rendering.
2. **Synchronous Per-Shot Mirroring**: Immediately after `sd-cli` finishes each individual shot (`shot_01`, `shot_02`, ...), transcode to `.mp4`, copy both `.webm` and `.mp4` + extracted last frame (`shot_XX_last.png`) to Google Drive, and invoke `os.sync()`.
3. **Checkpoint Auto-Resume**: On restart, scan `/content/drive/MyDrive/MiniMax_H3_Outputs/raw_shots/` and automatically skip completed shots, picking up seamlessly from the last extracted keyframe.
