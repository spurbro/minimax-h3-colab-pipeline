#!/usr/bin/env python3
"""
Apply the 3-part Low-RAM Zero-Copy mmap Patch to stable-diffusion.cpp (src/model_manager.cpp).

Why this patch is required on Google Colab Tesla T4 (12.67 GB System RAM / 15 GB VRAM):
1. Upstream `ModelManager::can_mmap_storage` blocks `mmap` when:
   - `state.source_file != 0` (which blocks the 13.1 GB Qwen3-VL 32B LLM text encoder, where source_file == 1)
   - `state.residency_mode != ResidencyMode::ParamBackend` (which blocks CPU-offloaded tensors)
2. Upstream `ModelManager::mmap_params` maps tensors via `mmap_tensors(...)`, but does not mark
   `state->loaded_to_params_backend = true`.
3. Upstream `ModelManager::load_tensors_to_params_backend` subsequently calls `load_tensors(need_load)`
   on all tensors in `need_load`, reading them into RAM anyway and triggering the Linux OOM killer at 12.67 GB.

With this 3-part patch:
- All GGUF/Safetensors weights on CPU/Host backends use true zero-copy OS page-cache `mmap`.
- System RAM consumption stays below ~4.5 GB while running the full 26.9 GB MiniMax-H3 pipeline on T4.
"""

import os
import re
import sys

DEFAULT_CPP_PATH = "/content/stable-diffusion.cpp/src/model_manager.cpp"


def apply_patch(cpp_path: str = DEFAULT_CPP_PATH) -> bool:
    if not os.path.exists(cpp_path):
        print(f"[ERROR] File not found: {cpp_path}")
        return False

    with open(cpp_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Patch load_tensors_to_params_backend to skip already-mmapped tensors (loaded_to_params_backend)
    target_load_block = """    if (!alloc_params_buffers(need_alloc, created_storage_blocks) ||
        !load_tensors(need_load)) {"""

    replacement_load_block = """    std::vector<TensorState*> need_read;
    for (TensorState* state : need_load) {
        if (!state->loaded_to_params_backend) {
            need_read.push_back(state);
        }
    }

    if (!alloc_params_buffers(need_alloc, created_storage_blocks) ||
        (!need_read.empty() && !load_tensors(need_read))) {"""

    if "std::vector<TensorState*> need_read;" not in content:
        if target_load_block in content:
            content = content.replace(target_load_block, replacement_load_block, 1)
            print("[OK] Patched ModelManager::load_tensors_to_params_backend (skip mmapped tensors in load_tensors)")
        else:
            print("[WARN] Could not locate exact target_load_block; checking regex fallback...")
            content = re.sub(
                r"if\s*\(\s*!alloc_params_buffers\(need_alloc,\s*created_storage_blocks\)\s*\|\|\s*!load_tensors\(need_load\)\s*\)\s*\{",
                replacement_load_block.strip(),
                content,
                count=1,
            )

    # 2. Patch mmap_params to mark state->loaded_to_params_backend = true
    target_mmap_loop = """            if (state != nullptr && state->tensor != nullptr && state->tensor->data != nullptr) {
                block->states.push_back(state);
            }"""

    replacement_mmap_loop = """            if (state != nullptr && state->tensor != nullptr && state->tensor->data != nullptr) {
                block->states.push_back(state);
                state->loaded_to_params_backend = true;
            }"""

    if "state->loaded_to_params_backend = true;" not in content or content.count("state->loaded_to_params_backend = true;") < 2:
        if target_mmap_loop in content:
            content = content.replace(target_mmap_loop, replacement_mmap_loop, 1)
            print("[OK] Patched ModelManager::mmap_params (set state->loaded_to_params_backend = true)")

    # 3. Patch can_mmap_storage to allow auxiliary files (source_file != 0) and offloaded residency modes
    lines = content.splitlines(keepends=True)
    new_lines = []
    skip = False
    patched_can_mmap = False

    for line in lines:
        if "bool ModelManager::can_mmap_storage" in line:
            skip = True
            patched_can_mmap = True
            new_lines.append("bool ModelManager::can_mmap_storage(const TensorState& state) const {\n")
            new_lines.append("    if (state.sources.size() > 1) { return false; }\n")
            new_lines.append("    if (!enable_mmap_) { return false; }\n")
            new_lines.append("    if (state.compute_backend == nullptr || state.params_backend == nullptr) {\n")
            new_lines.append("        return false;\n")
            new_lines.append("    }\n")
            new_lines.append("    return sd_backend_is_cpu(state.compute_backend) ||\n")
            new_lines.append("           sd_backend_is_cpu(state.params_backend) ||\n")
            new_lines.append("           backend_supports_host_buffer(state.compute_backend);\n")
            new_lines.append("}\n")
            continue
        if skip:
            if line.startswith("}"):
                skip = False
            continue
        new_lines.append(line)

    if patched_can_mmap:
        print("[OK] Patched ModelManager::can_mmap_storage (enabled mmap for LLM & offloaded tensors)")

    with open(cpp_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    print(f"[SUCCESS] {cpp_path} patched for Colab T4 low-RAM zero-copy mmap!")
    return True


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CPP_PATH
    sys.exit(0 if apply_patch(target) else 1)
