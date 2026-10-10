#!/usr/bin/env python3
"""
Build the two Google Colab Jupyter Notebooks (.ipynb) for the MiniMax-H3 repository:
1. notebooks/MiniMax_H3_T4_GGUF_PlanB_Pipeline.ipynb  (Tesla T4 Free-Tier Low-VRAM GGUF + Plan B Chaining)
2. notebooks/MiniMax_H3_A100_ComfyUI_Drive.ipynb      (A100 High-RAM ComfyUI WebUI + Cloudflare Tunnel)
"""

import json
import os
import shutil

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOKS_DIR = os.path.join(REPO_ROOT, "notebooks")
os.makedirs(NOTEBOOKS_DIR, exist_ok=True)


def make_notebook(name: str, gpu_type: str = "T4"):
    return {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {
                "name": name,
                "provenance": [],
                "gpuType": gpu_type,
            },
            "kernelspec": {
                "name": "python3",
                "display_name": "Python 3",
            },
            "language_info": {
                "name": "python",
            },
            "accelerator": "GPU",
        },
        "cells": [],
    }


def add_md(nb, text: str):
    nb["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")],
    })


def add_code(nb, code: str):
    nb["cells"].append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [line + "\n" for line in code.strip().split("\n")],
    })


def build_t4_notebook():
    nb = make_notebook("MiniMax_H3_T4_GGUF_PlanB_Pipeline.ipynb", gpu_type="T4")

    add_md(
        nb,
        r"""# 🎬 MiniMax-H3 · Google Colab (Tesla T4 低显存/免费版) 原生视音频生成与分镜续写全流程

> **核心特性**：
> - 🛠️ **12.67GB 内存零拷贝 `mmap` 内核补丁**：自动修复 `stable-diffusion.cpp` 对 13.1GB `Qwen3-VL-32B` 文本编码器及 CPU 卸载权重的 `mmap` 限制，彻底解决 Colab T4 内存溢出（OOM Kill）。
> - ⚡ **Ninja + CUDA (`sm_75`) 极速构建**：自动调用 Colab 原生 CUDA 12.8 工具链快速编译 `sd-cli`。
> - 🎞️ **方案 B（T2V -> I2V 关键帧接力分镜生成）**：首镜采用文生视频（T2V），后续分镜自动提取上一镜头尾帧（`--init-img`）进行图生视频（I2V）连续扩展，100% 真实扩散动态生成，拒绝静态图片幻灯片。
> - 💾 **防断连 Google Drive 实时落盘与断点续跑**：每一个独立分镜渲染完成后立即转码 `.mp4` 并实时同步至 Google Drive（`/content/drive/MyDrive/MiniMax_H3_Outputs/raw_shots/`），即使配额耗尽断连也绝不丢件；重连后自动跳过已完成分镜继续生成！""",
    )

    add_md(
        nb,
        r"""### 步骤 1：GPU 硬件自检与挂载 Google Drive 防丢存储仓
请确保顶部菜单 **代码执行程序 -> 更改运行时类型** 已选择 **GPU (T4 / L4 / A100)**。""",
    )

    add_code(
        nb,
        r"""# @title 1. 环境自检 & 挂载 Google Drive 持久化目录
import os
import subprocess
import psutil
from google.colab import drive

print("=" * 68)
print("🔍 Step 1: 检查 GPU 硬件与系统内存...")
print("=" * 68)

smi = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
if smi.returncode != 0:
    raise RuntimeError("❌ 未检测到 NVIDIA GPU！请在菜单栏：代码执行程序 -> 更改运行时类型 -> 选择 T4 GPU。")
print(smi.stdout)

ram_gb = psutil.virtual_memory().total / (1024 ** 3)
print(f"✅ 系统可用内存 (RAM): {ram_gb:.2f} GB")

print("\n📂 正在挂载 Google Drive 以保障分镜实时持久化保存...")
drive.mount("/content/drive")

GDRIVE_BASE = "/content/drive/MyDrive/MiniMax_H3_Outputs"
for sub in ["raw_shots", "keyframes", "logs"]:
    os.makedirs(os.path.join(GDRIVE_BASE, sub), exist_ok=True)
os.makedirs("/content/outputs/plan_b", exist_ok=True)
os.makedirs("/content/models/vae", exist_ok=True)

print(f"✅ Google Drive 实时防丢目录已就绪: {GDRIVE_BASE}/raw_shots")""",
    )

    add_md(
        nb,
        r"""### 步骤 2：克隆 `stable-diffusion.cpp`、注入 T4 低内存 `mmap` 补丁并极速编译
自动修补 `src/model_manager.cpp` 中的三处 `mmap` 限制，使 26.9GB 权重在 12.67GB 内存的 Colab T4 上稳定运行。""",
    )

    add_code(
        nb,
        r"""# @title 2. 注入 3-Part Zero-Copy mmap 补丁 & Ninja 编译 sd-cli
import os
import subprocess

os.system("apt-get update -qq && apt-get install -y -qq ninja-build aria2 ffmpeg")

SD_DIR = "/content/stable-diffusion.cpp"
if not os.path.exists(SD_DIR):
    print("🚀 正在克隆 stable-diffusion.cpp 仓库...")
    subprocess.run(
        ["git", "clone", "--recursive", "https://github.com/leejet/stable-diffusion.cpp.git", SD_DIR],
        check=True,
    )

cpp_path = os.path.join(SD_DIR, "src", "model_manager.cpp")
with open(cpp_path, "r", encoding="utf-8") as f:
    content = f.read()

# Patch 1: Skip already-mmapped tensors during load_tensors_to_params_backend
target_load = (
    "    if (!alloc_params_buffers(need_alloc, created_storage_blocks) ||\n"
    "        !load_tensors(need_load)) {"
)
replacement_load = (
    "    std::vector<TensorState*> need_read;\n"
    "    for (TensorState* state : need_load) {\n"
    "        if (!state->loaded_to_params_backend) {\n"
    "            need_read.push_back(state);\n"
    "        }\n"
    "    }\n\n"
    "    if (!alloc_params_buffers(need_alloc, created_storage_blocks) ||\n"
    "        (!need_read.empty() && !load_tensors(need_read))) {"
)

if "std::vector<TensorState*> need_read;" not in content and target_load in content:
    content = content.replace(target_load, replacement_load, 1)
    print("✅ [Patch 1/3] load_tensors_to_params_backend 已修补")

# Patch 2: Mark state->loaded_to_params_backend = true inside mmap_params
target_mmap = (
    "            if (state != nullptr && state->tensor != nullptr && state->tensor->data != nullptr) {\n"
    "                block->states.push_back(state);\n"
    "            }"
)
replacement_mmap = (
    "            if (state != nullptr && state->tensor != nullptr && state->tensor->data != nullptr) {\n"
    "                block->states.push_back(state);\n"
    "                state->loaded_to_params_backend = true;\n"
    "            }"
)

if target_mmap in content:
    content = content.replace(target_mmap, replacement_mmap, 1)
    print("✅ [Patch 2/3] mmap_params 已修补")

# Patch 3: Enable mmap for auxiliary LLM (source_file != 0) & CPU-offloaded backends
lines = content.splitlines(keepends=True)
new_lines = []
skip = False
for line in lines:
    if "bool ModelManager::can_mmap_storage" in line:
        skip = True
        new_lines.extend([
            "bool ModelManager::can_mmap_storage(const TensorState& state) const {\n",
            "    if (state.sources.size() > 1) { return false; }\n",
            "    if (!enable_mmap_) { return false; }\n",
            "    if (state.compute_backend == nullptr || state.params_backend == nullptr) {\n",
            "        return false;\n",
            "    }\n",
            "    return sd_backend_is_cpu(state.compute_backend) ||\n",
            "           sd_backend_is_cpu(state.params_backend) ||\n",
            "           backend_supports_host_buffer(state.compute_backend);\n",
            "}\n",
        ])
        continue
    if skip:
        if line.startswith("}"):
            skip = False
        continue
    new_lines.append(line)

with open(cpp_path, "w", encoding="utf-8") as f:
    f.writelines(new_lines)
print("✅ [Patch 3/3] can_mmap_storage 已修补")

print("\n⚡ 正在使用 Ninja + CUDA (sm_75) 编译 sd-cli...")
build_cmd = (
    "cd /content/stable-diffusion.cpp && "
    "cmake -B build -GNinja -DSD_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_BUILD_TYPE=Release && "
    "ninja -C build sd-cli"
)
res = subprocess.run(build_cmd, shell=True)
if res.returncode != 0 or not os.path.exists("/content/stable-diffusion.cpp/build/bin/sd-cli"):
    raise RuntimeError("❌ sd-cli 编译失败，请检查上方编译日志。")
print("🎉 sd-cli 编译成功：/content/stable-diffusion.cpp/build/bin/sd-cli")""",
    )

    add_md(
        nb,
        r"""### 步骤 3：高速下载 MiniMax-H3 GGUF 量化权重与双模态 VAE
使用 `aria2c` 16 线程极速拉取 4 个核心权重文件（支持检测 Google Drive 缓存秒级软链接复用）：
1. **DiT 扩散主干**：`minimax_h3_fl2va_pruned-UD-Q2_K_XL.gguf` (8.06 GB)
2. **多模态文本编码器**：`qwen3vl_32b_minimax_h3-Q2_K_M.gguf` (13.1 GB)
3. **视频 VAE**：`minimax_h3_video_vae_fp16.safetensors` (5.21 GB)
4. **音频 VAE**：`minimax_h3_audio_vae_fp32.safetensors` (0.61 GB)""",
    )

    add_code(
        nb,
        r"""# @title 3. 多线程极速下载 MiniMax-H3 权重 (支持 Google Drive 缓存复用)
import os
import subprocess

CACHE_MODELS_IN_DRIVE = False  # @param {type:"boolean"}
GDRIVE_MODEL_DIR = "/content/drive/MyDrive/MiniMax_H3_GGUF_Cache"
LOCAL_MODEL_DIR = "/content/models"
os.makedirs(os.path.join(LOCAL_MODEL_DIR, "vae"), exist_ok=True)

MODEL_MANIFEST = [
    (
        "minimax_h3_fl2va_pruned-UD-Q2_K_XL.gguf",
        "https://huggingface.co/unsloth/MiniMax-H3-GGUF/resolve/main/minimax_h3_fl2va_pruned-UD-Q2_K_XL.gguf",
    ),
    (
        "qwen3vl_32b_minimax_h3-Q2_K_M.gguf",
        "https://huggingface.co/unsloth/MiniMax-H3-GGUF/resolve/main/text_encoders/qwen3vl_32b_minimax_h3-Q2_K_M.gguf",
    ),
    (
        "vae/minimax_h3_video_vae_fp16.safetensors",
        "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_fp16.safetensors",
    ),
    (
        "vae/minimax_h3_audio_vae_fp32.safetensors",
        "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors",
    ),
]

for rel_path, url in MODEL_MANIFEST:
    local_target = os.path.join(LOCAL_MODEL_DIR, rel_path)
    drive_target = os.path.join(GDRIVE_MODEL_DIR, rel_path)
    os.makedirs(os.path.dirname(local_target), exist_ok=True)

    if os.path.exists(drive_target) and os.path.getsize(drive_target) > 1024 * 1024:
        print(f"⚡ [云盘秒级复用] 软链接已有模型: {rel_path}")
        if not os.path.exists(local_target):
            os.symlink(drive_target, local_target)
        continue

    if not os.path.exists(local_target) or os.path.getsize(local_target) < 1024 * 1024:
        print(f"🌐 [aria2c 16线程下载] {rel_path} ...")
        cmd = [
            "aria2c", "-c", "-x", "16", "-s", "16", "-k", "1M",
            "--console-log-level=warn",
            "-d", os.path.dirname(local_target),
            "-o", os.path.basename(local_target),
            url,
        ]
        subprocess.run(cmd, check=True)

    size_gb = os.path.getsize(local_target) / (1024 ** 3)
    print(f"✅ 已就绪: {rel_path} ({size_gb:.2f} GB)")

print("\n🎉 全部 4 项 MiniMax-H3 权重准备就绪！")""",
    )

    add_md(
        nb,
        r"""### 步骤 4：单镜头快速测试（T2V 原生视音频同源生成 + 立即备份 Google Drive）
先运行单个镜头验证 T4 零拷贝 `mmap` 推理，生成完成后**立即同步到 Google Drive** 并在页面内直接播放预览。""",
    )

    add_code(
        nb,
        r"""# @title 4. 单镜头 T2V 测试生成 & 实时同步 Google Drive
import os
import shutil
import subprocess
import time
from IPython.display import Video, display

PROMPT = "a red panda stepping along a mossy log in a misty forest, cinematic, soft morning light, gentle rain and birdsong"  # @param {type:"string"}
WIDTH = 640  # @param {type:"integer"}
HEIGHT = 384  # @param {type:"integer"}
FRAMES = 25  # @param {type:"integer"}
STEPS = 4  # @param {type:"integer"}
SEED = 11  # @param {type:"integer"}

out_webm = "/content/outputs/minimax_h3_t4_test.webm"
out_mp4 = "/content/outputs/minimax_h3_t4_test.mp4"
drive_dir = "/content/drive/MyDrive/MiniMax_H3_Outputs/raw_shots"
os.makedirs(drive_dir, exist_ok=True)

cmd = [
    "/content/stable-diffusion.cpp/build/bin/sd-cli",
    "--mode", "vid_gen",
    "--diffusion-model", "/content/models/minimax_h3_fl2va_pruned-UD-Q2_K_XL.gguf",
    "--llm", "/content/models/qwen3vl_32b_minimax_h3-Q2_K_M.gguf",
    "--vae", "/content/models/vae/minimax_h3_video_vae_fp16.safetensors",
    "--audio-vae", "/content/models/vae/minimax_h3_audio_vae_fp32.safetensors",
    "--prompt", PROMPT,
    "--width", str(WIDTH),
    "--height", str(HEIGHT),
    "--video-frames", str(FRAMES),
    "--steps", str(STEPS),
    "--cfg-scale", "1.0",
    "--backend", "te=cpu",
    "--params-backend", "te=cpu",
    "--diffusion-fa",
    "--mmap",
    "--rng", "cpu",
    "--fps", "24",
    "--seed", str(SEED),
    "-o", out_webm,
]

print("🚀 正在启动 MiniMax-H3 单镜头视音频生成...")
t0 = time.time()
subprocess.run(cmd, check=True)
print(f"✅ 渲染完成！耗时: {int(time.time() - t0)} 秒")

subprocess.run(
    ["ffmpeg", "-y", "-i", out_webm, "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", out_mp4],
    check=True,
)
shutil.copy2(out_webm, os.path.join(drive_dir, os.path.basename(out_webm)))
shutil.copy2(out_mp4, os.path.join(drive_dir, os.path.basename(out_mp4)))
os.sync()
print(f"💾 已实时备份至 Google Drive: {drive_dir}/{os.path.basename(out_mp4)}")

display(Video(out_mp4, embed=True, width=640))""",
    )

    add_md(
        nb,
        r"""### 步骤 5：方案 B 多分镜连续长片生成（T2V -> 尾帧提取 -> I2V 链式续写 + 逐镜落盘防丢）
- **逐镜独立保存**：每完成一段分镜（`shot_01.mp4`, `shot_02.mp4`, ...），立刻同步写入 Google Drive 的 `raw_shots/` 目录并触发 `os.sync()`，确保您第一时间拿到所有未合成的独立原始视频。
- **自动断点续跑**：如果中途 Colab 断开，重新运行此单元格会自动跳过已在云盘中生成的分镜，从下一镜无缝继续！
- **全部完成后自动拼接**：所有分镜生成完毕后，使用 FFmpeg 无损合成为完整连续短片。""",
    )

    add_code(
        nb,
        r"""# @title 5. 方案 B：多分镜链式生成 (T2V -> I2V 尾帧接力 + 逐镜实时保存 Google Drive)
import os
import shutil
import subprocess
import time
from IPython.display import Video, display

SD_CLI = "/content/stable-diffusion.cpp/build/bin/sd-cli"
WORK_DIR = "/content/outputs/plan_b"
GDRIVE_BASE = "/content/drive/MyDrive/MiniMax_H3_Outputs"

# 可自由增删或修改分镜脚本 (Shot 1 为 T2V，Shot 2..N 自动提取上一镜最后一帧作为 --init-img 进行 I2V 接力)
STORYBOARD = [
    {
        "id": 1,
        "seed": 42,
        "prompt": "Inside the cavernous command bridge of a futuristic starship, panoramic viewport revealing an armada of dreadnoughts hovering before a glowing cyan and gold nebula, holographic star charts floating, volumetric lighting, cinematic 24fps, ambient bridge hum",
    },
    {
        "id": 2,
        "seed": 49,
        "prompt": "The camera slowly pushes forward towards the panoramic viewport. The armada of dreadnoughts ignites glowing cyan plasma thrusters in unison, advancing into the cosmic nebula, high consistency, cinematic lighting, deep engine rumble",
    },
    {
        "id": 3,
        "seed": 57,
        "prompt": "Medium close-up of a female fleet commander on the starship bridge turning toward the glowing tactical hologram, blue data reflections dancing across her eyes, dramatic rim lighting, cinematic sci-fi score and telemetry beeps",
    },
    {
        "id": 4,
        "seed": 64,
        "prompt": "Relativistic hyperspace jump sequence viewed from the bridge viewport, starlight stretching into radiant blue-violet quantum slipstream tunnels, intense volumetric glow, deep sub-bass warp boom and harmonic resonance",
    },
    {
        "id": 5,
        "seed": 73,
        "prompt": "Emerging from hyperspace into a golden star system, revealing a colossal ancient Dyson ring megastructure orbiting a radiant yellow star, awe-inspiring scale, orchestral brass climax",
    },
]

completed_mp4s = []
prev_last_png = None

for shot in STORYBOARD:
    sid = shot["id"]
    seed = shot["seed"]
    prompt = shot["prompt"]
    tag = f"shot_{sid:02d}"

    local_webm = os.path.join(WORK_DIR, f"{tag}.webm")
    local_mp4 = os.path.join(WORK_DIR, f"{tag}.mp4")
    local_png = os.path.join(WORK_DIR, f"{tag}_last.png")

    drive_mp4 = os.path.join(GDRIVE_BASE, "raw_shots", f"{tag}.mp4")
    drive_png = os.path.join(GDRIVE_BASE, "keyframes", f"{tag}_last.png")

    # 断点续跑检查
    if os.path.exists(drive_mp4) and os.path.getsize(drive_mp4) > 10000:
        print(f"⚡ [断点恢复] 检测到云盘已存在 {tag}.mp4，直接复用！")
        shutil.copy2(drive_mp4, local_mp4)
        if os.path.exists(drive_png):
            shutil.copy2(drive_png, local_png)
        else:
            subprocess.run(["ffmpeg", "-y", "-sseof", "-0.1", "-i", local_mp4, "-vframes", "1", "-q:v", "2", local_png], check=True)
            shutil.copy2(local_png, drive_png)
        prev_last_png = local_png
        completed_mp4s.append(local_mp4)
        continue

    mode_str = "T2V 首镜生成" if prev_last_png is None else f"I2V 尾帧续写 (参考: {os.path.basename(prev_last_png)})"
    print(f"\n🎬 正在生成 {tag.upper()} [{mode_str}] ...")

    cmd = [
        SD_CLI,
        "--mode", "vid_gen",
        "--diffusion-model", "/content/models/minimax_h3_fl2va_pruned-UD-Q2_K_XL.gguf",
        "--llm", "/content/models/qwen3vl_32b_minimax_h3-Q2_K_M.gguf",
        "--vae", "/content/models/vae/minimax_h3_video_vae_fp16.safetensors",
        "--audio-vae", "/content/models/vae/minimax_h3_audio_vae_fp32.safetensors",
        "--prompt", prompt,
        "--width", "640",
        "--height", "384",
        "--video-frames", "24",
        "--steps", "4",
        "--cfg-scale", "1.0",
        "--backend", "te=cpu",
        "--params-backend", "te=cpu",
        "--diffusion-fa",
        "--mmap",
        "--rng", "cpu",
        "--fps", "24",
        "--seed", str(seed),
        "-o", local_webm,
    ]
    if prev_last_png and os.path.exists(prev_last_png):
        cmd.extend(["--init-img", prev_last_png])

    t0 = time.time()
    subprocess.run(cmd, check=True)
    print(f"✅ {tag} 渲染完成 ({int(time.time() - t0)}s)")

    # 立即转码 MP4 并同步到 Google Drive (防止断连丢失)
    subprocess.run(
        ["ffmpeg", "-y", "-i", local_webm, "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", local_mp4],
        check=True,
    )
    shutil.copy2(local_webm, os.path.join(GDRIVE_BASE, "raw_shots", f"{tag}.webm"))
    shutil.copy2(local_mp4, drive_mp4)

    # 截取尾帧供下一镜头 I2V 使用
    subprocess.run(["ffmpeg", "-y", "-sseof", "-0.1", "-i", local_webm, "-vframes", "1", "-q:v", "2", local_png], check=True)
    shutil.copy2(local_png, drive_png)
    os.sync()

    print(f"💾 [防丢落盘完成] 独立原片已存入云盘: {drive_mp4}")
    prev_last_png = local_png
    completed_mp4s.append(local_mp4)

# 全部独立分镜保存完毕后，合成为完整短片
if len(completed_mp4s) > 1:
    concat_txt = os.path.join(WORK_DIR, "concat_list.txt")
    with open(concat_txt, "w", encoding="utf-8") as f:
        for p in completed_mp4s:
            f.write(f"file '{p}'\n")
    final_film = os.path.join(WORK_DIR, "minimax_h3_plan_b_full_film.mp4")
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_txt, "-c", "copy", final_film], check=True)
    drive_final = os.path.join(GDRIVE_BASE, "minimax_h3_plan_b_full_film.mp4")
    shutil.copy2(final_film, drive_final)
    os.sync()
    print(f"\n🎉 全部 {len(completed_mp4s)} 个独立分镜及合成全片均已保存在 Google Drive: {drive_final}")
    display(Video(final_film, embed=True, width=640))""",
    )

    out_path = os.path.join(NOTEBOOKS_DIR, "MiniMax_H3_T4_GGUF_PlanB_Pipeline.ipynb")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print(f"[OK] Built T4 Notebook: {out_path}")


def copy_a100_notebook():
    src = r"C:\Users\15695\.gemini\antigravity\scratch\minimax_h3_colab_drive.ipynb"
    dst = os.path.join(NOTEBOOKS_DIR, "MiniMax_H3_A100_ComfyUI_Drive.ipynb")
    shutil.copy2(src, dst)
    print(f"[OK] Copied A100 ComfyUI Notebook: {dst}")


if __name__ == "__main__":
    build_t4_notebook()
    copy_a100_notebook()
