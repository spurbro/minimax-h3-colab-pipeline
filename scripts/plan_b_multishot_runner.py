#!/usr/bin/env python3
"""
Plan B Multi-Shot Chaining & Real-Time Google Drive Persistence Runner for MiniMax-H3.

Key Features:
1. Genuine Diffusion Video Generation (Zero Slideshow):
   - Shot 01 is rendered via Text-to-Video (T2V, `--mode vid_gen`).
   - Subsequent Shots 02..N extract the exact last frame of the preceding shot (`shot_{k-1}_last.png`)
     and pass it via `--init-img` (I2V first-frame conditioning) to preserve scene, character, and lighting continuity.
2. Zero Data Loss (Immediate Per-Shot Google Drive Persistence):
   - Every shot is immediately transcoded to `.mp4` and copied to Google Drive (`/content/drive/MyDrive/MiniMax_H3_Outputs/raw_shots/`)
     with `os.sync()` right after completion—before the next shot starts.
3. Checkpoint Auto-Resume:
   - If Colab disconnects due to quota or idle timeout, re-running this script automatically detects
     already-completed shots in Google Drive and resumes from the next unfinished shot.
4. Raw Segments First + Final Stitched Film:
   - Preserves all individual unstitched `.mp4` clips in `raw_shots/` and compiles the concatenated
     continuous video at the end.
"""

import os
import shutil
import subprocess
import time
from typing import Dict, List, Optional

SD_CLI_BIN = "/content/stable-diffusion.cpp/build/bin/sd-cli"
MODELS_DIR = "/content/models"
LOCAL_WORK_DIR = "/content/outputs/plan_b"
GDRIVE_BASE_DIR = "/content/drive/MyDrive/MiniMax_H3_Outputs"

DEFAULT_STORYBOARD: List[Dict[str, object]] = [
    {
        "id": 1,
        "seed": 42,
        "prompt": (
            "Inside the cavernous command bridge of a futuristic starship, panoramic viewport "
            "revealing an armada of dreadnoughts hovering before a glowing cyan and gold nebula, "
            "holographic star charts floating, volumetric lighting, cinematic 24fps, ambient bridge hum"
        ),
    },
    {
        "id": 2,
        "seed": 49,
        "prompt": (
            "The camera slowly pushes forward towards the panoramic viewport. The armada of dreadnoughts "
            "ignites glowing cyan plasma thrusters in unison, advancing into the cosmic nebula, "
            "high consistency, cinematic lighting, deep engine rumble"
        ),
    },
    {
        "id": 3,
        "seed": 57,
        "prompt": (
            "Medium close-up of a female fleet commander on the starship bridge turning toward the glowing "
            "tactical hologram, blue data reflections dancing across her eyes, dramatic rim lighting, "
            "cinematic sci-fi score and telemetry beeps"
        ),
    },
    {
        "id": 4,
        "seed": 64,
        "prompt": (
            "Relativistic hyperspace jump sequence viewed from the bridge viewport, starlight stretching "
            "into radiant blue-violet quantum slipstream tunnels, intense volumetric glow, "
            "deep sub-bass warp boom and harmonic resonance"
        ),
    },
    {
        "id": 5,
        "seed": 73,
        "prompt": (
            "Emerging from hyperspace into a golden star system, revealing a colossal ancient Dyson ring "
            "megastructure orbiting a radiant yellow star, awe-inspiring scale, orchestral brass climax"
        ),
    },
]


def ensure_dirs():
    os.makedirs(LOCAL_WORK_DIR, exist_ok=True)
    os.makedirs(os.path.join(GDRIVE_BASE_DIR, "raw_shots"), exist_ok=True)
    os.makedirs(os.path.join(GDRIVE_BASE_DIR, "keyframes"), exist_ok=True)
    os.makedirs(os.path.join(GDRIVE_BASE_DIR, "logs"), exist_ok=True)


def extract_last_frame(video_path: str, out_png: str) -> bool:
    cmd = [
        "ffmpeg", "-y",
        "-sseof", "-0.1",
        "-i", video_path,
        "-vframes", "1",
        "-q:v", "2",
        out_png,
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return os.path.exists(out_png) and os.path.getsize(out_png) > 0


def transcode_to_mp4(webm_path: str, mp4_path: str) -> bool:
    cmd = [
        "ffmpeg", "-y",
        "-i", webm_path,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        mp4_path,
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return os.path.exists(mp4_path) and os.path.getsize(mp4_path) > 1000


def sync_file_to_drive(src_path: str, subdir: str) -> Optional[str]:
    if not os.path.exists("/content/drive/MyDrive"):
        print("[WARN] Google Drive not mounted at /content/drive/MyDrive. Skipping Drive copy.")
        return None
    dest_dir = os.path.join(GDRIVE_BASE_DIR, subdir)
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, os.path.basename(src_path))
    shutil.copy2(src_path, dest_path)
    os.sync()
    print(f"  💾 [Drive Synced] -> {dest_path} ({os.path.getsize(dest_path) / 1024:.1f} KB)")
    return dest_path


def run_plan_b(
    storyboard: List[Dict[str, object]] = DEFAULT_STORYBOARD,
    width: int = 640,
    height: int = 384,
    frames: int = 25,
    steps: int = 4,
    fps: int = 24,
):
    ensure_dirs()
    completed_mp4s: List[str] = []
    prev_last_frame: Optional[str] = None

    print("=" * 72)
    print(f"🎬 Starting MiniMax-H3 Plan B Multi-Shot Pipeline ({len(storyboard)} shots)")
    print(f"📁 Google Drive Real-Time Vault: {GDRIVE_BASE_DIR}/raw_shots")
    print("=" * 72)

    for shot in storyboard:
        shot_id = int(shot["id"])
        seed = int(shot.get("seed", 42 + shot_id))
        prompt = str(shot["prompt"])
        tag = f"shot_{shot_id:02d}"

        local_webm = os.path.join(LOCAL_WORK_DIR, f"{tag}.webm")
        local_mp4 = os.path.join(LOCAL_WORK_DIR, f"{tag}.mp4")
        local_last_png = os.path.join(LOCAL_WORK_DIR, f"{tag}_last.png")

        drive_mp4 = os.path.join(GDRIVE_BASE_DIR, "raw_shots", f"{tag}.mp4")
        drive_last_png = os.path.join(GDRIVE_BASE_DIR, "keyframes", f"{tag}_last.png")

        # Check if already completed in Google Drive (Checkpoint Resume)
        if os.path.exists(drive_mp4) and os.path.getsize(drive_mp4) > 10000:
            print(f"\n⚡ [RESUME] {tag}.mp4 already exists in Google Drive! Reusing checkpoint.")
            if not os.path.exists(local_mp4):
                shutil.copy2(drive_mp4, local_mp4)
            if os.path.exists(drive_last_png):
                shutil.copy2(drive_last_png, local_last_png)
            else:
                extract_last_frame(local_mp4, local_last_png)
                sync_file_to_drive(local_last_png, "keyframes")
            prev_last_frame = local_last_png
            completed_mp4s.append(local_mp4)
            continue

        mode_label = "T2V (First Shot)" if prev_last_frame is None else f"I2V (Conditioned on {os.path.basename(prev_last_frame)})"
        print(f"\n🚀 Rendering {tag.upper()} [{mode_label}] | Seed={seed}")
        print(f"   Prompt: {prompt}")

        cmd = [
            SD_CLI_BIN,
            "--mode", "vid_gen",
            "--diffusion-model", os.path.join(MODELS_DIR, "minimax_h3_fl2va_pruned-UD-Q2_K_XL.gguf"),
            "--llm", os.path.join(MODELS_DIR, "qwen3vl_32b_minimax_h3-Q2_K_M.gguf"),
            "--vae", os.path.join(MODELS_DIR, "vae", "minimax_h3_video_vae_fp16.safetensors"),
            "--audio-vae", os.path.join(MODELS_DIR, "vae", "minimax_h3_audio_vae_fp32.safetensors"),
            "--prompt", prompt,
            "--width", str(width),
            "--height", str(height),
            "--video-frames", str(frames),
            "--steps", str(steps),
            "--cfg-scale", "1.0",
            "--backend", "te=cpu",
            "--params-backend", "te=cpu",
            "--diffusion-fa",
            "--mmap",
            "--rng", "cpu",
            "--fps", str(fps),
            "--seed", str(seed),
            "-o", local_webm,
        ]

        if prev_last_frame and os.path.exists(prev_last_frame):
            cmd.extend(["--init-img", prev_last_frame])

        log_file = os.path.join(LOCAL_WORK_DIR, f"{tag}.log")
        t0 = time.time()
        with open(log_file, "w", encoding="utf-8") as lf:
            proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
            while proc.poll() is None:
                time.sleep(5)
            rc = proc.returncode

        elapsed = int(time.time() - t0)
        sync_file_to_drive(log_file, "logs")

        if rc != 0 or not os.path.exists(local_webm) or os.path.getsize(local_webm) < 1000:
            print(f"❌ [ERROR] {tag} failed (rc={rc}, elapsed={elapsed}s). Check {log_file}.")
            break

        print(f"✅ {tag} rendered in {elapsed}s ({os.path.getsize(local_webm) / (1024*1024):.2f} MB)")

        # Immediately persist raw .webm and transcoded .mp4 to Google Drive BEFORE starting next shot
        sync_file_to_drive(local_webm, "raw_shots")
        if transcode_to_mp4(local_webm, local_mp4):
            sync_file_to_drive(local_mp4, "raw_shots")
            completed_mp4s.append(local_mp4)

        # Extract last frame for next shot I2V conditioning
        if extract_last_frame(local_webm, local_last_png):
            sync_file_to_drive(local_last_png, "keyframes")
            prev_last_frame = local_last_png

    # Final Stitching (only after all individual raw clips are safely persisted)
    if len(completed_mp4s) > 1:
        concat_list = os.path.join(LOCAL_WORK_DIR, "concat_list.txt")
        with open(concat_list, "w", encoding="utf-8") as f:
            for mp4 in completed_mp4s:
                f.write(f"file '{mp4}'\n")

        final_mp4 = os.path.join(LOCAL_WORK_DIR, "minimax_h3_plan_b_stitched.mp4")
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list, "-c", "copy", final_mp4],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if os.path.exists(final_mp4):
            sync_file_to_drive(final_mp4, "")
            print(f"\n🎉 Final stitched film saved to Drive: {GDRIVE_BASE_DIR}/minimax_h3_plan_b_stitched.mp4")


if __name__ == "__main__":
    run_plan_b()
