import os
import subprocess

INPUT_DIR = "input"
OUTPUT_DIR = "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

for job in os.listdir(INPUT_DIR):
    job_path = os.path.join(INPUT_DIR, job)
    if not os.path.isdir(job_path):
        continue

    images = sorted(os.listdir(job_path))
    if len(images) == 0:
        continue

    first_image = os.path.join(job_path, images[0])

    out_path = os.path.join(OUTPUT_DIR, f"{job}.mp4")

    cmd = [
        "python", "scripts/sampling/simple_video_sample.py",
        "--input_path", first_image,
        "--num_frames", "14",
        "--fps", "6",
        "--output_path", out_path,
        "--device", "cuda"
    ]

    subprocess.run(cmd)
