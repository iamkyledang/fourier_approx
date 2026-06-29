import subprocess
import os

SCENES = [
    "as_novel_view",
    "basin_novel_view",
    "bell_novel_view",
    "cup_novel_view",
    "plate_novel_view",
    "press_novel_view",
    "sieve_novel_view",
]

def run(cmd):
    print(f"\n>>> {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)

for scene in SCENES:
    source = f"../nerf_ds/{scene}"
    output = f"output/{scene}"

    print(f"\n{'='*60}")
    print(f"Scene: {scene}")
    print(f"{'='*60}")

    if os.path.exists(output):
        print(f"Output folder '{output}' already exists — skipping.")
        continue

    # Train
    run(["python", "train.py", "-s", source, "-m", output])

    # Render (test set only)
    run(["python", "render.py", "-s", source, "-m", output, "--skip_train"])

    # FFmpeg comparison video
    renders_dir = os.path.join(output, "test", "ours_30000", "renders", "%05d.png")
    gt_dir      = os.path.join(output, "test", "ours_30000", "gt",      "%05d.png")
    out_video   = os.path.join(output, "test", "ours_30000", "comparison.mp4")
    run([
        "ffmpeg", "-y",
        "-framerate", "30", "-i", renders_dir,
        "-framerate", "30", "-i", gt_dir,
        "-filter_complex", "hstack,pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        out_video,
    ])

    # Metrics
    run(["python", "metrics.py", "-m", output])

print("\nAll scenes done.")

