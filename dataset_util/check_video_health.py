from videollama3.mm_utils import load_video
from glob import glob
import argparse
from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument("--root_path", type=str, help="Root path containing videos")
parser.add_argument("--output_log_path", type=str, help="Path to output log file")
args = parser.parse_args()

if __name__ == "__main__":
    root_path = args.root_path
    output_log_path = args.output_log_path
    all_video_paths = glob(root_path + "/*")
    deprecated_count = 0
    for video_path in tqdm(glob(root_path + "/**", recursive=True), desc="Checking video health"):
        if video_path.endswith((".mp4", ".avi", ".mkv", ".webm", ".mov")):
            try:
                frames, timestamps = load_video(video_path, fps=1, max_frames=200)
                if len(frames) == 0 or frames is None or timestamps is None:
                    with open(output_log_path, "a") as f:
                        f.write(video_path + "\n")
                    deprecated_count += 1
            except Exception as e:
                with open(output_log_path, "a") as f:
                    f.write(f"{video_path} (Error: {str(e)})\n")
                deprecated_count += 1
    print(f"Total deprecated videos: {deprecated_count}")