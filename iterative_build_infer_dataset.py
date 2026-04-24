from glob import glob
import argparse
import json
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--video_root", type=str, required=True, help="Root directory containing video files.")
arg_parser.add_argument("--output_file", type=str, required=True, help="Path to save the output JSON file.")
arg_parser.add_argument("--validation_prompt", type=str, default="<video>\nSummarize all the events in this video. For each event, include the duration. Format your response as: <start time> - <end time>, <description>", help="Prompt to use for generating descriptions.")
args = arg_parser.parse_args()
v_root = args.video_root
output_file = args.output_file
validation_prompt = args.validation_prompt
videos_list = []
for videos in glob(f"{v_root}/**/*.mp4", recursive=True):
    video_name = videos.split("/")[-1]
    videos_list.append({
        "video":[video_name],
        "conversations":[
            {
                "from": "human", 
                "value": validation_prompt
            },
            {
                "from": "assistant", 
                "value": ""
            }
        ]
    })
    
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(videos_list, f, indent=4)