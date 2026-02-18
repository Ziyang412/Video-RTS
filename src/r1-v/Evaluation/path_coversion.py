import json

# Update these variables as needed
input_json = "/nas-ssd2/ziyang/Streaming_Video_understanding/Video-R1/src/r1-v/Evaluation/eval_minerva_old.json"  # Path to your input JSON file
output_json = "/nas-ssd2/ziyang/Streaming_Video_understanding/Video-R1/src/r1-v/Evaluation/eval_minerva.json"  # Path to your output JSON file
old_prefix = "/fsx/sfr/data/multimodal/video_reasoning_ziyang/Minerva/videos/download/"
new_prefix = "/nas-ssd2/ziyang/data/Minerva/download/"

# Read the JSON file
with open(input_json, "r", encoding="utf-8") as f:
    data = json.load(f)

# Update the "path" field in each problem
for item in data:
    if "path" in item and item["path"].startswith(old_prefix):
        item["path"] = item["path"].replace(old_prefix, new_prefix, 1)

# Write the updated data back to a new JSON file
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)
