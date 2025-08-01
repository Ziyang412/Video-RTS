model_paths=(
    "YOUR_MODEL_PATH_HERE"
)

file_names=(
    "YOUR_OUTPUT_FILE_NAME_HERE"  
)

export DECORD_EOF_RETRY_MAX=100480

for i in "${!model_paths[@]}"; do
    model="${model_paths[$i]}"
    file_name="${file_names[$i]}"
    CUDA_VISIBLE_DEVICES=i python ./src/video_rts_eval.py --model_path "$model" --file_name "$file_name"
done


