import os
import json
import re
from tqdm import tqdm
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import torch
import traceback
import random  # For seed setting

from transformers import AutoProcessor, AutoTokenizer
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import argparse


# regex to extract the answer inside <answer>...</answer>
ANSWER_RE = re.compile(r'<answer>(.*?)</answer>')

def extract_ground_truth(solution_str):
    """Return the text inside <answer>…</answer>, or None if not found."""
    m = ANSWER_RE.search(solution_str)
    return m.group(1) if m else None

def compute_accuracies(data):
    total = 0
    correct_pred = 0
    correct_majority = 0

    for item in data.get('results', []):
        gt = extract_ground_truth(item.get('solution', ''))
        if gt is None:
            # skip if no ground truth found
            continue

        total += 1
        if item.get('prediction') == gt:
            correct_pred += 1
        if item.get('majority_vote') == gt:
            correct_majority += 1

    if total == 0:
        return 0.0, 0.0

    single_run_acc = correct_pred / total
    majority_vote_acc = correct_majority / total
    return single_run_acc, majority_vote_acc

BSZ = 16  # Reduced from 64 to handle multiple samples per prompt
NUM_SAMPLES = 5  # Number of samples to generate for majority voting

# ─── Frame-sampling policy ────────────────────────────────────────────────────
BASE_NFRAMES = 32
MAX_NFRAMES = {          # dataset-specific hard limits
    "mmvu": 64,
    "videommmu": 64,
    "videoholmes": 128,   
    "videomme": 128,
    "lvb": 128}


parser = argparse.ArgumentParser(description="Evaluation benchmark")
parser.add_argument('--model_path', type=str, required=True, help="Path to the model")
parser.add_argument('--file_name', type=str, required=True, help="Name of the file")
args = parser.parse_args()

MODEL_PATH = args.model_path
file_name = args.file_name


# Helper functions for processing outputs
def extract_answer(text):
    """Extract answer from response text."""
    answer_pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
    match = answer_pattern.search(text)
    if match:
        return match.group(1).strip()
    return "No answer found"

def get_vote_counts(outputs):
    """Count votes for each unique answer."""
    answers = [extract_answer(output) for output in outputs]
    return dict(Counter(answers))

def majority_vote(outputs):
    """Get the majority vote from multiple outputs."""
    answers = [extract_answer(output) for output in outputs]
    vote_count = Counter(answers)
    if not vote_count:
        return "No valid answers found"
    
    # Get the most common answer
    majority_answer, count = vote_count.most_common(1)[0]
    return majority_answer

def reward_fn(sample, output, problem_type, comparison_target=None):
    """Calculate reward based on problem type and correct answer.
    
    Args:
        sample: The sample data containing the ground truth
        output: The model's output to evaluate
        problem_type: The type of problem (multiple choice, numerical, etc.)
        comparison_target: If provided, use this as the target instead of ground truth
    """
    prediction = extract_answer(output)
    
    # Use the provided comparison target or get the ground truth from the sample
    if comparison_target is not None:
        ground_truth = comparison_target
    else:
        # Check both "answer" and "solution" fields, as dataset might use either
        ground_truth = ""
        if "answer" in sample:
            ground_truth = sample.get("answer", "").strip()
        elif "solution" in sample:
            # Extract answer from solution field which contains full <answer>X</answer> format
            ground_truth = extract_answer(sample.get("solution", ""))
    
    # For multiple choice, just check exact match
    if problem_type == "multiple choice":
        return 1.0 if prediction.upper() == ground_truth.upper() else 0.0
    
    # For numerical, allow small differences
    elif problem_type == "numerical" or problem_type == "regression":
        try:
            pred_val = float(prediction)
            gt_val = float(ground_truth)
            # Allow 1% tolerance
            if abs(pred_val - gt_val) <= 0.01 * abs(gt_val):
                return 1.0
            else:
                return 0.0
        except:
            return 0.0
    
    # For OCR and free-form, use BLEU and ROUGE
    elif problem_type == "OCR" or problem_type == "free-form":
        # BLEU score
        smoothie = SmoothingFunction().method1
        bleu_score = sentence_bleu([ground_truth.split()], prediction.split(), 
                                  smoothing_function=smoothie)
        
        # ROUGE score
        scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
        rouge_scores = scorer.score(ground_truth, prediction)
        rouge1_score = rouge_scores['rouge1'].fmeasure
        
        # Combine scores
        combined_score = (bleu_score + rouge1_score) / 2
        return combined_score
    
    # Default case
    else:
        return 1.0 if prediction == ground_truth else 0.0

def save_final_results(output_path, results, total_samples):
    """Save final results to file."""
    # Calculate overall statistics
    total_processed = len(results)
    print(f"Processed {total_processed}/{total_samples} samples")
    
    # Save to file
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"results": results}, f, indent=2, ensure_ascii=False)
        print(f"Final results saved to {output_path}")
    except Exception as e:
        print(f"Error saving final results: {e}")
        
# Set different random seeds for each run to ensure diversity
random.seed(42)  # Master seed
torch_seeds = [random.randint(1, 10000) for _ in range(NUM_SAMPLES)]

# Initialize LLM with proper parameters
llm = LLM(
    model=MODEL_PATH,
    tensor_parallel_size=torch.cuda.device_count(),
    max_model_len=65536,  # Increased to accommodate multi-modal embeddings
    gpu_memory_utilization=0.8,  # Reduced slightly to prevent OOM
    limit_mm_per_prompt={"image": 1, "video": 1},
    max_num_seqs=BSZ * NUM_SAMPLES,  # Ensure we can handle all samples
    trust_remote_code=True,  # Required for some model configurations
    enforce_eager=True,  # Ensure synchronous execution
    dtype="auto"  # Use model's native dtype
)

print(f"\nLLM Configuration:")
print(f"Max sequences: {BSZ * NUM_SAMPLES}")
print(f"Batch size: {BSZ}")
print(f"Samples per input: {NUM_SAMPLES}")

# Use high temperature for diverse sampling
sampling_params = SamplingParams(
    temperature=0.7,  # Higher temperature for more diversity
    top_p=0.2,  # Higher top_p to allow more diverse token selection
    max_tokens=1024,
    stop=["</answer>"]  # Add stop token to prevent overgeneration
)

processor = AutoProcessor.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.padding_side = "left"
processor.tokenizer = tokenizer


for dataset_name in ['videoholmes','videommmu', 'lvb', 'mmvu', 'videomme']:


    OUTPUT_PATH = f"./src/test_results/eval_{dataset_name}_{file_name}_greedy_output.json"
    PROMPT_PATH = f"./src/r1-v/Evaluation/eval_{dataset_name}.json"
    
    # Initialize data list
    all_data = []
    
    if PROMPT_PATH.endswith('.jsonl'):
        with open(PROMPT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                all_data.append(json.loads(line))
    elif PROMPT_PATH.endswith('.json'):
        with open(PROMPT_PATH, "r", encoding="utf-8") as f:
            all_data = json.load(f)
    else:
        raise ValueError("Input file must be .json or .jsonl")

    solved_data_list = []  # Start with all data as unsolved

    #### our training prompt
    QUESTION_TEMPLATE = (
        "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
        "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
        "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
        "<think> reasoning process here </think><answer> answer here </answer>"
        "{Question}\n"
    )

    TYPE_TEMPLATE = {
        "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
        "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
        "OCR": " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
        "free-form": " Please provide your text answer within the <answer> </answer> tags.",
        "regression": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags."
    }

    nframes = BASE_NFRAMES

    # Initialize results storage
    final_output = []

    while nframes <= MAX_NFRAMES[dataset_name]:
        print(f"\nProcessing dataset: {dataset_name} with nframes={nframes}")
        messages = []

        ### filter out the samples that have been solved
        data = [x for x in all_data if x not in solved_data_list]
        for x in data:
            if x["problem_type"] == 'multiple choice':
                question = x['problem'] + "Options:\n"
                for op in x["options"]:
                    question += op + "\n"
            else:
                question = x['problem']

            msg = [{
                "role": "user",
                "content": [
                    {
                        "type": x['data_type'],
                        x['data_type']: x['path'],
                        "nframes": nframes,
                        # x['data_type']: os.getcwd() + "/src/r1-v/Evaluation" + x['path'][1:]

                    },
                    {
                        "type": "text",
                        "text": QUESTION_TEMPLATE.format(Question=question) + TYPE_TEMPLATE[x['problem_type']]
                    }
                ]
            }]
            messages.append(msg)
            

        # Initialize results storage
        start_idx = 0

        # # Load existing results if any
        # if os.path.exists(OUTPUT_PATH):
        #     try:
        #         with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        #             existing = json.load(f)
        #             final_output = existing.get("results", [])
        #             start_idx = len(final_output)
        #             print(f"Resuming from sample index {start_idx}")
        #     except Exception as e:
        #         print(f"Error reading existing output file: {e}")

        # Make sure output directory exists
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

        # Calculate total batches and show configuration
        total_samples = len(messages)
        total_batches = (total_samples - start_idx) // BSZ + (1 if (total_samples - start_idx) % BSZ else 0)
        print(f"\nProcessing Configuration:")
        print(f"Total samples to process: {total_samples}")
        print(f"Starting from index: {start_idx}")
        print(f"Batch size: {BSZ}")
        print(f"Total batches: {total_batches}")
        print(f"Samples per input: {NUM_SAMPLES}")

        pbar = tqdm(total=total_batches, desc="Processing batches", unit="batch")
        current_batch = 0

        def clear_gpu_memory():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        try:
            for i in range(start_idx, len(messages), BSZ):
                current_batch += 1
                batch_messages = messages[i:i + BSZ]
                current_batch_size = len(batch_messages)
                
                print(f"\nProcessing batch {current_batch}/{total_batches}")
                print(f"Batch range: samples {i} to {i + current_batch_size - 1}")
                
                # Process each sample one by one to collect all outputs
                all_sample_outputs = []  # Store all outputs for this batch
                
                for sample_idx in range(current_batch_size):
                    # Get the message for this sample
                    sample_message = batch_messages[sample_idx]
                    sample_outputs = []  # Store outputs for this specific sample
                    
                    print(f"  Processing sample {i + sample_idx} ({NUM_SAMPLES} times)...")
                    
                    # Process vision info for this sample
                    image_inputs = []
                    video_inputs = []
                    video_kwargs = {}
                    
                    mm_type = sample_message[0]['content'][0]['type']
                    
                    try:
                        image_inputs, video_inputs, video_kwargs = process_vision_info(
                            [sample_message], 
                            return_video_kwargs=True
                        )
                    except Exception as e:
                        print(f"Error processing vision info for sample {i + sample_idx}: {e}")
                        # If we can't process vision info, we'll skip this sample
                        all_sample_outputs.append([])
                        continue
                    
                    # Run inference multiple times for this sample
                    for run_idx in range(NUM_SAMPLES):
                        # Set a different random seed for each run
                        torch.manual_seed(torch_seeds[run_idx])
                        
                        # Apply chat template
                        prompt = processor.apply_chat_template(
                            sample_message, 
                            tokenize=False, 
                            add_generation_prompt=True
                        )
                        
                        # Apply a small variation to prompts to encourage diversity
                        # Add a unique prefix that doesn't affect the model's understanding
                        prompt_variation = f"Sample run {run_idx+1}: {prompt}"
                        
                        # Prepare multi-modal data
                        sample_mm_data = {}
                        sample_video_kw = {}
                        
                        if mm_type == 'image' and image_inputs:
                            sample_mm_data["image"] = image_inputs[0]
                        elif mm_type == 'video' and video_inputs:
                            sample_mm_data["video"] = video_inputs[0]
                            for key, value in video_kwargs.items():
                                sample_video_kw[key] = value[0]
                        
                        # Create input for vLLM with unique sampling parameters for this run
                        run_sampling_params = SamplingParams(
                            # raise the ceiling: 0.8 → 1.4 across 5 runs
                            temperature       = 0.8 + 0.15 * run_idx,
                            # gradually widen the nucleus (70 % → 95 %)
                            top_p             = min(0.95, 0.7 + 0.05 * run_idx),
                            # keep at most the 50 most‑likely tokens in play
                            top_k             = 50,
                            # soft repetition penalties (optional but often helpful)
                            presence_penalty  = 0.4,
                            frequency_penalty = 0.2,
                            max_tokens        = 1024,
                            stop              = ["</answer>"]
                        )

                        
                        
                        sample_input = {
                            "prompt": prompt_variation,
                            "multi_modal_data": sample_mm_data,
                            "mm_processor_kwargs": sample_video_kw,
                        }
                        
                        # Generate single output
                        try:
                            outputs = llm.generate([sample_input], sampling_params=run_sampling_params)
                            if outputs and len(outputs) > 0:
                                output_text = outputs[0].outputs[0].text
                                # Make sure the answer tag is properly closed
                                if "<answer>" in output_text and "</answer>" not in output_text:
                                    output_text = output_text + "</answer>"
                                sample_outputs.append(output_text)
                                
                                # Print the extracted answer for debugging
                                print(f"    Run {run_idx+1} answer: {extract_answer(output_text)}")
                            else:
                                sample_outputs.append("<answer>no output generated</answer>")
                        except Exception as e:
                            print(f"Error generating output {run_idx+1} for sample {i + sample_idx}: {e}")
                            sample_outputs.append(f"<answer>error: {str(e)[:50]}</answer>")
                    
                    # Add outputs for this sample to the batch results
                    all_sample_outputs.append(sample_outputs)
                    
                    # Check if we got diverse answers
                    answers = [extract_answer(out) for out in sample_outputs if out]
                    unique_answers = len(set(answers))
                    print(f"    Sample diversity: {unique_answers} unique answers from {len(sample_outputs)} outputs")

                    ### only save the samples that have one unique answer

                    if unique_answers == 1 or (nframes*2) > MAX_NFRAMES[dataset_name]:
                        try:
                            # Add this sample to final output
                            sample_copy = data[i + sample_idx].copy()
                            
                            # Make sure we have NUM_SAMPLES outputs
                            while len(sample_outputs) < NUM_SAMPLES:
                                sample_outputs.append("<answer>insufficient samples</answer>")
                            
                            majority = majority_vote(sample_outputs)
                            sample_copy["output"] = sample_outputs[0]
                            sample_copy["prediction"] = extract_answer(sample_outputs[0])
                            sample_copy["majority_vote"] = majority
                            sample_copy["all_outputs"] = sample_outputs
                            sample_copy["vote_counts"] = get_vote_counts(sample_outputs)
                            
                            # Calculate reward against ground truth solution
                            sample_copy["reward"] = reward_fn(data[i + sample_idx], sample_outputs[0], 
                                                            data[i + sample_idx].get("problem_type", ""))
                            
                            # Calculate reward against majority vote
                            sample_copy["majority_vote_reward"] = reward_fn(
                                data[i + sample_idx], 
                                sample_outputs[0], 
                                data[i + sample_idx].get("problem_type", ""),
                                comparison_target=majority
                            )
                            
                            final_output.append(sample_copy)

                            ### add the idx to the solved data list
                            solved_data_list.append(data[i + sample_idx])
                            print(f"    Sample {i + sample_idx} saved with prediction: {sample_copy['prediction']}")
                            
                            # Save progress
                            with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                                json.dump({"results": final_output}, f, indent=2, ensure_ascii=False)
                            # print(f"    Saved progress: {len(final_output)}/{total_samples} samples")

                        
                                            
                        except Exception as e:
                            print(f"Error saving sample {i + sample_idx}: {e}")
                            traceback.print_exc()
                    
                    # Clear GPU memory after each sample
                    clear_gpu_memory()
                
                pbar.update(1)

        except Exception as e:
            print(f"Fatal error in main processing loop: {str(e)}")
            traceback.print_exc()
        finally:
            pbar.close()

        ### add nframes
        nframes = nframes * 2

    # Save final results
    save_final_results(OUTPUT_PATH, final_output, total_samples)

    # Compute accuracies 
    with open(OUTPUT_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    _, maj_acc = compute_accuracies(data)
    print("dataset:", dataset_name)
    print(f'Overall accuracy: {maj_acc:.4f} ({maj_acc*100:.2f}%)')

    # print(f"Evaluation completed for {dataset_name}")

