# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from datasets import load_dataset
from transformers import Qwen2VLForConditionalGeneration,Qwen2_5_VLForConditionalGeneration

from math_verify import parse, verify
from open_r1_video.trainer import Qwen2VLGRPOTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    jsonl_path: Optional[str] = field(
        default=None,
        metadata={"help": "json file path"},
    )



# ####### original version  ########
# def accuracy_reward(completions, solution, **kwargs):
#     """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
#     contents = [completion[0]["content"] for completion in completions]
#     rewards = []
#     current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
#     for content, sol in zip(contents, solution):
#         reward = 0.0
#         # Try symbolic verification first
#         try:
#             answer = parse(content)
#             if float(verify(answer, parse(sol))) > 0:
#                 reward = 1.0
#         except Exception:
#             pass  # Continue to next verification method if this fails

#         # If symbolic verification failed, try string matching
#         if reward == 0.0:
#             try:
#                 # Extract answer from solution if it has think/answer tags
#                 sol_match = re.search(r"<answer>(.*?)</answer>", sol, re.DOTALL)
#                 ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

#                 # Extract answer from content if it has think/answer tags
#                 content_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
#                 student_answer = content_match.group(1).strip() if content_match else content.strip()

#                 # Compare the extracted answers
#                 if student_answer == ground_truth:
#                     reward = 1.0
#             except Exception:
#                 pass  # Keep reward as 0.0 if both methods fail

#         rewards.append(reward)
#         if os.getenv("DEBUG_MODE") == "true":
#             log_path = os.getenv("LOG_PATH")
#             with open(log_path, "a") as f:
#                 f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
#                 f.write(f"Content: {content}\n")
#                 f.write(f"Solution: {sol}\n")
#         print(f"Content: {content}\n")
#         print(f"Solution: {sol}\n")
#         print(f"Reward: {reward}\n")
#     return rewards





####### modified version after qwen-2.5-vl ########


# Helper to extract just the first letter from a string.
def extract_answer_letter(text):
    m = re.match(r'\s*([A-Za-z])', text)
    return m.group(1).upper() if m else text.strip().upper()

# Enhanced extraction to pull the student answer from various possible formats.
def extract_student_answer(text):
    # First, try to extract from <answer> tags.
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # If no tag, look for a "Final answer:" pattern.
    m = re.search(r"(?i)final answer.*?:\s*([A-Za-z])", text)
    if m:
        return m.group(1).strip()
    # Look for a "correct answer" pattern.
    m = re.search(r"(?i)correct answer.*?:\s*([A-Za-z])", text)
    if m:
        return m.group(1).strip()
    # Fallback: extract the first letter of the text.
    return extract_answer_letter(text)

def accuracy_reward(completions, solution, **kwargs):
    """
    Reward function that checks if the completion is correct using either symbolic 
    verification or answer letter matching.
    """
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    
    for content, sol in zip(contents, solution):
        reward = 0.0
        
        # Attempt symbolic verification (if applicable)
        try:
            answer = parse(content)
            if float(verify(answer, parse(sol))) > 0:
                reward = 1.0
        except Exception:
            pass  # If symbolic verification fails, try the next method
        
        # If symbolic verification did not produce a reward, try answer letter matching.
        if reward == 0.0:
            try:
                # Extract the ground truth answer from the solution (assumed in <answer> tags).
                sol_match = re.search(r"<answer>(.*?)</answer>", sol, re.DOTALL)
                ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
                
                # Extract the student answer using the enhanced function.
                student_answer = extract_student_answer(content)
                
                # Extract only the answer letters.
                solution_letter = extract_answer_letter(ground_truth)
                student_letter = extract_answer_letter(student_answer)
                
                if student_letter == solution_letter:
                    reward = 1.0
            except Exception:
                pass  # Leave reward as 0.0 if any error occurs
        
        rewards.append(reward)
        
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
        # print(f"Content: {content}\n")
        # print(f"Solution: {sol}\n")
        # print(f"Reward: {reward}\n")
    return rewards

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


# def format_reward(completions, **kwargs):
#     """Reward function that checks if the completion has a specific format."""
#     pattern = r"^<think>.*?</think>.*?<answer>.*?</answer>$"
#     completion_contents = [completion[0]["content"] for completion in completions]
#     matches = [re.search(pattern, content, re.DOTALL) for content in completion_contents]
#     format_rewards = []
#     for match, content in zip(matches, completion_contents):
#         local_reward = 0
#         if not match:
#             local_reward = local_reward - 0.5
#         elif len(content) > 10:
#             ans = content.split('<answer>', 1)[-1]
#         format_rewards.append(local_reward)
#     return format_rewards




reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


from datasets import Dataset, DatasetDict
import json

def create_dataset_from_jsonl_simple(jsonl_path):
    base_dataset = Dataset.from_json(jsonl_path)
    return DatasetDict({
        "train": base_dataset
    })


def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    
    if script_args.jsonl_path:
        # # load dataset from jsonl
        dataset = create_dataset_from_jsonl_simple(script_args.jsonl_path)
    else:
        # Load the dataset
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    # Format into conversation
    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
        }

    # QUESTION_TEMPLATE = "{Question}  Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."


    #### our training prompt
    QUESTION_TEMPLATE = (
        "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
        "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
        "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
        "<think> reasoning process here </think><answer> answer here </answer>"
        "{Question}\n"
        " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags."
    )



    def make_conversation_image(example):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])},
                    ],
                },
            ],
        }
    
    def make_conversation_video(example):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "video"},
                        # {"type": "video", "video": example["video"]},
                        # {"type": "video", "bytes": open(example["video"],"rb").read()},
                        {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])},
                    ],
                },
            ],
    }

    if "image" in dataset[script_args.dataset_train_split].features:
        dataset = dataset.map(make_conversation_image)  # Utilize multiprocessing for faster mapping
    elif "video" in dataset[script_args.dataset_train_split].features:
        dataset = dataset.map(
            make_conversation_video,
        )
    else:
        dataset = dataset.map(make_conversation)
        dataset = dataset.remove_columns("messages")
    
    # import pdb; pdb.set_trace()

    trainer_cls = Qwen2VLGRPOTrainer

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args, remain_args  = parser.parse_args_and_config(return_remaining_strings=True)
    print("remain_args", remain_args)
    main(script_args, training_args, model_args)

# if __name__ == "__main__":
#     parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
#     script_args, training_args, model_args, remain_args = parser.parse_args_and_config(
#         return_remaining_strings=True
#     )
    
#     # Just override the attribute on the parsed object
#     training_args.scale_rewards = False

#     print("remain_args", remain_args)
#     main(script_args, training_args, model_args)
