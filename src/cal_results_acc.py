import json
import re

# path to your output JSON file
JSON_PATH = ""

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

    majority_vote_acc = correct_majority / total
    return majority_vote_acc

def main():
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    maj_acc = compute_accuracies(data)
    print(f'Total items evaluated: {int(data.get("results", []) and len(data["results"]))}')
    print(f'Video QA accuracy: {maj_acc:.4f} ({maj_acc*100:.2f}%)')

if __name__ == '__main__':
    main()
