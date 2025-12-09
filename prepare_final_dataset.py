"""
Final Script: Multi-Task Dataset Preparation for CodeT5+

This script combines bug and code-NL datasets to create final training files
for the 4 tasks:
1. Code Repair (fix bug)
2. Bug Detection (classify code)
3. Code Summarization (summarize code)
4. Code Search (search code)

Sources:
- data/bug_dataset.json: Bugs and fixes (Tasks 1 & 2)
- data/nl_pl_dataset.json: Code-docstring pairs (Tasks 3 & 4)

Output:
- final_data/train.jsonl: Training data (80%)
- final_data/val.jsonl: Validation data (10%)
- final_data/test.jsonl: Test data (10%)

Tasks are balanced (same number of samples) and shuffled in batches of 4.
"""

import json
import random
from pathlib import Path
from typing import List, Dict

# Configuration
RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def clean_code(code: str) -> str:
    """Remove docstrings from code"""
    lines = code.split("\n")
    result = []
    in_docstring = False
    docstring_char = None

    for line in lines:
        if '"""' in line or "'''" in line:
            char = '"""' if '"""' in line else "'''"
            if not in_docstring:
                in_docstring = True
                docstring_char = char
                if line.count(char) >= 2:
                    in_docstring = False
                continue
            elif char == docstring_char:
                in_docstring = False
                continue

        if not in_docstring:
            result.append(line)

    return "\n".join(result).strip()


def clean_docstring(docstring: str) -> str:
    """Clean and simplify docstring to get a concise summary"""
    lines = [line.strip() for line in docstring.split("\n") if line.strip()]

    summary = ""
    for line in lines:
        if line.startswith(
            (">>>", "...", "Args:", "Returns:", "Note:", "Example:", "Raises:")
        ):
            break
        if line and not line.startswith(":"):
            summary = line
            break

    if not summary and lines:
        summary = lines[0]

    return " ".join(summary.split())


# ===== TASK 1: CODE REPAIR =====
def create_repair_samples(
    bug_data: List[Dict], include_tests: bool = False
) -> List[Dict]:
    """Create samples for bug repair

    Args:
        bug_data: List of bug fix samples
        include_tests: If True, include unit tests for evaluation (for test split)
    """
    samples = []
    for item in bug_data:
        sample = {
            "task": "repair",
            "input": f"fix bug:\n{item['pl']}",
            "output": item["fixed_code"],
        }
        # Add tests if available and requested (for test split evaluation)
        if include_tests and "tests" in item and item["tests"]:
            sample["tests"] = item["tests"]
        samples.append(sample)
    return samples


# ===== TASK 2: BUG DETECTION =====
def create_detection_samples(bug_data: List[Dict]) -> List[Dict]:
    """Create samples for bug detection"""
    samples = []
    for item in bug_data:
        # Buggy code
        samples.append(
            {
                "task": "detection",
                "input": f"classify code:\n{item['pl']}",
                "output": "BUGGY",
            }
        )
        # Correct code
        samples.append(
            {
                "task": "detection",
                "input": f"classify code:\n{item['fixed_code']}",
                "output": "CORRECT",
            }
        )
    return samples


# ===== TASK 3: CODE SUMMARIZATION =====
def create_summary_samples(nl_pl_data: List[Dict]) -> List[Dict]:
    """Create samples for code summarization"""
    samples = []
    for item in nl_pl_data:
        code = clean_code(item["code"])
        summary = clean_docstring(item["docstring"])

        if not code or not summary:
            continue

        sample = {
            "task": "summary",
            "input": f"summarize code:\n{code}",
            "output": summary,
        }
        samples.append(sample)
    return samples


# ===== TASK 4: CODE SEARCH =====
def create_search_samples(
    nl_pl_data: List[Dict], num_distractors: int = 2
) -> List[Dict]:
    """Create samples for code search (multiple choice)"""
    samples = []

    for i, item in enumerate(nl_pl_data):
        summary = clean_docstring(item["docstring"])
        correct_code = clean_code(item["code"])

        if not summary or not correct_code:
            continue

        # Select distractors
        available = [x for j, x in enumerate(nl_pl_data) if j != i]
        if len(available) < num_distractors:
            continue

        distractors = random.sample(available, num_distractors)

        # Create choices
        choices = [{"code": correct_code, "is_correct": True}]
        for distractor in distractors:
            dist_code = clean_code(distractor["code"])
            if dist_code:
                choices.append({"code": dist_code, "is_correct": False})

        # Shuffle
        random.shuffle(choices)

        # Find correct index
        correct_idx = next(i for i, c in enumerate(choices) if c["is_correct"])

        # Format input
        input_text = f"search code:\n{summary}\n\nChoices:"
        for idx, choice in enumerate(choices):
            code_preview = str(choice["code"]).replace("\n", " ")
            input_text += f"\n{idx}: {code_preview}"

        sample = {"task": "search", "input": input_text, "output": str(correct_idx)}
        samples.append(sample)

    return samples


def balance_tasks(task_samples: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
    """Balance tasks to have the same number of samples"""
    # Find minimum
    min_samples = min(len(samples) for samples in task_samples.values())

    print(f"\n📊 Balancing tasks to minimum: {min_samples} samples per task")

    # Limit each task to minimum
    balanced = {}
    for task, samples in task_samples.items():
        random.shuffle(samples)
        balanced[task] = samples[:min_samples]
        print(f"  {task}: {len(task_samples[task])} → {len(balanced[task])}")

    return balanced


def create_batches(task_samples: Dict[str, List[Dict]]) -> List[Dict]:
    """Create batches of 4 samples (1 per task) shuffled"""
    tasks = list(task_samples.keys())
    num_samples = len(task_samples[tasks[0]])  # All tasks have the same size

    all_samples = []

    # Create batches of 4 (1 from each task)
    for i in range(num_samples):
        batch = []
        for task in tasks:
            if i < len(task_samples[task]):
                batch.append(task_samples[task][i])

        # Shuffle the 4 samples in the batch
        random.shuffle(batch)
        all_samples.extend(batch)

    return all_samples


def split_dataset(
    samples: List[Dict],
    train_ratio: float,
    val_ratio: float,
    bug_data: List[Dict] = None,
) -> tuple:
    """Split dataset into train/val/test

    Args:
        samples: All samples to split
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        bug_data: Original bug data (to add tests to test split repair samples)
    """
    total = len(samples)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)

    train = samples[:train_size]
    val = samples[train_size : train_size + val_size]
    test = samples[train_size + val_size :]

    # Add tests to repair samples in test split for pass@1 evaluation
    if bug_data:
        # Create a mapping from buggy code to tests
        bug_to_tests = {
            item["pl"]: item.get("tests", []) for item in bug_data if "tests" in item
        }

        for sample in test:
            if sample["task"] == "repair":
                # Extract buggy code from input (after "fix bug:\n")
                buggy_code = sample["input"].replace("fix bug:\n", "", 1)
                if buggy_code in bug_to_tests and bug_to_tests[buggy_code]:
                    sample["tests"] = bug_to_tests[buggy_code]

    return train, val, test


def save_jsonl(samples: List[Dict], filepath: Path):
    """Save samples to JSONL format"""
    with open(filepath, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def main():
    print("=" * 70)
    print("🚀 Final Multi-Task Dataset Preparation for CodeT5+")
    print("=" * 70)

    # Set random seed
    random.seed(RANDOM_SEED)

    # Load data
    print("\n📂 Loading source data...")

    with open("data/bug_dataset.json", "r", encoding="utf-8") as f:
        bug_data = json.load(f)
    print(f"  ✅ bug_dataset.json: {len(bug_data)} bugs")

    with open("data/nl_pl_dataset.json", "r", encoding="utf-8") as f:
        nl_pl_data = json.load(f)
    print(f"  ✅ nl_pl_dataset.json: {len(nl_pl_data)} code-NL pairs")

    # Generate samples for each task
    print("\n🔧 Generating samples per task...")

    print("  1️⃣  Code Repair...")
    repair_samples = create_repair_samples(bug_data)
    print(f"      ✅ {len(repair_samples)} samples")

    print("  2️⃣  Bug Detection...")
    detection_samples = create_detection_samples(bug_data)
    print(f"      ✅ {len(detection_samples)} samples")

    print("  3️⃣  Code Summarization...")
    summary_samples = create_summary_samples(nl_pl_data)
    print(f"      ✅ {len(summary_samples)} samples")

    print("  4️⃣  Code Search...")
    search_samples = create_search_samples(nl_pl_data)
    print(f"      ✅ {len(search_samples)} samples")

    # Balance tasks
    task_samples = {
        "repair": repair_samples,
        "detection": detection_samples,
        "summary": summary_samples,
        "search": search_samples,
    }

    balanced_samples = balance_tasks(task_samples)

    # Create shuffled batches
    print("\n🔀 Creating shuffled batches (4 tasks per batch)...")
    all_samples = create_batches(balanced_samples)
    print(f"  ✅ {len(all_samples)} total samples")

    # Split train/val/test (pass bug_data to add tests to test split)
    print(
        f"\n📊 Dataset split ({TRAIN_RATIO*100:.0f}% train / {VAL_RATIO*100:.0f}% val / {TEST_RATIO*100:.0f}% test)..."
    )
    train, val, test = split_dataset(all_samples, TRAIN_RATIO, VAL_RATIO, bug_data)

    print(f"  Train: {len(train)} samples")
    print(f"  Val:   {len(val)} samples")
    print(f"  Test:  {len(test)} samples")

    # Count tasks in each split
    for split_name, split_data in [("Train", train), ("Val", val), ("Test", test)]:
        task_counts = {}
        for sample in split_data:
            task = sample["task"]
            task_counts[task] = task_counts.get(task, 0) + 1
        print(f"  {split_name} breakdown: {task_counts}")

    # Save files
    print("\n💾 Saving final files...")
    output_dir = Path("final_data")
    output_dir.mkdir(exist_ok=True)

    save_jsonl(train, output_dir / "train.jsonl")
    save_jsonl(val, output_dir / "val.jsonl")
    save_jsonl(test, output_dir / "test.jsonl")

    print(f"  ✅ train.jsonl")
    print(f"  ✅ val.jsonl")
    print(f"  ✅ test.jsonl")

    # Display examples
    print("\n" + "=" * 70)
    print("📋 Sample examples (1 per task):")
    print("=" * 70)

    for task in ["repair", "detection", "summary", "search"]:
        sample = next((s for s in train if s["task"] == task), None)
        if sample:
            print(f"\n🔹 Task: {task.upper()}")
            print(f"Input: {sample['input'][:150]}...")
            print(f"Output: {sample['output'][:100]}...")

    print("\n" + "=" * 70)
    print("✨ Final dataset ready for training!")
    print("=" * 70)
    print(f"\n📁 Files available in: {output_dir}/")
    print(f"  • train.jsonl ({len(train)} samples)")
    print(f"  • val.jsonl ({len(val)} samples)")
    print(f"  • test.jsonl ({len(test)} samples)")
    print(
        f"\n💡 Format: Each line = {{'task': '...', 'input': '...', 'output': '...'}}"
    )
    print(f"💡 Tasks are shuffled in batches of 4 for balanced training")


if __name__ == "__main__":
    main()
