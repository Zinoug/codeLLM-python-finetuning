---
marp: true
theme: default
paginate: true
---

# Multi-Task Fine-Tuning of CodeT5+
## Code Search, Repair, Summarization & Bug Detection

**Mid-Project Presentation**
**Team 5**
November 13, 2025

---

## Project Goal

**Fine-tune a single CodeT5+ model for 4 complementary tasks:**

1. **Code Search** 
2. **Code Repair** 
3. **Code Summarization** 
4. **Bug Detection** 


---

## Choice of CodeLLM

### CodeT5+ 

**Why CodeT5+?**
- Encoder-decoder architecture (supports all tasks)
- Pre-trained on code + natural language
- Task-prefix based multi-task learning



---

## Dataset Overview

| Task | Dataset Type | Status | Source |
|------|--------------|--------|--------|
| Code Repair | Bug fixes | 🚧 **Mined (36)** (in progress) | GitHub commits |
| Bug Detection | Buggy/Correct | 🚧 **Mined (36)**(in progress) | Same as Repair |
| Code Search | Code-NL pairs | ⏳ **To Mine** | Docstrings |
| Code Summary | Code-NL pairs | ⏳ **To Mine** | Docstrings |

---

## Dataset 1: Bug Fixes (In Progress)

### Mining Methodology
- **Tool**: PyDriller (commit history analysis)
- **Target**: Bug fix commits with top-level functions
- **Sources**: 
  - TheAlgorithms/Python (28 samples)
  - more-itertools (5 samples)
  - toolz (2 samples)
  - boltons/iterutils (1 sample)

**Total**: 36 validated bug samples

---

## Dataset 1: Validation Process

**Strict Validation (eliminates ~95% of candidates):**

1. Buggy code must **FAIL** tests
2. Fixed code must **PASS** tests
3. Tests extracted from:
   - Doctest examples

Result: High-quality, real bugs with verified fixes

---

## Dataset 1: Sample Format

```json
{
  "function": "binary_search",
  "buggy_code": "def binary_search(arr, target):\n ...   ",
  "fixed_code": "def binary_search(arr, target):\n ...   ",
  "tests": ["assert binary_search([1,2,3], 2) == 1"],
}
```

---
## How Same Dataset Serves Both Tasks
## Task Training Format
```json
// Code Repair : Pairs of buggy/fixed and fixed/fixed from the dataset
{"input": "fix a bug:\ndef add(a,b): return a-b", 
 "output": "def add(a,b): return a+b"}

// Bug Detection : choose buggy and fixed code from the dataset
{"input": "classify BUGGY OR CORRECT :\ndef add(a,b): return a-b",
 "output": "BUGGY"}
```
---

## Dataset 1: Data Augmentation Plan

**Challenge**: 36 samples insufficient for robust training

**Solution**: Find others Github repos or Synthetic Bug Injection
- Extract fixed code + tests from GitHub (Easier than buggy + fixed + tests)
- Inject realistic bugs using mutation operators:
  - Operator changes (`+` → `-`, `==` → `!=`)
  - Off-by-one errors (`i` → `i+1`)
  - Boundary modifications

---

## Dataset 2: Code-NL Pairs (In Progress)

### Target Repositories (Simple, Well-documented Python projects)

**Top Candidates**:
1. **TheAlgorithms/Python** - Educational algorithms with clear docstrings
2. **more-itertools** - Iterator utilities, simple functions
3. **python-string-utils** - String manipulation, straightforward
4. **humanize** - Simple formatting functions
5. **boltons** - Utility functions, well-documented

**Selection Criteria**: 
- Simple, focused functions (not complex classes)
- Clear, concise docstrings, Educational/utility functions (easier to understand)


---

## Dataset 2: Mining Methodology

### Extraction Pipeline

**Phase 1: Repository Selection**
- Focus on 2-3 well-documented repos
- Target: 400-500 function-docstring pairs

**Phase 2: Code-NL Extraction**

For each function in the repository:
1. **Extract the code** (function body)
2. **Extract the docstring** (text between `"""`)
3. **Validate** (simple code + clear description)
4. **Add to dataset** (if valid)

---
Example:
```
def multiply(a, b):
    """Multiply two numbers"""
    return a * b

→ Extract: code + "Multiply two numbers" → Valid 
```


---

## How Same Dataset Serves Both Tasks

### Base Data: Function-Docstring Pairs

```python
def multiply(a, b):
    """Multiply two numbers and return the result"""
    return a * b

# Extraction:
code = "def multiply(a, b):\n    return a * b"
docstring = "Multiply two numbers and return the result"
```

### Task 1: Code Summary (Direct)
```json
{"input": "summarize code:\ndef multiply(a,b): return a*b",
 "output": "Multiply two numbers and return the result"}
```

---

## How Same Dataset Serves Both Tasks

### Task 2: Code Search (Multiple-Choice)

**For each pair, create choices:**
1. **Correct code** (from the pair)
2. **2 Distractor codes** (random samples from dataset)

```json
{"input": "search code:\nMultiply two numbers\n 
Choices:
0: def add(a,b): return a+b
1: def multiply(a,b): return a*b
2: def subtract(a,b): return a-b",
 "output": "1"}
```

**Advantage**: Same mining effort, two tasks

---

## Multi-Task Training Format

**Task-Specific Prefixes**:

```json
// Code Repair
{"input": "fix a bug:\ndef add(a,b): return a-b", 
 "output": "def add(a,b): return a+b"}

// Bug Detection
{"input": "classify BUGGY OR CORRECT :\ndef add(a,b): return a-b",
 "output": "BUGGY"}

// Code Search
{"input": "search code:\nAdd two numbers\nChoices:...",
 "output": "2"}

// Code Summary
{"input": "summarize code:\ndef add(a,b): return a+b",
 "output": "Add two numbers"}
```

---

## Fine-Tuning Plan

### Step 1: Data Preparation
1. Prepare Code Repair + Bug Detection 
2. Mine Code Search + Summary datasets 
3. Augment Repair dataset
4. Merge all tasks with prefixes
5. Split: 70% train, 15% val, 15% test

---

## Fine-Tuning Plan

### Step 2: Training Configuration
- **Base Model**: CodeT5+ 
- **Strategy**: Mixed-task batches



---

## Evaluation Plan

### Task 1: Code Repair
**Metrics**:
- **Pass@1**: % samples where ALL tests pass

**Comparison**: Base CodeT5+ vs. Fine-tuned

---

## Evaluation Plan

### Task 2: Bug Detection
**Metrics**:

- **F1-Score**: Harmonic mean

**Method**: Parse "BUGGY"/"CORRECT" from generated text

---

## Evaluation Plan

### Task 3: Code Search
**Metrics**:
- **Accuracy**: Correct top-1 retrievals

**Setup**: Multiple-choice (1 correct + 2 distractors)

---

## Evaluation Plan

### Task 4: Code Summarization
**Metrics**:
- **BLEU**: N-gram overlap with reference
- **ROUGE-L**: Longest common subsequence


---


## Current Challenges

1. **Limited bug samples** (36 → need augmentation)
   - Solution: Find new repos or synthetic bug injection

2. **Code-NL pairs to mine** (need 400-500)
   - Solution: Target well-documented repos 

3. **Multi-task balancing** (different task difficulties)
   - Solution: Weighted sampling or task-specific learning rates

4. **Evaluation complexity** (4 different metrics)
   - Solution: Automated evaluation pipeline

---


## Next Steps 

1. **Mine Code-NL pairs** from selected repositories
   - Extract 400-500 function-docstring pairs

2. **Augment bug dataset**

3. **Prepare multi-task dataset** 
   - Merge all 4 tasks with prefixes
   - Create train/val/test splits

---

## Next Steps 

4. **Fine-tune CodeT5+ model**
   - Train on mixed-task batches
   - Monitor validation loss for all tasks

5. **Run comprehensive evaluations**
   - Test on all 4 tasks separately
   - Compare base vs. fine-tuned

6. **Analyze results & iterate**
   - Identify weak points
   - Adjust training if needed

---

# Thank You!

Questions ? 

