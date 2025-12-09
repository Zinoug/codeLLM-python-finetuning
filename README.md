# CodeLLM Python Fine-tuning Project

Fine-tuning CodeT5+ for 4 complementary tasks on Python code.

## 📋 Overview

This project fine-tunes a single CodeT5+ model to handle 4 tasks:

1. **Code Repair** (`fix bug:`) - Fix bugs in code
2. **Bug Detection** (`classify code:`) - Detect if code contains a bug (BUGGY/CORRECT)
3. **Code Summarization** (`summarize code:`) - Generate code descriptions
4. **Code Search** (`search code:`) - Search code from a description (multiple choice)

## 📁 Project Structure

```
.
├── data/                          # Source data
│   ├── bug_dataset.json          # 402 bugs with fixes (source: TheAlgorithms/Python)
│   └── nl_pl_dataset.json        # 1479 code-docstring pairs (source: boltons, more-itertools)
│
├── final_data/                    # ⭐ FINAL DATASETS FOR TRAINING
│   ├── train.jsonl               # 1286 samples (70%)
│   ├── val.jsonl                 # 241 samples (15%)
│   └── test.jsonl                # 241 samples (15%)
│
├── prepare_final_dataset.py      # ⭐ MAIN SCRIPT - Creates final datasets
│
└── README_PROJECT.md             # This file
```

## 🚀 Usage

### Option A: Google Colab (Recommended - Free GPU!)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Zinoug/codeLLM-python-finetuning/blob/main/colab_finetune.ipynb)

**Steps:**
1. Click the badge above or open `colab_finetune.ipynb` in Colab
2. Go to **Runtime > Change runtime type > Select GPU (T4)**
3. Run all cells
4. Training takes ~2-3 hours on T4 GPU (free tier)
5. Download trained model at the end

**Advantages:**
- ✅ Free GPU (Tesla T4)
- ✅ No setup required
- ✅ All dependencies pre-installed
- ✅ Save directly to Google Drive

### Option B: Local Setup

#### 1. Install PyTorch with CUDA

Check your CUDA version first:
```bash
nvidia-smi
```

Install PyTorch (choose one):
```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1+
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU only (not recommended)
pip install torch torchvision torchaudio
```

Verify CUDA:
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

#### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare Data

The final dataset is already created in `final_data/`, but to regenerate it:

```bash
python prepare_final_dataset.py
```

This script:
- Loads `bug_dataset.json` and `nl_pl_dataset.json`
- Generates 4 task types
- **Balances** all tasks to the same number of samples (402 per task)
- **Shuffles** samples in batches of 4 (1 sample from each task)
- Splits into train/val/test (70/15/15)
- Saves to `final_data/` with unit tests for repair samples

### 4. Data Format

Each JSONL line:

```json
{
  "task": "repair|detection|summary|search",
  "input": "fix bug:\n<code>",
  "output": "<fixed_code>",
  "tests": ["assert ..."]  // Only in test.jsonl for repair task
}
```

**Examples:**

```json
// Code Repair
{"task": "repair", "input": "fix bug:\ndef add(a,b): return a-b", "output": "def add(a,b): return a+b"}

// Bug Detection
{"task": "detection", "input": "classify code:\ndef add(a,b): return a-b", "output": "BUGGY"}

// Code Summarization
{"task": "summary", "input": "summarize code:\ndef add(a,b): return a+b", "output": "Add two numbers"}

// Code Search
{"task": "search", "input": "search code:\nAdd two numbers\n\nChoices:\n0: def sub(a,b)...\n1: def add(a,b)...", "output": "1"}
```

### 5. Fine-tuning

Run the fine-tuning script:

```bash
python finetune_codet5_multitask.py
```

**Training Configuration:**
- Base model: `Salesforce/codet5-base`
- Batch size: 2 per device (effective: 8 with gradient accumulation)
- Learning rate: 5e-5
- Epochs: 3
- Mixed precision (fp16): Enabled for faster training
- Checkpoints: Saved every 500 steps to `codet5_multitask_checkpoint/`
- Final model: Saved to `codet5_multitask_final/`

### 6. Evaluation

After training, evaluate the model:

```bash
python evaluate_model.py
```

**Evaluation Metrics:**
- **Code Search**: Accuracy (exact match of choice index)
- **Bug Detection**: Accuracy (BUGGY vs CORRECT classification)
- **Code Summarization**: ROUGE-1/2/L + BLEU-4
- **Code Repair**: **pass@1** (unit test execution) + Exact Match + BLEU-4

**Output:**
Results are saved to `eval_outputs/`:
- `search_errors.json` - Mispredicted search samples
- `detection_errors.json` - Mispredicted detection samples
- `summary_predictions.json` - All summary predictions vs references
- `repair_predictions.json` - All repair predictions with **test execution results**

**pass@1 Metric:**
The pass@1 score measures the percentage of repair samples where the generated code passes **ALL** unit tests. This is the gold standard for code repair evaluation!

## 📊 Dataset Statistics

| Split | Total | Repair | Detection | Summary | Search |
|-------|-------|--------|-----------|---------|--------|
| Train | 1126  | ~281   | ~281      | ~282    | ~282   |
| Val   | 241   | ~60    | ~60       | ~60     | ~61    |
| Test  | 241   | ~61    | ~61       | ~60     | ~59    |

**Total: 1608 samples** (402 per task × 4 tasks)

## 🔍 Mining Methodology

### Bugs (data/bug_dataset.json)
- **Source**: GitHub repository `TheAlgorithms/Python`
- **Method**: PyDriller to analyze commit history
- **Validation**: 
  - Buggy code must FAIL tests
  - Fixed code must PASS tests
  - Tests extracted from doctests
- **Result**: 402 validated bugs with fixes

### Code-NL Pairs (data/nl_pl_dataset.json)
- **Sources**: `boltons`, `more-itertools`, `toolz`
- **Method**: AST extraction + docstring parsing
- **Filtering**: Simple functions with clear docstrings
- **Result**: 1479 function-docstring pairs

## 🎯 Task Balancing

**Problem**: Unequal number of samples per task
- Repair: 402 samples
- Detection: 804 samples (2× repair because buggy + correct)
- Summary: 1479 samples
- Search: 1479 samples

**Solution**: Limit all tasks to 402 samples (the minimum)

**Advantage**: The model learns each task fairly without bias.

## 📝 Important Files

### Main Script
- `prepare_final_dataset.py` - **MAIN SCRIPT** to create final datasets

### Source Data
- `data/bug_dataset.json` - Original bugs (402 samples)
- `data/nl_pl_dataset.json` - Original code-NL pairs (1479 samples)

### Final Datasets (Ready to Use)
- `final_data/train.jsonl` - **USE FOR TRAINING** (1126 samples, 70%)
- `final_data/val.jsonl` - **USE FOR VALIDATION** (241 samples, 15%)
- `final_data/test.jsonl` - **USE FOR EVALUATION** (241 samples, 15%)

## 🔧 Dependencies

See `requirements.txt` for the complete list. Key dependencies:

- **PyTorch** (2.0.0+) with CUDA support
- **Transformers** (4.30.0+) for CodeT5
- **Evaluate** (0.4.0+) for metrics (ROUGE, BLEU)
- **pydriller** for data mining (optional)

## 🐛 Troubleshooting

### CUDA Out of Memory
If you get OOM errors during training:
1. Reduce `per_device_train_batch_size` to 1
2. Increase `gradient_accumulation_steps` to maintain effective batch size
3. Disable fp16: `fp16=False` in TrainingArguments
4. Reduce `max_input_len` or `max_output_len` in the dataset class

### CUDA Not Available
If `torch.cuda.is_available()` returns False:
1. Check NVIDIA driver: `nvidia-smi`
2. Verify CUDA installation: `nvcc --version`
3. Reinstall PyTorch with correct CUDA version (see Setup section)
4. Check that PyTorch and CUDA versions are compatible

### Slow Training
To speed up training:
1. Enable fp16 mixed precision (requires CUDA >= 7.0)
2. Increase batch size if GPU memory allows
3. Use `torch.compile()` if PyTorch >= 2.0
4. Consider using multiple GPUs with `accelerate`

## 📚 References

- **CodeT5**: [Salesforce/codet5-base](https://huggingface.co/Salesforce/codet5-base)
- **Dataset Sources**: TheAlgorithms/Python, boltons, more-itertools
- **Evaluation**: pass@1 metric for code repair with unit test execution

---

Academic project - Seoul National University of Technology and Science

