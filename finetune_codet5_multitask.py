"""
Fine-tuning CodeT5 for Multi-Task Learning

This script fine-tunes CodeT5-base on 4 code-related tasks:
1. Code Repair (fix bug)
2. Bug Detection (classify code)
3. Code Summarization (summarize code)
4. Code Search (search code)

The model is trained on the balanced dataset from final_data/
"""

import json
import torch
from dataclasses import dataclass
from typing import Dict, List
from collections import Counter
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from torch.utils.data import Dataset

# Configuration
MODEL_NAME = "Salesforce/codet5-base"
TRAIN_PATH = "final_data/train.jsonl"
VAL_PATH = "final_data/val.jsonl"
TEST_PATH = "final_data/test.jsonl"  # por si luego quieres usarlo aquí también


# ============================================
# Dataset
# ============================================


class JsonlSeq2SeqDataset(Dataset):
    """Dataset for loading JSONL files with input/output pairs"""

    def __init__(self, path: str, tokenizer, max_input_len=768, max_output_len=256, 
                 oversample_detection: int = 3):
        # Load all raw examples
        raw_examples = []
        with open(path, "r", encoding="utf8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                raw_examples.append(obj)

        # Oversample detection task (hardest task, needs more exposure)
        examples = []
        for ex in raw_examples:
            examples.append(ex)
            # Add extra copies of detection samples
            if ex.get("task") == "detection" and oversample_detection > 1:
                for _ in range(oversample_detection - 1):
                    examples.append(ex)
        
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        inp = ex["input"]
        out = ex["output"]

        # Tokenize input and output together
        model_inputs = self.tokenizer(
            inp,
            text_target=out,
            max_length=self.max_input_len,
            truncation=True,
            padding=False,
        )
        
        # Truncate labels separately if needed
        if len(model_inputs["labels"]) > self.max_output_len:
            model_inputs["labels"] = model_inputs["labels"][:self.max_output_len]
        
        return model_inputs


# ============================================
# Data Collator
# ============================================


@dataclass
class DataCollator:
    """Collator to pad batches dynamically"""

    tokenizer: AutoTokenizer
    label_pad_token_id: int = -100

    def __call__(self, features: List[Dict]) -> Dict[str, List[int]]:
        # Pad inputs
        batch = self.tokenizer.pad(
            {
                "input_ids": [f["input_ids"] for f in features],
                "attention_mask": [f["attention_mask"] for f in features],
            },
            padding=True,
            return_tensors="pt",
        )

        # Pad labels
        labels = [f["labels"] for f in features]
        labels_batch = self.tokenizer.pad(
            {"input_ids": labels},
            padding=True,
            return_tensors="pt",
        )["input_ids"]

        # Replace pad_token_id with -100 so they don't contribute to loss
        labels_batch[labels_batch == self.tokenizer.pad_token_id] = (
            self.label_pad_token_id
        )
        batch["labels"] = labels_batch
        return batch


# ============================================
# Progress Callback
# ============================================


class ProgressCallback(TrainerCallback):
    """Custom callback for better progress logging"""

    def __init__(self, total_steps):
        self.total_steps = total_steps
        self.pbar = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.pbar = tqdm(total=self.total_steps, desc="Training", unit="step")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.pbar and logs:
            current_step = state.global_step
            self.pbar.n = current_step
            self.pbar.refresh()

            if "loss" in logs:
                self.pbar.set_postfix({"loss": f"{logs['loss']:.4f}"})

            if "eval_loss" in logs:
                print(f"\n📊 Eval at step {current_step}: loss={logs['eval_loss']:.4f}")

    def on_train_end(self, args, state, control, **kwargs):
        if self.pbar:
            self.pbar.close()


# ============================================
# Main Training Function
# ============================================


def main():
    print("=" * 70)
    print("🚀 Fine-tuning CodeT5 for Multi-Task Learning")
    print("=" * 70)

    # Check CUDA
    print("\n💻 System Information:")
    print(f"   • PyTorch version: {torch.__version__}")
    print(f"   • CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   • CUDA version: {torch.version.cuda}")
        print(f"   • GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"   • GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )
    else:
        print("   ⚠️  Warning: CUDA not available, training will be slow!")

    # Load tokenizer and model
    print("\n📥 Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   ✅ Model: {MODEL_NAME}")
    print(f"   • Total parameters: {total_params:,}")
    print(f"   • Trainable parameters: {trainable_params:,}")

    # Load datasets (unbalanced - using ALL available data)
    print("\n📂 Loading datasets...")
    train_dataset = JsonlSeq2SeqDataset(TRAIN_PATH, tokenizer, oversample_detection=1)
    val_dataset = JsonlSeq2SeqDataset(VAL_PATH, tokenizer, oversample_detection=1)
    print(f"   ✅ Train: {len(train_dataset)} samples")
    print(f"   ✅ Val:   {len(val_dataset)} samples")

    # Analyze task distribution (sobre el dataset ya oversampleado)
    print("\n📊 Task distribution in training set (after oversampling):")
    task_counts = Counter(ex["task"] for ex in train_dataset.examples)
    for task, count in sorted(task_counts.items()):
        percentage = (count / len(train_dataset)) * 100
        print(f"   • {task:12s}: {count:4d} samples ({percentage:.1f}%)")

    # Data collator
    data_collator = DataCollator(tokenizer=tokenizer)

    # Training arguments (optimized for ~2649 samples, unbalanced tasks)
    print("\n⚙️  Training configuration:")
    training_args = TrainingArguments(
        output_dir="codet5_multitask_checkpoint",

        # Batch & grad accumulation
        per_device_train_batch_size=4,          # ↑ Increased (more data = can use larger batch)
        per_device_eval_batch_size=8,           # ↑ Increased for faster eval
        gradient_accumulation_steps=4,          # Effective batch size = 16

        # Optimization
        learning_rate=5e-5,                     # Higher LR for larger dataset
        num_train_epochs=5,                     # ↓ Reduced (2.3x more data)
        warmup_ratio=0.06,                      # ~100 warmup steps
        weight_decay=0.01,
        lr_scheduler_type="cosine",             # Smoother convergence
        optim="adamw_torch",                    # Native PyTorch optimizer
        max_grad_norm=1.0,                      # Gradient clipping
        # Note: label_smoothing_factor removed - incompatible with T5 + fp16

        # Logging / reporting
        logging_steps=50,                       # Log every 50 steps
        report_to="none",

        # Evaluation & checkpoints (step-based for better model selection)
        eval_strategy="steps",
        eval_steps=165,                         # ~1x per epoch (2649/16=165)
        save_strategy="steps",
        save_steps=165,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # Mixed precision
        fp16=torch.cuda.is_available(),
        
        # Reproducibility
        seed=42,
        data_seed=42,
    )

    print(f"   • Batch size per device: {training_args.per_device_train_batch_size}")
    print(f"   • Gradient accumulation: {training_args.gradient_accumulation_steps}")
    print(
        f"   • Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}"
    )
    print(f"   • Learning rate: {training_args.learning_rate}")
    print(f"   • Warmup ratio: {training_args.warmup_ratio}")
    print(f"   • LR scheduler: {training_args.lr_scheduler_type}")
    print(f"   • Weight decay: {training_args.weight_decay}")
    print(f"   • Epochs: {training_args.num_train_epochs}")
    print(f"   • Max grad norm: {training_args.max_grad_norm}")
    print(f"   • Mixed precision (fp16): {training_args.fp16}")

    # Calculate training steps (aproximado)
    steps_per_epoch = len(train_dataset) // (
        training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
    )
    total_steps = steps_per_epoch * training_args.num_train_epochs
    print(f"\n📈 Training plan:")
    print(f"   • Steps per epoch: ~{steps_per_epoch}")
    print(f"   • Total training steps: ~{total_steps}")
    # `TrainingArguments` uses `eval_strategy` parameter name
    print(f"   • Evaluation strategy: {getattr(training_args, 'eval_strategy', getattr(training_args, 'evaluation_strategy', 'unknown'))}")
    print(f"   • Save strategy: {training_args.save_strategy}")

    # Initialize trainer with progress callback
    print("\n🔧 Initializing trainer...")
    progress_callback = ProgressCallback(total_steps)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[progress_callback],
    )

    # Train
    print("\n" + "=" * 70)
    print("🏋️  Starting training...")
    print("=" * 70)

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user!")
        print("💾 Saving current model state...")
        trainer.save_model("codet5_multitask_interrupted")
        tokenizer.save_pretrained("codet5_multitask_interrupted")
        print("   ✅ Model saved to: codet5_multitask_interrupted")
        return
    except Exception as e:
        print(f"\n\n❌ Training failed with error: {e}")
        raise

    # Save final model (mejor checkpoint si load_best_model_at_end=True)
    print("\n" + "=" * 70)
    print("💾 Saving final model...")
    trainer.save_model("codet5_multitask_final")
    tokenizer.save_pretrained("codet5_multitask_final")
    print(f"   ✅ Model saved to: codet5_multitask_final/")

    # Final summary
    print("\n" + "=" * 70)
    print("✨ Training completed successfully!")
    print("=" * 70)
    print(f"\n📊 Training summary:")
    print(f"   • Total steps completed: {trainer.state.global_step}")

    best = trainer.state.best_metric
    if best is not None:
        print(f"   • Best eval loss: {best:.4f}")
    else:
        print("   • Best eval loss: (no eval metrics tracked)")

    print(f"   • Checkpoints saved in: codet5_multitask_checkpoint/")
    print(f"   • Final model saved in: codet5_multitask_final/")

    print(f"\n🎯 Next steps:")
    print(f"   1. Evaluate the model: python evaluate_model.py")
    print(f"   2. Check results in: eval_outputs/")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
