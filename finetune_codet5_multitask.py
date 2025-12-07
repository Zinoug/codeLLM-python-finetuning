import json
from dataclasses import dataclass
from typing import Dict, List

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
)
from torch.utils.data import Dataset

MODEL_NAME = "Salesforce/codet5-base"  # o el checkpoint que estés usando

TRAIN_PATH = "multitask_dataset/train.jsonl"
VAL_PATH = "multitask_dataset/val.jsonl"
TEST_PATH = "multitask_dataset/test.jsonl"


# -------- Dataset --------

class JsonlSeq2SeqDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_input_len=512, max_output_len=128):
        self.examples = []
        with open(path, "r", encoding="utf8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                self.examples.append(obj)

        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        inp = ex["input"]
        out = ex["output"]

        model_inputs = self.tokenizer(
            inp,
            max_length=self.max_input_len,
            truncation=True,
            padding=False,
        )

        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                out,
                max_length=self.max_output_len,
                truncation=True,
                padding=False,
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


# -------- Collator simple (pad a batch) --------

@dataclass
class DataCollator:
    tokenizer: AutoTokenizer
    label_pad_token_id: int = -100

    def __call__(self, features: List[Dict]) -> Dict[str, List[int]]:
        # padding para inputs
        batch = self.tokenizer.pad(
            {"input_ids": [f["input_ids"] for f in features],
             "attention_mask": [f["attention_mask"] for f in features]},
            padding=True,
            return_tensors="pt",
        )

        # padding para labels
        labels = [f["labels"] for f in features]
        labels_batch = self.tokenizer.pad(
            {"input_ids": labels},
            padding=True,
            return_tensors="pt",
        )["input_ids"]

        # reemplazar pad_token_id por -100 para que no cuenten en la loss
        labels_batch[labels_batch == self.tokenizer.pad_token_id] = self.label_pad_token_id
        batch["labels"] = labels_batch
        return batch


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    # datasets
    train_dataset = JsonlSeq2SeqDataset(TRAIN_PATH, tokenizer)
    val_dataset = JsonlSeq2SeqDataset(VAL_PATH, tokenizer)

    data_collator = DataCollator(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir="codet5_multitask_ckpt",
        per_device_train_batch_size=2,   # súbelo si tu GPU aguanta
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,   # batch efectivo = 2 * 4 = 8
        learning_rate=5e-5,
        num_train_epochs=3,
        logging_steps=50,
        report_to="none",    
        # OJO: sin evaluation_strategy, eval_steps, save_steps,
        # save_total_limit, predict_with_generate, fp16
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model("codet5_multitask_finetuned")


if __name__ == "__main__":
    main()
