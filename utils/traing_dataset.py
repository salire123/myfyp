import json
from typing import List, Dict
from datasets import Dataset



def load_json(path: str) -> List[List[Dict[str, str]]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_single_chat(chat: List[Dict[str, str]]) -> Dict[str, str]:
    prompt_parts = []
    for msg in chat:
        role = msg.get("role")
        content = msg.get("content", "").strip()

        if role == "system":
            prompt_parts.append(f"<|system|>\n{content}\n")
        elif role == "user":
            prompt_parts.append(f"<|user|>\n{content}\n")
        elif role == "assistant":
            prompt_parts.append(f"<|assistant|>\n")

    prompt = "".join(prompt_parts)
    output = chat[-1]["content"].strip() if chat and chat[-1]["role"] == "assistant" else ""
    return {"prompt": prompt, "output": output}


def convert_dataset(data: List[List[Dict[str, str]]]) -> List[Dict[str, str]]:
    return list(map(format_single_chat, data))


def save_as_jsonl(data: List[Dict[str, str]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for entry in data:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")

# use

def convert_dataset_and_save_as_file(input_path: str, output_path: str) -> None:
    raw_data = load_json(input_path)
    formatted = convert_dataset(raw_data)
    save_as_jsonl(formatted, output_path)

def load_jsonl(path: str) -> Dataset:
    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return Dataset.from_list(data)
    
def tokenize_the_data(examples, tokenizer):
    texts = [p + o for p, o in zip(examples["prompt"], examples["output"])]
    model_inputs = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=512
    )
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs


# ✅ Usage example
if __name__ == "__main__":
    convert_dataset_and_save_as_file(
        input_path="train.json",
        output_path="train.jsonl"
    )
