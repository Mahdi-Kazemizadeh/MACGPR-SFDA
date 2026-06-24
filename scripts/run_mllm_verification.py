import argparse
import base64
import json
import subprocess
from pathlib import Path
from typing import Any
import requests


CLASS_NAMES = [
    "aeroplane",
    "bicycle",
    "bus",
    "car",
    "horse",
    "knife",
    "motorcycle",
    "person",
    "plant",
    "skateboard",
    "train",
    "truck",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local MLLM verification on sanitized VisDA-C queries."
    )

    parser.add_argument(
        "--input",
        type=str,
        default="outputs/cgpr/mllm_queries_sanitized.jsonl",
        help="Path to sanitized MLLM query JSONL file.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="outputs/cgpr/mllm_responses_qwen.jsonl",
        help="Path to output MLLM response JSONL file.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="qwen2.5vl:7b",
        help="Ollama vision model name.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of samples to verify.",
    )

    return parser.parse_args()


def load_jsonl(path: Path, limit: int) -> list[dict[str, Any]]:
    records = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            records.append(json.loads(line))

            if len(records) >= limit:
                break

    return records


def image_to_base64(image_path: Path) -> str:
    with open(image_path, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")


def build_prompt(model_label: str) -> str:
    class_list = ", ".join(CLASS_NAMES)

    return (
        "You are verifying an image classification result for the VisDA-C dataset.\n"
        "You must choose exactly one label from the following list:\n"
        f"{class_list}\n\n"
        f"The current model predicted: {model_label}\n\n"
        "Look at the image carefully. If the model prediction is correct, return the same label. "
        "If it is wrong, return the correct label from the list.\n\n"
        "Return only one label name. Do not explain."
    )


def call_ollama(model: str, prompt: str, image_path: Path) -> str:

    image_b64 = image_to_base64(image_path)

    payload = {
        "model": model,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
    }

    response = requests.post(
        "http://localhost:11434/api/generate",
        json=payload,
        timeout=180,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Ollama HTTP {response.status_code}: {response.text}"
        )

    data = response.json()

    return str(data.get("response", "")).strip()


def normalize_label(response_text: str) -> tuple[str | None, bool]:
    text = response_text.strip().lower()

    text = text.replace(".", "")
    text = text.replace(",", "")
    text = text.replace("'", "")
    text = text.replace('"', "")

    for class_name in CLASS_NAMES:
        if text == class_name:
            return class_name, True

    return None, False


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input query file not found: {input_path}")

    records = load_jsonl(input_path, limit=args.limit)

    print("=" * 70)
    print("Local MLLM Verification")
    print("=" * 70)
    print(f"Model : {args.model}")
    print(f"Input : {input_path}")
    print(f"Output: {output_path}")
    print(f"Limit : {args.limit}")
    print("=" * 70)

    valid_count = 0
    invalid_count = 0

    with open(output_path, "w", encoding="utf-8") as out_file:
        for i, record in enumerate(records, start=1):
            image_path = Path(record["image_path"])
            model_label = record["model_label"]

            prompt = build_prompt(model_label=model_label)

            try:
                raw_response = call_ollama(
                    model=args.model,
                    prompt=prompt,
                    image_path=image_path,
                )
                verified_label, is_valid = normalize_label(raw_response)

            except Exception as exc:
                raw_response = str(exc)
                verified_label = None
                is_valid = False

            if is_valid:
                valid_count += 1
            else:
                invalid_count += 1

            output_record = {
                "sample_id": record["sample_id"],
                "target_index": record["target_index"],
                "image_path": record["image_path"],
                "model_label": model_label,
                "model_label_index": record["model_label_index"],
                "reliability_score": record["reliability_score"],
                "mllm_model": args.model,
                "raw_response": raw_response,
                "verified_label": verified_label,
                "is_valid": is_valid,
            }

            out_file.write(json.dumps(
                output_record, ensure_ascii=False) + "\n")

            print(
                f"[{i}/{len(records)}] "
                f"model={model_label} | "
                f"mllm={verified_label} | "
                f"valid={is_valid}"
            )

    print("=" * 70)
    print(f"Valid responses  : {valid_count}")
    print(f"Invalid responses: {invalid_count}")
    print("=" * 70)


if __name__ == "__main__":
    main()
