"""将 OutdoorClothingCatalog_1000.csv 批量翻译为中文。"""

import csv
import json
import time
from pathlib import Path

from deep_translator import GoogleTranslator

SRC = Path(__file__).with_name("OutdoorClothingCatalog_1000.csv")
DST = Path(__file__).with_name("OutdoorClothingCatalog_1000_zh.csv")
CHECKPOINT = Path(__file__).with_name(".translate_checkpoint.json")
BATCH_SIZE = 30
SLEEP_SECONDS = 0.5


def load_rows() -> list[dict[str, str]]:
    with SRC.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_checkpoint() -> dict[str, dict[str, str]]:
    if CHECKPOINT.exists():
        return json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    return {}


def save_checkpoint(data: dict[str, dict[str, str]]) -> None:
    CHECKPOINT.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_csv(rows: list[dict[str, str]]) -> None:
    with DST.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["", "name", "description"])
        writer.writeheader()
        writer.writerows(rows)


def translate_batch(
    translator: GoogleTranslator, batch: list[dict[str, str]]
) -> list[dict[str, str]]:
    names = translator.translate_batch([row["name"] for row in batch])
    descriptions = translator.translate_batch([row["description"] for row in batch])
    return [
        {
            "": row[""],
            "name": name,
            "description": description,
        }
        for row, name, description in zip(batch, names, descriptions, strict=True)
    ]


def main() -> None:
    rows = load_rows()
    done = load_checkpoint()
    translated_rows: list[dict[str, str] | None] = [None] * len(rows)

    for row in rows:
        idx = row[""]
        if idx in done:
            translated_rows[int(idx)] = done[idx]

    pending = [row for row in rows if row[""] not in done]
    print(f"共 {len(rows)} 条，已完成 {len(done)} 条，待翻译 {len(pending)} 条", flush=True)

    translator = GoogleTranslator(source="en", target="zh-CN")

    for start in range(0, len(pending), BATCH_SIZE):
        batch = pending[start : start + BATCH_SIZE]
        end = start + len(batch)
        print(f"翻译 {start + 1}-{end} / {len(pending)} ...", flush=True)
        batch_result = translate_batch(translator, batch)
        for item in batch_result:
            done[item[""]] = item
            translated_rows[int(item[""])] = item
        save_checkpoint(done)
        write_csv([row for row in translated_rows if row is not None])
        time.sleep(SLEEP_SECONDS)

    write_csv([row for row in translated_rows if row is not None])
    if CHECKPOINT.exists():
        CHECKPOINT.unlink()
    print(f"已保存: {DST}", flush=True)


if __name__ == "__main__":
    main()
