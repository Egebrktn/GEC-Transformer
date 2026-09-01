"""
Prepare the bilingual Turkish + English GEC dataset.

Sources:
- Turkish: mcemilg/GECTurk-generation
- English: martinsr/c4_200m

The resulting DatasetDict contains:
    train / validation / test
with columns:
    source / target / language
"""

import argparse
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets


def prepare_dataset(output_dir: str, seed: int = 42):
    # -----------------------------
    # Load Turkish GECTurk
    # -----------------------------
    tr_train = load_dataset(
        "mcemilg/GECTurk-generation",
        split="train"
    )

    tr_val = load_dataset(
        "mcemilg/GECTurk-generation",
        split="validation"
    )

    tr_test = load_dataset(
        "mcemilg/GECTurk-generation",
        split="test"
    )

    # -----------------------------
    # Load English C4-200M
    # Streaming avoids downloading
    # the complete dataset.
    # -----------------------------
    c4 = load_dataset(
        "martinsr/c4_200m",
        split="train",
        streaming=True
    )

    # Match the number of English
    # examples to the Turkish splits.
    n_train = len(tr_train)
    n_val = len(tr_val)
    n_test = len(tr_test)
    total = n_train + n_val + n_test

    rows = []

    for i, row in enumerate(c4):
        if i >= total:
            break

        rows.append({
            "source": row["input"],
            "target": row["output"],
            "language": "en"
        })

    c4_data = Dataset.from_list(rows)

    en_train = c4_data.select(range(n_train))

    en_val = c4_data.select(
        range(n_train, n_train + n_val)
    )

    en_test = c4_data.select(
        range(n_train + n_val, total)
    )

    # Add language labels to Turkish data.
    tr_train = tr_train.map(lambda x: {"language": "tr"})
    tr_val = tr_val.map(lambda x: {"language": "tr"})
    tr_test = tr_test.map(lambda x: {"language": "tr"})

    # -----------------------------
    # Combine Turkish + English
    # -----------------------------
    train = concatenate_datasets([
        tr_train,
        en_train
    ]).shuffle(seed=seed)

    val = concatenate_datasets([
        tr_val,
        en_val
    ]).shuffle(seed=seed)

    test = concatenate_datasets([
        tr_test,
        en_test
    ]).shuffle(seed=seed)

    print("Train:", len(train))
    print("Validation:", len(val))
    print("Test:", len(test))

    dataset = DatasetDict({
        "train": train,
        "validation": val,
        "test": test
    })

    dataset.save_to_disk(output_dir)

    print(f"\nDataset saved to: {output_dir}")
    print(dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        default="./data/gec_dataset",
        help="Directory where the DatasetDict will be saved."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()
    prepare_dataset(args.output_dir, args.seed)
