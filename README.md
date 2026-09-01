# Bilingual Turkish-English Grammar Error Correction Transformer

A custom Transformer encoder-decoder model for **Grammar Error Correction (GEC)** in Turkish and English.

The project was developed as a baseline GEC system using a bilingual training dataset and a Transformer architecture implemented directly in PyTorch.

## Project Structure

```text
GEC-Transformer/
├── 01_dataset_preparation.py
├── 02_model_training.py
├── 03_inference_and_benchmark.py
├── README.md
├── requirements.txt
├── data/
│   └── gec_dataset/
└── models/
    └── gec_transformer_final.pt
```

> The dataset and trained model files are not included in the repository because of their size.

## Dataset

The training data combines Turkish and English GEC datasets.

### Turkish

**GECTurk-generation**

- Hugging Face dataset: `mcemilg/GECTurk-generation`
- Contains source/target grammatical correction pairs.
- The dataset is labeled with `language = "tr"`.

### English

**C4-200M**

- Hugging Face dataset: `martinsr/c4_200m`
- The dataset is streamed rather than downloaded completely.
- The number of English examples is matched to the corresponding Turkish train/validation/test split sizes.
- English examples are normalized to the same format:
  - `source`
  - `target`
  - `language = "en"`

The final dataset is stored as a Hugging Face `DatasetDict` with:

```text
train
validation
test
```

Each split contains:

```text
source
target
language
```

## 1. Dataset Preparation

Run:

```bash
python 01_dataset_preparation.py --output_dir ./data/gec_dataset
```

This downloads the required datasets, combines Turkish and English examples, shuffles the splits with seed `42`, and saves the resulting `DatasetDict` to disk.

The original implementation used the same pipeline with Google Drive in Colab; the GitHub version makes the output directory configurable.

## 2. Model Training

Run:

```bash
python 02_model_training.py \
    --data_dir ./data/gec_dataset \
    --output_dir ./models \
    --batch_size 32 \
    --epochs 3 \
    --learning_rate 1e-4
```

### Model architecture

```text
XLM-R tokenizer
       │
       ▼
Token Embedding (d_model=256)
       │
       +
Positional Embedding
       │
       ▼
3 × Encoder Layer
       │
       ▼
Encoder Representation
       │
       ▼
3 × Decoder Layer
       │
       ├── Masked Self-Attention
       ├── Cross-Attention
       └── Feed-Forward Network
       │
       ▼
Linear(256 → vocabulary size)
       │
       ▼
Token logits
```

Configuration:

| Parameter | Value |
|---|---:|
| Tokenizer | `xlm-roberta-base` |
| Model dimension | 256 |
| Encoder layers | 3 |
| Decoder layers | 3 |
| Attention heads | 8 |
| Feed-forward | 256 → 1024 → 256 |
| Max sequence length | 128 |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Loss | CrossEntropyLoss |
| Padding ignored in loss | -100 |
| Decoding | Greedy |

The model uses **teacher forcing during training**: the decoder receives the correct previous target token while learning to generate the corrected sentence.

## 3. Inference and Benchmark

Interactive correction:

```bash
python 03_inference_and_benchmark.py \
    --model_path ./models/gec_transformer_final.pt
```

Then enter sentences:

```text
You: Ben bugün okula gitcem.
Model: ...
```

Type `exit` to quit.

### Evaluation

To evaluate on a validation subset:

```bash
python 03_inference_and_benchmark.py \
    --data_dir ./data/gec_dataset \
    --model_path ./models/gec_transformer_final.pt \
    --evaluate \
    --samples 5000
```

The evaluation calculates:

- BLEU
- SARI

The reported scores are validation-subset measurements and should not be interpreted as directly comparable to published GEC benchmark results unless the same evaluation protocol is used.

## Results

The baseline model was trained for 3 epochs.

Observed training:

```text
Epoch 1  Train Loss ≈ 2.995
Epoch 2  Train Loss ≈ 1.337
Epoch 3  Train Loss ≈ 1.039
```

Final validation loss:

```text
≈ 1.05
```

On a 5,000-example validation subset:

```text
BLEU ≈ 0.67
SARI ≈ 69.41
```

These results represent the current baseline implementation.

## Limitations

The model does not perform equally well on every error type.

During manual testing, several everyday Turkish error patterns were poorly represented in the training data. For example, informal forms such as:

```text
gitcem
gelicem
yapıyon
```

were absent or very rare in the examined dataset.

This suggests that **error-type coverage and data distribution** are important bottlenecks for the current baseline.

## Future Work

Possible improvements for a second version:

- Targeted synthetic error augmentation
- Better coverage of informal Turkish errors
- More balanced error-type distribution
- Improved decoding strategies such as beam search
- More systematic test sets by error type
- Comparison with pretrained multilingual GEC models
- Separate evaluation for Turkish and English
- Error-type-specific metrics

## Requirements

The project uses:

- Python 3
- PyTorch
- Hugging Face Datasets
- Hugging Face Transformers
- NLTK
- Evaluate
- SacreBLEU
- Sacremoses

See `requirements.txt`.

## License

MIT
