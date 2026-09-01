"""
Run inference and evaluate the trained GEC Transformer.

Current generation method:
- greedy decoding (argmax)
- autoregressive token-by-token generation

Metrics:
- BLEU
- SARI
"""

import argparse

import torch
import torch.nn as nn
from datasets import load_from_disk
from nltk.translate.bleu_score import corpus_bleu
from transformers import AutoTokenizer


class EncoderLayer(nn.Module):
    def __init__(self, d_model=256, num_heads=8):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            d_model,
            num_heads,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 1024),
            nn.ReLU(),
            nn.Linear(1024, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x, _ = self.attention(x, x, x)
        x = self.norm1(residual + x)

        residual = x
        x = self.feed_forward(x)
        x = self.norm2(residual + x)

        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model=256, num_heads=8):
        super().__init__()

        self.self_attention = nn.MultiheadAttention(
            d_model,
            num_heads,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)

        self.cross_attention = nn.MultiheadAttention(
            d_model,
            num_heads,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 1024),
            nn.ReLU(),
            nn.Linear(1024, d_model)
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output):
        seq_len = x.size(1)

        causal_mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                device=x.device,
                dtype=torch.bool
            ),
            diagonal=1
        )

        residual = x

        x, _ = self.self_attention(
            x, x, x,
            attn_mask=causal_mask
        )
        x = self.norm1(residual + x)

        residual = x

        x, _ = self.cross_attention(
            x,
            encoder_output,
            encoder_output
        )
        x = self.norm2(residual + x)

        residual = x

        x = self.feed_forward(x)
        x = self.norm3(residual + x)

        return x


class GECTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=256,
        num_layers=3,
        max_length=128,
        num_heads=8
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            d_model
        )

        self.positional_embedding = nn.Embedding(
            max_length,
            d_model
        )

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads)
            for _ in range(num_layers)
        ])

        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads)
            for _ in range(num_layers)
        ])

        self.output_layer = nn.Linear(
            d_model,
            vocab_size
        )

    def forward(self, source_ids, target_ids):
        batch_size, source_len = source_ids.shape
        _, target_len = target_ids.shape

        source_positions = torch.arange(
            source_len,
            device=source_ids.device
        ).unsqueeze(0).expand(batch_size, source_len)

        target_positions = torch.arange(
            target_len,
            device=target_ids.device
        ).unsqueeze(0).expand(batch_size, target_len)

        source = (
            self.embedding(source_ids)
            + self.positional_embedding(source_positions)
        )

        for layer in self.encoder_layers:
            source = layer(source)

        target = (
            self.embedding(target_ids)
            + self.positional_embedding(target_positions)
        )

        for layer in self.decoder_layers:
            target = layer(target, source)

        return self.output_layer(target)


def generate_text(
    text,
    model,
    tokenizer,
    device,
    max_length=128
):
    model.eval()

    source = tokenizer(
        text,
        return_tensors="pt",
        max_length=max_length,
        truncation=True
    )

    source_ids = source["input_ids"].to(device)

    decoder_ids = torch.tensor(
        [[tokenizer.bos_token_id]],
        device=device
    )

    with torch.no_grad():
        for _ in range(max_length - 1):
            logits = model(
                source_ids,
                decoder_ids
            )

            next_token = logits[:, -1, :].argmax(
                dim=-1,
                keepdim=True
            )

            decoder_ids = torch.cat(
                [decoder_ids, next_token],
                dim=1
            )

            if next_token.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(
        decoder_ids[0],
        skip_special_tokens=True
    )


def main(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "xlm-roberta-base"
    )

    model = GECTransformer(
        vocab_size=len(tokenizer),
        d_model=256
    ).to(device)

    model.load_state_dict(
        torch.load(
            args.model_path,
            map_location=device
        )
    )

    model.eval()

    print("Model loaded.")
    print("Device:", device)

    # -----------------------------
    # Interactive inference
    # -----------------------------
    if not args.evaluate:
        print("\nType a sentence to correct.")
        print("Type 'exit' to quit.\n")

        while True:
            text = input("You: ").strip()

            if text.lower() in {"exit", "quit", "çık"}:
                break

            if not text:
                continue

            result = generate_text(
                text,
                model,
                tokenizer,
                device
            )

            print("Model:", result)

        return

    # -----------------------------
    # BLEU + SARI evaluation
    # -----------------------------
    dataset = load_from_disk(args.data_dir)
    validation = dataset["validation"]

    sample_count = min(
        args.samples,
        len(validation)
    )

    references = []
    predictions = []
    sources = []

    for i in range(sample_count):
        sample = validation[i]

        source = sample["source"]
        target = sample["target"]

        prediction = generate_text(
            source,
            model,
            tokenizer,
            device
        )

        sources.append(source)
        references.append([target.split()])
        predictions.append(prediction.split())

        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{sample_count} evaluated")

    bleu = corpus_bleu(
        references,
        predictions
    )

    print("\nBLEU:", bleu)

    # SARI
    try:
        import evaluate

        sari = evaluate.load("sari")

        sari_result = sari.compute(
            sources=sources,
            predictions=[
                " ".join(p)
                for p in predictions
            ],
            references=[
                [" ".join(r[0])]
                for r in references
            ]
        )

        print("SARI:", sari_result)

    except Exception as e:
        print("\nSARI could not be calculated.")
        print("Install: pip install evaluate sacremoses sacrebleu")
        print("Error:", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        default="./data/gec_dataset"
    )

    parser.add_argument(
        "--model_path",
        default="./models/gec_transformer_final.pt"
    )

    parser.add_argument(
        "--evaluate",
        action="store_true"
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=5000
    )

    args = parser.parse_args()
    main(args)
