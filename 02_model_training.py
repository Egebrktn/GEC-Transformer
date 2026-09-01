"""
Train the custom Transformer encoder-decoder for bilingual GEC.

Architecture:
- XLM-R tokenizer
- d_model = 256
- 3 encoder layers
- 3 decoder layers
- 8 attention heads
- FFN: 256 -> 1024 -> 256
- AdamW
- CrossEntropyLoss(ignore_index=-100)
- Teacher forcing
"""

import argparse
import os

import torch
import torch.nn as nn
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForSeq2Seq


class EncoderLayer(nn.Module):
    def __init__(self, d_model=256, num_heads=8):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
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
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(d_model)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
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


def shift_right(labels, bos_token_id, pad_token_id):
    decoder_input = torch.full_like(
        labels,
        pad_token_id
    )

    decoder_input[:, 0] = bos_token_id
    decoder_input[:, 1:] = labels[:, :-1]

    decoder_input[decoder_input == -100] = pad_token_id

    return decoder_input


def tokenize_dataset(dataset, tokenizer):
    def tokenize_function(batch):
        inputs = tokenizer(
            batch["source"],
            max_length=128,
            truncation=True
        )

        targets = tokenizer(
            batch["target"],
            max_length=128,
            truncation=True
        )

        inputs["labels"] = targets["input_ids"]

        return inputs

    return dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )


def train(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    dataset = load_from_disk(args.data_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        "xlm-roberta-base"
    )

    tokenized_train = tokenize_dataset(
        dataset["train"],
        tokenizer
    )

    tokenized_val = tokenize_dataset(
        dataset["validation"],
        tokenizer
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=None,
        padding=True
    )

    train_loader = DataLoader(
        tokenized_train,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=data_collator
    )

    val_loader = DataLoader(
        tokenized_val,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=data_collator
    )

    model = GECTransformer(
        vocab_size=len(tokenizer),
        d_model=256
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate
    )

    criterion = nn.CrossEntropyLoss(
        ignore_index=-100
    )

    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            source_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            decoder_ids = shift_right(
                labels,
                tokenizer.bos_token_id,
                tokenizer.pad_token_id
            )

            optimizer.zero_grad()

            logits = model(
                source_ids,
                decoder_ids
            )

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1)
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (batch_idx + 1) % 100 == 0:
                print(
                    f"Epoch {epoch + 1} | "
                    f"Batch {batch_idx + 1}/{len(train_loader)} | "
                    f"Loss {loss.item():.4f}"
                )

        avg_train_loss = total_loss / len(train_loader)

        # -----------------------------
        # Validation loss
        # -----------------------------
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                source_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                decoder_ids = shift_right(
                    labels,
                    tokenizer.bos_token_id,
                    tokenizer.pad_token_id
                )

                logits = model(
                    source_ids,
                    decoder_ids
                )

                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1)
                )

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(
            f"\nEpoch {epoch + 1} finished"
            f"\nTrain Loss: {avg_train_loss:.4f}"
            f"\nValidation Loss: {avg_val_loss:.4f}\n"
        )

    model_path = os.path.join(
        args.output_dir,
        "gec_transformer_final.pt"
    )

    torch.save(
        model.state_dict(),
        model_path
    )

    print("Model saved to:", model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        default="./data/gec_dataset"
    )

    parser.add_argument(
        "--output_dir",
        default="./models"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4
    )

    args = parser.parse_args()
    train(args)
