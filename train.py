import os

import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset

from model import Lumiere


CONFIG = {
    "vocab_size": 50257,
    "context_len": 128,
    "embedding_dim": 128,
    "n_head": 4,
    "n_layers": 8,
    "drop_rate": 0.1,
    "qkv_bias": False,
}

N_EPOCHS = 100
BATCH_SIZE = 4
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.1
CHECKPOINT_PATH = "./model_checkpoints/lumiere.checkpoint.pt"
HISTORY_PATH = "./logs/training_history.pt"


class TextDataset(Dataset):

    def __init__(self, enc_tokens, context_len):
        self.context_len = context_len
        self.total_len = len(enc_tokens)
        self.enc_tokens = enc_tokens

        if self.total_len <= self.context_len:
            raise ValueError(
                f"Dataset needs more than {self.context_len} tokens, got {self.total_len}."
            )

    def __len__(self):
        return self.total_len - self.context_len

    def __getitem__(self, idx):
        x = self.enc_tokens[idx : idx + self.context_len]
        y = self.enc_tokens[idx + 1 : idx + self.context_len + 1]

        return x, y


@torch.no_grad()
def model_evaluation(model, dataloader, loss_fn, device):
    was_training = model.training
    model.eval()

    total_loss = 0.0

    for xx, yy in dataloader:
        xx = xx.to(device)
        yy = yy.to(device)

        logits = model(xx)
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), yy.reshape(-1))

        total_loss += loss.item()

    if was_training:
        model.train()

    if len(dataloader) == 0:
        raise ValueError("Cannot evaluate with an empty dataloader.")

    return total_loss / len(dataloader)


def load_checkpoint(model, optimizer, checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        print("No checkpoint found.")
        return 0, 0

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    epoch = checkpoint["epoch"]
    global_step = checkpoint["global_step"]

    print(f"Loaded checkpoint {checkpoint_path}")

    return epoch, global_step


def load_history(history_path):
    if not os.path.exists(history_path):
        return empty_history()

    return torch.load(history_path)


def empty_history():
    return {
        "train_steps": [],
        "train_losses": [],
        "validation_steps": [],
        "validation_losses": [],
    }


def record_losses(history, step, train_loss, validation_loss):
    losses_by_step = {
        step_id: (train_loss_value, validation_loss_value)
        for step_id, train_loss_value, validation_loss_value in zip(
            history["train_steps"],
            history["train_losses"],
            history["validation_losses"],
        )
    }
    losses_by_step[step] = (train_loss, validation_loss)

    sorted_steps = sorted(losses_by_step)
    history["train_steps"] = sorted_steps
    history["train_losses"] = [losses_by_step[step_id][0] for step_id in sorted_steps]
    history["validation_steps"] = sorted_steps
    history["validation_losses"] = [
        losses_by_step[step_id][1] for step_id in sorted_steps
    ]


def save_checkpoint(model, optimizer, epoch, global_step, checkpoint_path):
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        checkpoint_path,
    )


def load_encoded_text(path):
    tokenizer = tiktoken.get_encoding("gpt2")

    with open(path, "r") as file:
        file_content = file.read()

    return torch.tensor(tokenizer.encode(file_content))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    text_encoded = load_encoded_text("./the_verdict.txt")
    total_tokens = len(text_encoded)
    print(total_tokens)

    split_idx = int(total_tokens * 0.9)
    train_text_encoded = text_encoded[:split_idx]
    test_text_encoded = text_encoded[split_idx:]

    train_dataset = TextDataset(train_text_encoded, context_len=CONFIG["context_len"])
    test_dataset = TextDataset(test_text_encoded, context_len=CONFIG["context_len"])

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = Lumiere(config=CONFIG).to(device=device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters in Lumiere model = {num_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    epoch, global_step = load_checkpoint(model, optimizer, CHECKPOINT_PATH, device)
    history = load_history(HISTORY_PATH)

    for _ in range(N_EPOCHS):
        epoch += 1
        model.train()

        for _, (x, y) in enumerate(train_dataloader):
            global_step += 1

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits = model(x)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

            loss.backward()
            optimizer.step()

            if global_step % 10 == 0:
                print(f"Epoch: {epoch} Step: {global_step}, Training Loss: {loss.item()}")

            if global_step % 100 == 0:
                test_loss = model_evaluation(model, test_dataloader, loss_fn, device)

                print(f"Step={global_step}, test_loss={test_loss}, train_loss={loss.item()}")

                record_losses(history, global_step, loss.item(), test_loss)

                os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
                torch.save(history, HISTORY_PATH)

            if global_step % 1000 == 0:
                save_checkpoint(model, optimizer, epoch, global_step, CHECKPOINT_PATH)


if __name__ == "__main__":
    main()
