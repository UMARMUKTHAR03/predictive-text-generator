import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from ml.training.model import BiLSTMLanguageModel

def train_model(
    artifacts_dir: str = "ml/artifacts",
    models_dir: str = "models/current",
    epochs: int = 20,
    batch_size: int = 256,
    learning_rate: float = 0.005,
    seed: int = 42
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Load split info
    with open(os.path.join(artifacts_dir, "split_info.json"), "r", encoding="utf-8") as f:
        split_info = json.load(f)

    vocab_size = split_info["total_words"]
    max_sequence_len = split_info["max_sequence_len"]

    # Load data
    x_train = np.load(os.path.join(artifacts_dir, "x_train.npy"))
    y_train = np.load(os.path.join(artifacts_dir, "y_train.npy"))
    x_val = np.load(os.path.join(artifacts_dir, "x_val.npy"))
    y_val = np.load(os.path.join(artifacts_dir, "y_val.npy"))

    train_dataset = TensorDataset(torch.from_numpy(x_train).long(), torch.from_numpy(y_train).long())
    val_dataset = TensorDataset(torch.from_numpy(x_val).long(), torch.from_numpy(y_val).long())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = BiLSTMLanguageModel(vocab_size=vocab_size, embedding_dim=100, hidden_dim=150).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "epoch_times": []
    }

    best_val_loss = float("inf")
    best_weights_path = os.path.join(artifacts_dir, "best_model.pt")

    print(f"Starting training for {epochs} epochs (vocab: {vocab_size}, train: {len(x_train)}, val: {len(x_val)})...")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * len(batch_y)
            preds = logits.argmax(dim=-1)
            train_correct += (preds == batch_y).sum().item()
            train_total += len(batch_y)

        train_loss = total_train_loss / train_total
        train_acc = train_correct / train_total

        # Validation
        model.eval()
        total_val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                total_val_loss += loss.item() * len(batch_y)
                preds = logits.argmax(dim=-1)
                val_correct += (preds == batch_y).sum().item()
                val_total += len(batch_y)

        val_loss = total_val_loss / val_total
        val_acc = val_correct / val_total
        epoch_time = time.time() - t0

        scheduler.step(val_loss)

        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 4))
        history["epoch_times"].append(round(epoch_time, 2))

        print(f"Epoch {epoch:02d}/{epochs} [{epoch_time:.1f}s] - "
              f"train_loss: {train_loss:.4f} - train_acc: {train_acc*100:.2f}% | "
              f"val_loss: {val_loss:.4f} - val_acc: {val_acc*100:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_weights_path)

    # Save final model state
    final_model_path = os.path.join(artifacts_dir, "model.pt")
    torch.save(model.state_dict(), final_model_path)
    # Also save to production models directory
    os.makedirs(models_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(models_dir, "model.pt"))

    # Save training history
    with open(os.path.join(artifacts_dir, "training_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining completed! Best val_loss: {best_val_loss:.4f}")
    return history

if __name__ == "__main__":
    train_model(epochs=15, batch_size=256, learning_rate=0.005)

