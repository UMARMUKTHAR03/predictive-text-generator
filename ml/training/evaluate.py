import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from ml.training.model import BiLSTMLanguageModel
from ml.preprocessing.tokenizer import TextTokenizer

def evaluate_model(
    artifacts_dir: str = "ml/artifacts",
    models_dir: str = "models/current",
    batch_size: int = 256
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    # Load split info and tokenizer
    with open(os.path.join(artifacts_dir, "split_info.json"), "r", encoding="utf-8") as f:
        split_info = json.load(f)

    vocab_size = split_info["total_words"]
    tokenizer = TextTokenizer.load_json(os.path.join(artifacts_dir, "tokenizer.json"))

    # Load test set
    x_test = np.load(os.path.join(artifacts_dir, "x_test.npy"))
    y_test = np.load(os.path.join(artifacts_dir, "y_test.npy"))

    test_dataset = TensorDataset(torch.from_numpy(x_test).long(), torch.from_numpy(y_test).long())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = BiLSTMLanguageModel(vocab_size=vocab_size, embedding_dim=100, hidden_dim=150).to(device)
    model_path = os.path.join(artifacts_dir, "best_model.pt")
    if not os.path.exists(model_path):
        model_path = os.path.join(artifacts_dir, "model.pt")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    top10_correct = 0
    total_samples = 0

    latencies = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            t0 = time.perf_counter()
            logits = model(batch_x)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) / len(batch_x))

            loss = criterion(logits, batch_y)
            total_loss += loss.item() * len(batch_y)

            # Top-K accuracies
            _, top10_preds = logits.topk(10, dim=-1)
            targets = batch_y.view(-1, 1)

            top1_correct += (top10_preds[:, :1] == targets).sum().item()
            top3_correct += (top10_preds[:, :3] == targets).sum().item()
            top5_correct += (top10_preds[:, :5] == targets).sum().item()
            top10_correct += (top10_preds[:, :10] == targets).sum().item()
            total_samples += len(batch_y)

    test_loss = total_loss / total_samples
    perplexity = float(np.exp(min(test_loss, 20.0)))
    top1_acc = top1_correct / total_samples
    top3_acc = top3_correct / total_samples
    top5_acc = top5_correct / total_samples
    top10_acc = top10_correct / total_samples
    avg_latency_ms = float(np.mean(latencies) * 1000)

    # Historical notebook metrics for comparison
    historical_metrics = {
        "dataset_source": "medium_data.csv (6,508 rows, Medium article titles)",
        "notebook_epoch_50_train_loss": 2.0090,
        "notebook_epoch_50_train_accuracy": 0.5460,
        "notebook_validation_split": "None (trained on full dataset without holdout split)",
        "notebook_leakage_note": "No test/val split was performed in the original Final.ipynb notebook. Evaluation on train set alone overstates generalization."
    }

    metrics = {
        "test_loss": round(test_loss, 4),
        "test_perplexity": round(perplexity, 2),
        "top1_accuracy": round(top1_acc, 4),
        "top3_accuracy": round(top3_acc, 4),
        "top5_accuracy": round(top5_acc, 4),
        "top10_accuracy": round(top10_acc, 4),
        "test_samples": total_samples,
        "avg_inference_latency_ms": round(avg_latency_ms, 3),
        "parameters": model.get_config(),
        "split_summary": split_info,
        "historical_comparison": historical_metrics
    }

    # Save to artifacts & production
    metrics_path = os.path.join(artifacts_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(models_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Copy tokenizer and model_metadata to models/current
    import shutil
    shutil.copy(os.path.join(artifacts_dir, "tokenizer.json"), os.path.join(models_dir, "tokenizer.json"))

    model_metadata = {
        "name": "NeuroText AI BiLSTM Language Model",
        "version": "1.0.0",
        "framework": "PyTorch",
        "architecture": "Bidirectional LSTM (Embedding 100 -> BiLSTM 150 -> Linear Head)",
        "vocab_size": vocab_size,
        "embedding_dim": 100,
        "hidden_dim": 150,
        "max_sequence_len": split_info["max_sequence_len"],
        "metrics": {
            "top1_accuracy": round(top1_acc, 4),
            "top3_accuracy": round(top3_acc, 4),
            "top5_accuracy": round(top5_acc, 4),
            "top10_accuracy": round(top10_acc, 4),
            "test_loss": round(test_loss, 4),
            "perplexity": round(perplexity, 2)
        },
        "dataset": {
            "name": "Medium Articles Dataset",
            "domain": "Data science, machine learning, and tech headline titles",
            "total_articles": split_info["total_titles"],
            "split": "80% Train, 10% Val, 10% Test (Title-level leak-free split)"
        }
    }
    with open(os.path.join(models_dir, "model_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(model_metadata, f, indent=2)

    print("\n--- Model Evaluation Complete ---")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Perplexity: {perplexity:.2f}")
    print(f"Top-1 Accuracy: {top1_acc*100:.2f}%")
    print(f"Top-3 Accuracy: {top3_acc*100:.2f}%")
    print(f"Top-5 Accuracy: {top5_acc*100:.2f}%")
    print(f"Top-10 Accuracy: {top10_acc*100:.2f}%")
    print(f"Latency per token: {avg_latency_ms:.3f} ms")
    return metrics

if __name__ == "__main__":
    evaluate_model()

