import os
import json
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
from ml.preprocessing.tokenizer import TextTokenizer, clean_text

def prepare_splits_and_sequences(
    data_csv_path: str,
    output_dir: str,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    random_seed: int = 42,
    max_sequence_len: int = 40
) -> Dict[str, any]:
    """
    Split the dataset titles into train, validation, and test splits BEFORE n-gram extraction.
    This guarantees 0 data leakage between splits.
    """
    df = pd.read_csv(data_csv_path)
    # Extract non-null cleaned titles
    titles = df["title"].dropna().astype(str).map(clean_text).tolist()
    titles = [t for t in titles if len(t.strip()) > 0]
    total_titles = len(titles)

    # Deterministic train/val/test split by title
    np.random.seed(random_seed)
    indices = np.random.permutation(total_titles)

    train_end = int(total_titles * train_ratio)
    val_end = int(total_titles * (train_ratio + val_ratio))

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    train_titles = [titles[i] for i in train_indices]
    val_titles = [titles[i] for i in val_indices]
    test_titles = [titles[i] for i in test_indices]

    print(f"Total titles: {total_titles}")
    print(f"Train titles: {len(train_titles)} ({len(train_titles)/total_titles*100:.1f}%)")
    print(f"Val titles: {len(val_titles)} ({len(val_titles)/total_titles*100:.1f}%)")
    print(f"Test titles: {len(test_titles)} ({len(test_titles)/total_titles*100:.1f}%)")

    # Fit tokenizer ONLY on the train_titles
    tokenizer = TextTokenizer(oov_token="<oov>")
    tokenizer.fit_on_texts(train_titles)

    os.makedirs(output_dir, exist_ok=True)
    tokenizer_path = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save_json(tokenizer_path)
    print(f"Vocabulary size (train-fitted): {tokenizer.total_words}")
    print(f"Saved tokenizer to {tokenizer_path}")

    # Generate n-grams for a split
    def build_ngrams(split_titles: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        sequences = []
        for line in split_titles:
            token_list = tokenizer.texts_to_sequences([line])[0]
            for i in range(1, len(token_list)):
                n_gram = token_list[:i+1]
                sequences.append(n_gram)

        # Pad sequences (pre-padding)
        padded = np.zeros((len(sequences), max_sequence_len), dtype=np.int32)
        for i, seq in enumerate(sequences):
            if len(seq) > max_sequence_len:
                seq = seq[-max_sequence_len:]
            padded[i, -len(seq):] = seq

        xs = padded[:, :-1]
        ys = padded[:, -1]
        return xs, ys

    x_train, y_train = build_ngrams(train_titles)
    x_val, y_val = build_ngrams(val_titles)
    x_test, y_test = build_ngrams(test_titles)

    print(f"N-gram sequences generated:")
    print(f" - Train: {x_train.shape[0]} sequences")
    print(f" - Val:   {x_val.shape[0]} sequences")
    print(f" - Test:  {x_test.shape[0]} sequences")

    # Save arrays
    np.save(os.path.join(output_dir, "x_train.npy"), x_train)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "x_val.npy"), x_val)
    np.save(os.path.join(output_dir, "y_val.npy"), y_val)
    np.save(os.path.join(output_dir, "x_test.npy"), x_test)
    np.save(os.path.join(output_dir, "y_test.npy"), y_test)

    # Save split metadata
    split_info = {
        "total_titles": total_titles,
        "train_titles_count": len(train_titles),
        "val_titles_count": len(val_titles),
        "test_titles_count": len(test_titles),
        "train_sequences_count": int(x_train.shape[0]),
        "val_sequences_count": int(x_val.shape[0]),
        "test_sequences_count": int(x_test.shape[0]),
        "max_sequence_len": max_sequence_len,
        "total_words": tokenizer.total_words,
        "vocab_size": len(tokenizer.word_index),
        "random_seed": random_seed
    }
    with open(os.path.join(output_dir, "split_info.json"), "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)

    return split_info

if __name__ == "__main__":
    csv_path = os.path.join("ml", "data", "medium_data.csv")
    out_dir = os.path.join("ml", "artifacts")
    prepare_splits_and_sequences(csv_path, out_dir)

