import json
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def load_training_data(data_path: str):
    with open(data_path, "r") as f:
        data = json.load(f)

    texts, labels, label_set = [], [], []

    # Collect all unique intents from the file (skip "unknown")
    seen = {}
    for item in data:
        intent = item["intent"]
        if intent == "unknown":
            continue
        if intent not in seen:
            seen[intent] = len(seen)

    label_names = list(seen.keys())
    print(f"Found intents: {label_names}")

    for item in data:
        intent = item["intent"]
        if intent == "unknown" or intent not in seen:
            continue
        texts.append(item["text"])
        labels.append(seen[intent])

    print(f"Loaded {len(texts)} training examples across {len(label_names)} intents")
    return texts, labels, label_names


def train_model(texts, labels, label_names, model_save_path: str):
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.20, random_state=42, stratify=labels
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=10000)),
        ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
    ])

    print("Training TF-IDF + LogisticRegression classifier...")
    pipeline.fit(X_train, y_train)
    pred_labels = pipeline.predict(X_val)

    Path(model_save_path).mkdir(parents=True, exist_ok=True)

    # Save model
    model_file = Path(model_save_path) / "sklearn_pipeline.pkl"
    with open(model_file, "wb") as f:
        pickle.dump(pipeline, f)

    # Save label mappings
    id2label = {i: name for i, name in enumerate(label_names)}
    label2id = {name: i for i, name in enumerate(label_names)}
    with open(Path(model_save_path) / "label_mappings.json", "w") as f:
        json.dump({
            "id2label": id2label,
            "label2id": label2id,
            "model_type": "sklearn",
            "intents": label_names
        }, f, indent=2)

    print(f"Model saved to {model_file}")
    return pred_labels, y_val, label_names


def evaluate_and_plot(pred_labels, true_labels, label_names, output_dir: str):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("INTENT CLASSIFICATION - EVALUATION REPORT")
    print("="*70)
    print(classification_report(true_labels, pred_labels, target_names=label_names, digits=3))

    accuracy = accuracy_score(true_labels, pred_labels)
    macro_f1 = f1_score(true_labels, pred_labels, average="macro")
    print(f"Overall Accuracy : {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Macro F1-Score   : {macro_f1:.4f}")
    print("="*70)

    # Confusion Matrix
    cm = confusion_matrix(true_labels, pred_labels)
    short_labels = [i.replace("_", "\n") for i in label_names]

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=short_labels, yticklabels=short_labels,
        ax=ax, linewidths=0.5
    )
    ax.set_title("Intent Classification — Confusion Matrix", fontsize=16, fontweight="bold", pad=20)
    ax.set_ylabel("True Intent", fontsize=12)
    ax.set_xlabel("Predicted Intent", fontsize=12)
    plt.tight_layout()
    cm_path = output_path / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")

    # F1 Bar Chart
    report = classification_report(
        true_labels, pred_labels,
        target_names=label_names, digits=3, output_dict=True
    )
    f1_scores = [report[intent]["f1-score"] for intent in label_names if intent in report]

    fig2, ax2 = plt.subplots(figsize=(12, 6))
    bars = ax2.bar(
        range(len(label_names)), f1_scores,
        color=plt.cm.viridis(np.linspace(0.2, 0.8, len(label_names)))
    )
    ax2.set_xticks(range(len(label_names)))
    ax2.set_xticklabels([i.replace("_", "\n") for i in label_names], fontsize=9)
    ax2.set_ylabel("F1-Score")
    ax2.set_title("F1-Score per Intent", fontsize=14, fontweight="bold")
    ax2.set_ylim(0, 1.15)
    ax2.axhline(y=macro_f1, color='red', linestyle='--', label=f'Macro F1: {macro_f1:.3f}')
    ax2.legend()
    for bar, score in zip(bars, f1_scores):
        ax2.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.02,
            f"{score:.2f}", ha='center', va='bottom', fontsize=8
        )
    plt.tight_layout()
    f1_path = output_path / "f1_scores.png"
    plt.savefig(f1_path, dpi=150, bbox_inches="tight")
    plt.close()
    pr