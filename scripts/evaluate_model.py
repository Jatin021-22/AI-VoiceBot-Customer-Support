"""Evaluation script — generates metrics and confusion matrix."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def run_evaluation():
    from app.models.intent_classifier import IntentClassifier, INTENT_LABELS
    from app.services.asr_service import ASRService

    print("\n" + "="*60)
    print("  VoiceBot Model Evaluation Report")
    print("="*60)

    with open("data/intents/training_data.json") as f:
        all_data = json.load(f)

    split = int(len(all_data) * 0.8)
    train_data, test_data = all_data[:split], all_data[split:]
    print(f"\nTrain: {len(train_data)} | Test: {len(test_data)}")

    clf = IntentClassifier()
    print("\nTraining...")
    result = clf.train(train_data)
    print(f"Backend: {result.get('backend')}")

    print("\nEvaluating...")
    metrics = clf.evaluate(test_data)
    print(f"\n  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1_score']:.4f}")

    report = metrics.get("classification_report", {})
    print(f"\n{'Intent':<30} {'P':>6} {'R':>6} {'F1':>6} {'N':>6}")
    print("-"*55)
    for intent in INTENT_LABELS:
        if intent in report:
            r = report[intent]
            print(f"  {intent:<28} {r['precision']:>5.3f} {r['recall']:>5.3f} {r['f1-score']:>5.3f} {int(r['support']):>5}")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt, numpy as np
        cm = np.array(metrics["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(14, 12))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(INTENT_LABELS))); ax.set_yticks(range(len(INTENT_LABELS)))
        ax.set_xticklabels(INTENT_LABELS, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(INTENT_LABELS, fontsize=8)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title("Intent Classification — Confusion Matrix", fontsize=13, fontweight="bold")
        plt.colorbar(im)
        for i in range(len(INTENT_LABELS)):
            for j in range(len(INTENT_LABELS)):
                ax.text(j, i, cm[i][j], ha="center", va="center",
                        color="white" if cm[i][j] > cm.max()/2 else "black", fontsize=8)
        plt.tight_layout()
        os.makedirs("outputs", exist_ok=True)
        plt.savefig("outputs/confusion_matrix.png", dpi=150, bbox_inches="tight")
        print("\n✓ Confusion matrix → outputs/confusion_matrix.png")
    except ImportError:
        print("\n[matplotlib not installed]")

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/evaluation_metrics.json", "w") as f:
        json.dump({k: v for k, v in metrics.items() if k != "classification_report"}, f, indent=2)
    print("✓ Metrics → outputs/evaluation_metrics.json")

if __name__ == "__main__":
    run_evaluation()
