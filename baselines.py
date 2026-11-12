import os
import json
import time
import numpy as np
import pandas as pd
from PIL import Image

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

SPLIT_DIR = 'splits'
CLASS_IDX_FILE = 'class_indices.json'
RESULTS_FILE = 'baseline_results.json'
SEED = 42


def extract_features(path: str) -> np.ndarray:
    img = Image.open(path).convert('RGB')
    small = np.asarray(img.resize((32, 32)), dtype=np.float32).ravel() / 255.0
    mid = np.asarray(img.resize((64, 64)), dtype=np.float32) / 255.0
    hist = np.concatenate([
        np.histogram(mid[:, :, c], bins=32, range=(0, 1))[0] / mid[:, :, c].size
        for c in range(3)
    ])
    return np.concatenate([small, hist])


def load_split(name: str, class_to_idx: dict):
    df = pd.read_csv(os.path.join(SPLIT_DIR, f'{name}.csv'))
    feats, labels = [], []
    skipped = 0
    for path, label in zip(df['image_path'], df['pest_type']):
        try:
            feats.append(extract_features(path))
            labels.append(class_to_idx[label])
        except Exception:
            skipped += 1
    if skipped:
        print(f"[Warning] {skipped} unreadable images skipped in {name}")
    print(f"{name:<5}: {len(labels)} samples")
    return np.array(feats), np.array(labels)


def score(model, X, y, label_ids, class_names, show_report=False):
    pred = model.predict(X)
    acc = accuracy_score(y, pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y, pred, labels=label_ids, average='weighted', zero_division=0
    )
    if show_report:
        print(classification_report(y, pred, labels=label_ids,
                                    target_names=class_names, zero_division=0))
    return {
        'accuracy': float(acc),
        'weighted_precision': float(p),
        'weighted_recall': float(r),
        'weighted_f1': float(f1),
    }


def main():
    with open(CLASS_IDX_FILE, 'r') as f:
        class_to_idx = json.load(f)
    class_names = [c for c, _ in sorted(class_to_idx.items(), key=lambda kv: kv[1])]
    label_ids = list(range(len(class_names)))

    X_train, y_train = load_split('train', class_to_idx)
    X_val, y_val = load_split('val', class_to_idx)
    X_test, y_test = load_split('test', class_to_idx)

    models = {
        'SVM (RBF)': make_pipeline(
            StandardScaler(),
            PCA(n_components=100, random_state=SEED),
            SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced',
                random_state=SEED)
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=300, class_weight='balanced',
            n_jobs=-1, random_state=SEED
        ),
    }

    results = {}
    for name, model in models.items():
        print(f"\n=== {name} ===")
        start = time.time()
        model.fit(X_train, y_train)
        print(f"Training time: {time.time() - start:.1f}s")

        val_scores = score(model, X_val, y_val, label_ids, class_names)
        print(f"Validation accuracy: {val_scores['accuracy'] * 100:.2f}%")

        print("\nTest classification report:")
        test_scores = score(model, X_test, y_test, label_ids, class_names, show_report=True)
        print(f"Test accuracy: {test_scores['accuracy'] * 100:.2f}%")

        results[name] = {'validation': val_scores, 'test': test_scores}

    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {RESULTS_FILE}")

    print("\n=== Summary (test set) ===")
    for name, res in results.items():
        t = res['test']
        print(f"{name:<15} acc={t['accuracy'] * 100:.2f}%  "
              f"P={t['weighted_precision']:.3f}  R={t['weighted_recall']:.3f}  "
              f"F1={t['weighted_f1']:.3f}")


if __name__ == '__main__':
    main()
