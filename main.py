import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
import keras
from keras import layers
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

DATA_CSV = r'pests_labels.csv'
IMAGE_ROOT = r"C:\Users\HP\Documents\data mining\project\pest-identification system"
MODEL_SAVE = 'pest_model.keras'
CLASS_IDX_FILE = 'class_indices.json'
SPLIT_DIR = 'splits'
METRICS_FILE = 'test_metrics.json'
TEST_IMAGE = 'images/test_aphid.jpg'

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_HEAD = 10
EPOCHS_FINE = 10
UNFREEZE_LAST = 30
TEST_FRACTION = 0.15
VAL_FRACTION = 0.15
CONFIDENCE_THRESHOLD = 0.60
SEED = 42

AUTOTUNE = tf.data.AUTOTUNE

augmenter = keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(20 / 360),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
    layers.RandomBrightness(0.2, value_range=(0, 255)),
])


def resolve_path(path: str) -> str:
    if not IMAGE_ROOT:
        return path
    parts = [p for p in path.replace('\\', '/').split('/') if p]
    return os.path.join(IMAGE_ROOT, *parts[-2:])


def load_data(csv_path: str) -> pd.DataFrame:
    data = pd.read_csv(csv_path)
    if 'pest_type' not in data.columns and 'label' in data.columns:
        data = data.rename(columns={'label': 'pest_type'})

    required = {'image_path', 'pest_type'}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(
            f"CSV is missing columns: {missing}\n"
            f"Found columns: {list(data.columns)}"
        )

    data['image_path'] = data['image_path'].apply(resolve_path)

    exists = data['image_path'].apply(os.path.exists)
    dropped = int((~exists).sum())
    if dropped:
        print(f"[Warning] {dropped} rows dropped because the image file was not found")
    if not exists.any():
        raise FileNotFoundError(
            f"No images found. Check IMAGE_ROOT ('{IMAGE_ROOT}') points to the folder "
            f"that contains the class sub-folders."
        )
    data = data[exists].reset_index(drop=True)

    print(f"Dataset loaded  : {len(data)} images across "
          f"{data['pest_type'].nunique()} classes")
    print(f"Classes         : {sorted(data['pest_type'].unique())}")
    print("\nClass distribution:")
    print(data['pest_type'].value_counts().to_string())
    return data


def split_data(data: pd.DataFrame):
    train_val_df, test_df = train_test_split(
        data,
        test_size=TEST_FRACTION,
        stratify=data['pest_type'],
        random_state=SEED
    )
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=VAL_FRACTION / (1 - TEST_FRACTION),
        stratify=train_val_df['pest_type'],
        random_state=SEED
    )

    os.makedirs(SPLIT_DIR, exist_ok=True)
    train_df.to_csv(os.path.join(SPLIT_DIR, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(SPLIT_DIR, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(SPLIT_DIR, 'test.csv'), index=False)

    print(f"\nTrain: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return train_df, val_df, test_df


def load_image(path, label):
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, IMG_SIZE)
    return tf.cast(img, tf.float32), label


def make_dataset(df: pd.DataFrame, class_to_idx: dict, training: bool):
    paths = df['image_path'].values
    labels = df['pest_type'].map(class_to_idx).values.astype('int32')

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(df), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(load_image, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE)
    if training:
        ds = ds.map(lambda x, y: (augmenter(x, training=True), y),
                    num_parallel_calls=AUTOTUNE)
    ds = ds.map(lambda x, y: (preprocess_input(x), y),
                num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


def build_model(num_classes: int):
    base = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(*IMG_SIZE, 3)
    )
    base.trainable = False

    inputs = Input(shape=(*IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)

    print(f"\nModel built — {num_classes} output classes")
    return model, base


def unfreeze_for_finetuning(base_model):
    base_model.trainable = True
    for layer in base_model.layers[:-UNFREEZE_LAST]:
        layer.trainable = False
    for layer in base_model.layers:
        if isinstance(layer, BatchNormalization):
            layer.trainable = False


def count_lr_drops(history: dict) -> int:
    lrs = history.get('learning_rate', history.get('lr', []))
    return int(sum(1 for a, b in zip(lrs, lrs[1:]) if b < a * 0.999))


def run_phase(model, train_ds, val_ds, epochs, lr, tag, class_weight):
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    ckpt = f'best_{tag}.weights.h5'
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=5, verbose=1),
        ModelCheckpoint(ckpt, monitor='val_accuracy', save_best_only=True,
                        save_weights_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3,
                          min_lr=1e-7, verbose=1)
    ]
    hist = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weight,
        callbacks=callbacks
    )
    model.load_weights(ckpt)
    return hist.history


def train_model(model, base_model, train_ds, val_ds, class_weight):
    print("\n=== Phase 1: Training head (base frozen) ===")
    h1 = run_phase(model, train_ds, val_ds, EPOCHS_HEAD, 1e-3, 'head', class_weight)

    print(f"\n=== Phase 2: Fine-tuning last {UNFREEZE_LAST} layers ===")
    unfreeze_for_finetuning(base_model)
    h2 = run_phase(model, train_ds, val_ds, EPOCHS_FINE, 1e-5, 'finetune', class_weight)

    print(f"\nLR reductions — phase 1: {count_lr_drops(h1)} | phase 2: {count_lr_drops(h2)}")

    keys = ['accuracy', 'val_accuracy', 'loss', 'val_loss']
    combined = {k: h1[k] + h2[k] for k in keys}
    boundary = len(h1['accuracy'])
    return combined, boundary


def plot_history(history: dict, boundary: int):
    epochs = range(1, len(history['accuracy']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history['accuracy'], label='Train')
    ax1.plot(epochs, history['val_accuracy'], label='Validation')
    ax1.axvline(boundary, linestyle='--', color='gray', label='Fine-tune start')
    ax1.set_title('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend()

    ax2.plot(epochs, history['loss'], label='Train')
    ax2.plot(epochs, history['val_loss'], label='Validation')
    ax2.axvline(boundary, linestyle='--', color='gray', label='Fine-tune start')
    ax2.set_title('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()

    plt.suptitle('MobileNetV2 — Pest Classification', fontsize=13)
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150)
    plt.show()
    print("Training curves saved → training_curves.png")


def evaluate_model(model, test_ds, test_df, class_names, class_to_idx):
    print("\n=== Final evaluation on held-out TEST set ===")
    loss, acc = model.evaluate(test_ds, verbose=0)
    print(f"Test Loss    : {loss:.4f}")
    print(f"Test Accuracy: {acc * 100:.2f}%")

    preds = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(preds, axis=1)
    y_true = test_df['pest_type'].map(class_to_idx).values
    label_ids = list(range(len(class_names)))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=label_ids,
                                target_names=class_names, zero_division=0))

    report = classification_report(y_true, y_pred, labels=label_ids,
                                   target_names=class_names, zero_division=0,
                                   output_dict=True)
    metrics = {
        'test_loss': float(loss),
        'test_accuracy': float(acc),
        'weighted_precision': report['weighted avg']['precision'],
        'weighted_recall': report['weighted avg']['recall'],
        'weighted_f1': report['weighted avg']['f1-score'],
        'macro_f1': report['macro avg']['f1-score'],
        'per_class_f1': {c: report[c]['f1-score'] for c in class_names},
    }
    with open(METRICS_FILE, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved → {METRICS_FILE}")

    cm = confusion_matrix(y_true, y_pred, labels=label_ids)
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, annot_kws={"size": 7})
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix (Test Set)')
    plt.xticks(rotation=45, ha='right', fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    plt.show()
    print("Confusion matrix saved → confusion_matrix.png")


def predict_image(img_path: str, model: Model, class_indices_path: str,
                  threshold: float = CONFIDENCE_THRESHOLD):
    if not os.path.exists(class_indices_path):
        raise FileNotFoundError(
            f"'{class_indices_path}' not found — run training first."
        )
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image '{img_path}' not found.")

    with open(class_indices_path, 'r') as f:
        class_indices = json.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}

    img, _ = load_image(tf.constant(img_path), 0)
    x = preprocess_input(tf.expand_dims(img, 0))

    probs = model.predict(x, verbose=0)[0]
    top = np.argsort(probs)[::-1][:3]
    best = int(top[0])
    confidence = float(probs[best])

    print("\nTop predictions:")
    for i in top:
        print(f"  {idx_to_class[int(i)]:<30} {probs[int(i)] * 100:5.1f}%")

    if confidence < threshold:
        print(f"\nResult uncertain (confidence {confidence * 100:.1f}% is below "
              f"{threshold * 100:.0f}%). Retake the photo in good light with the "
              f"pest clearly visible.")
        return None, confidence

    pest_name = idx_to_class[best]
    print(f"\nPredicted pest : {pest_name} ({confidence * 100:.1f}% confidence)")
    return pest_name, confidence


if __name__ == '__main__':
    tf.keras.utils.set_random_seed(SEED)

    data = load_data(DATA_CSV)
    train_df, val_df, test_df = split_data(data)

    class_names = sorted(data['pest_type'].unique())
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    with open(CLASS_IDX_FILE, 'w') as f:
        json.dump(class_to_idx, f, indent=2)
    print(f"Class indices saved → {CLASS_IDX_FILE}")

    train_ds = make_dataset(train_df, class_to_idx, training=True)
    val_ds = make_dataset(val_df, class_to_idx, training=False)
    test_ds = make_dataset(test_df, class_to_idx, training=False)

    train_labels = train_df['pest_type'].map(class_to_idx).values
    weights = compute_class_weight('balanced',
                                   classes=np.arange(len(class_names)),
                                   y=train_labels)
    class_weight = {i: float(w) for i, w in enumerate(weights)}

    model, base_model = build_model(len(class_names))

    history, boundary = train_model(model, base_model, train_ds, val_ds, class_weight)
    model.save(MODEL_SAVE)
    print(f"Model saved → {MODEL_SAVE}")

    plot_history(history, boundary)
    evaluate_model(model, test_ds, test_df, class_names, class_to_idx)

    if os.path.exists(TEST_IMAGE):
        predict_image(TEST_IMAGE, model, CLASS_IDX_FILE)
    else:
        print(f"[Info] '{TEST_IMAGE}' not found — skipping sample prediction.")
