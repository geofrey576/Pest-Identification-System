"""
AI-Based Pest Identification 
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from keras.applications import MobileNetV2
from keras.models import Model, load_model
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam


from PIL import Image as PILImage

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
DATA_CSV       = 'pests_labels.csv'     
PEST_DETAILS   = 'Pest_Details.csv'     
MODEL_SAVE     = 'pest_model.keras'
CLASS_IDX_FILE = 'class_indices.json'
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
EPOCHS_HEAD = 10    
EPOCHS_FINE = 10   


def load_data(csv_path: str) -> pd.DataFrame:
    data = pd.read_csv(csv_path)
    if 'pest_type' not in data.columns and 'label' in data.columns:
        data.rename(columns={'label': 'pest_type'}, inplace=True)

    required = {'image_path', 'pest_type'}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(
            f"CSV is missing columns: {missing}\n"
            f"Found columns: {list(data.columns)}"
        )

    print(f"Dataset loaded  : {len(data)} images across "
          f"{data['pest_type'].nunique()} classes")
    print(f"Classes         : {sorted(data['pest_type'].unique())}")
    return data

def build_generators(train_df: pd.DataFrame, val_df: pd.DataFrame):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        x_col='image_path',
        y_col='pest_type',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    val_generator = val_datagen.flow_from_dataframe(
        val_df,
        x_col='image_path',
        y_col='pest_type',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False  
    )

    return train_generator, val_generator

def build_model(num_classes: int):
    base = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(*IMG_SIZE, 3)
    )
    base.trainable = False  
    x      = GlobalAveragePooling2D()(base.output)
    x      = Dense(256, activation='relu')(x)
    x      = Dropout(0.5)(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base.input, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f"\nModel built — {num_classes} output classes")
    print(f"Trainable params (head only): "
          f"{sum(np.prod(v.shape) for v in model.trainable_variables):,}")

    return model, base   
def train_model(model: Model, base_model, train_gen, val_gen) -> dict:
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=5,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(MODEL_SAVE, monitor='val_accuracy',
                        save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=3, min_lr=1e-6, verbose=1)
    ]

    print("\n=== Phase 1: Training head (base frozen) ===")
    h1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_HEAD,
        callbacks=callbacks
    )
    print("\n=== Phase 2: Fine-tuning last 30 layers ===")
    base_model.trainable = True                      
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    h2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_FINE,
        callbacks=callbacks
    )
    combined_history = {}
    for key in h1.history:
        combined_history[key] = h1.history[key] + h2.history[key]

    return combined_history

def evaluate_model(model: Model, val_gen, class_names: list):
    print("\n=== Evaluation ===")
    loss, acc = model.evaluate(val_gen, verbose=0)
    print(f"Validation Loss    : {loss:.4f}")
    print(f"Validation Accuracy: {acc*100:.2f}%")

    val_gen.reset()
    preds  = model.predict(val_gen, verbose=1)
    y_pred = np.argmax(preds, axis=1)
    y_true = val_gen.classes

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, annot_kws={"size": 7})
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    plt.xticks(rotation=45, ha='right', fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    plt.show()
    print("Confusion matrix saved → confusion_matrix.png")

def plot_history(history: dict):
    epochs         = range(1, len(history['accuracy']) + 1)
    phase_boundary = EPOCHS_HEAD

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history['accuracy'],     label='Train')
    ax1.plot(epochs, history['val_accuracy'], label='Validation')
    ax1.axvline(phase_boundary, linestyle='--', color='gray', label='Fine-tune start')
    ax1.set_title('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend()

    ax2.plot(epochs, history['loss'],     label='Train')
    ax2.plot(epochs, history['val_loss'], label='Validation')
    ax2.axvline(phase_boundary, linestyle='--', color='gray', label='Fine-tune start')
    ax2.set_title('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()

    plt.suptitle('MobileNetV2 — Pest Classification', fontsize=13)
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150)
    plt.show()
    print("Training curves saved → training_curves.png")

def _create_default_pest_details(csv_path: str, class_names: list):
    """Auto-generate a starter Pest_Details.csv if one does not exist."""
    default_data = {
        'pest_name'         : class_names,
        'pesticide'         : ['Consult extension officer'] * len(class_names),
        'dosage'            : ['As per label']              * len(class_names),
        'application_method': ['Foliar spray']              * len(class_names),
        'safety_notes'      : ['Wear PPE, follow label']    * len(class_names),
    }
    pd.DataFrame(default_data).to_csv(csv_path, index=False)
    print(f"[Info] Starter '{csv_path}' created — fill in pesticide details.")


def load_pesticide_db(csv_path: str, class_names: list = None) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        if class_names:
            _create_default_pest_details(csv_path, class_names)
        else:
            print(f"[Warning] {csv_path} not found and no class names provided.")
            return pd.DataFrame()

    db = pd.read_csv(csv_path)
    db['pest_name_lower'] = db['pest_name'].str.lower().str.strip()
    print(f"Pesticide database loaded: {len(db)} entries.")
    return db


def recommend_pesticide(pest_name: str, pest_db: pd.DataFrame) -> str:
    if pest_db.empty:
        return "Pesticide database not available."

    match = pest_db[pest_db['pest_name_lower'] == pest_name.lower().strip()]
    if match.empty:
        return (f"No record found for '{pest_name}'.\n"
                f"Consult your local agricultural extension officer (KALRO).")

    row = match.iloc[0]
    return "\n".join([
        f"Pest identified : {pest_name}",
        f"Pesticide       : {row.get('pesticide', 'N/A')}",
        f"Dosage          : {row.get('dosage', 'N/A')}",
        f"Application     : {row.get('application_method', 'N/A')}",
        f"Safety notes    : {row.get('safety_notes', 'Follow label')}",
    ])

def predict_image(img_path: str, model: Model,
                  class_indices_path: str, pest_db: pd.DataFrame):
    if not os.path.exists(class_indices_path):
        raise FileNotFoundError(
            f"'{class_indices_path}' not found — run training first."
        )
    with open(class_indices_path, 'r') as f:
        class_indices = json.load(f)

    idx_to_class = {v: k for k, v in class_indices.items()}

    img = PILImage.open(img_path).convert('RGB').resize(IMG_SIZE)
    x   = np.array(img, dtype=np.float32) / 255.0
    x   = np.expand_dims(x, axis=0)

    preds      = model.predict(x, verbose=0)
    class_idx  = int(np.argmax(preds, axis=1)[0])
    confidence = float(np.max(preds)) * 100
    pest_name  = idx_to_class[class_idx]

    print(f"\nPredicted pest : {pest_name} ({confidence:.1f}% confidence)")
    print(recommend_pesticide(pest_name, pest_db))
    return pest_name, confidence

if __name__ == '__main__':

    
    data = load_data(DATA_CSV)

    train_df, val_df = train_test_split(
        data,
        test_size=0.2,
        stratify=data['pest_type'],
        random_state=42
    )
    print(f"Train: {len(train_df)} | Val: {len(val_df)}")

    train_gen, val_gen = build_generators(train_df, val_df)

    with open(CLASS_IDX_FILE, 'w') as f:
        json.dump(train_gen.class_indices, f, indent=2)
    print(f"Class indices saved → {CLASS_IDX_FILE}")

    num_classes = len(train_gen.class_indices)
    class_names = list(train_gen.class_indices.keys())

    model, base_model = build_model(num_classes)

    history = train_model(model, base_model, train_gen, val_gen)

    plot_history(history)

    evaluate_model(model, val_gen, class_names)

    pest_db = load_pesticide_db(PEST_DETAILS, class_names)
        
        predict_image('images/test_aphid.jpg', model, CLASS_IDX_FILE, pest_db)