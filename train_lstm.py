"""
train_lstm.py — Bidirectional LSTM + Attention Training Script for Google Colab.

Architecture:
    Input:     (10, 24) — 10-game lookback, 24 features per timestep
    BiLSTM-1:  64 units bidirectional (128 total), return_sequences=True
    BiLSTM-2:  32 units bidirectional (64 total), return_sequences=True
    Attention: Learned weighted sum across timesteps
    Dense:     16 units, ReLU
    Output:    1 unit, Sigmoid → Win probability [0.0, 1.0]

Loss: Binary Crossentropy (classification)
Evaluation: Accuracy, AUC, Classification Report

Usage (Colab):
    1. Upload X_train.npy, X_val.npy, y_train.npy, y_val.npy, models/scaler.pkl
    2. Run this script
    3. Download models/lstm_model.keras and models/scaler.pkl

Usage (Local quick test):
    python train_lstm.py --epochs 5
"""

import numpy as np
import os
import argparse

# Suppress TF warnings for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, BatchNormalization,
    Bidirectional, Layer
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.metrics import classification_report, roc_auc_score

# CONFIG
MODELS_DIR = "models"
BATCH_SIZE = 32
DEFAULT_EPOCHS = 100
LEARNING_RATE = 0.001


class Attention(Layer):
    """
    Temporal Attention Layer for sequence models.
    
    Given a sequence of hidden states (batch, timesteps, features),
    learns a weight for each timestep and returns the weighted sum
    as a single context vector (batch, features).
    
    This allows the model to dynamically focus on specific games
    in the lookback window rather than relying solely on the final state.
    """
    
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], 1),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(input_shape[1], 1),
            initializer='zeros',
            trainable=True
        )
        super(Attention, self).build(input_shape)
    
    def call(self, x):
        # x shape: (batch, timesteps, features)
        # Score each timestep
        score = tf.nn.tanh(tf.matmul(x, self.W) + self.b)  # (batch, timesteps, 1)
        attention_weights = tf.nn.softmax(score, axis=1)     # (batch, timesteps, 1)
        
        # Weighted sum of input sequence
        context = x * attention_weights                       # (batch, timesteps, features)
        context = tf.reduce_sum(context, axis=1)              # (batch, features)
        
        return context
    
    def get_config(self):
        return super(Attention, self).get_config()


def build_model(input_shape):
    """
    Build a Bidirectional LSTM with Attention for win probability prediction.
    
    Architecture:
        BiLSTM(64) → BN → Dropout → BiLSTM(32) → BN → Dropout
        → Attention → Dense(16) → Dropout → Dense(1, sigmoid)
    """
    inputs = Input(shape=input_shape)
    
    # BiLSTM Layer 1: 64 units per direction = 128 total
    x = Bidirectional(
        LSTM(64, return_sequences=True, kernel_regularizer=l2(0.001)),
        merge_mode='concat'
    )(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    # BiLSTM Layer 2: 32 units per direction = 64 total
    x = Bidirectional(
        LSTM(32, return_sequences=True, kernel_regularizer=l2(0.001)),
        merge_mode='concat'
    )(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    # Attention: learn which games in the window matter most
    x = Attention(name='attention')(x)
    
    # Dense classifier head
    x = Dense(16, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.2)(x)
    
    # Output: win probability
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Train BiLSTM+Attention model for NBA predictions')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS,
                        help=f'Max training epochs (default: {DEFAULT_EPOCHS})')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help=f'Batch size (default: {BATCH_SIZE})')
    args = parser.parse_args()
    
    print("=" * 60)
    print("  BiLSTM + ATTENTION — WIN PROBABILITY TRAINING")
    print("=" * 60)
    
    # Check for GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"\n  GPU detected: {gpus[0].name}")
    else:
        print("\n  No GPU detected. Training on CPU.")
    
    # 1. Load data
    print("\n  Loading data...")
    X_train = np.load('X_train.npy')
    X_val = np.load('X_val.npy')
    y_train = np.load('y_train.npy')
    y_val = np.load('y_val.npy')
    
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val:   {X_val.shape}")
    print(f"  y_train: {y_train.shape} (win rate: {y_train.mean():.3f})")
    print(f"  y_val:   {y_val.shape} (win rate: {y_val.mean():.3f})")
    
    input_shape = (X_train.shape[1], X_train.shape[2])  # (10, 21)
    
    # 2. Build model
    print(f"\n  Building model with input shape {input_shape}...")
    model = build_model(input_shape)
    model.summary()
    
    # 3. Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # 4. Train
    print(f"\n  Training for up to {args.epochs} epochs...")
    print(f"  Batch size: {args.batch_size}")
    print("-" * 60)
    
    # Compute class weights to balance Win/Loss recall
    n_losses = (y_train == 0).sum()
    n_wins = (y_train == 1).sum()
    total = n_losses + n_wins
    weight_loss = total / (2.0 * n_losses)
    weight_win = total / (2.0 * n_wins)
    class_weights = {0: weight_loss, 1: weight_win}
    print(f"  Class weights: Loss={weight_loss:.3f}, Win={weight_win:.3f}")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # 5. Evaluate
    print("\n" + "=" * 60)
    print("  EVALUATION")
    print("=" * 60)
    
    # Validation predictions
    y_pred_prob = model.predict(X_val, verbose=0).flatten()
    y_pred = (y_pred_prob >= 0.5).astype(int)
    
    # Metrics
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    auc = roc_auc_score(y_val, y_pred_prob)
    
    print(f"\n  Validation Loss:     {val_loss:.4f}")
    print(f"  Validation Accuracy: {val_acc:.4f}")
    print(f"  Validation AUC:      {auc:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(
        y_val, y_pred, 
        target_names=['Loss', 'Win'],
        digits=3
    ))
    
    # 6. Save model
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, 'lstm_model.keras')
    model.save(model_path)
    
    print(f"\n  [SAVED] {model_path}")
    print(f"  [NOTE] Copy models/lstm_model.keras and models/scaler.pkl to your local machine.")
    
    # 7. Training summary
    best_epoch = np.argmin(history.history['val_loss']) + 1
    best_val_loss = min(history.history['val_loss'])
    print(f"\n  Best epoch: {best_epoch} (val_loss: {best_val_loss:.4f})")
    print(f"  Total epochs run: {len(history.history['loss'])}")


if __name__ == "__main__":
    main()
