import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ==========================================
# GESTIONE PERCORSI E SALVATAGGIO
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

dataset_path = os.path.join(parent_dir, "dataset", "master_dataset.csv")
models_dir = os.path.join(parent_dir, "models")
os.makedirs(models_dir, exist_ok=True)

# ==========================================
# DIMENSIONE INPUT (esportabile per altri moduli)
# speedX, trackPos, angle, rpm, speedY, speedZ, wheelSpinAvg + 19 track sensors
# ==========================================
INPUT_DIM = 26


# --- ARCHITETTURA RETE ---
class ActorCritic(nn.Module):
    def __init__(self, input_dim=INPUT_DIM):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256),  nn.LayerNorm(256),  nn.GELU(),
            nn.Linear(256, 512),        nn.LayerNorm(512),  nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),        nn.LayerNorm(256),  nn.GELU(),
            nn.Linear(256, 128),        nn.LayerNorm(128),  nn.GELU(),
        )
        # Head separate per output semanticamente diversi
        self.steer_head = nn.Linear(128, 1)    # tanh → [-1, 1]
        self.accel_head = nn.Linear(128, 1)     # sigmoid → [0, 1]
        self.brake_head = nn.Linear(128, 1)     # sigmoid → [0, 1]
        self.critic_head = nn.Linear(128, 1)    # Valore di stato (per RL)

    def forward(self, x):
        feat = self.shared(x)
        steer = torch.tanh(self.steer_head(feat))
        accel = torch.sigmoid(self.accel_head(feat))
        brake = torch.sigmoid(self.brake_head(feat))
        action_means = torch.cat([steer, accel, brake], dim=-1)
        value = self.critic_head(feat)
        return action_means, value


def main():
    print("🧠 Inizio Addestramento Behavioral Cloning...")

    if not os.path.exists(dataset_path):
        print(f"❌ Errore: Dataset non trovato in {dataset_path}")
        print("Assicurati di aver registrato almeno una sessione con collect_data.py")
        sys.exit(1)

    try:
        df = pd.read_csv(dataset_path)
        # Protezione vitale: rimuove righe corrotte o con valori mancanti per evitare Loss NaN
        df = df.dropna()
    except Exception as e:
        print(f"❌ Errore nella lettura del CSV: {e}")
        sys.exit(1)

    # --- Feature espanse: 7 scalari + 19 sensori di pista ---
    feature_cols = ['speedX', 'trackPos', 'angle', 'rpm',
                    'speedY', 'speedZ', 'wheelSpinAvg'] + [f'track_{i}' for i in range(19)]
    target_cols = ['steer', 'accel', 'brake']

    # Compatibilità all'indietro: se le nuove colonne non esistono, riempile con 0
    for col in feature_cols:
        if col not in df.columns:
            print(f"⚠️  Colonna '{col}' non trovata nel dataset, riempita con 0.")
            df[col] = 0.0

    X_raw = df[feature_cols].values.astype(np.float32)
    Y_raw = df[target_cols].values.astype(np.float32)

    print(f"📊 Dati caricati con successo: {X_raw.shape[0]} campioni, {X_raw.shape[1]} feature.")

    # --- Normalizzazione feature ---
    X_mean = X_raw.mean(axis=0)
    X_std = X_raw.std(axis=0) + 1e-8

    np.save(os.path.join(models_dir, "feat_mean.npy"), X_mean)
    np.save(os.path.join(models_dir, "feat_std.npy"), X_std)

    X_norm = (X_raw - X_mean) / X_std

    # --- Split train/val (90/10) ---
    n = len(X_norm)
    indices = np.random.permutation(n)
    split = int(0.9 * n)
    train_idx, val_idx = indices[:split], indices[split:]

    train_ds = TensorDataset(torch.tensor(X_norm[train_idx]), torch.tensor(Y_raw[train_idx]))
    val_ds = TensorDataset(torch.tensor(X_norm[val_idx]), torch.tensor(Y_raw[val_idx]))
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)

    print(f"📂 Train: {len(train_ds)} campioni | Val: {len(val_ds)} campioni")

    # --- Modello, ottimizzatore, scheduler ---
    model = ActorCritic()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10
    epochs = 100

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            actions, _ = model(bx)
            # Loss combinata: MSE per steer (peso doppio), MSE per accel/brake
            steer_loss = nn.MSELoss()(actions[:, 0], by[:, 0])
            accel_loss = nn.MSELoss()(actions[:, 1], by[:, 1])
            brake_loss = nn.MSELoss()(actions[:, 2], by[:, 2])
            loss = steer_loss * 2.0 + accel_loss + brake_loss  # Steer ha peso doppio
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                actions, _ = model(bx)
                steer_loss = nn.MSELoss()(actions[:, 0], by[:, 0])
                accel_loss = nn.MSELoss()(actions[:, 1], by[:, 1])
                brake_loss = nn.MSELoss()(actions[:, 2], by[:, 2])
                loss = steer_loss * 2.0 + accel_loss + brake_loss
                val_loss += loss.item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        lr_now = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:03d}/{epochs} | Train: {avg_train:.5f} | Val: {avg_val:.5f} | LR: {lr_now:.6f}")

        # --- Early stopping e salvataggio miglior modello ---
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(models_dir, "bc_model.pth"))
            print(f"  💾 Nuovo miglior modello salvato! (Val Loss: {best_val_loss:.5f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏹ Early stopping! Nessun miglioramento per {patience} epoche.")
                break

    print(f"\n✅ Addestramento completato! Miglior Val Loss: {best_val_loss:.5f}")
    print(f"   Modello salvato in {models_dir}/bc_model.pth")


if __name__ == "__main__":
    main()