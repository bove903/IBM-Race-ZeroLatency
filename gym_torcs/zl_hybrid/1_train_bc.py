import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, csv_path):
        print(f"Caricamento dataset da: {csv_path}")
        self.data = pd.read_csv(csv_path)

        # Estrazione feature (X) e target (Y)
        X_raw = self.data[['speedX', 'trackPos', 'angle', 'rpm']].values
        self.Y = self.data[['steer', 'accel', 'brake']].values

        # NORMALIZZAZIONE CORRETTA: asse 0 significa colonna per colonna
        # Aggiungiamo un piccolo epsilon (1e-8) per evitare divisioni per zero
        self.X_mean = X_raw.mean(axis=0)
        self.X_std = X_raw.std(axis=0) + 1e-8
        self.X = (X_raw - self.X_mean) / self.X_std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Usiamo self.X e self.Y per risolvere l'errore di scope
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.Y[idx], dtype=torch.float32)


class TorcsNet(nn.Module):
    def __init__(self):
        super(TorcsNet, self).__init__()
        self.layer1 = nn.Linear(4, 64)
        self.layer2 = nn.Linear(64, 128)
        self.layer3 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, 3)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        # Tanh mappa tutto tra -1 e 1 (perfetto per i pedali e lo sterzo)
        return torch.tanh(self.output_layer(x))


if __name__ == "__main__":
    # Trova automaticamente il file più recente per non dover cambiare il nome a mano
    list_of_files = glob.glob('dataset/*.csv')
    if not list_of_files:
        raise FileNotFoundError("Nessun file CSV trovato nella cartella 'dataset'. Guida l'auto prima di addestrare!")
    latest_csv_path = max(list_of_files, key=os.path.getctime)

    # Inizializzazione
    dataset = CustomDataset(latest_csv_path)

    # DATALOADER: fondamentale per la velocità e la stabilità (Batch size = 64)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = TorcsNet()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("Inizio addestramento della Rete Neurale...")
    epochs = 50
    for epoch in range(epochs):
        epoch_loss = 0.0

        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoca [{epoch + 1}/{epochs}], Loss Media: {epoch_loss / len(dataloader):.4f}")

    # Creazione cartella models se non esiste
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/bc_model.pth')
    print("Modello salvato con successo in 'models/bc_model.pth'")