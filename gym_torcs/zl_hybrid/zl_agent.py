import os
import sys
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- FIX INGEGNERISTICO PER IL PATH ---
# Permette allo script di "vedere" i moduli nella cartella padre (gym_torcs)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import snakeoil3_jm2 as snakeoil3


# 1. Definizione della Rete Neurale (deve essere IDENTICA a 1_train_bc.py)
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
        return torch.tanh(self.output_layer(x))


def main():
    print("Inizializzazione Agente Autonomo ZL Hybrid...")

    # 2. Ricalcolo esatto di Media e Deviazione Standard dal Dataset
    list_of_files = glob.glob('dataset/*.csv')
    if not list_of_files:
        raise FileNotFoundError("Nessun dataset trovato per ricalcolare la normalizzazione!")
    latest_csv_path = max(list_of_files, key=os.path.getctime)

    data = pd.read_csv(latest_csv_path)
    X_raw = data[['speedX', 'trackPos', 'angle', 'rpm']].values

    # Usiamo asse 0 come in addestramento
    feature_mean = X_raw.mean(axis=0)
    feature_std = X_raw.std(axis=0) + 1e-8

    # 3. Caricamento del Modello Addestrato
    model = TorcsNet()
    model.load_state_dict(torch.load('models/bc_model.pth'))
    model.eval()  # Imposta la rete in modalità inferenza (blocca l'addestramento)
    print("Cervello IA caricato e allineato ai sensori.")

    # 4. Connessione al simulatore TORCS
    client = snakeoil3.Client(p=3001, vision=False)
    gear = 1

    print("\n[IN ATTESA DI TORCS] - Apri il gioco e avvia una gara in modalità Practice...")

    smoothed_steer = 0.0  # <-- AGGIUNGI QUESTA RIGA QUI

    while True:
        # Ricezione pacchetto dati dal gioco (a 50Hz)
        client.get_servers_input()
        S = client.S.d
        if not S:
            print("Connessione con TORCS terminata.")
            break

        # Estrazione e formattazione dei 4 sensori critici
        current_state = np.array([
            S.get('speedX', 0),
            S.get('trackPos', 0),
            S.get('angle', 0),
            S.get('rpm', 0)
        ])

        # Normalizzazione Z-score
        normalized_state = (current_state - feature_mean) / feature_std

        # Trasformazione in Tensore PyTorch e inferenza
        inputs = torch.tensor(normalized_state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():  # Nessun calcolo dei gradienti per massima velocità
            outputs = model(inputs)

        # Estrazione dei 3 comandi dall'array bidimensionale
        steer_cmd, accel_cmd, brake_cmd = outputs[0].numpy()

        # --- FILTRI DI SICUREZZA INGEGNERISTICI (GUARDRAILS) ---

        # 0. Filtro Passa-Basso (Smoothing dello Sterzo)
        # alpha = 0.2 significa: fidati al 20% del nuovo comando, mantieni l'80% di quello vecchio.
        alpha = 0.2
        smoothed_steer = (alpha * steer_cmd) + ((1.0 - alpha) * smoothed_steer)
        steer_cmd = smoothed_steer  # Sovrascriviamo il comando grezzo con quello fluido

        # 1. Filtro Rumore Freno (Deadzone)
        if brake_cmd < 0.1:
            brake_cmd = 0.0

        # 2. Launch Control Assoluto (Ignora l'IA alla partenza)
        speed = S.get('speedX', 0)
        if speed < 15.0:
            accel_cmd = max(accel_cmd, 0.6)  # Forza l'acceleratore
            brake_cmd = 0.0  # Niente freno
            steer_cmd = 0.0  # <-- FORZA LO STERZO DRITTO!
            gear = 1  # <-- FORZA LA PRIMA MARCIA!

        # 5. Logica del Cambio Marce Sequenziale
        rpm = S.get('rpm', 0)
        if speed >= 15.0:  # Permetti il cambio marce solo dopo la partenza
            if gear < 6 and rpm > 7000:
                gear += 1
            elif gear > 1 and rpm < 1500:
                gear -= 1

        # --- TELEMETRIA DIAGNOSTICA AVANZATA ---
        track_pos = S.get('trackPos', 0)
        angle = S.get('angle', 0)
        print(f"V: {speed:.0f} km/h | Pos: {track_pos:.2f} | Angolo: {angle:.2f}")
        print(f"IA -> Sterzo: {steer_cmd:.2f} | Accel: {accel_cmd:.2f} | Freno: {brake_cmd:.2f}\n")

        # 6. Invio dei comandi al veicolo
        client.R.d['steer'] = float(steer_cmd)
        client.R.d['accel'] = float(max(0.0, min(1.0, accel_cmd)))
        client.R.d['brake'] = float(max(0.0, min(1.0, brake_cmd)))
        client.R.d['gear'] = gear
        client.R.d['clutch'] = 0.0
        client.R.d['meta'] = 0

        client.respond_to_server()


if __name__ == "__main__":
    main()