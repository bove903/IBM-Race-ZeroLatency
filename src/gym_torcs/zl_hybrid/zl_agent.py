"""
Agente Autonomo Ibrido (Inference Script) - Team ZeroLatency
Questo script carica il modello neurale addestrato (Behavioral Cloning)
e lo utilizza per guidare il veicolo in tempo reale nel simulatore TORCS.
Applica un "Safety Layer" deterministico (ABS, TCS, High-Speed Center Correction)
per garantire stabilità e sicurezza ad alte velocità.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import time
import torch
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import snakeoil3_gym as snakeoil3
from train_bc import ActorCritic

# Path assoluti (funzionano da qualsiasi directory)
models_dir = os.path.join(parent_dir, "models")
ppo_path = os.path.join(models_dir, "ppo_model.pth")
bc_path = os.path.join(models_dir, "bc_model.pth")
feat_mean_path = os.path.join(models_dir, "feat_mean.npy")
feat_std_path = os.path.join(models_dir, "feat_std.npy")


def extract_state(S, feat_mean, feat_std):
    """
    Estrae le 26 features in tempo reale dal pacchetto UDP di TORCS
    e le normalizza usando media e deviazione standard del dataset di training.
    
    Args:
        S (dict): Dizionario di stato proveniente da TORCS.
        feat_mean (np.ndarray): Array delle medie calcolate in fase di training.
        feat_std (np.ndarray): Array delle deviazioni standard calcolate in fase di training.
        
    Returns:
        np.ndarray: Vettore di stato normalizzato pronto per la Rete Neurale.
    """
    trk = S.get('track', [100.0] * 19)
    if len(trk) < 19: trk += [100.0] * (19 - len(trk))
    speed_y = S.get('speedY', 0.0)
    speed_z = S.get('speedZ', 0.0)
    wheels = S.get('wheelSpinVel', [0.0, 0.0, 0.0, 0.0])
    wheel_spin_avg = sum(wheels) / max(len(wheels), 1)
    raw = [
        S.get('speedX', 0.0), S.get('trackPos', 0.0),
        S.get('angle', 0.0), S.get('rpm', 0.0),
        speed_y, speed_z, wheel_spin_avg
    ] + trk[:19]
    return (np.array(raw, dtype=np.float32) - feat_mean) / feat_std


def check_model_health(model, feat_mean, feat_std):
    """Verifica che il modello acceleri da fermo in rettilineo."""
    model.eval()
    raw = np.array([0.0, 0.0, 0.0, 1000, 0, 0, 0] + [200.0]*19, dtype=np.float32)
    x = torch.tensor((raw - feat_mean) / feat_std).unsqueeze(0)
    with torch.no_grad():
        actions, _ = model(x)
    accel = actions[0, 1].item()
    brake = actions[0, 2].item()
    return accel > 0.3 and brake < 0.3  # Deve accelerare e non frenare


def main():
    """
    Ciclo di vita principale dell'Agente:
    1. Carica la Rete Neurale (Pesi BC).
    2. Istanzia il client UDP verso TORCS (snakeoil3).
    3. Esegue un loop a ~50Hz: riceve sensori -> inferenza -> applica safety layer -> invia comandi.
    """
    print("🏎️ AGENTE ZERO LATENCY v3.0 (Corkscrew)")
    torch.set_num_threads(4)

    feat_mean = np.load(feat_mean_path)
    feat_std = np.load(feat_std_path)

    # Carica modello con health check automatico
    model = ActorCritic()

    model_loaded = None
    if os.path.exists(ppo_path):
        model.load_state_dict(torch.load(ppo_path, map_location='cpu'))
        if check_model_health(model, feat_mean, feat_std):
            model_loaded = "PPO"
            print(f"✅ Modello PPO caricato e verificato.")
        else:
            print(f"⚠️  Modello PPO CORROTTO (non accelera). Uso il BC...")

    if model_loaded is None:
        if os.path.exists(bc_path):
            model.load_state_dict(torch.load(bc_path, map_location='cpu'))
            model_loaded = "BC"
            print(f"✅ Modello BC caricato (saltata la verifica di salute automatica).")
        else:
            print("❌ Nessun modello trovato!")
            sys.exit(1)

    model.eval()

    # Test rapido output modello
    raw_test = np.array([0.0, 0.0, 0.0, 1000, 0, 0, 0] + [200.0]*19, dtype=np.float32)
    x_test = torch.tensor((raw_test - feat_mean) / feat_std).unsqueeze(0)
    with torch.no_grad():
        act_test, _ = model(x_test)
    print(f"   Test da fermo: Gas={act_test[0,1]:.2f} Freno={act_test[0,2]:.2f} Sterzo={act_test[0,0]:.2f}")

    client = snakeoil3.Client(p=3001, vision=False)

    tick_vuoti = 0
    tick_globale = 0
    shift_cooldown = 0
    lap_speeds = []
    last_lap_time = 0.0
    lap_start = time.time()

    try:
        while True:
            try:
                client.get_servers_input()
            except Exception:
                tick_vuoti += 1
                if tick_vuoti >= 50: break
                continue

            S = client.S.d
            if not S or 'track' not in S: continue
            tick_vuoti = 0

            speed_kmh = S.get('speedX', 0.0)
            track_pos = S.get('trackPos', 0.0)
            rpm = S.get('rpm', 0)
            gear = S.get('gear', 1)

            # Inferenza: Rete Neurale Pura
            s_t = torch.tensor(extract_state(S, feat_mean, feat_std)).unsqueeze(0)
            with torch.no_grad():
                act_t, _ = model(s_t)

            sterzo, accel, brake = act_t[0].numpy()

            # === SAFETY LAYER (solo ad alta velocità) ===
            abs_track_pos = abs(track_pos)
            if abs_track_pos > 0.85 and speed_kmh > 80:
                safety_factor = min(1.0, (abs_track_pos - 0.85) / 0.15)
                accel *= (1.0 - safety_factor * 0.5)
                center_correction = -np.sign(track_pos) * safety_factor * 0.3
                sterzo += center_correction
                if abs_track_pos > 0.95:
                    brake = max(brake, safety_factor * 0.3)

            # === TCS (Traction Control System) ===
            wheels = S.get('wheelSpinVel', [0, 0, 0, 0])
            wheel_slip = (wheels[2] + wheels[3]) - (wheels[0] + wheels[1])
            if wheel_slip > 8.0:
                accel = max(0.0, accel - 0.4) 

            # Esecuzione
            client.R.d['steer'] = float(np.clip(sterzo, -1.0, 1.0))
            client.R.d['accel'] = float(np.clip(accel, 0.0, 1.0))

            # ABS
            steer_mag = abs(sterzo)
            if steer_mag > 0.1:
                max_brake = max(0.2, 1.0 - (steer_mag * 1.5))
                brake = min(brake, max_brake)

            brake = float(np.clip(brake, 0.0, 1.0))
            brake_threshold = 0.01 if speed_kmh > 100 else 0.03
            client.R.d['brake'] = brake if brake > brake_threshold else 0.0

            # === CAMBIO MARCE ===
            if shift_cooldown > 0:
                shift_cooldown -= 1
            
            # Impediamo all'auto di mettere la 1a a meno che non sia quasi ferma
            if speed_kmh < 25:
                gear = 1
            elif shift_cooldown == 0:
                if gear < 6 and rpm > 16500:
                    gear += 1
                    shift_cooldown = 5
                # Scaliamo in 2a solo se i giri scendono sotto 6000
                elif gear > 2 and rpm < 6000:
                    gear -= 1
                    shift_cooldown = 5
                # Scaliamo in 1a solo se i giri in 2a scendono sotto i 4000
                elif gear == 2 and rpm < 4000:
                    gear -= 1
                    shift_cooldown = 5
            client.R.d['gear'] = gear

            client.respond_to_server()

            # === TELEMETRIA ===
            lap_speeds.append(speed_kmh)
            cur_lap_time = S.get('lastLapTime', 0.0)
            if cur_lap_time > 0 and cur_lap_time != last_lap_time:
                avg_speed = np.mean(lap_speeds) if lap_speeds else 0
                print(f"\n🏁 GIRO COMPLETATO! Tempo: {cur_lap_time:.2f}s | V media: {avg_speed:.0f} km/h")
                last_lap_time = cur_lap_time
                lap_speeds = []
                lap_start = time.time()

            tick_globale += 1
            if tick_globale % 10 == 0:
                elapsed = time.time() - lap_start
                safety_str = "⚠️" if abs_track_pos > 0.85 else "✅"
                tcs_str = "🔴TCS" if wheel_slip > 8.0 else ""
                print(
                    f"\r🤖 {safety_str} | St:{sterzo:+.2f} Gas:{accel:.2f} Fr:{brake:.2f} M:{gear} V:{speed_kmh:.0f}km/h Pos:{track_pos:+.2f} T:{elapsed:.0f}s {tcs_str}",
                    end="")

    except KeyboardInterrupt:
        print("\n\n🛑 Guida interrotta.")
    finally:
        try:
            client.R.d.update({'steer': 0.0, 'accel': 0.0, 'brake': 1.0, 'gear': 0})
            client.respond_to_server()
            client.shutdown()
        except:
            pass
        if lap_speeds:
            print(f"   Velocità media ultimo stint: {np.mean(lap_speeds):.0f} km/h")
        print("   Spegnimento completato.")


if __name__ == "__main__":
    main()