import os
import torch
import numpy as np
import snakeoil3_gym as snakeoil3
from train_bc import ActorCritic


def extract_state(S, feat_mean, feat_std):
    trk = S.get('track', [100.0] * 19)
    if len(trk) < 19: trk += [100.0] * (19 - len(trk))

    # Estrazione feature extra (compatibile con il nuovo schema a 26 feature)
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


def main():
    print("🏎️ AGENTE ZERO LATENCY v2.0 (Ottimizzato Michigan)")
    torch.set_num_threads(4)  # Efficienza CPU su Mac

    model = ActorCritic()
    model_path = 'models/ppo_model.pth' if os.path.exists('models/ppo_model.pth') else 'models/bc_model.pth'

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    print(f"✅ Modello {model_path} caricato.")

    feat_mean = np.load("models/feat_mean.npy")
    feat_std = np.load("models/feat_std.npy")

    client = snakeoil3.Client(p=3001, vision=False)

    tick_vuoti = 0
    tick_globale = 0

    # Stato persistente per cambio marce e telemetria
    shift_cooldown = 0
    lap_speeds = []
    last_lap_time = 0.0

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

            # Inferenza Veloce
            s_t = torch.tensor(extract_state(S, feat_mean, feat_std)).unsqueeze(0)
            with torch.no_grad():
                act_t, _ = model(s_t)

            sterzo, accel, brake = act_t[0].numpy()

            # === SAFETY LAYER INTELLIGENTE ===
            # Override progressivo se troppo vicini al bordo pista
            abs_track_pos = abs(track_pos)
            if abs_track_pos > 0.85:
                # Fattore di sicurezza crescente (0.0 a 0.85, 1.0 a 1.0)
                safety_factor = min(1.0, (abs_track_pos - 0.85) / 0.15)
                # Riduce gas proporzionalmente
                accel *= (1.0 - safety_factor * 0.7)
                # Aggiunge correzione sterzo verso il centro
                center_correction = -np.sign(track_pos) * safety_factor * 0.3
                sterzo += center_correction
                # Frenata di emergenza se molto fuori
                if abs_track_pos > 0.92:
                    brake = max(brake, safety_factor * 0.4)

            # === ANTI-SPIN TCS (Traction Control) ===
            wheels = S.get('wheelSpinVel', [0, 0, 0, 0])
            wheel_slip = (wheels[2] + wheels[3]) - (wheels[0] + wheels[1])
            if wheel_slip > 8.0:
                accel = max(0.0, accel - 0.3)

            # Sicurezza ed Esecuzione
            client.R.d['steer'] = float(np.clip(sterzo, -1.0, 1.0))
            client.R.d['accel'] = float(np.clip(accel, 0.0, 1.0))

            # === ABS & Filtro Freno ===
            # Evita bloccaggio ruote anteriori in curva (sottosterzo e dritto)
            steer_mag = abs(sterzo)
            if steer_mag > 0.05:
                max_brake = max(0.05, 1.0 - (steer_mag * 3.0))
                brake = min(brake, max_brake)

            brake = float(np.clip(brake, 0.0, 1.0))
            brake_threshold = 0.01 if speed_kmh > 100 else 0.03
            client.R.d['brake'] = brake if brake > brake_threshold else 0.0

            # === LOGICA CAMBIO — ALLINEATA AL TRAINING DATA ===
            # FIX CRITICO: soglie RPM identiche a collect_data.py e train_rl.py
            if shift_cooldown > 0:
                shift_cooldown -= 1

            if speed_kmh < 20:
                gear = 1
            elif shift_cooldown == 0:
                if gear < 6 and rpm > 16000:
                    gear += 1
                    shift_cooldown = 5
                elif gear > 1 and rpm < 7000:
                    gear -= 1
                    shift_cooldown = 5
            client.R.d['gear'] = gear

            client.respond_to_server()

            # === TELEMETRIA AVANZATA ===
            lap_speeds.append(speed_kmh)
            cur_lap_time = S.get('lastLapTime', 0.0)
            if cur_lap_time > 0 and cur_lap_time != last_lap_time:
                avg_speed = np.mean(lap_speeds) if lap_speeds else 0
                print(f"\n🏁 GIRO COMPLETATO! Tempo: {cur_lap_time:.2f}s | V media: {avg_speed:.0f} km/h")
                last_lap_time = cur_lap_time
                lap_speeds = []

            tick_globale += 1
            if tick_globale % 10 == 0:
                safety_str = "⚠️" if abs_track_pos > 0.85 else "✅"
                tcs_str = "🔴TCS" if wheel_slip > 8.0 else ""
                print(
                    f"\r🤖 {safety_str} | St:{sterzo:+.2f} Gas:{accel:.2f} Fr:{brake:.2f} M:{gear} V:{speed_kmh:.0f}km/h Pos:{track_pos:+.2f} {tcs_str}",
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
        print("   Comandi azzerati. Spegnimento completato.")


if __name__ == "__main__":
    main()