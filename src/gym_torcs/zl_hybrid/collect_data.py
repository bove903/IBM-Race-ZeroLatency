import os
import sys
import time
import numpy as np
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import snakeoil3_gym as snakeoil3

PI = 3.14159265359

dataset_dir = os.path.join(parent_dir, "dataset")
os.makedirs(dataset_dir, exist_ok=True)
master_file = os.path.join(dataset_dir, "master_dataset.csv")

class RaceFinished(Exception): pass
class BotStuck(Exception): pass

def drive_and_record(c, session_data, t0, start_damage, bot_state):
    S, R = c.S.d, c.R.d
    if not S:
        raise RaceFinished()

    speed_kmh = S.get('speedX', 0.0)
    angle = S.get('angle', 0.0)
    track_pos = S.get('trackPos', 0.0)
    rpm = S.get('rpm', 0.0)
    gear = S.get('gear', 1)

    current_damage = S.get('damage', 0) - start_damage
    trk = S.get('track', [100.0] * 19)
    if len(trk) < 19: trk += [100.0] * (19 - len(trk))
    front_distance = trk[9]

    speed_y = S.get('speedY', 0.0)
    speed_z = S.get('speedZ', 0.0)
    wheels = S.get('wheelSpinVel', [0, 0, 0, 0])
    wheel_spin_avg = float(np.mean(wheels))

    # ============================================================
    # PROTEZIONE 1: Fuoripista → segnala immediatamente
    # ============================================================
    is_on_track = abs(track_pos) < 1.0

    # ============================================================
    # PROTEZIONE 2: Auto bloccata → conta i secondi ferma
    # ============================================================
    if abs(speed_kmh) < 5.0:
        bot_state['stuck_frames'] = bot_state.get('stuck_frames', 0) + 1
    else:
        bot_state['stuck_frames'] = 0

    # Se ferma per più di 200 frame (~10 secondi) → termina
    if bot_state['stuck_frames'] > 200:
        raise BotStuck("Auto ferma da troppo tempo!")

    # ============================================================
    # PROTEZIONE 3: Fuoripista per più di 100 frame → termina
    # ============================================================
    if not is_on_track:
        bot_state['offtrack_frames'] = bot_state.get('offtrack_frames', 0) + 1
    else:
        bot_state['offtrack_frames'] = 0

    if bot_state['offtrack_frames'] > 100:
        raise BotStuck("Fuoripista da troppo tempo!")

    # STERZO ADATTIVO (Reattivo sui tornanti, stabile sull'ovale)
    steer_gain = max(6.0, 25.0 - (speed_kmh * 0.08))
    target_steer = (angle * steer_gain / PI) - (track_pos * 0.25)
    
    # Smoothing dinamico: reattivo a bassa velocità (Corkscrew), fluido ad alta velocità (Michigan)
    if speed_kmh < 120:
        smooth_factor = 0.4 # 40% vecchio, 60% nuovo -> Molto agile per le curve strette
    else:
        smooth_factor = 0.8 # 80% vecchio, 20% nuovo -> Stabile per i rettilinei
        
    smooth_steer = bot_state.get('prev_steer', 0.0) * smooth_factor + target_steer * (1.0 - smooth_factor)
    R['steer'] = float(np.clip(smooth_steer, -1.0, 1.0))

    # VELOCITÀ AGGRESSIVE (Ottimizzato per i tempi sul giro a Laguna Seca)
    if front_distance > 120.0: target_speed = 280.0
    elif front_distance > 80.0: target_speed = 230.0
    elif front_distance > 50.0: target_speed = 160.0
    elif front_distance > 30.0: target_speed = 100.0
    else: target_speed = 50.0

    # FRENATA SPECIFICA "CORKSCREW" E TORNANTI
    # Il Corkscrew si riconosce perché la distanza frontale crolla per via della discesa cieca.
    # Se stiamo affrontando una curva stretta (angolo elevato) e la distanza è poca:
    if abs(angle) > 0.15 and front_distance < 60.0:
        target_speed = 45.0
        
    target_speed = max(45.0, target_speed) # Velocità minima in pista

    if speed_kmh < target_speed:
        R['accel'] = float(np.clip(R.get('accel', 0) + 0.2, 0.0, 1.0))
        R['brake'] = 0.0
    else:
        R['accel'] = 0.0
        steer_mag = abs(R['steer'])
        raw_brake = (speed_kmh - target_speed) * 0.05
        # ABS più dolce
        max_brake = max(0.1, 1.0 - (steer_mag * 2.0)) 
        R['brake'] = float(np.clip(min(raw_brake, max_brake), 0.0, 1.0))

    if speed_kmh < 10.0:
        R['accel'] = 1.0
        R['brake'] = 0.0

    # TCS
    wheel_slip = (wheels[2] + wheels[3]) - (wheels[0] + wheels[1])
    if wheel_slip > 8.0:
        R['accel'] = max(0.0, R['accel'] - 0.3)

    # MARCE
    if bot_state['shift_cooldown'] > 0:
        bot_state['shift_cooldown'] -= 1

    if speed_kmh < 20:
        gear = 1
    elif bot_state['shift_cooldown'] == 0:
        if gear < 6 and rpm > 16000:
            gear += 1
            bot_state['shift_cooldown'] = 5
        elif gear > 1 and rpm < 7000:
            gear -= 1
            bot_state['shift_cooldown'] = 5

    R['gear'] = gear

    # ============================================================
    # REGISTRA SOLO SE IN PISTA (filtro qualità dato)
    # ============================================================
    if is_on_track:
        row_data = {
            'time': time.time() - t0,
            'steer': R['steer'],
            'accel': R['accel'],
            'brake': R['brake'],
            'gear': R['gear'],
            'speedX': speed_kmh,
            'speedY': speed_y,
            'speedZ': speed_z,
            'trackPos': track_pos,
            'angle': angle,
            'rpm': rpm,
            'wheelSpinAvg': wheel_spin_avg,
            'damage': max(0, current_damage),
        }

        for i in range(19):
            row_data[f'track_{i}'] = trk[i]

        session_data.append(row_data)
        
    bot_state['prev_steer'] = R['steer']
    return is_on_track


if __name__ == "__main__":
    C = snakeoil3.Client(p=3001, vision=False)
    session_data = []
    bot_state = {'shift_cooldown': 0, 'prev_steer': 0.0, 'stuck_frames': 0, 'offtrack_frames': 0}
    last_autosave = 0
    AUTOSAVE_EVERY = 50000  # Salva automaticamente ogni 50.000 frame

    print("\n🤖 Bot di raccolta dati avviato. In attesa di TORCS...")
    print(f"   💾 Autosave ogni {AUTOSAVE_EVERY} frame per non perdere dati.")
    C.get_servers_input()
    t0 = time.time()
    start_damage = C.S.d.get('damage', 0) if C.S.d else 0

    def append_to_master(data):
        """Salva i dati accumulati nel CSV master (append)."""
        if not data:
            return
        df = pd.DataFrame(data)
        header_flag = not os.path.exists(master_file)
        df.to_csv(master_file, mode='a', header=header_flag, index=False)
        print(f"\n   💾 Autosave: {len(data)} frame scritti su disco. Totale: {sum(1 for _ in open(master_file)) - 1} righe.")

    try:
        while True:
            try:
                drive_and_record(C, session_data, t0, start_damage, bot_state)
                C.respond_to_server()
                C.get_servers_input()
            except RaceFinished:
                print("\n🏁 Gara terminata dal server.")
                break
            except BotStuck as e:
                # Auto-reset: chiudi la connessione e aspetta che TORCS riparta il giro
                print(f"\n⚠️  {e} — Auto-reset! Aspetto riavvio gara TORCS...")
                try:
                    C.R.d['meta'] = 1
                    C.respond_to_server()
                    C.shutdown()
                except:
                    pass
                time.sleep(3)
                # Riconnetti
                try:
                    C = snakeoil3.Client(p=3001, vision=False)
                    C.get_servers_input()
                    bot_state = {'shift_cooldown': 0, 'prev_steer': 0.0, 'stuck_frames': 0, 'offtrack_frames': 0}
                    t0 = time.time()
                    start_damage = C.S.d.get('damage', 0) if C.S.d else 0
                    print("✅ Riconnesso! Raccolta dati ripresa.")
                except Exception as e2:
                    print(f"❌ Impossibile riconnettersi: {e2}. Premi INVIO dopo aver riavviato TORCS.")
                    input()
                continue
            except Exception as e:
                print(f"\n⚠️ Errore generico: {e}")
                break

            if len(session_data) % 10 == 0 and session_data:
                last = session_data[-1]
                on_track_str = "✅" if abs(last['trackPos']) < 0.8 else "⚠️ bordo"
                print(f"\r[Bot] Frame: {len(session_data)} | V: {last['speedX']:3.0f} km/h | M: {last['gear']} | Accel: {last['accel']:.2f} | Brake: {last['brake']:.2f} | Pos: {last['trackPos']:+.2f} {on_track_str}", end="")

            # Autosave progressivo
            if len(session_data) - last_autosave >= AUTOSAVE_EVERY:
                nuovi = session_data[last_autosave:]
                append_to_master(nuovi)
                last_autosave = len(session_data)

    except KeyboardInterrupt:
        print("\n⏹ Interruzione manuale.")
    finally:
        try:
            C.R.d.update({'steer': 0.0, 'accel': 0.0, 'brake': 1.0, 'gear': 0})
            C.respond_to_server()
            C.shutdown()
        except:
            pass

    print("\n" + "=" * 50)
    print(f"Frame raccolti in sessione: {len(session_data)}")
    print("=" * 50)

    # Salva i frame rimanenti non ancora scritti su disco
    rimanenti = session_data[last_autosave:]
    if len(rimanenti) >= 100:
        while True:
            choice = input(f"\nVuoi salvare i {len(rimanenti)} frame rimanenti? [y/n]: ").strip().lower()
            if choice == 'y':
                append_to_master(rimanenti)
                print("-> SUCCESS!")
                break
            elif choice == 'n':
                print("-> Scartati.")
                break
    elif last_autosave > 0:
        print(f"-> Tutti i dati già salvati automaticamente ({last_autosave} frame totali).")
    else:
        print("Sessione troppo breve. Dati scartati.")

