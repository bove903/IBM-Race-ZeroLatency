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


if __name__ == "__main__":
    C = snakeoil3.Client(p=3001, vision=False)
    session_data = []
    bot_state = {'shift_cooldown': 0, 'prev_steer': 0.0}

    print("\n🤖 Bot di raccolta dati avviato. In attesa di TORCS...")
    C.get_servers_input()
    t0 = time.time()
    start_damage = C.S.d.get('damage', 0) if C.S.d else 0

    try:
        for step in range(C.maxSteps, 0, -1):
            drive_and_record(C, session_data, t0, start_damage, bot_state)
            C.respond_to_server()
            C.get_servers_input()

            if step % 10 == 0 and len(session_data) > 0:
                print(f"\r[Bot in azione] Buffer: {len(session_data)} tick | V: {session_data[-1]['speedX']:3.0f} km/h | Marcia: {session_data[-1]['gear']}", end="")

        print("\n🏁 Raggiunto il limite massimo di step simulazione.")
    except RaceFinished:
        print("\n🏁 Il gioco ha chiuso la gara (o è stato messo in pausa/restart).")
    except KeyboardInterrupt:
        print("\n⏹ Interruzione manuale da tastiera.")
    finally:
        try:
            C.R.d.update({'steer': 0.0, 'accel': 0.0, 'brake': 1.0, 'gear': 0})
            C.respond_to_server()
            C.shutdown()
        except:
            pass

    print("\n" + "=" * 50)
    print("SESSIONE AUTOMATICA TERMINATA.")
    print(f"Frame raccolti : {len(session_data)}")
    if len(session_data) > 0:
        print(f"Danni subiti: {session_data[-1]['damage']}")
    print("=" * 50)

    if len(session_data) < 100:
        print("Sessione troppo breve. Dati scartati automaticamente.")
        sys.exit()

    while True:
        choice = input(f"Vuoi salvare i dati nel Master Dataset? [y/n]: ").strip().lower()
        if choice == 'y':
            df = pd.DataFrame(session_data)
            header_flag = not os.path.exists(master_file)
            df.to_csv(master_file, mode='a', header=header_flag, index=False)
            print(f"-> SUCCESS: {len(session_data)} frame accodati con successo!")
            break
        elif choice == 'n':
            print("-> Dati scartati. Memoria RAM svuotata.")
            break
        else:
            print("Rispondi con 'y' per sì o 'n' per no.")
