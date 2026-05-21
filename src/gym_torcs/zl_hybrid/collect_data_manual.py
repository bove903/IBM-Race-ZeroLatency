import os
import sys
import time
from pynput.keyboard import Key, Listener
import pandas as pd
import numpy as np

# ==========================================
# GESTIONE PERCORSI (Dipendenze e Salvataggio)
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import snakeoil3_gym as snakeoil3

dataset_dir = os.path.join(parent_dir, "dataset")
os.makedirs(dataset_dir, exist_ok=True)
master_file = os.path.join(dataset_dir, "master_dataset.csv")


class RaceFinished(Exception): pass


# ==========================================
# SMART CONTROLLER (INTATTO)
# ==========================================
class SmartController:
    def __init__(self):
        self.keys = set()
        self.state = {'steer': 0.0, 'accel': 0.0, 'brake': 0.0, 'gear': 1}
        self.steer_speed = 0.05
        self.return_speed = 0.10
        self.exit_flag = False
        self.shift_cooldown = 0

        self.listener = Listener(on_press=self.press, on_release=self.release)
        self.listener.start()

    def press(self, key):
        self.keys.add(key)
        if key == Key.esc:
            self.exit_flag = True

    def release(self, key):
        self.keys.discard(key)

    def update(self, S):
        speed = S.get('speedX', 0)
        rpm = S.get('rpm', 0)

        # Telemetria Gomme (TCS)
        wheels = S.get('wheelSpinVel', [0, 0, 0, 0])
        front_speed = wheels[0] + wheels[1]
        rear_speed = wheels[2] + wheels[3]
        wheel_slip = rear_speed - front_speed

        if self.shift_cooldown > 0:
            self.shift_cooldown -= 1

        # Cambio F1/IBM Mode
        if speed < 20:
            self.state['gear'] = 1
        elif self.shift_cooldown == 0:
            if self.state['gear'] < 6 and rpm > 16000:
                self.state['gear'] += 1
                self.shift_cooldown = 5
            elif self.state['gear'] > 1 and rpm < 7000:
                self.state['gear'] -= 1
                self.shift_cooldown = 5

        # Elettronica TCS e ABS
        raw_accel = 1.0 if Key.up in self.keys else 0.0
        raw_brake = 1.0 if Key.down in self.keys else 0.0

        steer_magnitude = abs(self.state['steer'])
        max_accel = 1.0
        max_brake = 1.0

        if self.state['gear'] <= 3 and speed < 100:
            max_accel = max(0.1, 1.0 - (steer_magnitude * 2.5))

        if wheel_slip > 15.0 and raw_accel > 0:
            max_accel = 0.0

        # ABS: Evita il bloccaggio ruote anteriori (sottosterzo) in curva
        if steer_magnitude > 0.05:
            max_brake = max(0.05, 1.0 - (steer_magnitude * 3.0))

        target_accel = raw_accel * max_accel
        target_brake = raw_brake * max_brake

        self.state['accel'] += (target_accel - self.state['accel']) * 0.2
        self.state['brake'] += (target_brake - self.state['brake']) * 0.3

        if speed < 60:
            steer_limit = 0.35
        else:
            steer_limit = max(0.2, 1.0 - (speed / 180.0))

        if Key.left in self.keys:
            self.state['steer'] += self.steer_speed
        elif Key.right in self.keys:
            self.state['steer'] -= self.steer_speed
        else:
            if self.state['steer'] > 0.05:
                self.state['steer'] -= self.return_speed
            elif self.state['steer'] < -0.05:
                self.state['steer'] += self.return_speed
            else:
                self.state['steer'] = 0.0

        self.state['steer'] = max(-steer_limit, min(steer_limit, self.state['steer']))


def main():
    print("=== Avvio Sistema di Raccolta Dati VISIVI ===")
    print("Guida con le Frecce. Il cambio marce è AUTOMATICO.")

    client = snakeoil3.Client(p=3001, vision=False)

    def disarmed_shutdown():
        try:
            client.sock.close()
        except:
            pass
        raise RaceFinished()

    client.shutdown = disarmed_shutdown

    controller = SmartController()
    client.get_servers_input()

    session_data = []
    t0 = time.time()
    damages_taken = 0
    start_damage = None

    try:
        while not controller.exit_flag:
            S = client.S.d
            if not S: break

            if start_damage is None:
                start_damage = S.get('damage', 0)
            current_damage = S.get('damage', 0) - start_damage
            damages_taken = max(0, current_damage)

            controller.update(S)
            a = controller.state

            if S.get('lastLapTime', 0) > 0:
                print("\n🏁 TRAGUARDO RAGGIUNTO!")
                break

            client.R.d['steer'] = a['steer']
            client.R.d['accel'] = a['accel']
            client.R.d['brake'] = a['brake']
            client.R.d['gear'] = a['gear']
            client.respond_to_server()
            client.get_servers_input()

            trk = S.get('track', [100.0] * 19)
            if len(trk) < 19:
                trk += [100.0] * (19 - len(trk))

            # Registriamo solo se l'auto è in movimento
            if S.get('speedX', 0) > 1.0 or a['accel'] > 0.1:
                wheels = S.get('wheelSpinVel', [0, 0, 0, 0])
                wheel_spin_avg = float(np.mean(wheels))

                row_data = {
                    'time': time.time() - t0,
                    'steer': a['steer'],
                    'accel': a['accel'],
                    'brake': a['brake'],
                    'gear': a['gear'],
                    'speedX': S.get('speedX', 0.0),
                    'speedY': S.get('speedY', 0.0),
                    'speedZ': S.get('speedZ', 0.0),
                    'trackPos': S.get('trackPos', 0.0),
                    'angle': S.get('angle', 0.0),
                    'rpm': S.get('rpm', 0.0),
                    'wheelSpinAvg': wheel_spin_avg,
                    'damage': current_damage,
                }

                for i in range(19):
                    row_data[f'track_{i}'] = trk[i]

                session_data.append(row_data)

                if len(session_data) % 10 == 0:
                    print(f"\r[Registrazione] Buffer: {len(session_data)} tick | V: {S.get('speedX', 0):.0f} km/h",
                          end="")

    except RaceFinished:
        print("\n🏁 Il gioco ha chiuso la gara. Chiusura gestita con successo.")
    except KeyboardInterrupt:
        print("\n⏹ Interruzione manuale.")
    finally:
        try:
            client.R.d.update({'steer': 0.0, 'accel': 0.0, 'brake': 1.0, 'gear': 0})
            client.respond_to_server()
        except:
            pass

    print("\n" + "=" * 50)
    print("SESSIONE TERMINATA.")
    print(f"Frame raccolti in memoria : {len(session_data)}")
    print(f"Danni subiti in questo run: {damages_taken}")
    print("=" * 50)

    if len(session_data) < 100:
        print("Sessione troppo breve. Dati scartati in automatico.")
        return

    while True:
        choice = input(f"Vuoi salvare i dati nel Master Dataset ({master_file})? [y/n]: ").strip().lower()
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


if __name__ == "__main__":
    main()