from pynput.keyboard import Key, Listener
import snakeoil3_jm2 as snakeoil3
import time
import json


class ArcadeController:
    def __init__(self):
        self.keys = set()

        self.state = {
            'steer': 0.0,
            'accel': 0.0,
            'brake': 0.0,
            'gear': 1
        }

        self.listener = Listener(on_press=self.press, on_release=self.release)
        self.listener.start()

    def press(self, key):
        self.keys.add(key)

        if hasattr(key, "char"):
            if key.char == 'w':
                self.state['gear'] += 1
            elif key.char == 's':
                self.state['gear'] -= 1

    def release(self, key):
        self.keys.discard(key)

    def update(self, sensors):
        speed = sensors.get('speedX', 0)
        angle = sensors.get('angle', 0)

        # ========================
        # ACCELERAZIONE SMOOTH
        # ========================
        target_accel = 1.0 if Key.up in self.keys else 0.0
        self.state['accel'] += (target_accel - self.state['accel']) * 0.1

        # ========================
        # FRENO
        # ========================
        target_brake = 1.0 if Key.down in self.keys else 0.0
        self.state['brake'] += (target_brake - self.state['brake']) * 0.2

        # ========================
        # STEERING INPUT
        # ========================
        # INPUT
        steer_input = 0.0
        if Key.left in self.keys:
            steer_input += 0.6
        if Key.right in self.keys:
            steer_input -= 0.6

        # LIMITE VELOCITÀ
        max_steer = max(0.25, 1.0 - speed / 200.0)
        steer_input *= max_steer

        # SE NON STAI STERZANDO → VAI DRITTO
        if abs(steer_input) < 0.01:
            steer_target = 0.0
        else:
            stability = angle * 0.3
            steer_target = steer_input - stability

        # SMOOTH
        self.state['steer'] += (steer_target - self.state['steer']) * 0.2

        # DEAD ZONE
        if abs(self.state['steer']) < 0.02:
            self.state['steer'] = 0.0

        # clamp
        self.state['steer'] = max(-1.0, min(1.0, self.state['steer']))
        self.state['accel'] = max(0.0, min(1.0, self.state['accel']))
        self.state['brake'] = max(0.0, min(1.0, self.state['brake']))

        # marce
        self.state['gear'] = max(-1, min(6, self.state['gear']))


# ============================================================
# MAIN
# ============================================================

def main():
    client = snakeoil3.Client(p=3001, vision=False)
    controller = ArcadeController()

    client.get_servers_input()

    print("Arcade driving mode attivo")
    print("Freccie per guidare, W/S per marce")


    # CSV log
    log_csv = open("manual_log.csv", "w")
    log_csv.write("time,steer,accel,brake,gear,speedX,trackPos,angle,rpm,damage\n")

    # JSON log (step-by-step strutturato)
    log_json = []
    
    t0 = time.time()
    step = 0

    while True:
        S = client.S.d

        controller.update(S)
        a = controller.state
        
        print(f"steer={a['steer']:.2f} accel={a['accel']:.2f} brake={a['brake']:.2f} gear={a['gear']}")

        client.R.d['steer'] = a['steer']
        client.R.d['accel'] = a['accel']
        client.R.d['brake'] = a['brake']
        client.R.d['gear'] = a['gear']
        client.R.d['clutch'] = 0.0
        client.R.d['meta'] = 0

        client.respond_to_server()
        client.get_servers_input()

        #S = client.S.d

        current_time = time.time() - t0

        # ===== CSV LOG =====
        log_csv.write(
            f"{current_time},{a['steer']},{a['accel']},{a['brake']},{a['gear']},"
            f"{S.get('speedX',0)},{S.get('trackPos',0)},{S.get('angle',0)},"
            f"{S.get('rpm',0)},{S.get('damage',0)}\n"
        )

        # ===== JSON LOG (STEP-BY-STEP) =====
        log_json.append({
            "step": step,
            "time": current_time,
            "action": {
                "steer": a['steer'],
                "accel": a['accel'],
                "brake": a['brake'],
                "gear": a['gear']
            },
            "state": {
                "speedX": S.get('speedX', 0),
                "trackPos": S.get('trackPos', 0),
                "angle": S.get('angle', 0),
                "rpm": S.get('rpm', 0),
                "damage": S.get('damage', 0)
            }
        })

        step += 1

        # salva JSON ogni tot step (evita perdita dati)
        if step % 100 == 0:
            with open("manual_log.json", "w") as f:
                json.dump(log_json, f, indent=2)
                
                
        time.sleep(0.02)


if __name__ == "__main__":
    main()