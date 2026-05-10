from pynput.keyboard import Key, Listener
import snakeoil3_jm2 as snakeoil3
import time
import json

class OptimizedArcadeController:
    def __init__(self):
        self.keys = set()

        self.state = {
            'steer': 0.0,
            'accel': 0.0,
            'brake': 0.0,
            'gear': 1
        }

        # Parametri di smoothing
        self.steer_speed_in = 0.06   # Velocità di sterzata
        self.steer_speed_out = 0.12  # Velocità di ritorno al centro (più veloce)
        self.accel_speed = 0.15
        self.brake_speed = 0.25

        self.listener = Listener(on_press=self.press, on_release=self.release)
        self.listener.start()

    def press(self, key):
        self.keys.add(key)
        if hasattr(key, "char"):
            if key.char == 'w':
                self.state['gear'] = min(6, self.state['gear'] + 1)
            elif key.char == 's':
                self.state['gear'] = max(-1, self.state['gear'] - 1)

    def release(self, key):
        self.keys.discard(key)

    def update(self, sensors):
        speed = sensors.get('speedX', 0)
        
        # ========================
        # ACCELERATORE & TRACTION CONTROL
        # ========================
        target_accel = 1.0 if Key.up in self.keys else 0.0
        self.state['accel'] += (target_accel - self.state['accel']) * self.accel_speed
        
        # Se le ruote motrici slittano rispetto a quelle anteriori, taglia potenza
        wheel_spin = ((sensors.get('wheelSpinVel', [0]*4)[2] + sensors.get('wheelSpinVel', [0]*4)[3]) - 
                      (sensors.get('wheelSpinVel', [0]*4)[0] + sensors.get('wheelSpinVel', [0]*4)[1]))
        
        if wheel_spin > 2.0 and self.state['accel'] > 0:
            self.state['accel'] -= 0.15 # Intervento TCS

        # ========================
        # FRENO
        # ========================
        target_brake = 1.0 if Key.down in self.keys else 0.0
        self.state['brake'] += (target_brake - self.state['brake']) * self.brake_speed

        # ========================
        # STERZO PROGRESSIVO SPEED-SENSITIVE
        # ========================
        # Calcolo il limite di sterzo: a 150km/h lo sterzo massimo è molto ridotto
        steer_limit = max(0.15, 1.0 - (speed / 150.0))
        
        if Key.left in self.keys:
            self.state['steer'] += self.steer_speed_in
        elif Key.right in self.keys:
            self.state['steer'] -= self.steer_speed_in
        else:
            # Auto-centramento rapido se non premo nulla
            if self.state['steer'] > 0.02:
                self.state['steer'] -= self.steer_speed_out
            elif self.state['steer'] < -0.02:
                self.state['steer'] += self.steer_speed_out
            else:
                self.state['steer'] = 0.0

        # Clamp finale dello sterzo in base alla velocità
        self.state['steer'] = max(-steer_limit, min(steer_limit, self.state['steer']))

        # Clamp di sicurezza generico
        self.state['accel'] = max(0.0, min(1.0, self.state['accel']))
        self.state['brake'] = max(0.0, min(1.0, self.state['brake']))

# ============================================================
# MAIN
# ============================================================

def main():
    client = snakeoil3.Client(p=3001, vision=False)
    controller = OptimizedArcadeController()

    client.get_servers_input()

    print("Arcade driving mode OTTIMIZZATO attivo")
    print("Frecce per guidare, W/S per marce. Timing di rete corretto.")

    # CSV log
    log_csv = open("manual_log.csv", "w")
    log_csv.write("time,steer,accel,brake,gear,speedX,trackPos,angle,rpm,damage\n")

    # JSON log
    log_json = []
    
    t0 = time.time()
    step = 0

    while True:
        # 1. Il loop è dettato dalla ricezione dati dal server (50Hz)
        # NESSUN TIME.SLEEP NECESSARIO
        S = client.S.d
        if not S:
            break # Sicurezza se il server si spegne

        # 2. Aggiorna la logica con i sensori attuali
        controller.update(S)
        a = controller.state
        
        # 3. Prepara e invia la risposta
        client.R.d['steer'] = a['steer']
        client.R.d['accel'] = a['accel']
        client.R.d['brake'] = a['brake']
        client.R.d['gear'] = a['gear']
        client.R.d['clutch'] = 0.0
        client.R.d['meta'] = 0

        client.respond_to_server()
        
        # 4. Chiama il blocco per il prossimo ciclo
        client.get_servers_input()

        # ==========================
        # LOGGING
        # ==========================
        current_time = time.time() - t0

        log_csv.write(
            f"{current_time:.3f},{a['steer']:.3f},{a['accel']:.3f},{a['brake']:.3f},{a['gear']},"
            f"{S.get('speedX',0):.2f},{S.get('trackPos',0):.3f},{S.get('angle',0):.3f},"
            f"{S.get('rpm',0):.0f},{S.get('damage',0)}\n"
        )

        log_json.append({
            "step": step,
            "time": current_time,
            "action": {
                "steer": round(a['steer'], 3),
                "accel": round(a['accel'], 3),
                "brake": round(a['brake'], 3),
                "gear": a['gear']
            },
            "state": {
                "speedX": S.get('speedX', 0),
                "trackPos": S.get('trackPos', 0),
                "angle": S.get('angle', 0),
                "rpm": S.get('rpm', 0)
            }
        })

        step += 1

        if step % 200 == 0:
            with open("manual_log.json", "w") as f:
                json.dump(log_json, f, indent=2)

if __name__ == "__main__":
    main()