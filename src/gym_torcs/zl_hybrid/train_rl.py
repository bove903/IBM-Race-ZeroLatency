import os
import sys
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pynput.keyboard import Controller

# ==========================================
# GESTIONE PERCORSI E DIPENDENZE
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import snakeoil3_gym as snakeoil3
from train_bc import ActorCritic, INPUT_DIM

# ==========================================
# IPERPARAMETRI RL v5.0 — CURRICULUM LEARNING
# ==========================================
# PPO Core
GAMMA = 0.99
LAMBDA_GAE = 0.95
CLIP_EPS = 0.1
FRAME_SKIP = 4
MAX_EPISODES = 2000          # Più episodi per dare tempo al curriculum
MAX_STEPS = 4000             # Giro completo richiede ~800 step, margine per errori
PPO_EPOCHS = 3
MINI_BATCH_SIZE = 64         # Batch più piccoli = update più frequenti
MAX_GRAD_NORM = 0.5

# Learning Rate
LR_ACTOR = 1e-5              # Leggermente più alto per imparare più in fretta
LR_CRITIC = 3e-4

# Esplorazione ALTA: l'AI deve poter provare a staccare più tardi o accelerare di più
INITIAL_LOG_STD = -2.3       # std ≈ 0.10 (10% di esplorazione)
FINAL_LOG_STD = -4.0         # std ≈ 0.018

# BC Anchor: DEBOLE per permettere di slegarsi dal 1:54
BC_ANCHOR_WEIGHT = 0.4
BC_ANCHOR_MIN_PHASE1 = 0.2   
BC_ANCHOR_MIN_PHASE2 = 0.05  # A fine addestramento l'ancora sarà quasi inesistente
BC_ANCHOR_DECAY = 0.95       # Decadimento rapido

# Warm-up (Rapido)
WARMUP_EPISODES = 2

# Anti-camping
MIN_SPEED_THRESHOLD = 5.0
CAMPING_STEP_LIMIT = 50

# Curriculum: soglia per passare da Fase 1 a Fase 2
PHASE2_THRESHOLD_LAPS = 0    # L'agente sa già guidare, partiamo SUBITO con Fase 2 (Velocità)

# Paths
models_dir = os.path.join(parent_dir, "models")
bc_model_path = os.path.join(models_dir, "bc_model.pth")
ppo_model_path = os.path.join(models_dir, "ppo_model.pth")
ppo_latest_path = os.path.join(models_dir, "ppo_latest.pth")
feat_mean_path = os.path.join(models_dir, "feat_mean.npy")
feat_std_path = os.path.join(models_dir, "feat_std.npy")


def extract_state(S, feat_mean, feat_std):
    """Estrae e normalizza lo stato — Schema a 26 feature."""
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

    raw = np.array(raw, dtype=np.float32)
    return (raw - feat_mean) / feat_std


def get_log_std(episode, max_episodes):
    """Schedule di esplorazione con warm-up iniziale."""
    if episode <= WARMUP_EPISODES:
        return -7.0  # Quasi deterministico nel warm-up
    effective_ep = episode - WARMUP_EPISODES
    effective_max = max_episodes - WARMUP_EPISODES
    progress = min(1.0, effective_ep / max(1, effective_max * 0.5))
    return INITIAL_LOG_STD + (FINAL_LOG_STD - INITIAL_LOG_STD) * progress


def get_reward_phase1(S, prev_action=None):
    """
    FASE 1: SOPRAVVIVENZA — Impara a completare il giro.
    
    Priorità:
    1. Stare in pista (al centro)
    2. Andare avanti (qualsiasi velocità va bene)
    3. NON uscire di pista
    
    Il bonus velocità è MINIMO: l'importante è sopravvivere.
    """
    speedX = S.get('speedX', 0.0)
    angle = S.get('angle', 0.0)
    trackPos = S.get('trackPos', 0.0)
    lastLapTime = S.get('lastLapTime', 0.0)
    distFromStart = S.get('distFromStart', 0.0)

    # 1. REWARD BASE: progresso in avanti (qualsiasi velocità positiva)
    forward_speed = speedX * np.cos(angle)
    reward = max(0.0, forward_speed) * 0.5  # Peso dimezzato rispetto a prima

    # 2. BONUS CENTRATURA — il premio più alto è stare al centro
    abs_tp = abs(trackPos)
    if abs_tp < 0.3:
        reward += 2.0       # Premio per stare centrato
    elif abs_tp < 0.6:
        reward += 1.0
    elif abs_tp < 0.85:
        reward -= abs_tp * 5.0
    elif abs_tp < 1.0:
        reward -= abs_tp ** 2 * 40.0   # Penalità forte al bordo
    else:
        reward -= 200.0                 # Quasi fuori!

    # 3. PENALITÀ angolo — stai dritto!
    reward -= (np.sin(angle) ** 2) * 30.0

    # 4. PENALITÀ sterzate brusche
    if prev_action is not None:
        steer_mag = abs(prev_action[0])
        if steer_mag > 0.4:
            reward -= steer_mag * 5.0

    done = False

    # TERMINAZIONE: fuoripista
    if abs(trackPos) > 1.1:
        # Penalità proporzionale alla velocità (ma moderata in fase 1)
        crash_speed = max(0, speedX)
        reward = -(500.0 + crash_speed * 10.0)
        done = True

    # Retromarcia
    if speedX < -10.0:
        reward = -500.0
        done = True

    # VITTORIA: giro completato!!! Premio ENORME in fase 1
    if lastLapTime > 0:
        reward = 50000.0  # Premio colossale per aver completato il giro
        done = True
        print(f"\n🏆🏆🏆 GIRO COMPLETATO! Tempo: {lastLapTime:.2f}s 🏆🏆🏆")

    return reward / 100.0, done


def get_reward_phase2(S, prev_action=None):
    """
    FASE 2: VELOCITÀ — Reward super-lineare.
    """
    speedX = S.get('speedX', 0.0)
    angle = S.get('angle', 0.0)
    trackPos = S.get('trackPos', 0.0)
    lastLapTime = S.get('lastLapTime', 0.0)

    # 1. IL SEGRETO: Reward Super-Lineare
    # Invece di usare una penalità di tempo (che porta l'agente a suicidarsi),
    # premiamo il quadrato della velocità.
    # Dato che Distanza = Velocità * Tempo, usare Velocità^2 fa sì che 
    # completare la stessa distanza a 200km/h dia il DOPPIO dei punti totali
    # rispetto a completarla a 100km/h. Addio "crawling"!
    
    forward_speed = speedX * np.cos(angle)
    v_norm = max(0.0, forward_speed) / 100.0  # 1.0 a 100km/h, 2.0 a 200km/h
    
    reward = (v_norm ** 2) * 5.0  

    # 2. PENALITÀ Posizione (Ammorbidita per permettere i "tagli" sui cordoli)
    abs_tp = abs(trackPos)
    if abs_tp > 1.0:
        reward -= (abs_tp ** 2) * 5.0  # Penalizza solo se davvero oltre il cordolo

    # 3. PENALITÀ sterzate brusche
    if prev_action is not None:
        steer_mag = abs(prev_action[0])
        if steer_mag > 0.3:
            reward -= steer_mag * 2.0

    done = False

    # TERMINAZIONE
    if abs(trackPos) > 1.15:
        # Penalità fissa per il crash. L'agente capirà che schiantarsi 
        # non conviene perché si perde l'opportunità di accumulare altri punti V^2.
        reward = -100.0
        done = True

    if speedX < -10.0:
        reward = -100.0
        done = True

    # VITTORIA
    if lastLapTime > 0:
        time_bonus = max(0, 150 - lastLapTime) * 10.0
        reward = 500.0 + time_bonus
        done = True
        print(f"\n🏆 GIRO COMPLETATO! Tempo: {lastLapTime:.2f}s (Bonus tempo: {time_bonus:.0f})")

    return reward / 100.0, done


def compute_gae(rewards, values, gamma=GAMMA, lam=LAMBDA_GAE):
    """Calcola Generalized Advantage Estimation."""
    advantages = []
    gae = 0.0
    next_value = 0.0

    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * next_value - values[i]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
        next_value = values[i]

    advantages = torch.tensor(advantages, dtype=torch.float32)
    returns = advantages + torch.tensor(values, dtype=torch.float32)
    if len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages, returns


def check_model_health(model, feat_mean, feat_std):
    """
    Verifica che il modello non sia corrotto.
    Simula uno scenario da fermo in rettilineo: deve accelerare.
    """
    model.eval()
    raw = np.array([0.0, 0.0, 0.0, 1000, 0, 0, 0] + [200.0]*19, dtype=np.float32)
    x = torch.tensor((raw - feat_mean) / feat_std).unsqueeze(0)
    with torch.no_grad():
        actions, _ = model(x)
    accel = actions[0, 1].item()
    model.train()
    return accel > 0.2


def main():
    print("🚀 Addestramento RL v5.0 — CURRICULUM LEARNING")
    print(f"   📚 Fase 1: Impara a completare il giro (sopravvivenza)")
    print(f"   🏎️  Fase 2: Impara ad andare veloce (dopo {PHASE2_THRESHOLD_LAPS} giri completati)")
    print(f"   Frame Skip: {FRAME_SKIP} | LR Actor: {LR_ACTOR} | LR Critic: {LR_CRITIC}")
    print(f"   Esplorazione: std {np.exp(INITIAL_LOG_STD):.3f} → {np.exp(FINAL_LOG_STD):.3f}")
    print(f"   BC Anchor: {BC_ANCHOR_WEIGHT} (min Fase1: {BC_ANCHOR_MIN_PHASE1}, min Fase2: {BC_ANCHOR_MIN_PHASE2})")

    # Carica il modello BC originale come ANCORA
    bc_anchor = ActorCritic()
    if not os.path.exists(bc_model_path):
        print("❌ Errore: Nessun modello BC trovato. Esegui prima train_bc.py.")
        sys.exit(1)
    bc_anchor.load_state_dict(torch.load(bc_model_path))
    bc_anchor.eval()
    for p in bc_anchor.parameters():
        p.requires_grad = False
    print("⚓ BC Anchor caricato (riferimento fisso)")

    # Carica il modello attivo
    model = ActorCritic()
    if os.path.exists(ppo_model_path):
        model.load_state_dict(torch.load(ppo_model_path))
        print("🧠 Cervello PPO caricato. Ripresa addestramento.")
    else:
        model.load_state_dict(torch.load(bc_model_path))
        print("🧠 Cervello BC clonato come punto di partenza.")

    feat_mean = np.load(feat_mean_path)
    feat_std = np.load(feat_std_path)

    # Health check all'avvio
    if not check_model_health(model, feat_mean, feat_std):
        print("⚠️  MODELLO CORROTTO! Reset al BC...")
        model.load_state_dict(torch.load(bc_model_path))
        if os.path.exists(ppo_model_path):
            os.remove(ppo_model_path)
        print("✅ Reset completato.")

    # Optimizer
    actor_params = []
    critic_params = []
    for name, param in model.named_parameters():
        if 'critic' in name:
            critic_params.append(param)
        else:
            actor_params.append(param)

    optimizer = optim.Adam([
        {'params': actor_params, 'lr': LR_ACTOR},
        {'params': critic_params, 'lr': LR_CRITIC},
    ])

    # Stato addestramento
    best_reward = -float('inf')
    reward_history = []
    bc_anchor_weight = BC_ANCHOR_WEIGHT
    consecutive_bad = 0
    torcs_just_restarted = True

    # CURRICULUM STATE
    completed_laps = 0
    current_phase = 2 if PHASE2_THRESHOLD_LAPS == 0 else 1
    best_lap_time = float('inf')

    print(f"\n{'='*60}")
    if current_phase == 1:
        print(f"  📚 FASE 1 ATTIVA — Obiettivo: completare {PHASE2_THRESHOLD_LAPS} giri")
    else:
        print(f"  🏎️ FASE 2 ATTIVA — Velocità Massima!")
    print(f"{'='*60}")

    for episode in range(1, MAX_EPISODES + 1):
        print(f"\n{'='*60}")
        is_warmup = episode <= WARMUP_EPISODES

        if is_warmup:
            phase_str = "🔵 WARM-UP"
        elif current_phase == 1:
            phase_str = f"📚 FASE 1 (Giri completati: {completed_laps}/{PHASE2_THRESHOLD_LAPS})"
        else:
            phase_str = f"🏎️  FASE 2 (Best: {best_lap_time:.1f}s)"

        print(f"--- Episodio {episode}/{MAX_EPISODES} [{phase_str}] ---")

        current_log_std = get_log_std(episode, MAX_EPISODES)
        current_std = np.exp(current_log_std)
        print(f"    std={current_std:.4f} | BC anchor={bc_anchor_weight:.3f}")

        # Connessione a TORCS
        try:
            client = snakeoil3.Client(p=3001, vision=False)
            client.get_servers_input()
        except Exception as e:
            print(f"   ⚠️ TORCS non raggiungibile: {e}")
            print("   Riavvia TORCS, clicca sulla sua finestra, poi premi INVIO...")
            input()
            torcs_just_restarted = True
            continue

        # Accelerazione TORCS
        if torcs_just_restarted:
            print("\n⏳ PREPARAZIONE...")
            time.sleep(1)
            torcs_just_restarted = False

        keyboard = Controller()
        keyboard.press('+')
        keyboard.release('+')
        time.sleep(0.1)

        # Variabili episodio
        states, actions, log_probs, rewards, values_list = [], [], [], [], []
        done = False
        step = 0
        total_episode_reward = 0.0
        prev_action_np = None
        shift_cooldown = 0
        max_speed_ep = 0.0
        camping_counter = 0
        max_distance = 0.0

        while not done and step < MAX_STEPS:
            if client.so is None:
                break

            try:
                client.get_servers_input()
            except Exception:
                break

            S = client.S.d
            if not S or 'track' not in S:
                break

            s_np = extract_state(S, feat_mean, feat_std)
            s_tensor = torch.tensor(s_np, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                act_means, value = model(s_tensor)
                std = torch.exp(torch.ones_like(act_means) * current_log_std)
                dist = torch.distributions.Normal(act_means, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)

            raw_action = action[0].numpy().copy()

            # Smoothing sterzo
            smoothed = raw_action.copy()
            if prev_action_np is not None:
                smoothed[0] = prev_action_np[0] * 0.3 + raw_action[0] * 0.7

            steer, accel, brake = smoothed

            # ABS dolce
            steer_mag = abs(steer)
            if steer_mag > 0.1:
                max_brake = max(0.25, 1.0 - (steer_mag * 1.2))
                brake = min(brake, max_brake)
                smoothed[2] = brake

            # Esecuzione con frame skip
            reward_sum = 0.0
            for _ in range(FRAME_SKIP):
                client.R.d['steer'] = float(np.clip(steer, -1.0, 1.0))
                client.R.d['accel'] = float(np.clip(accel, 0.0, 1.0))
                client.R.d['brake'] = float(np.clip(brake, 0.0, 1.0))
                client.R.d['meta'] = 0

                rpm = S.get('rpm', 0)
                gear = S.get('gear', 1)
                speed_kmh = S.get('speedX', 0)
                max_speed_ep = max(max_speed_ep, speed_kmh)

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

                try:
                    client.respond_to_server()
                    client.get_servers_input()
                except Exception:
                    done = True
                    break

                S_next = client.S.d
                if not S_next:
                    done = True
                    break

                # Scegli la reward function in base alla fase
                if current_phase == 1:
                    r, done_flag = get_reward_phase1(S_next, smoothed)
                else:
                    r, done_flag = get_reward_phase2(S_next, smoothed)

                # Traccia distanza percorsa
                dist_now = S_next.get('distFromStart', 0.0)
                max_distance = max(max_distance, dist_now)

                # Anti-camping
                if speed_kmh < MIN_SPEED_THRESHOLD and step > 50:
                    camping_counter += 1
                    if camping_counter > CAMPING_STEP_LIMIT:
                        r = -100.0
                        done_flag = True
                        print(f"\n   🐢 Auto ferma! Reset.")
                else:
                    camping_counter = 0

                reward_sum += r

                if done_flag:
                    done = True
                    tp = S_next.get('trackPos', 0)
                    lastLapTime = S_next.get('lastLapTime', 0.0)

                    if lastLapTime > 0:
                        # GIRO COMPLETATO!
                        completed_laps += 1
                        if lastLapTime < best_lap_time:
                            best_lap_time = lastLapTime

                        # Check promozione a Fase 2
                        if current_phase == 1 and completed_laps >= PHASE2_THRESHOLD_LAPS:
                            current_phase = 2
                            print(f"\n{'='*60}")
                            print(f"  🎓 PROMOZIONE A FASE 2! 🏎️")
                            print(f"  Il pilota sa completare il giro. Ora spingiamo sulla velocità!")
                            print(f"  Miglior tempo finora: {best_lap_time:.2f}s")
                            print(f"{'='*60}")

                    elif abs(tp) > 1.1:
                        print(f"\n   💥 Fuoripista a {speed_kmh:.0f} km/h (trackPos: {tp:.2f}) | Distanza: {max_distance:.0f}m")

                    client.R.d['meta'] = 1
                    try:
                        client.respond_to_server()
                    except:
                        pass
                    break

                S = S_next

            prev_action_np = smoothed.copy()
            total_episode_reward += reward_sum

            states.append(s_tensor)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward_sum)
            values_list.append(value.item())

            step += 1
            if step % 200 == 0:
                tp_now = S.get('trackPos', 0)
                print(f"  Step {step:4d} | V: {speed_kmh:3.0f} km/h | Pos: {tp_now:+.2f} | R: {total_episode_reward:.1f} | Dist: {max_distance:.0f}m")

        try:
            client.shutdown()
        except:
            pass

        # === AGGIORNAMENTO PPO ===
        if len(rewards) > 10:
            advantages, returns = compute_gae(rewards, values_list)

            s_batch = torch.cat(states)
            a_batch = torch.cat(actions)
            old_log_probs = torch.cat(log_probs).detach()
            returns_tensor = returns.unsqueeze(1)
            adv_tensor = advantages.unsqueeze(1)
            n_samples = len(states)

            if is_warmup:
                print(f"🔵 Warm-up: solo Critic su {n_samples} frame...")
                for _ in range(PPO_EPOCHS):
                    _, new_values = model(s_batch)
                    critic_loss = nn.MSELoss()(new_values, returns_tensor)
                    optimizer.zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                    optimizer.step()
            else:
                print(f"🔄 PPO su {n_samples} frame | BC anchor={bc_anchor_weight:.3f}...")

                for ppo_epoch in range(PPO_EPOCHS):
                    indices = np.random.permutation(n_samples)

                    for start in range(0, n_samples, MINI_BATCH_SIZE):
                        end = min(start + MINI_BATCH_SIZE, n_samples)
                        idx = indices[start:end]

                        mb_states = s_batch[idx]
                        mb_actions = a_batch[idx]
                        mb_old_lp = old_log_probs[idx]
                        mb_returns = returns_tensor[idx]
                        mb_advs = adv_tensor[idx]

                        new_means, new_values = model(mb_states)
                        std = torch.exp(torch.ones_like(new_means) * current_log_std)
                        dist = torch.distributions.Normal(new_means, std)
                        new_log_probs = dist.log_prob(mb_actions).sum(dim=-1)

                        # PPO Loss
                        ratio = torch.exp(new_log_probs - mb_old_lp)
                        surr1 = ratio * mb_advs.squeeze()
                        surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * mb_advs.squeeze()
                        actor_loss = -torch.min(surr1, surr2).mean()

                        critic_loss = nn.MSELoss()(new_values, mb_returns)
                        entropy = dist.entropy().mean()

                        # BC Anchor Loss
                        with torch.no_grad():
                            bc_means, _ = bc_anchor(mb_states)
                        bc_loss = nn.MSELoss()(new_means, bc_means)

                        loss = (actor_loss
                                + 0.5 * critic_loss
                                - 0.005 * entropy
                                + bc_anchor_weight * bc_loss)

                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                        optimizer.step()

                # Decay BC anchor con floor diverso per fase
                if current_phase == 1:
                    bc_anchor_weight = max(BC_ANCHOR_MIN_PHASE1, bc_anchor_weight * BC_ANCHOR_DECAY)
                else:
                    bc_anchor_weight = max(BC_ANCHOR_MIN_PHASE2, bc_anchor_weight * BC_ANCHOR_DECAY)

            # Health check
            if not check_model_health(model, feat_mean, feat_std):
                print("⚠️  CORRUZIONE! Rollback al BC...")
                model.load_state_dict(torch.load(bc_model_path))
                bc_anchor_weight = BC_ANCHOR_WEIGHT
                consecutive_bad = 0
                continue

            # Statistiche
            reward_history.append(total_episode_reward)
            avg_recent = np.mean(reward_history[-20:])

            info = f"Steps: {step} | Vmax: {max_speed_ep:.0f} km/h | Dist: {max_distance:.0f}m"

            # Tracking bad episodes
            if total_episode_reward < -50:
                consecutive_bad += 1
            else:
                consecutive_bad = 0

            if consecutive_bad >= 30:
                print("🔴 30 episodi pessimi! L'esplorazione si è spinta troppo oltre. Reset al BC...")
                model.load_state_dict(torch.load(bc_model_path))
                bc_anchor_weight = BC_ANCHOR_WEIGHT
                consecutive_bad = 0
                continue

            # Salvataggio intelligente
            if total_episode_reward > best_reward:
                best_reward = total_episode_reward
                torch.save(model.state_dict(), ppo_model_path)
                print(f"💾 🌟 RECORD! ppo_model.pth | R: {total_episode_reward:.1f} | Media: {avg_recent:.1f} | {info}")
            else:
                torch.save(model.state_dict(), ppo_latest_path)
                print(f"💾 ppo_latest.pth | R: {total_episode_reward:.1f} | Media: {avg_recent:.1f} | Best: {best_reward:.1f} | {info}")

            time.sleep(0.1)


if __name__ == "__main__":
    main()