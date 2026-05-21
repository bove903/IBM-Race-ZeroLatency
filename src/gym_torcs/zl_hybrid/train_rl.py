import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

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
# IPERPARAMETRI RL (v3.0 — Fix convergenza)
# ==========================================
GAMMA = 0.99          # Fattore di sconto temporale
LAMBDA_GAE = 0.95     # Lambda per GAE
CLIP_EPS = 0.2        # Clipping PPO
FRAME_SKIP = 4        # Frame skip (era 3, alzato per più stabilità su ovale)
LR = 3e-5             # Learning rate
MAX_EPISODES = 1000   # Episodi massimi
MAX_STEPS = 3000      # Step massimi per episodio
PPO_EPOCHS = 6        # Epoche PPO per episodio
MINI_BATCH_SIZE = 64  # Dimensione mini-batch
MAX_GRAD_NORM = 0.5   # Gradient clipping

# === FIX #1: Esplorazione molto più bassa ===
# Il modello BC guida già bene (loss 0.0006). Non serve esplorare tanto,
# basta fare micro-aggiustamenti. std=0.22 distruggeva il BC!
INITIAL_LOG_STD = -2.5   # std iniziale ≈ 0.08 (era 0.22 — TROPPO ALTO)
FINAL_LOG_STD = -4.0     # std finale ≈ 0.018 (era 0.03)

# === FIX #2: Warm-up senza esplorazione ===
# I primi N episodi usano il modello BC puro (std ≈ 0) per stabilizzare
# il critic e dare episodi lunghi come baseline
WARMUP_EPISODES = 10     # Episodi iniziali senza rumore

# === FIX #3: Smoothing azioni ===
# Mescola l'azione corrente con quella precedente per ridurre jitter
ACTION_SMOOTHING = 0.6   # Aumentato al 60% per eliminare le oscillazioni destra/sinistra

models_dir = os.path.join(parent_dir, "models")
bc_model_path = os.path.join(models_dir, "bc_model.pth")
ppo_model_path = os.path.join(models_dir, "ppo_model.pth")
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
        # Warm-up: quasi zero esplorazione, lascia che il BC guidi pulito
        return -6.0  # std ≈ 0.0025, praticamente deterministico
    # Dopo il warm-up: esplorazione graduale
    effective_ep = episode - WARMUP_EPISODES
    effective_max = max_episodes - WARMUP_EPISODES
    progress = min(1.0, effective_ep / max(1, effective_max * 0.5))
    return INITIAL_LOG_STD + (FINAL_LOG_STD - INITIAL_LOG_STD) * progress


def get_reward_and_done(S, prev_action=None):
    """
    Reward function v3 — Michigan Oval.
    FIX #4: NON termina al fuoripista. Usa penalità progressive crescenti
    così l'agente impara a stare in pista invece di avere episodi cortissimi.
    """
    speedX = S.get('speedX', 0.0)
    angle = S.get('angle', 0.0)
    trackPos = S.get('trackPos', 0.0)
    lastLapTime = S.get('lastLapTime', 0.0)

    # === 1. COMPONENTE PRINCIPALE: Velocità proiettata in avanti ===
    forward_thrust = speedX * np.cos(angle)

    # === 2. PENALITÀ ANGOLO (quadratica) ===
    angle_penalty = speedX * (np.sin(angle) ** 2) * 3.0

    # === 3. PENALITÀ POSIZIONE — PROGRESSIVA (la chiave del fix!) ===
    # Zona sicura (|trackPos| < 0.7): penalità leggera e lineare
    # Zona pericolo (0.7-0.9): penalità quadratica crescente
    # Zona critica (0.9-1.0): penalità esponenziale fortissima
    abs_tp = abs(trackPos)
    if abs_tp < 0.7:
        track_penalty = speedX * (trackPos ** 2) * 1.0  # Leggera
    elif abs_tp < 0.9:
        track_penalty = speedX * (trackPos ** 2) * 4.0  # Media
    else:
        # Esponenziale: da 0.9 a 1.0 scala da 4x a ~50x
        danger = (abs_tp - 0.9) / 0.1  # 0.0 a 1.0
        multiplier = 4.0 + danger * 46.0
        track_penalty = speedX * (trackPos ** 2) * multiplier

    # === 4. BONUS VELOCITÀ ALTA ===
    speed_bonus = max(0.0, speedX - 80.0) * 0.3

    # === 5. RADAR DI SICUREZZA ===
    trk = S.get('track', [100.0] * 19)
    if len(trk) < 19: trk += [100.0] * (19 - len(trk))
    front_distance = trk[9]

    brake_penalty = 0.0
    if front_distance < 40.0 and speedX > 100.0:
        brake_penalty = (speedX - 100.0) * 2.0

    # === 6. PENALITÀ SMOOTHNESS ===
    smoothness_penalty = 0.0
    if prev_action is not None:
        steer_magnitude = abs(prev_action[0])
        smoothness_penalty = steer_magnitude * speedX * 0.03

    # === REWARD TOTALE ===
    reward = forward_thrust + speed_bonus - angle_penalty - track_penalty - brake_penalty - smoothness_penalty
    done = False

    # === TERMINAZIONE: Solo in casi estremi ===
    # FIX: NON terminare a 0.95! Termina solo se COMPLETAMENTE fuori (1.1+)
    # o se va in retromarcia. L'agente deve imparare a RECUPERARE.
    if abs(trackPos) > 1.1:
        reward = -500.0  # Penalità ridotta (era -3000), ma comunque forte
        done = True
    elif speedX < -5.0:
        reward = -500.0
        done = True

    # === VITTORIA: Giro Completato ===
    if lastLapTime > 0:
        time_bonus = max(0, 120 - lastLapTime) * 100
        reward = 5000.0 + time_bonus
        done = True
        print(f"\n🏆 TRAGUARDO RAGGIUNTO! Tempo: {lastLapTime:.2f}s (Bonus: {time_bonus:.0f})")

    return reward / 100.0, done


def compute_gae(rewards, values, gamma=GAMMA, lam=LAMBDA_GAE):
    """Calcola Generalized Advantage Estimation per PPO."""
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


def main():
    print("🚀 Inizio Addestramento RL (PPO v3.0 — Fix Convergenza)")
    print(f"   Frame Skip: {FRAME_SKIP} | LR: {LR} | GAE λ: {LAMBDA_GAE}")
    print(f"   Esplorazione: std {np.exp(INITIAL_LOG_STD):.3f} → {np.exp(FINAL_LOG_STD):.3f}")
    print(f"   Warm-up: {WARMUP_EPISODES} episodi senza rumore")
    print(f"   Action Smoothing: {ACTION_SMOOTHING:.0%}")
    print(f"   Fuoripista: penalità progressiva (NO terminazione a 0.95)")

    model = ActorCritic()

    if os.path.exists(ppo_model_path):
        model.load_state_dict(torch.load(ppo_model_path))
        print("🧠 Cervello PPO caricato. Ripresa dell'addestramento.")
    elif os.path.exists(bc_model_path):
        model.load_state_dict(torch.load(bc_model_path))
        print("🧠 Cervello BC clonato con successo come punto di partenza.")
    else:
        print("❌ Errore: Nessun modello BC trovato. Esegui prima train_bc.py.")
        sys.exit(1)

    optimizer = optim.Adam(model.parameters(), lr=LR)

    feat_mean = np.load(feat_mean_path)
    feat_std = np.load(feat_std_path)

    best_reward = -float('inf')
    reward_history = []

    for episode in range(1, MAX_EPISODES + 1):
        print(f"\n{'='*60}")
        is_warmup = episode <= WARMUP_EPISODES
        phase_str = "🔵 WARM-UP (BC puro)" if is_warmup else "🟢 TRAINING"
        print(f"--- Episodio {episode}/{MAX_EPISODES} [{phase_str}] ---")

        current_log_std = get_log_std(episode, MAX_EPISODES)
        current_std = np.exp(current_log_std)
        print(f"    Esplorazione: std = {current_std:.4f}")

        client = snakeoil3.Client(p=3001, vision=False)
        client.get_servers_input()

        states, actions, log_probs, rewards, values_list = [], [], [], [], []
        done = False
        step = 0
        total_episode_reward = 0.0
        prev_action_np = None  # Per smoothing

        shift_cooldown = 0
        max_speed_ep = 0.0
        off_track_count = 0

        print("   Gara in corso...")

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

            # Inferenza Rete Neurale
            s_np = extract_state(S, feat_mean, feat_std)
            s_tensor = torch.tensor(s_np, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                act_means, value = model(s_tensor)
                std = torch.exp(torch.ones_like(act_means) * current_log_std)
                dist = torch.distributions.Normal(act_means, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)

            raw_action = action[0].numpy().copy()

            # === ACTION SMOOTHING E ABS ===
            # Mescola con l'azione precedente per ridurre jitter da rumore
            if prev_action_np is not None:
                smoothed = prev_action_np * ACTION_SMOOTHING + raw_action * (1.0 - ACTION_SMOOTHING)
            else:
                smoothed = raw_action
            steer, accel, brake = smoothed

            # ABS: limita il freno se stiamo sterzando per evitare sottosterzo estremo
            steer_mag = abs(steer)
            if steer_mag > 0.05:
                max_brake = max(0.05, 1.0 - (steer_mag * 3.0))
                brake = min(brake, max_brake)
                smoothed[2] = brake # Aggiorna array per consistency

            # Esecuzione Azione nell'ambiente
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

                r, done_flag = get_reward_and_done(S_next, smoothed)
                reward_sum += r

                # Conta le volte in zona critica
                tp = abs(S_next.get('trackPos', 0))
                if tp > 0.9:
                    off_track_count += 1

                if done_flag:
                    done = True
                    tp_val = S_next.get('trackPos', 0)
                    if abs(tp_val) > 1.1:
                        print(f"\n   💥 Fuoripista estremo a {speed_kmh:.0f} km/h (trackPos: {tp_val:.2f})")
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
            if step % 100 == 0:
                tp_now = S.get('trackPos', 0)
                print(f"  Step {step:4d} | V: {speed_kmh:3.0f} km/h | Pos: {tp_now:+.2f} | Reward: {total_episode_reward:.1f} | OffTrack: {off_track_count}")

        try:
            client.shutdown()
        except:
            pass

        # === AGGIORNAMENTO PPO ===
        if len(rewards) > 1:
            advantages, returns = compute_gae(rewards, values_list)

            s_batch = torch.cat(states)
            a_batch = torch.cat(actions)
            old_log_probs = torch.cat(log_probs).detach()
            returns_tensor = returns.unsqueeze(1)
            adv_tensor = advantages.unsqueeze(1)

            n_samples = len(states)

            # Durante il warm-up aggiorna SOLO il critic (non toccare l'actor)
            if is_warmup:
                print(f"🔵 Warm-up: aggiorno solo Critic su {n_samples} frame...")
                for _ in range(PPO_EPOCHS):
                    _, new_values = model(s_batch)
                    critic_loss = nn.MSELoss()(new_values, returns_tensor)
                    optimizer.zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                    optimizer.step()
            else:
                print(f"🔄 PPO Update su {n_samples} frame ({PPO_EPOCHS} epoche × mini-batch {MINI_BATCH_SIZE})...")
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

                        ratio = torch.exp(new_log_probs - mb_old_lp)
                        surr1 = ratio * mb_advs.squeeze()
                        surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * mb_advs.squeeze()

                        actor_loss = -torch.min(surr1, surr2).mean()
                        critic_loss = nn.MSELoss()(new_values, mb_returns)
                        entropy = dist.entropy().mean()

                        loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                        optimizer.step()

            # Statistiche episodio
            reward_history.append(total_episode_reward)
            avg_recent = np.mean(reward_history[-20:]) if len(reward_history) >= 20 else np.mean(reward_history)

            status = f"Steps: {step} | Vmax: {max_speed_ep:.0f} km/h | OffTrack: {off_track_count}"

            if total_episode_reward > best_reward:
                best_reward = total_episode_reward
                torch.save(model.state_dict(), ppo_model_path)
                print(f"💾 🌟 RECORD! Reward: {total_episode_reward:.1f} | Media: {avg_recent:.1f} | {status}")
            else:
                torch.save(model.state_dict(), ppo_model_path)
                print(f"💾 Reward: {total_episode_reward:.1f} | Media: {avg_recent:.1f} | Best: {best_reward:.1f} | {status}")

            time.sleep(1.5)


if __name__ == "__main__":
    main()