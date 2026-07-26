#!/usr/bin/env python3
"""
Script de entrenamiento de PPO utilizando Stable-Baselines3 y RacetrackEnv.
"""

import os
import sys
import argparse

def main():
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CheckpointCallback
    except ImportError:
        print("[ERROR] Stable-Baselines3 no está instalado. Ejecuta:")
        print("  pip install stable-baselines3 gymnasium tensorboard")
        sys.exit(1)

    from sim_vision_test.racetrack_env import RacetrackEnv

    model_dir = os.path.expanduser("~/sb3_models")
    os.makedirs(model_dir, exist_ok=True)

    log_dir = os.path.expanduser("~/tensorboard_logs")
    os.makedirs(log_dir, exist_ok=True)

    print("=" * 60)
    print("   INICIANDO ENTRENAMIENTO CON STABLE-BASELINES3 (PPO)")
    print("=" * 60)

    # Crear Entorno Gymnasium
    env = RacetrackEnv(random_spawn=True, max_steps=1500)

    # Callback para guardar checkpoints cada 10,000 pasos
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=model_dir,
        name_prefix="ppo_racetrack_model"
    )

    # Configurar Agente PPO
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=log_dir
    )

    try:
        print("[INFO] Entrenando modelo por 100,000 pasos de tiempo...")
        model.learn(total_timesteps=100000, callback=checkpoint_callback)
        
        final_model_path = os.path.join(model_dir, "ppo_racetrack_final.zip")
        model.save(final_model_path)
        print(f"[ÉXITO] Modelo final guardado en {final_model_path}")
    except KeyboardInterrupt:
        print("\n[INFO] Entrenamiento interrumpido por el usuario. Guardando checkpoint actual...")
        model.save(os.path.join(model_dir, "ppo_racetrack_interrupted.zip"))
    finally:
        env.close()

if __name__ == "__main__":
    main()
