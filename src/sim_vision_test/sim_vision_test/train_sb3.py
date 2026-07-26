#!/usr/bin/env python3
"""
Script de entrenamiento de PPO utilizando Stable-Baselines3 y RacetrackEnv.
Configurado para ejecución optimizada en GPU (NVIDIA Tesla T4 / CUDA).
"""

import os
import sys

def main():
    try:
        import torch
    except ImportError:
        print("[ERROR] PyTorch no está instalado.")
        sys.exit(1)

    print("=" * 60)
    print("   VERIFICACIÓN DE HARDWARE GPU (CUDA)")
    print("=" * 60)
    cuda_available = torch.cuda.is_available()
    print(f"  torch.cuda.is_available() : {cuda_available}")

    if not cuda_available:
        print("\n[ERROR CRÍTICO] CUDA no está disponible en PyTorch.")
        print("Para habilitar la GPU Tesla T4 en AWS, reinstala PyTorch con soporte CUDA:")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --force-reinstall")
        sys.exit(1)

    device_name = torch.cuda.get_device_name(0)
    print(f"  torch.cuda.get_device_name(0) : {device_name}")
    print("=" * 60)

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

    print("\n   INICIANDO ENTRENAMIENTO PPO EN GPU (CUDA: Tesla T4)")
    print("=" * 60)

    # Crear Entorno Gymnasium
    env = RacetrackEnv(random_spawn=False, max_steps=1500)

    # Callback para guardar checkpoints cada 10,000 pasos
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=model_dir,
        name_prefix="ppo_racetrack_model_gpu"
    )

    # Configurar Agente PPO explícitamente en GPU CUDA con parámetros optimizados
    model = PPO(
        policy="MlpPolicy",
        env=env,
        device="cuda",      # Forzar uso explícito de GPU CUDA
        n_steps=2048,       # Rollout más largo optimizado para GPU
        batch_size=256,     # Tamaño de batch alto para maximizar uso de VRAM T4
        n_epochs=10,        # 10 épocas de gradiente por iteración
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=log_dir
    )

    try:
        print(f"[INFO] Entrenando modelo en GPU ({device_name}) por 50,000 pasos de prueba...")
        model.learn(total_timesteps=50000, callback=checkpoint_callback)
        
        final_model_path = os.path.join(model_dir, "ppo_racetrack_final_gpu.zip")
        model.save(final_model_path)
        print(f"[ÉXITO] Modelo final guardado en {final_model_path}")
    except KeyboardInterrupt:
        print("\n[INFO] Entrenamiento interrumpido por el usuario. Guardando checkpoint actual...")
        model.save(os.path.join(model_dir, "ppo_racetrack_interrupted.zip"))
    finally:
        env.close()

if __name__ == "__main__":
    main()
