#!/usr/bin/env python3
"""
Entrenador por Clonación de Comportamiento (PyTorch CUDA T4)
Entrena una Red Neuronal MLP en segundos usando el dataset de 20+ vueltas
guardado por artudo_data_recorder_node.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class ArtudoNeuralDriver(nn.Module):
    def __init__(self, input_dim=10, output_dim=2):
        super(ArtudoNeuralDriver, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim) # [steer, speed]
        )

    def forward(self, x):
        return self.net(x)

def main():
    print("=" * 60)
    print("   ENTRENAMIENTO SUPERVISADO POR CLONACIÓN DE COMPORTAMIENTO")
    print("=" * 60)

    dataset_path = os.path.expanduser('~/dataset_artudo/artudo_expert_dataset.npz')
    model_path = os.path.expanduser('~/dataset_artudo/artudo_expert_model.pth')

    if not os.path.exists(dataset_path):
        print(f"[ERROR] No se encontró el dataset en: {dataset_path}")
        print("Asegúrate de haber corrido artudo_data_recorder y presionado Ctrl+C para guardar.")
        sys.exit(1)

    print(f"Cargando dataset desde: {dataset_path}")
    data = np.load(dataset_path)
    obs = data['obs']       # (N, 10)
    actions = data['actions'] # (N, 2)

    print(f"Total de Muestras Experta Capturadas: {len(obs)}")
    print(f"Formato de Observaciones           : {obs.shape}")
    print(f"Formato de Acciones                : {actions.shape}")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo de Entrenamiento       : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Convertir Tensors
    X = torch.tensor(obs, dtype=torch.float32)
    Y = torch.tensor(actions, dtype=torch.float32)

    # Split Train/Val (90% / 10%)
    dataset = TensorDataset(X, Y)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    model = ArtudoNeuralDriver(input_dim=obs.shape[1], output_dim=actions.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    epochs = 40
    print("\nIniciando entrenamiento por 40 épocas...")
    print("-" * 60)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)

        train_loss /= train_size

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)
        val_loss /= val_size

        if epoch % 5 == 0 or epoch == 1:
            print(f"Época {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    # Guardar modelo
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print("=" * 60)
    print(f"[ÉXITO] Modelo de Red Neuronal Clonado guardado en:\n  {model_path}")
    print("=" * 60)

if __name__ == '__main__':
    main()
