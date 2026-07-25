import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import cv2
import os
import sys

# Modelo CNN multivariable
class RacerCNN(nn.Module):
    def __init__(self):
        super(RacerCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2), # 120x160 -> 58x78
            nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2), # 58x78 -> 27x37
            nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2), # 27x37 -> 12x17
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3),           # 12x17 -> 10x15
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),           # 10x15 -> 8x13
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 13, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 2) # [linear_x, angular_z]
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

class RacerDataset(Dataset):
    def __init__(self, dataframe, root_dir):
        self.data_df = dataframe
        self.root_dir = root_dir

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data_df.iloc[idx, 0])
        image = cv2.imread(img_name)
        # Normalizar imagen
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))

        linear_x = float(self.data_df.iloc[idx, 1])
        angular_z = float(self.data_df.iloc[idx, 2])

        # --- NORMALIZACIÓN DE ESCALA DE TARGETS ---
        # Escalamos ambos al rango [-1.0, 1.0] para que pesen igual en la loss
        linear_x_norm = linear_x / 2.0
        angular_z_norm = angular_z / 3.0
        targets = np.array([linear_x_norm, angular_z_norm], dtype=np.float32)

        return torch.tensor(image), torch.tensor(targets, dtype=torch.float32)

def train():
    data_dir = os.path.expanduser('~/training_data')
    csv_path = os.path.join(data_dir, 'data.csv')

    if not os.path.exists(csv_path):
        print(f"ERROR: No se encontro el archivo de datos {csv_path}")
        sys.exit(1)

    print("Cargando dataset...")
    df = pd.read_csv(csv_path)

    # --- BALANCEO DE DATOS ---
    rectas = df[(df['angular_z'].abs() < 0.05) & (df['linear_x'] > 0.0)]
    curvas = df[df['angular_z'].abs() >= 0.05]
    reversa = df[df['linear_x'] < -0.01]
    detenido = df[(df['linear_x'].abs() < 0.01) & (df['angular_z'].abs() < 0.01)]

    # Filtramos las rectas y quietas redundantes
    rectas_filtradas = rectas.sample(frac=0.15, random_state=42) if len(rectas) > 0 else rectas
    detenidos_filtrados = detenido.sample(frac=0.10, random_state=42) if len(detenido) > 0 else detenido

    df_balanceado = pd.concat([rectas_filtradas, detenidos_filtrados, curvas, reversa]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    print(f"Dataset original: {len(df)} frames")
    print(f"Dataset balanceado -> Total: {len(df_balanceado)} (Curvas: {len(curvas)} | Reversas: {len(reversa)})")
    
    dataset = RacerDataset(dataframe=df_balanceado, root_dir=data_dir)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando hardware: {device}")

    model = RacerCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    epochs = 80
    print("Iniciando entrenamiento en la GPU...")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * images.size(0)

        train_loss /= len(train_dataset)
        val_loss /= len(val_dataset)

        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")

    model_path = os.path.join(data_dir, 'racer_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Entrenamiento completado. Modelo guardado en {model_path}")

if __name__ == '__main__':
    train()
