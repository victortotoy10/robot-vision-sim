import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import cv2
import os
import sys

# Modelo CNN enfocado 100% en predecir la direccion (steering angle)
class RacerCNN(nn.Module):
    def __init__(self):
        super(RacerCNN, self).__init__()
        # Entrada: (3 channels, 120 height, 160 width)
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
            nn.Linear(50, 1) # Unica salida: angular_z (direccion)
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
        image = np.transpose(image, (2, 0, 1)) # HWC -> CHW

        angular_z = float(self.data_df.iloc[idx, 2])
        return torch.tensor(image), torch.tensor([angular_z], dtype=torch.float32)

def train():
    data_dir = os.path.expanduser('~/training_data')
    csv_path = os.path.join(data_dir, 'data.csv')

    if not os.path.exists(csv_path):
        print(f"ERROR: No se encontro el archivo de datos {csv_path}")
        print("Graba algunos datos de entrenamiento primero.")
        sys.exit(1)

    print("Cargando dataset...")
    df = pd.read_csv(csv_path)

    # --- BALANCEO DINAMICO DE DATOS ---
    # Separamos rectas (giro cercano a 0) de las curvas
    rectas = df[df['angular_z'].abs() < 0.05]
    curvas = df[df['angular_z'].abs() >= 0.05]

    # Reducimos las rectas al 15% de forma aleatoria para evitar el sesgo de "ir siempre recto"
    rectas_filtradas = rectas.sample(frac=0.15, random_state=42) if len(rectas) > 0 else rectas
    df_balanceado = pd.concat([rectas_filtradas, curvas]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    print(f"Dataset original: {len(df)} frames | Dataset balanceado: {len(df_balanceado)} frames")
    
    dataset = RacerDataset(dataframe=df_balanceado, root_dir=data_dir)
    
    # Division Entrenamiento (80%) / Validacion (20%)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando hardware: {device}")

    model = RacerCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Subimos a 40 epocas para mejor aprendizaje
    epochs = 40
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

        # Validacion
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

        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")

    # Guardar modelo
    model_path = os.path.join(data_dir, 'racer_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Entrenamiento completado. Modelo guardado en {model_path}")

if __name__ == '__main__':
    train()
