import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from weather_classification.weather_dataset import WeatherDataset

def train_model(model, train_dir, val_dir, input_dim, num_classes=11, 
                batch_size=32, epochs=20, lr=1e-3, device="cuda"):
    
    # Data transforms (resize + flatten for MLP)
    transform = transforms.Compose([
        transforms.Resize((64, 64)),   # Resize images
        transforms.ToTensor(),         # Convert to tensor
        transforms.Lambda(lambda x: x.view(-1))  # Flatten for MLP
    ])

    # Datasets
    train_dataset = WeatherDataset(train_dir, transform=transform)
    val_dataset = WeatherDataset(val_dir, transform=transform)

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model
    model = model(input_dim=input_dim, num_classes=num_classes).to(device)

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    best_val_acc = 0.0

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss, running_corrects = 0.0, 0

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = outputs.argmax(1)
            running_loss += loss.item() * imgs.size(0)
            running_corrects += (preds == labels).sum().item()

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects / len(train_dataset)

        # Validation
        model.eval()
        val_loss, val_corrects = 0.0, 0
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)

                preds = outputs.argmax(1)
                val_loss += loss.item() * imgs.size(0)
                val_corrects += (preds == labels).sum().item()

        val_loss /= len(val_dataset)
        val_acc = val_corrects / len(val_dataset)

        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss={epoch_loss:.4f}, Train Acc={epoch_acc:.4f}, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f"Best model saved (val_acc={val_acc:.4f})")

    print(f"Training complete. Best val_acc={best_val_acc:.4f}")
    return model
