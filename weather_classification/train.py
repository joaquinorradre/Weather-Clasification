import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import pickle
import os
from collections import Counter

from weather_classification.weather_dataset import WeatherDataset
from weather_classification.modeling.mlp import MLPClassifier
from weather_classification.modeling.cnn import CNN_V2, CNN_V2_reg, CNN_V3, CNN_V3_reg


def train_model(model_class, train_dir, val_dir, input_dim, num_classes=11, 
                batch_size=32, epochs=20, lr=1e-3, device="cuda"):
    
    model_name = model_class.__name__

    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    metrics_dir = os.path.join(PROJECT_ROOT, "reports", "metrics")
    models_dir = os.path.join(PROJECT_ROOT, "models")

    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    metrics_path = os.path.join(metrics_dir, f"metrics_{model_name}.pkl")
    model_path = os.path.join(models_dir, f"best_model_{model_name}.pth")

    # Data transforms (resize + flatten para MLP)
    if model_class == MLPClassifier:
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1))  # Flatten para MLP
        ])
    elif (model_class == CNN_V2) or (model_class == CNN_V2_reg):
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
    elif model_class == CNN_V3 or model_class == CNN_V3_reg:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])

    # Datasets
    train_dataset = WeatherDataset(train_dir, transform=transform)
    val_dataset = WeatherDataset(val_dir, transform=transform)

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Calculate weights per class
    labels = [label for _, label in train_dataset.samples]
    class_counts = Counter(labels)
    num_classes = len(class_counts)

    total_samples = sum(class_counts.values())
    class_weights = [total_samples / class_counts[i] for i in range(num_classes)]
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)

    print("Class counts:", class_counts)
    print("Class weights:", class_weights_tensor)

    # Modelo
    model = model_class(input_dim=input_dim, num_classes=num_classes).to(device)

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

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

        # Save metrics
        history["train_loss"].append(epoch_loss)
        history["train_acc"].append(epoch_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss={epoch_loss:.4f}, Train Acc={epoch_acc:.4f}, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path)
            print(f"Best model saved in {model_path} (val_acc={val_acc:.4f})")

    # Save metrics to .pkl
    with open(metrics_path, "wb") as f:
        pickle.dump(history, f)

    print(f"Training complete. Best val_acc={best_val_acc:.4f}")
    print(f"MMetrics saved in {metrics_path}")
    return model, history
