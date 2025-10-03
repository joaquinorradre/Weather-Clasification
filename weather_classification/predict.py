import torch
from sklearn.metrics import accuracy_score
from PIL import Image
import os

from torch.utils.data import DataLoader
from weather_classification.weather_dataset import WeatherDataset

# Use python -m weather_classification.predict to execute

def predict(model, model_path, transform, set_path, batch_size=32):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found at: {os.path.abspath(model_path)}")

    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    val_dataset = WeatherDataset(set_path, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    print("Validation accuracy:", acc)
    return acc