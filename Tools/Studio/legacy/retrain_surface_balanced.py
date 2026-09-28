import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATASET_PATH = os.path.join(DATA_DIR, "surface_dataset5.npz")
ONNX_PATH = os.path.join(DATA_DIR, "surface_nn_model.onnx")

SURFACE_CLASSES = ["HARD_FLOOR", "CARPET", "SPECULAR", "VOID"]
SURFACE_CANONICAL_BINS = 16

def augment_surface_canonical(canonical_16):
    augmented = [np.array(canonical_16, dtype=np.float32)]
    base = np.array(canonical_16, dtype=np.float32)

    for scale in [0.92, 1.08]:
        augmented.append(np.clip(base * scale, 0.0, 1.0))

    for _ in range(2):
        noise = np.random.normal(0.0, 0.02 * (np.max(base) + 0.2), base.shape)
        augmented.append(np.clip(base + noise, 0.0, 1.0))

    b_wide = np.convolve(base, [0.10, 0.80, 0.10], mode='same')
    augmented.append(b_wide / (np.max(b_wide) + 1e-4))

    return augmented

class Surface1DCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 8, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(8, 16, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(16 * 4, 16)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

def main():
    print(f"Loading dataset from: {DATASET_PATH}")
    data = np.load(DATASET_PATH)

    counts = [len(data[c]) for c in SURFACE_CLASSES]
    for c, cnt in zip(SURFACE_CLASSES, counts):
        print(f"  {c:12s}: {cnt} samples")

    min_samples = min(counts)
    print(f"\nBalancing dataset to {min_samples} samples per class to prevent bias...")

    X_list = []
    y_list = []

    for class_idx, class_name in enumerate(SURFACE_CLASSES):
        pulses = data[class_name][:min_samples]
        for p in pulses:
            augmented = augment_surface_canonical(p)
            for a in augmented:
                X_list.append(a)
                y_list.append(class_idx)

    X = np.array(X_list, dtype=np.float32).reshape(-1, 1, SURFACE_CANONICAL_BINS)
    y = np.array(y_list, dtype=np.int64)

    print(f"Total balanced augmented samples: {len(y)} (Shape: {X.shape})")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    torch.manual_seed(42)
    np.random.seed(42)
    model = Surface1DCNN(num_classes=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()

    dataset = torch.utils.data.TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    print("Training Surface1DCNN model (40 epochs)...")
    model.train()
    for epoch in range(40):
        total_loss = 0.0
        for bx, by in loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch [{epoch+1:2d}/40] - Loss: {total_loss/len(loader):.4f}")

    model.eval()
    with torch.no_grad():
        val_preds = torch.argmax(model(torch.tensor(X_val)), dim=1).numpy()
        acc = np.mean(val_preds == y_val) * 100.0

    print(f"\nTraining Complete! Validation Accuracy: {acc:.2f}%\n")
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, val_preds))
    print("\nClassification Report:")
    print(classification_report(y_val, val_preds, target_names=SURFACE_CLASSES))

    print(f"Exporting ONNX model to: {ONNX_PATH}")
    dummy = torch.randn(1, 1, SURFACE_CANONICAL_BINS)
    torch.onnx.export(
        model,
        dummy,
        ONNX_PATH,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input_canonical'],
        output_names=['output_logits'],
        dynamo=False
    )
    print("ONNX export successful!")

if __name__ == "__main__":
    main()
