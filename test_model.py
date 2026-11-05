import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import classification_report, confusion_matrix

# ===== CẤU HÌNH =====
DATA_DIR = './dataset_binary'
MODEL_PATH = './skin_acne_model.pth'
NUM_CLASSES = 2
BATCH_SIZE = 32

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Thiết bị: {device}")

    # ===== LOAD MODEL =====
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"✅ Loaded model: {MODEL_PATH}")

    # ===== LOAD TEST SET =====
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'test'), test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    class_names = test_dataset.classes
    print(f"📊 Test: {len(test_dataset)} ảnh | Classes: {class_names}")

    # ===== ĐÁNH GIÁ =====
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # ===== KẾT QUẢ =====
    print("\n" + "=" * 60)
    print("📈 KẾT QUẢ ĐÁNH GIÁ TRÊN TEST SET")
    print("=" * 60)

    print("\n📋 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    print("🔢 Confusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(f"              Predicted")
    print(f"              {class_names[0]:<10} {class_names[1]:<10}")
    print(f"Actual {class_names[0]:<6} {cm[0][0]:<10} {cm[0][1]:<10}")
    print(f"       {class_names[1]:<6} {cm[1][0]:<10} {cm[1][1]:<10}")

    acc = sum(1 for p, l in zip(all_preds, all_labels) if p == l) / len(all_labels)
    print(f"\n🎯 Test Accuracy: {acc:.4f}")

if __name__ == '__main__':
    main()