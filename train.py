import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torch.optim import lr_scheduler
import time
import copy

# ===== CẤU HÌNH =====
DATA_DIR = './dataset_binary'
MODEL_SAVE_PATH = './skin_acne_model.pth'
NUM_CLASSES = 2
BATCH_SIZE = 16              # Giảm cho RTX 3050 (4GB VRAM)
NUM_EPOCHS = 20              # Tăng lên 20 để học kỹ hơn
LEARNING_RATE = 0.001

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"🔥 Thiết bị: {device}")

# ===== TIỀN XỬ LÝ ẢNH (AUGMENTATION MẠNH) =====
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(30),
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.3,
            hue=0.1
        ),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# ===== LOAD DATASET =====
image_datasets = {
    x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x])
    for x in ['train', 'valid', 'test']
}
dataloaders = {
    x: DataLoader(
        image_datasets[x],
        batch_size=BATCH_SIZE,
        shuffle=(x == 'train'),
        num_workers=0        # Windows fix — tránh multiprocessing bug
    )
    for x in ['train', 'valid', 'test']
}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}
class_names = image_datasets['train'].classes

print(f"📊 Train: {dataset_sizes['train']} | Valid: {dataset_sizes['valid']} | Test: {dataset_sizes['test']}")
print(f"🏷️  Classes: {class_names}")

# ===== MODEL (ResNet50 Transfer Learning) =====
model_ft = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# Freeze các layer đầu (chỉ train FC layer + layer cuối) → ít overfit hơn
for name, param in model_ft.named_parameters():
    if 'layer4' not in name and 'fc' not in name:
        param.requires_grad = False

num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, NUM_CLASSES)
model_ft = model_ft.to(device)

# Label smoothing để giảm overfitting
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer_ft = optim.Adam(
    filter(lambda p: p.requires_grad, model_ft.parameters()),
    lr=LEARNING_RATE
)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# ===== HÀM TRAIN =====
def train_model(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 40)

        for phase in ['train', 'valid']:
            model.train() if phase == 'train' else model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print(f'{phase:5s} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}')

            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f'\n✅ Train xong trong {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'🏆 Best valid acc: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return model

# ===== CHẠY =====
if __name__ == '__main__':
    print("\n🚀 Bắt đầu train...\n")
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, NUM_EPOCHS)
    torch.save(model_ft.state_dict(), MODEL_SAVE_PATH)
    print(f"\n💾 Đã lưu model: {MODEL_SAVE_PATH}")