import os
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
from torch import nn
from module import *
from tqdm.auto import tqdm
from data import reorganize_dataset

if not os.path.exists('data/reorganized_chest_xray'):
    reorganize_dataset.main()
else:
    print("✅ Reorganized dataset already exists, skipping reorganization...")

def quick_count(data_dir):
    root = Path(data_dir)
    counts = {}
    
    for split in ['train', 'val', 'test']:
        split_path = root / split
        counts[split] = {}
        
        for cls in ['NORMAL', 'PNEUMONIA']:
            cls_path = split_path / cls
            if cls_path.exists():
                image_files = [f for f in cls_path.iterdir() 
                             if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}]
                counts[split][cls] = len(image_files)
            else:
                counts[split][cls] = 0
                
    return counts

data_dir = "./data/reorganized_chest_xray"
counts = quick_count(data_dir)

print("Dataset Distribution:")
print("-" * 40)

for split in counts:
    normal = counts[split].get('NORMAL', 0)
    pneumonia = counts[split].get('PNEUMONIA', 0)
    total = normal + pneumonia
    
    if total > 0:
        normal_pct = (normal / total) * 100
        pneumonia_pct = (pneumonia / total) * 100
        
        print(f"{split.upper():<10} NORMAL: {normal:>4} ({normal_pct:5.1f}%)  |  PNEUMONIA: {pneumonia:>4} ({pneumonia_pct:5.1f}%)")
    else:
        print(f"{split.upper():<10} No images found")


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05), fill=0),
    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.1),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

train_data = datasets.ImageFolder(
    root='data/reorganized_chest_xray/train',
    transform= train_transform
)

val_data = datasets.ImageFolder(
    root='data/reorganized_chest_xray/val',
    transform= val_test_transform
)

test_data = datasets.ImageFolder(
    root='data/reorganized_chest_xray/test',
    transform= val_test_transform
)

image0, label0 = train_data[0]
print(f"Image {0} of train data : shape = {image0.shape}, label = {label0}")

print(f"Dataset loaded!")
print("summary of train data :")
print(f"Number of images: {len(train_data)}")
print(f"Classes: {train_data.classes}")
print(f"Class to index mapping: {train_data.class_to_idx}")

#================================================================================
#================================================================================

device = 'cuda' if torch.cuda.is_available() else 'cpu'

BATCH_SIZE = 32
torch.manual_seed(42)

train_dataloader = DataLoader(dataset= train_data, batch_size= BATCH_SIZE, shuffle= True)
val_dataloader = DataLoader(dataset= val_data, batch_size= BATCH_SIZE, shuffle= False)
test_dataloader = DataLoader(dataset= test_data, batch_size= BATCH_SIZE, shuffle= False)


model1 = improved_cnn(input_shape= 1,
                          hidden_units=10,
                          output_shape= 2)
model1.to(device)

#=====================================================================
#=                                                                   =
#=         PUT EXISTING CHECK OF THE FILE AFTER THIS SECTION         =
#=                                                                   =
#=====================================================================

lossfn1 = nn.CrossEntropyLoss()
optimizer1 = torch.optim.SGD(params= model1.parameters(), lr= 0.01)

epochs = 3

for epoch in tqdm(range(epochs)):
    train_loss, train_auc = train_step(model=model1,
                                        dataloader=train_dataloader,
                                        loss_fn=lossfn1,
                                        optimizer=optimizer1,
                                        device=device)
    
    val_loss, val_auc = val_step(model=model1,
                                  dataloader=val_dataloader,
                                  loss_fn=lossfn1,
                                  device=device)
    
    print(f"\nEpoch {epoch+1}/{epochs}")
    print(f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")


model1_result = test_step(model= model1,
                          dataloader= test_dataloader,
                          loss_fn= lossfn1,
                          device= device,
                          verbose= True)
