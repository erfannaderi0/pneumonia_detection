import os
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
from torch import nn
from module import *
from tqdm.auto import tqdm
from data import reorganize_dataset
from sklearn.utils.class_weight import compute_class_weight

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

y_train_labels = [label for _, label in train_data.samples]
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_labels),
    y=y_train_labels)
class_weights = torch.FloatTensor(class_weights).to(device)

lossfn1 = nn.CrossEntropyLoss(weight=class_weights)

if not os.path.exists('models/model1_improved_cnn.pth'):

    optimizer1 = torch.optim.Adam(params= model1.parameters(), lr= 0.01)

    epochs = 20

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer1,
                                                        mode='min',
                                                        factor=0.5,
                                                        patience=3)

    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

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
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model1.state_dict(), 'models/model1_improved_cnn.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break


    model1_result = test_step(model= model1,
                            dataloader= test_dataloader,
                            loss_fn= lossfn1,
                            device= device,
                            verbose= True)
else :
    """
    model1.load_state_dict(torch.load('models/model1_improved_cnn.pth'))
    model1.to(device)
    model1.eval()
    print("Model loaded successfully!")
    
    # After training your model
    # 1. First find the optimal threshold using validation data
    optimal_threshold, best_score = find_optimal_threshold(
        model=model1,
        dataloader=val_dataloader,  # Use validation set!
        device=device)

    # 2. Then test with the optimal threshold
    test_loss, test_auc, test_acc, test_preds, test_labels = test_with_threshold(
        model=model1,
        dataloader=test_dataloader,
        loss_fn=lossfn1,
        device=device,
        threshold=optimal_threshold,
        verbose=True
    )

    # 3. Compare with default threshold (0.5)
    print("\n" + "="*60)
    print("COMPARING DEFAULT VS OPTIMAL THRESHOLD")
    print("="*60)

    # Test with default threshold for comparison
    default_loss, default_auc, default_acc, default_preds, default_labels = test_with_threshold(
        model=model1,
        dataloader=test_dataloader,
        loss_fn=lossfn1,
        device=device,
        threshold=0.5,
        verbose=False  # Don't print full report
    )

    print(f"Default threshold (0.50): Accuracy={default_acc:.4f}, AUC={default_auc:.4f}")
    print(f"Optimal threshold ({optimal_threshold:.2f}): Accuracy={test_acc:.4f}, AUC={test_auc:.4f}")
    print(f"Improvement: +{(test_acc - default_acc)*100:.2f}% accuracy")
    """
    example_usage()