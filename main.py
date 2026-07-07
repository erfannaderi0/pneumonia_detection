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
import sys
import io
from results_tracker import log_and_plot
from sklearn.metrics import f1_score as sk_f1_score
import numpy as np

# Fix encoding issues on Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')


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
    transforms.Resize((256,256)),
    transforms.CenterCrop(224),
    transforms.Grayscale(num_output_channels=1),
    #transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

val_test_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.CenterCrop(224),
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
                      hidden_units=32,
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

if not os.path.exists('models/model2.4_improved_cnn.pth'):

    # ── Ask for a run name before training starts ──────────────────
    print("\n" + "─" * 50)
    cnn_run_name = input("Enter a name for this CNN run (e.g. 'CNN v1 class-weights'): ").strip()
    if not cnn_run_name:
        cnn_run_name = "Improved CNN"
    print("─" * 50 + "\n")

    optimizer1 = torch.optim.Adam(params= model1.parameters(), lr= 1e-3)

    epochs = 50

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer1,
                                                        mode='min',
                                                        factor=0.5,
                                                        patience=3)

    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    train_loss_history, val_loss_history = [], []
    train_auc_history,  val_auc_history  = [], []

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
        
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        train_auc_history.append(train_auc)
        val_auc_history.append(val_auc)
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model1.state_dict(), 'models/model2_improved_cnn.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break


    print("\n" + "="*60)
    print("RUNNING FINAL TEST EVALUATION")
    print("="*60)
        
    model1.load_state_dict(torch.load('models/model2_improved_cnn.pth'))
        
    print("\nFinding optimal threshold on validation set...")
    optimal_threshold, best_f1_val = find_optimal_threshold(model1, val_dataloader, device)

    test_loss, test_auc, test_accuracy, all_preds, all_labels, all_preds_probs = test_with_threshold(model=model1,
                                                                                    dataloader=test_dataloader,
                                                                                    loss_fn=lossfn1,
                                                                                    device=device,
                                                                                    threshold=optimal_threshold,
                                                                                    verbose=True)

    # Calculate F1 score
    from sklearn.metrics import f1_score
    f1 = f1_score(all_labels, all_preds, average='weighted')
        
        
    epochs_run = len(train_loss_history)

    log_and_plot(run_name=cnn_run_name,
                 model_type="Custom CNN",
                 accuracy=test_auc * 100,
                 y_true=all_labels,
                 y_pred=all_preds,
                 y_prob=all_preds_probs,
                 auc_score=test_auc,
                 f1_score=f1_score(all_labels, all_preds, average='weighted'),
                 val_loss=best_val_loss,
                 epochs=epochs_run,
                 threshold=0.55,
                 train_losses=train_loss_history,
                 val_losses=val_loss_history,
                 train_aucs=train_auc_history,
                 val_aucs=val_auc_history,
                 hp_notes="hu=32, lr=1e-3, class_weights=True",
                 notes="Improved CNN with class weights")

os.system('python transfer_training_code/resnet152.py')

example_usage()
