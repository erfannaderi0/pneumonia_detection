import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import sys
import copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import module
from module import *
from results_tracker import (
    log_and_plot,
    plot_calibration_curve,
    start_terminal_log,
    stop_terminal_log
)
from tqdm.auto import tqdm
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from torchvision.transforms import functional as F
from sklearn.metrics import f1_score


class CropBorders:
    def __call__(self, img):
        w, h = img.size

        left   = int(w * 0.05)
        right  = int(w * 0.95)
        top    = int(h * 0.05)
        bottom = int(h * 0.95)

        return F.crop(
            img,
            top,
            left,
            bottom-top,
            right-left
        )

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    torch.manual_seed(42)

    # ImageNet statistics
    xray_mean = [0.482, 0.482, 0.482]
    xray_std  = [0.234, 0.234, 0.234]

    data_transforms = {
        'train': transforms.Compose([
            CropBorders(),
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(xray_mean, xray_std)]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(xray_mean, xray_std)]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(xray_mean, xray_std)])
    }

    data_dir = './data/reorganized_chest_xray'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'val', 'test']}
    
    train_dataloader = torch.utils.data.DataLoader(image_datasets['train'],
                                                    batch_size=32,
                                                    shuffle=True,
                                                    num_workers=0)
    val_test_dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                           batch_size=32,
                                                           shuffle=False,
                                                           num_workers=0)
                            for x in ['val', 'test']}
    val_dataloader = val_test_dataloaders['val']
    test_dataloader = val_test_dataloaders['test']
    if not os.path.exists('models/model2_resnet152.pth'):
        model_res152 = models.resnet152(weights='IMAGENET1K_V1')

        for params in model_res152.parameters():
            params.requires_grad = False

        num_features = model_res152.fc.in_features

        model_res152.fc = nn.Sequential(nn.Linear(num_features, 256),
                                        nn.ReLU(inplace=True), 
                                        nn.Dropout(0.4),
                                        nn.Linear(256, 2))

        #=====================================================================
        #=                        TRAINING PHASE                             =
        #=====================================================================
        model_res152 = model_res152.to(device)

        # ── Ask for a run name before training starts ──────────────────────
        print("\n" + "─" * 50)
        resnet_run_name = input("Enter a name for this ResNet152 run (e.g. 'ResNet152 v1'): ").strip()
        if not resnet_run_name:
            resnet_run_name = "ResNet152"
        print("─" * 50 + "\n")
        
        log_path = start_terminal_log(resnet_run_name)
        
        optimizer_ft = optim.Adam(model_res152.fc.parameters(), lr=0.001)
        exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft,
                                                        mode='min',
                                                        factor=0.5,
                                                        patience=3)

        y_train = [label for _, label in image_datasets['train'].samples]
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        
        print("Classes:", image_datasets['train'].classes)
        print("Class weights:", class_weights)
        
        class_weights_tensor = torch.FloatTensor(class_weights).to(device)

        loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)

        epochs = 10

        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0

        train_loss_history, val_loss_history = [], []
        train_auc_history,  val_auc_history  = [], []

        for epoch in tqdm(range(epochs)):
            train_loss, train_auc = train_step(model=model_res152,
                                            dataloader=train_dataloader,
                                            loss_fn=loss_fn,
                                            optimizer=optimizer_ft,
                                            device=device)
                
            val_loss, val_auc = val_step(model=model_res152,
                                        dataloader=val_dataloader,
                                        loss_fn=loss_fn,
                                        device=device)
                
            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")

            train_loss_history.append(train_loss)
            val_loss_history.append(val_loss)
            train_auc_history.append(train_auc)
            val_auc_history.append(val_auc)
                
            exp_lr_scheduler.step(val_loss)
                
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(model_res152.state_dict(), 'models/model2_resnet152.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        for param in model_res152.parameters():
            param.requires_grad = False  # re-freeze everything first
        for param in model_res152.layer4.parameters():
            param.requires_grad = True
        for param in model_res152.layer3.parameters():
            param.requires_grad = True
        for param in model_res152.fc.parameters():
            param.requires_grad = True
        print("\nPhase 2: Unfroze layer3, layer4, fc — fine-tuning with lr=1e-5")

        optimizer_ft = optim.Adam(model_res152.parameters(), lr=1e-5)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)

        train_loss_history_p2, val_loss_history_p2 = [], []
        train_auc_history_p2,  val_auc_history_p2  = [], []
        patience_counter = 0
        
        for epoch in tqdm(range(epochs)):
            
            train_loss, train_auc = train_step(model=model_res152,
                                            dataloader=train_dataloader,
                                            loss_fn=loss_fn,
                                            optimizer=optimizer_ft,
                                            device=device)
                
            val_loss, val_auc = val_step(model=model_res152,
                                        dataloader=val_dataloader,
                                        loss_fn=loss_fn,
                                        device=device)
                
            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")

            train_loss_history_p2.append(train_loss)
            val_loss_history_p2.append(val_loss)
            train_auc_history_p2.append(train_auc)
            val_auc_history_p2.append(val_auc)
                
            exp_lr_scheduler.step()
                
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(model_res152.state_dict(), 'models/model2_resnet152.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # ── Evaluate and log the freshly trained model ─────────────────────
        print("\n" + "="*60)
        print("RUNNING FINAL TEST EVALUATION")
        print("="*60)

        model_res152.load_state_dict(torch.load('models/model2_resnet152.pth'))

        #print("\nFinding optimal threshold on validation set...")
        #optimal_threshold, best_score = find_optimal_threshold(model_res152, val_dataloader, device)
        optimal_threshold = 0.7

        test_loss, test_auc, test_accuracy, all_preds, all_labels, all_preds_probs = test_with_threshold(
            model=model_res152,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device,
            threshold=optimal_threshold,
            verbose=True,
        )

        f1 = f1_score(all_labels, all_preds, average='weighted')

        combined_train_losses = train_loss_history + train_loss_history_p2
        combined_val_losses   = val_loss_history   + val_loss_history_p2
        combined_train_aucs   = train_auc_history  + train_auc_history_p2
        combined_val_aucs     = val_auc_history    + val_auc_history_p2

        log_and_plot(
            run_name=resnet_run_name,
            model_type="ResNet152",

            accuracy=test_accuracy * 100,
            y_true=all_labels,
            y_pred=all_preds,
            y_prob=all_preds_probs,

            auc_score=test_auc,
            f1_score=f1,
            val_loss=best_val_loss,
            test_loss=test_loss,

            epochs_trained=len(combined_train_losses),
            max_epochs=20,

            threshold=optimal_threshold,

            train_losses=combined_train_losses,
            val_losses=combined_val_losses,
            train_aucs=combined_train_aucs,
            val_aucs=combined_val_aucs,

            lr=1e-5,
            batch_size=32,
            optimizer="Adam",
            class_weighting=True,
            dropout_rate=0.4,
            pretrained=True,

            img_size=224,
            log_file=log_path,
            notes="Phase1 lr=1e-3 fc-only | Phase2 lr=1e-5 layer3+layer4+fc"
        )

    else:
        
        best_val_loss = None

        combined_train_losses = []
        combined_val_losses = []

        combined_train_aucs = []
        combined_val_aucs = []
        
        # ── Model already trained: load and evaluate ────────────────────────
        y_train = [label for _, label in image_datasets['train'].samples]
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        
        print("Classes:", image_datasets['train'].classes)
        print("Class weights:", class_weights)
        
        class_weights_tensor = torch.FloatTensor(class_weights).to(device)

        loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)
        
        model_res152 = models.resnet152(weights=None)
        
        num_features = model_res152.fc.in_features
        model_res152.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True), 
            nn.Dropout(0.4),
            nn.Linear(256, 2))
        
        model_path = 'models/model2_resnet152.pth'
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=device, weights_only=False)
            model_res152.load_state_dict(state_dict)
            print("✅ ResNet152 model loaded successfully")
        else:
            print(f"❌ Model file not found: {model_path}")
        
        model_res152.to(device)

        # ── Ask for a run name ──────────────────────────────────────────────
        print("\n" + "─" * 50)
        resnet_run_name = input("Enter a name for this ResNet152 evaluation run: ").strip()
        if not resnet_run_name:
            resnet_run_name = "ResNet152"
        print("─" * 50 + "\n")
        
        log_path = start_terminal_log(resnet_run_name)

        #print("\nFinding optimal threshold on validation set...")
        #optimal_threshold, best_score = find_optimal_threshold(model_res152, val_dataloader, device)
        optimal_threshold = 0.7

        test_loss, test_auc, test_accuracy, all_preds, all_labels, all_preds_probs = test_with_threshold(
            model=model_res152,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device,
            threshold=optimal_threshold,
            verbose=True,
        )

        f1 = f1_score(all_labels, all_preds, average='weighted')

        # Initialize plotter
        plotter = Plot(style='seaborn-v0_8-darkgrid', figsize=(12, 8))

        # 1. Plot confusion matrix
        plotter.plot_confusion_matrix(all_labels,
                                      all_preds,
                                      class_names=['Class 0', 'Class 1'],
                                      normalize=False,
                                      title='Test Set Confusion Matrix',
                                      save_path='pictures/plot_confusion_matrix_res152')

        # 2. Plot ROC curves
        plotter.plot_roc_curves(all_labels,
                                all_preds_probs,
                                n_classes=2,
                                class_names=['Class 0', 'Class 1'],
                                title='ROC Curves - Test Set',
                                save_path='pictures/plot_roc_curves')

        # 3. Plot Precision-Recall curves
        plotter.plot_precision_recall_curves(all_labels,
                                             all_preds_probs,
                                             n_classes=2,
                                             class_names=['Class 0', 'Class 1'],
                                             save_path='pictures/plot_oercision_recall_curves_res152')

        # 4. Error analysis
        plotter.plot_error_analysis(all_labels,
                                    all_preds,
                                    feature_names=['feature1', 'feature2'],
                                    save_path='pictures/plot_error_analysis_res152')

        log_and_plot(
            run_name=resnet_run_name,
            model_type="ResNet152",

            accuracy=test_accuracy * 100,
            y_true=all_labels,
            y_pred=all_preds,
            y_prob=all_preds_probs,

            auc_score=test_auc,
            f1_score=f1,
            val_loss=best_val_loss,
            test_loss=test_loss,

            epochs_trained=len(combined_train_losses),
            max_epochs=20,

            threshold=optimal_threshold,

            train_losses=combined_train_losses,
            val_losses=combined_val_losses,
            train_aucs=combined_train_aucs,
            val_aucs=combined_val_aucs,

            lr=1e-5,
            batch_size=32,
            optimizer="Adam",
            class_weighting=True,
            dropout_rate=0.4,
            pretrained=True,

            img_size=224,
            log_file=log_path,
            notes="Phase1 lr=1e-3 fc-only | Phase2 lr=1e-5 layer3+layer4+fc"
        )
    
    example_usage()
