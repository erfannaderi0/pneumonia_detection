import torch
import torchvision
from torch import nn
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix, f1_score, roc_curve, auc, precision_recall_curve
import numpy as np
from PIL import Image
from torchvision import transforms
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize
import itertools
import sys


class improved_cnn(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        
        dropout_rate = 0.2
        
        self.dropout1 = nn.Dropout2d(dropout_rate)
        self.dropout2 = nn.Dropout2d(dropout_rate)
        self.dropout3 = nn.Dropout2d(dropout_rate)
        self.dropout4 = nn.Dropout2d(dropout_rate)
        
        # Block 1: 224x224 -> 112x112
        self.block1 = nn.Sequential(
            nn.Conv2d(input_shape, hidden_units, 3, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Block 2: 112x112 -> 56x56
        self.block2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units*2, 3, padding=1),
            nn.BatchNorm2d(hidden_units*2),
            nn.ReLU(),
            nn.Conv2d(hidden_units*2, hidden_units*2, 3, padding=1),
            nn.BatchNorm2d(hidden_units*2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Block 3: 56x56 -> 28x28
        self.block3 = nn.Sequential(
            nn.Conv2d(hidden_units*2, hidden_units*4, 3, padding=1),
            nn.BatchNorm2d(hidden_units*4),
            nn.ReLU(),
            nn.Conv2d(hidden_units*4, hidden_units*4, 3, padding=1),
            nn.BatchNorm2d(hidden_units*4),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Block 4: 28x28 -> 14x14
        self.block4 = nn.Sequential(
            nn.Conv2d(hidden_units*4, hidden_units*8, 3, padding=1),
            nn.BatchNorm2d(hidden_units*8),
            nn.ReLU(),
            nn.Conv2d(hidden_units*8, hidden_units*8, 3, padding=1),
            nn.BatchNorm2d(hidden_units*8),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(hidden_units*8, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, output_shape)
        )
        
    def forward(self, x):
        x = self.block1(x)
        x = self.dropout1(x)
        x = self.block2(x)
        x = self.dropout2(x)
        x = self.block3(x)
        x = self.dropout3(x)
        x = self.block4(x)
        x = self.dropout4(x)
        x = self.classifier(x)
        return x


def train_step(model: nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device = None):
    
    train_loss = 0
    all_preds = []
    all_labels = []
    
    model.train()
    
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        
        train_loss += loss.item()
        
        if y_pred.shape[1] == 2:  # Binary classification
            # Use probability of positive class
            all_preds.extend(torch.softmax(y_pred, dim=1)[:, 1].detach().cpu().numpy())
        else:  # Multi-class
            all_preds.extend(torch.softmax(y_pred, dim=1).detach().cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    train_loss /= len(dataloader)
    
    # Calculate AUC
    if len(set(all_labels)) > 1:
        if len(np.unique(all_labels)) == 2:  # Binary
            train_auc = roc_auc_score(all_labels, all_preds)
        else:  # Multi-class
            train_auc = roc_auc_score(all_labels, all_preds, multi_class='ovr')
    else:
        train_auc = 0.0
    
    return train_loss, train_auc

def val_step(model: nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: nn.Module,
               device: torch.device = None):
    
    val_loss = 0
    all_preds = []
    all_labels = []
    
    model.eval()
    
    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            
            val_pred = model(X)
            loss = loss_fn(val_pred, y)
            
            val_loss += loss.item()
            
            if val_pred.shape[1] == 2:  # Binary classification
                # Use probability of positive class
                all_preds.extend(torch.softmax(val_pred, dim=1)[:, 1].detach().cpu().numpy())
            else:  # Multi-class
                all_preds.extend(torch.softmax(val_pred, dim=1).detach().cpu().numpy())
            all_labels.extend(y.cpu().numpy())
        
        val_loss /= len(dataloader)
        
        # Calculate AUC
        if len(set(all_labels)) > 1:
            if len(np.unique(all_labels)) == 2:  # Binary
                val_auc = roc_auc_score(all_labels, all_preds)
            else:  # Multi-class
                val_auc = roc_auc_score(all_labels, all_preds, multi_class='ovr')
        else:
            val_auc = 0.0

    return val_loss, val_auc


def test_step(model: nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: nn.Module,
              device: torch.device = None,
              verbose: bool = True):
    
    test_loss = 0
    all_preds = []
    all_preds_probs = []
    all_labels = []
    
    model.eval()
    
    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            
            test_pred = model(X)
            loss = loss_fn(test_pred, y)
            test_loss += loss.item()
            
            if test_pred.shape[1] == 2:  # Binary classification
                probs = torch.softmax(test_pred, dim=1)

                all_preds_probs.extend(probs[:, 1].detach().cpu().numpy())

                all_preds.extend(torch.argmax(test_pred, dim=1).detach().cpu().numpy())
            else:  # Multi-class
                probs = torch.softmax(test_pred, dim=1)
                all_preds_probs.extend(probs.detach().cpu().numpy())
                all_preds.extend(torch.argmax(test_pred, dim=1).detach().cpu().numpy())
            
            all_labels.extend(y.cpu().numpy())
    
    test_loss /= len(dataloader)
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    # Accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    
    if len(np.unique(all_labels)) > 1:
        if len(np.unique(all_labels)) == 2:  # Binary
            test_auc = roc_auc_score(all_labels, all_preds_probs)
        else:  # Multi-class
            test_auc = roc_auc_score(all_labels, all_preds_probs, multi_class='ovr')
    else:
        test_auc = 0.0
    
    # Print detailed metrics if verbose
    if verbose:
        print("\n" + "="*50)
        print("TEST RESULTS")
        print("="*50)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test AUC: {test_auc:.4f}")
        print("\nDetailed Classification Report:")
        print(classification_report(all_labels, all_preds))
        print("\nConfusion Matrix:")
        print(confusion_matrix(all_labels, all_preds))
        print("="*50)
    
    return test_loss, test_auc, accuracy, all_preds, all_labels, all_preds_probs

def find_optimal_threshold(model, dataloader, device):
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    thresholds = np.arange(0.3, 0.8, 0.05)
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        preds = (np.array(all_probs) >= threshold).astype(int)
        f1 = f1_score(all_labels, preds, average='weighted')
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"Optimal threshold: {best_threshold:.2f} (F1: {best_f1:.4f})")
    return best_threshold, best_f1

def test_with_threshold(model: nn.Module,
                        dataloader: torch.utils.data.DataLoader,
                        loss_fn: nn.Module,
                        device: torch.device = None,
                        threshold: float = 0.5,
                        verbose: bool = True):
    """
    Test the model with a custom probability threshold for binary classification.
    
    Args:
        model: The neural network model
        dataloader: DataLoader for test data
        loss_fn: Loss function
        device: Device to run on (cuda/cpu)
        threshold: Probability threshold for positive class (default: 0.5)
        verbose: Whether to print detailed results
    
    Returns:
        test_loss: Average loss on test set
        test_auc: AUC score
        accuracy: Accuracy score
        all_preds: All predictions
        all_labels: All ground truth labels
    """
    
    test_loss = 0
    all_preds_probs = []  # Store probabilities
    all_preds = []        # Store final predictions after threshold
    all_labels = []
    
    model.eval()
    
    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            
            # Forward pass
            test_pred = model(X)
            loss = loss_fn(test_pred, y)
            test_loss += loss.item()
            
            # Get probabilities
            if test_pred.shape[1] == 2:  # Binary classification
                probs = torch.softmax(test_pred, dim=1)
                # Get probability of positive class (class 1)
                prob_positive = probs[:, 1].detach().cpu().numpy()
                all_preds_probs.extend(prob_positive)
                
                # Apply custom threshold
                batch_preds = (prob_positive >= threshold).astype(int)
                all_preds.extend(batch_preds)
            else:  # Multi-class (fallback to argmax)
                probs = torch.softmax(test_pred, dim=1)
                all_preds_probs.extend(probs.detach().cpu().numpy())
                all_preds.extend(torch.argmax(test_pred, dim=1).detach().cpu().numpy())
            
            all_labels.extend(y.cpu().numpy())
    
    # Calculate average loss
    test_loss /= len(dataloader)
    
    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_preds_probs = np.array(all_preds_probs)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    
    # Calculate AUC (using probabilities, not thresholded predictions)
    if len(np.unique(all_labels)) > 1:
        if len(np.unique(all_labels)) == 2:  # Binary
            test_auc = roc_auc_score(all_labels, all_preds_probs)
        else:  # Multi-class
            test_auc = roc_auc_score(all_labels, all_preds_probs, multi_class='ovr')
    else:
        test_auc = 0.0
    
    # Print detailed metrics if verbose
    if verbose:
        print("\n" + "="*50)
        print(f"TEST RESULTS (Threshold = {threshold:.2f})")
        print("="*50)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test AUC: {test_auc:.4f}")
        print("\nDetailed Classification Report:")
        print(classification_report(all_labels, all_preds, target_names=['NORMAL', 'PNEUMONIA']))
        print("\nConfusion Matrix:")
        print(confusion_matrix(all_labels, all_preds))
        print("="*50)
    
    return test_loss, test_auc, accuracy, all_preds, all_labels, all_preds_probs


def print_result_simple(result):
    """
    Nicely print a prediction result.
    """
    print("\n" + "="*60)
    print(f"Image: {result['image']}")
    print("="*60)
    print(f"Prediction: {result['prediction']}")
    print(f"Probability: {result['probability_percent']} (threshold: 0.55)")
    print(f"Confidence: {result['confidence']}")
    print(f"Status: {result['review_message']}")
    print("="*60)


def save_results_simple(results, output_file='predictions.txt'):
    """
    Save results to a simple text file.
    """
    with open(output_file, 'w') as f:
        f.write("PNEUMONIA DETECTION RESULTS\n")
        f.write("="*60 + "\n\n")
        
        pneumonia_cases = [r for r in results if r['prediction'] == "PNEUMONIA"]
        normal_cases = [r for r in results if r['prediction'] == "NORMAL"]
        review_cases = [r for r in results if r['needs_review']]
        
        f.write(f"Total images: {len(results)}\n")
        f.write(f"Pneumonia detected: {len(pneumonia_cases)}\n")
        f.write(f"Normal: {len(normal_cases)}\n")
        f.write(f"Needs review: {len(review_cases)}\n\n")
        
        if pneumonia_cases:
            f.write("PNEUMONIA CASES:\n")
            f.write("-"*40 + "\n")
            for r in pneumonia_cases:
                f.write(f"{r['image']}: {r['probability_percent']} - {r['confidence']}\n")
            f.write("\n")
        
        if review_cases:
            f.write("CASES NEEDING REVIEW:\n")
            f.write("-"*40 + "\n")
            for r in review_cases:
                f.write(f"{r['image']}: {r['prediction']} ({r['probability_percent']})\n")
    
    print(f"\n✅ Results saved to {output_file}")


def prepare_image_for_model(image_path, model_type='custom'):
    """
    Prepare image for model input.
    
    Args:
        image_path: Path to image or PIL Image
        model_type: 'custom' for grayscale (1-channel) or 'resnet152' for 3-channel RGB
    """
    if isinstance(image_path, (str, Path)):
        img = Image.open(image_path).convert('RGB')
    else:
        img = image_path.convert('RGB')
    
    if model_type == 'custom':
        # For your original custom grayscale model
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    elif model_type == 'resnet152':
        # For your trained ResNet152 model
        # Note: This matches your training transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        # Generic pretrained model (like default ResNet)
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    image_tensor = transform(img).unsqueeze(0)
    return image_tensor


def get_pneumonia_probability(model, image_tensor, device, model_type='custom'):

    model.eval()
    image_tensor = image_tensor.to(device)
    
    with torch.inference_mode():
        output = model(image_tensor)
        
        # Both your custom model and ResNet152 have 2 output classes
        probabilities = torch.softmax(output, dim=1)
        pneumonia_prob = probabilities[:, 1].cpu().numpy()[0]
    
    return float(pneumonia_prob)


def simple_predict(model, image_path, device, threshold=0.55, model_type='custom'):

    image_tensor = prepare_image_for_model(image_path, model_type)
    prob = get_pneumonia_probability(model, image_tensor, device, model_type)
    
    # Rest of your function remains the same...
    prediction = "PNEUMONIA" if prob >= threshold else "NORMAL"
    
    if prediction == "PNEUMONIA":
        if prob >= 0.90:
            confidence = "VERY HIGH - Definitely pneumonia"
        elif prob >= 0.75:
            confidence = "HIGH - Likely pneumonia"
        elif prob >= threshold:
            confidence = "MEDIUM - Possible pneumonia, please review"
        else:
            confidence = "LOW - Just above threshold"
    else:  # NORMAL
        if prob <= 0.10:
            confidence = "VERY HIGH - Definitely normal"
        elif prob <= 0.25:
            confidence = "HIGH - Likely normal"
        elif prob < threshold:
            confidence = "MEDIUM - Possibly normal, please review"
        else:
            confidence = "LOW - Just below threshold"
    
    needs_review = 0.25 < prob < 0.75
    
    result = {
        'image': str(image_path),
        'prediction': prediction,
        'probability': round(prob, 3),
        'probability_percent': f"{prob:.1%}",
        'confidence': confidence,
        'needs_review': needs_review,
        'review_message': "⚠️ Please review this case" if needs_review else "✓ OK"
    }
    
    return result


def example_usage():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    while True :
    
        print(f"Using device: {device}")
        print("\nSelect model type:")
        print("1. Custom CNN (grayscale, 1-channel)")
        print("2. ResNet152 (3-channel, from your training)")
        print("0. Exit program")
        
        choice = input("Enter choice (1, 2 or 0): ").strip()
        if choice == '':  # Handle empty input
            continue
        
        if choice == '0':
            print("\n👋 Exiting program. Goodbye!")
            sys.exit(0)
        
        elif choice == '2':
            import torchvision.models as models
            
            model_res152 = models.resnet152(weights=None)
            num_features = model_res152.fc.in_features
            model_res152.fc = nn.Sequential(
                nn.Linear(num_features, 256),
                nn.ReLU(inplace=True), 
                nn.Dropout(0.4),
                nn.Linear(256, 2)
            )
            
            model_path = 'models/model2_resnet152.pth'
            if os.path.exists(model_path):
                model_res152.load_state_dict(torch.load(model_path, map_location=device))
                print("✅ ResNet152 model loaded successfully")
            else:
                print(f"❌ Model file not found: {model_path}")
                return
            
            model_to_use = model_res152
            model_type = 'resnet152'
            OPTIMAL_THRESHOLD = 0.5
            
        elif choice == '1':
            # Load your custom CNN model
            from module import improved_cnn
            
            custom_model = improved_cnn(input_shape=1, hidden_units=32, output_shape=2)
            model_path = 'models/model2_improved_cnn.pth'
            
            if os.path.exists(model_path):
                custom_model.load_state_dict(torch.load(model_path, map_location=device))
                print("✅ Custom CNN model loaded successfully")
            else:
                print(f"❌ Model file not found: {model_path}")
                return
            
            model_to_use = custom_model
            model_type = 'custom'
            OPTIMAL_THRESHOLD = 0.55
        
        model_to_use.to(device)
        model_to_use.eval()
        
        # Single image prediction
        try:
            pred_choice = input("\nEnter 1 for single image prediction, 0 for batch prediction: ").strip()
            if pred_choice == '':
                continue
            single_image_pred = int(pred_choice)
        except ValueError:
            print("❌ Invalid input. Please enter 1 or 0")
            continue
        
        if single_image_pred == 1:
            print("\n" + "="*60)
            print("SINGLE IMAGE PREDICTION")
            print("="*60)
            
            while True:
                print("\n" + "-"*40)
                user_input = input("Enter image path (or '0' to exit): ")
                
                if user_input == '0':
                    print("Exiting single image prediction...")
                    return
                
                if os.path.exists(user_input):
                    result = simple_predict(model_to_use, user_input, device, 
                                        OPTIMAL_THRESHOLD, model_type)
                    print_result_simple(result)
                else:
                    print(f"❌ File not found: {user_input}")
                    print("Please check the path and try again.")
        else:
            # Batch prediction
            print("\n" + "="*60)
            print("BATCH PREDICTION")
            print("="*60)
            
            test_normal_dir = "data/reorganized_chest_xray/test/NORMAL"
            test_pneumonia_dir = "data/reorganized_chest_xray/test/PNEUMONIA"
            
            all_results = []
            
            if os.path.exists(test_normal_dir):
                normal_images = list(Path(test_normal_dir).glob("*.jpeg"))[:5]  # First 5 normal images
                for img_path in normal_images:
                    result = simple_predict(model_to_use, img_path, device, 
                                        OPTIMAL_THRESHOLD, model_type)
                    all_results.append(result)
            
            if os.path.exists(test_pneumonia_dir):
                pneumonia_images = list(Path(test_pneumonia_dir).glob("*.jpeg"))[:5]  # First 5 pneumonia images
                for img_path in pneumonia_images:
                    result = simple_predict(model_to_use, img_path, device, 
                                        OPTIMAL_THRESHOLD, model_type)
                    all_results.append(result)
            
            # Print summary
            print(f"\nProcessed {len(all_results)} images:")
            for r in all_results:
                print(f"  {Path(r['image']).name:<30} → {r['prediction']:<10} ({r['probability_percent']}) {r['review_message']}")
            
            # Save results
            save_results_simple(all_results, f'prediction_samples/predictions_{model_type}.txt')


class Plot:
    def __init__(self, style='seaborn-v0_8-darkgrid', figsize=(12, 8), palette='viridis'):
        """
        Initialize Plot class with default styling
        
        Args:
            style: matplotlib style
            figsize: default figure size
            palette: color palette for plots
        """
        plt.style.use(style)
        self.figsize = figsize
        self.palette = palette
        sns.set_palette(palette)
        
    # ============= STAGE 3: Model Performance Evaluation =============
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names=None, 
                              normalize=False, title='Confusion Matrix', 
                              save_path=None):
        """
        Plot confusion matrix heatmap
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize if requested
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = f'Normalized {title}'
        else:
            fmt = 'd'
        
        # Plot
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', ax=ax,
                   xticklabels=class_names, yticklabels=class_names)
        
        ax.set_xlabel('Predicted Labels', fontsize=12)
        ax.set_ylabel('True Labels', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def plot_roc_curves(self, y_true, y_pred_probs, n_classes=None, 
                        class_names=None, title='ROC Curves',
                        save_path=None):
        """
        Plot ROC curves for binary or multi-class classification
        
        Args:
            y_true: true labels
            y_pred_probs: predicted probabilities
            n_classes: number of classes (auto-detected if None)
            class_names: names of classes
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        y_true = np.array(y_true)
        y_pred_probs = np.array(y_pred_probs)
        
        # Determine if binary or multi-class
        if n_classes is None:
            n_classes = len(np.unique(y_true))
        
        # Binary classification
        if n_classes == 2:
            if y_pred_probs.ndim == 2 and y_pred_probs.shape[1] == 2:
                # If we have probabilities for both classes, use positive class
                y_score = y_pred_probs[:, 1]
            else:
                y_score = y_pred_probs
            
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, 'b-', label=f'ROC curve (AUC = {roc_auc:.3f})', 
                   linewidth=2)
            
        # Multi-class classification
        else:
            # Binarize the labels
            y_true_bin = label_binarize(y_true, classes=range(n_classes))
            
            # Compute ROC curve and ROC area for each class
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
                roc_auc = auc(fpr, tpr)
                
                class_label = class_names[i] if class_names else f'Class {i}'
                ax.plot(fpr, tpr, linewidth=2,
                       label=f'{class_label} (AUC = {roc_auc:.3f})')
        
        # Formatting
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def plot_precision_recall_curves(self, y_true, y_pred_probs, n_classes=None,
                                     class_names=None, title='Precision-Recall Curves',
                                     save_path=None):
        """
        Plot Precision-Recall curves
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        y_true = np.array(y_true)
        y_pred_probs = np.array(y_pred_probs)
        
        if n_classes is None:
            n_classes = len(np.unique(y_true))
        
        # Binary classification
        if n_classes == 2:
            if y_pred_probs.ndim == 2 and y_pred_probs.shape[1] == 2:
                y_score = y_pred_probs[:, 1]
            else:
                y_score = y_pred_probs
            
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            
            ax.plot(recall, precision, 'b-', linewidth=2,
                   label='Precision-Recall curve')
            
        # Multi-class classification
        else:
            y_true_bin = label_binarize(y_true, classes=range(n_classes))
            
            for i in range(n_classes):
                precision, recall, _ = precision_recall_curve(
                    y_true_bin[:, i], y_pred_probs[:, i]
                )
                
                class_label = class_names[i] if class_names else f'Class {i}'
                ax.plot(recall, precision, linewidth=2, label=class_label)
        
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def plot_model_comparison(self, model_names, metrics_dict, 
                             title='Model Performance Comparison',
                             save_path=None):
        """
        Plot bar chart comparing multiple models
        
        Args:
            model_names: list of model names
            metrics_dict: dict with metric names as keys and lists of values as values
                        e.g., {'Accuracy': [0.85, 0.87, 0.86], 
                               'F1-Score': [0.84, 0.86, 0.85]}
        """
        n_metrics = len(metrics_dict)
        n_models = len(model_names)
        
        fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 6))
        if n_metrics == 1:
            axes = [axes]
        
        x = np.arange(n_models)
        width = 0.6
        
        for idx, (metric_name, values) in enumerate(metrics_dict.items()):
            ax = axes[idx]
            bars = ax.bar(x, values, width, color=sns.color_palette(self.palette, n_models))
            
            # Customize
            ax.set_xlabel('Models', fontsize=12)
            ax.set_ylabel(metric_name, fontsize=12)
            ax.set_title(f'{metric_name} Comparison', fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(model_names, rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=10)
            
            ax.set_ylim([0, max(values) * 1.1])
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.05)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    # ============= STAGE 4: Model Diagnostics =============
    
    def plot_learning_curves(self, train_losses, val_losses, train_accs=None, 
                            val_accs=None, title='Learning Curves',
                            save_path=None):
        """
        Plot learning curves showing training/validation metrics over epochs
        """
        epochs = range(1, len(train_losses) + 1)
        
        fig, axes = plt.subplots(1, 2 if train_accs else 1, 
                                 figsize=(14, 6) if train_accs else (8, 6))
        
        if train_accs:
            ax1, ax2 = axes
        else:
            ax1 = axes
        
        # Plot loss
        ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epochs', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Loss Curves', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot accuracy if provided
        if train_accs:
            ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2)
            ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
            ax2.set_xlabel('Epochs', fontsize=12)
            ax2.set_ylabel('Accuracy', fontsize=12)
            ax2.set_title('Accuracy Curves', fontsize=12, fontweight='bold')
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def plot_validation_curves(self, param_name, param_values, train_scores, 
                              val_scores, title='Validation Curve',
                              save_path=None):
        """
        Plot validation curves for hyperparameter tuning
        
        Args:
            param_name: name of the hyperparameter
            param_values: list of parameter values
            train_scores: list of training scores for each parameter value
            val_scores: list of validation scores for each parameter value
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate mean and std if scores are provided as arrays
        if isinstance(train_scores[0], (list, np.ndarray)):
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)
            
            ax.plot(param_values, train_mean, 'o-', label='Training score', 
                   linewidth=2, markersize=8)
            ax.fill_between(param_values, train_mean - train_std, 
                           train_mean + train_std, alpha=0.15)
            
            ax.plot(param_values, val_mean, 'o-', label='Validation score', 
                   linewidth=2, markersize=8)
            ax.fill_between(param_values, val_mean - val_std, 
                           val_mean + val_std, alpha=0.15)
        else:
            ax.plot(param_values, train_scores, 'o-', label='Training score', 
                   linewidth=2, markersize=8)
            ax.plot(param_values, val_scores, 'o-', label='Validation score', 
                   linewidth=2, markersize=8)
        
        ax.set_xlabel(param_name, fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Optional: mark the best parameter
        best_idx = np.argmax(val_scores if not isinstance(val_scores[0], (list, np.ndarray)) 
                            else np.mean(val_scores, axis=1))
        ax.axvline(x=param_values[best_idx], color='green', linestyle='--', 
                  alpha=0.5, label=f'Best: {param_values[best_idx]}')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def plot_error_analysis(self, y_true, y_pred, X_data=None, 
                           feature_names=None, title='Error Analysis',
                           save_path=None):
        """
        Plot error analysis: misclassification patterns
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Find misclassified samples
        misclassified = y_true != y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. Bar chart of misclassifications per class
        ax1 = axes[0, 0]
        classes = np.unique(y_true)
        misclass_counts = []
        
        for cls in classes:
            class_mask = y_true == cls
            misclass_in_class = np.sum((y_true == cls) & (y_pred != cls))
            total_in_class = np.sum(class_mask)
            misclass_counts.append(misclass_in_class)
            
            # Add percentage label
            percentage = (misclass_in_class / total_in_class) * 100 if total_in_class > 0 else 0
            print(f"Class {cls}: {misclass_in_class}/{total_in_class} misclassified ({percentage:.1f}%)")
        
        bars = ax1.bar(range(len(classes)), misclass_counts, 
                      color=sns.color_palette(self.palette, len(classes)))
        ax1.set_xlabel('Class', fontsize=12)
        ax1.set_ylabel('Number of Misclassifications', fontsize=12)
        ax1.set_title('Misclassifications per Class', fontsize=12, fontweight='bold')
        ax1.set_xticks(range(len(classes)))
        ax1.set_xticklabels([f'Class {c}' for c in classes])
        
        # Add value labels
        for bar, count in zip(bars, misclass_counts):
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom')
        
        # 2. Confusion matrix of errors only
        ax2 = axes[0, 1]
        cm = confusion_matrix(y_true, y_pred)
        
        # Create error matrix (non-diagonal entries only)
        error_cm = cm.copy()
        np.fill_diagonal(error_cm, 0)
        
        sns.heatmap(error_cm, annot=True, fmt='d', cmap='Reds', ax=ax2,
                   xticklabels=[f'Pred {c}' for c in classes],
                   yticklabels=[f'True {c}' for c in classes])
        ax2.set_title('Error Confusion Matrix\n(Diagonal removed)', fontsize=12, fontweight='bold')
        
        # 3. If feature data is provided, plot feature distributions for errors vs correct
        if X_data is not None and feature_names is not None:
            ax3 = axes[1, 0]
            X_data = np.array(X_data)
            
            # Select first few features for visualization (max 5)
            n_features_to_plot = min(5, X_data.shape[1])
            feature_indices = np.random.choice(X_data.shape[1], n_features_to_plot, replace=False)
            
            # Prepare data
            correct_mask = ~misclassified
            data_to_plot = []
            labels = []
            
            for idx in feature_indices:
                data_to_plot.extend([X_data[correct_mask, idx], X_data[misclassified, idx]])
                labels.extend([f'{feature_names[idx]}\n(Correct)', f'{feature_names[idx]}\n(Error)'])
            
            # Create box plot
            bp = ax3.boxplot(data_to_plot, labels=labels, patch_artist=True)
            
            # Color boxes
            colors = ['green', 'red'] * n_features_to_plot
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            
            ax3.set_ylabel('Feature Value', fontsize=12)
            ax3.set_title('Feature Distributions:\nCorrect vs Misclassified', 
                         fontsize=12, fontweight='bold')
            ax3.tick_params(axis='x', rotation=45)
        
        # 4. Confidence scores for errors vs correct
        ax4 = axes[1, 1]
        
        # If we have probability scores, we can plot them
        # This is a placeholder - you'd need to pass probability scores
        ax4.text(0.5, 0.5, 'Add prediction probabilities\nfor confidence analysis',
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Confidence Analysis', fontsize=12, fontweight='bold')
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def plot_calibration_curve(self, y_true, y_pred_probs, n_bins=10,
                              title='Calibration Curve (Reliability Diagram)',
                              save_path=None):
        """
        Plot calibration curve to check if predicted probabilities are well-calibrated
        """
        from sklearn.calibration import calibration_curve
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        y_true = np.array(y_true)
        y_pred_probs = np.array(y_pred_probs)
        
        # For binary classification
        if y_pred_probs.ndim == 2 and y_pred_probs.shape[1] == 2:
            y_pred_probs = y_pred_probs[:, 1]
        
        fraction_pos, mean_pred_val = calibration_curve(y_true, y_pred_probs, 
                                                        n_bins=n_bins)
        
        # Plot calibration curve
        ax.plot(mean_pred_val, fraction_pos, 's-', label='Model', 
               linewidth=2, markersize=8)
        
        # Plot perfectly calibrated line
        ax.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated', linewidth=1)
        
        # Plot histogram of predictions
        ax.hist(y_pred_probs, range=(0, 1), bins=n_bins, histtype='step', 
               lw=2, label='Distribution', alpha=0.7, color='gray')
        
        ax.set_xlabel('Mean Predicted Probability', fontsize=12)
        ax.set_ylabel('Fraction of Positives', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig


if __name__ == "__main__":
    example_usage()
