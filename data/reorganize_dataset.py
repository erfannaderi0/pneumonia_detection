"""
the original chest xray dataset images(pneumonia) from kaggle is a little imbalance
and out of total nearly 5800 samples only 16 of them are for validation which is a problem
this code creates a new structure inwhich 15% of the trainning data specifies for validation and
the test data will be untouch the same as the original dataset
by the way the ration of pneumonia to normal sample in train data is around 3 to 1 in the original dataset
which will be the same in the new one too and we try to deal with this issue with applying class weights
"""

import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1. UNDERSTAND THE ORIGINAL STRUCTURE

def analyze_dataset_structure(base_path='data/chest_xray'):
    """Analyzing the original dataset distribution"""
    print("Original Dataset Structure:")
    print("=" * 50)
    
    total_counts = {}
    for split in ['train', 'test', 'val']:
        split_path = os.path.join(base_path, split)
        normal_count = len(os.listdir(os.path.join(split_path, 'NORMAL'))) if os.path.exists(os.path.join(split_path, 'NORMAL')) else 0
        pneumonia_count = len(os.listdir(os.path.join(split_path, 'PNEUMONIA'))) if os.path.exists(os.path.join(split_path, 'PNEUMONIA')) else 0
        
        total_counts[split] = {
            'NORMAL': normal_count,
            'PNEUMONIA': pneumonia_count,
            'TOTAL': normal_count + pneumonia_count
        }
        
        print(f"\n{split.upper()} SET:")
        print(f"  Normal: {normal_count}")
        print(f"  Pneumonia: {pneumonia_count}")
        print(f"  Total: {normal_count + pneumonia_count}")
        print(f"  Pneumonia Ratio: {pneumonia_count/(normal_count + pneumonia_count):.1%}")
    
    return total_counts

original_stats = analyze_dataset_structure('data/chest_xray')

# 2. CREATE NEW VALIDATION SET FROM TRAINING DATA

def create_stratified_split(base_path='data/chest_xray', val_size=0.15, random_seed=42):
    """
    Creating a new stratified validation set from training data
    while keeping the original test set untouched
    """
    
    train_dir = os.path.join(base_path, 'train')
    normal_dir = os.path.join(train_dir, 'NORMAL')
    pneumonia_dir = os.path.join(train_dir, 'PNEUMONIA')
    
    normal_images = [os.path.join(normal_dir, f) for f in os.listdir(normal_dir) if f.endswith(('.jpeg', '.jpg', '.png'))]
    pneumonia_images = [os.path.join(pneumonia_dir, f) for f in os.listdir(pneumonia_dir) if f.endswith(('.jpeg', '.jpg', '.png'))]
    
    # create labels
    normal_labels = [0] * len(normal_images)  # 0 = NORMAL
    pneumonia_labels = [1] * len(pneumonia_images)  # 1 = PNEUMONIA
    
    # combine all data
    all_images = normal_images + pneumonia_images
    all_labels = normal_labels + pneumonia_labels
    
    # stratified split
    print(f"\nSplitting {len(all_images)} training images...")
    X_train, X_val, y_train, y_val = train_test_split(
        all_images, all_labels, 
        test_size=val_size, 
        stratify=all_labels,  # maintaining imbalance ratio
        random_state=random_seed
    )
    
    print(f"New training set: {len(X_train)} images")
    print(f"New validation set: {len(X_val)} images")
    
    # analyzing class distribution in new splits
    def analyze_split(images, labels, name):
        normal_count = sum(1 for label in labels if label == 0)
        pneumonia_count = sum(1 for label in labels if label == 1)
        print(f"\n{name}:")
        print(f"  Normal: {normal_count} ({normal_count/len(images):.1%})")
        print(f"  Pneumonia: {pneumonia_count} ({pneumonia_count/len(images):.1%})")
    
    analyze_split(X_train, y_train, "TRAINING SET")
    analyze_split(X_val, y_val, "VALIDATION SET")
    
    return X_train, X_val, y_train, y_val, all_images, all_labels

# 3. ORGANIZE INTO NEW DIRECTORY STRUCTURE

def create_new_directory_structure(base_path='data/chest_xray', X_train=None, X_val=None, y_train=None, y_val=None):
    """Create organized directories for the new split"""
    
    # create new directory structure
    new_base = 'data/reorganized_chest_xray'
    splits = ['train', 'val', 'test']
    classes = ['NORMAL', 'PNEUMONIA']
    
    # incase of rerunning the code remove old reorganized directory if exists
    if os.path.exists(new_base):
        shutil.rmtree(new_base)
    
    #creating new directories itself
    for split in splits:
        for cls in classes:
            os.makedirs(os.path.join(new_base, split, cls), exist_ok=True)
    
    def copy_images(image_paths, labels, destination_split):
        normal_count = 0
        pneumonia_count = 0
        
        for img_path, label in zip(image_paths, labels):
            filename = os.path.basename(img_path)
            if label == 0:  # NORMAL
                dest_dir = os.path.join(new_base, destination_split, 'NORMAL')
                normal_count += 1
            else:  # PNEUMONIA
                dest_dir = os.path.join(new_base, destination_split, 'PNEUMONIA')
                pneumonia_count += 1
            
            shutil.copy2(img_path, os.path.join(dest_dir, filename))
        
        return normal_count, pneumonia_count
    
    # copy training and validation images
    print(f"\nCopying images to new structure...")
    train_normal, train_pneumonia = copy_images(X_train, y_train, 'train')
    val_normal, val_pneumonia = copy_images(X_val, y_val, 'val')
    
    # copy original test set (unchanged)
    original_test_dir = os.path.join(base_path, 'test')
    for cls in classes:
        src_dir = os.path.join(original_test_dir, cls)
        if os.path.exists(src_dir):
            for img in os.listdir(src_dir):
                if img.endswith(('.jpeg', '.jpg', '.png')):
                    shutil.copy2(
                        os.path.join(src_dir, img),
                        os.path.join(new_base, 'test', cls, img)
                    )
    
    # get test counts
    test_normal = len(os.listdir(os.path.join(new_base, 'test', 'NORMAL')))
    test_pneumonia = len(os.listdir(os.path.join(new_base, 'test', 'PNEUMONIA')))
    
    # print summary
    print("\n" + "=" * 50)
    print("NEW DATASET STRUCTURE SUMMARY")
    print("=" * 50)
    
    data = [
        ("TRAIN", train_normal, train_pneumonia, train_normal + train_pneumonia),
        ("VAL", val_normal, val_pneumonia, val_normal + val_pneumonia),
        ("TEST", test_normal, test_pneumonia, test_normal + test_pneumonia)
    ]
    
    for split_name, normal, pneumonia, total in data:
        print(f"\n{split_name}:")
        print(f"  Normal: {normal} ({normal/total:.1%})")
        print(f"  Pneumonia: {pneumonia} ({pneumonia/total:.1%})")
        print(f"  Total: {total}")
    
    return new_base

# 4. VISUALIZE THE DISTRIBUTION

def visualize_distribution(original_stats, new_base='data/reorganized_chest_xray'):
    """comparing original vs new distributions"""
    
    new_stats = {}
    for split in ['train', 'val', 'test']:
        normal_dir = os.path.join(new_base, split, 'NORMAL')
        pneumonia_dir = os.path.join(new_base, split, 'PNEUMONIA')
        
        normal_count = len(os.listdir(normal_dir)) if os.path.exists(normal_dir) else 0
        pneumonia_count = len(os.listdir(pneumonia_dir)) if os.path.exists(pneumonia_dir) else 0
        
        new_stats[split] = {
            'NORMAL': normal_count,
            'PNEUMONIA': pneumonia_count,
            'TOTAL': normal_count + pneumonia_count
        }
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # original distribution
    splits = ['train', 'test', 'val']
    original_counts = [original_stats[s]['TOTAL'] for s in splits]
    new_counts = [new_stats[s]['TOTAL'] for s in ['train', 'val', 'test']]
    
    x = np.arange(len(splits))
    width = 0.35
    
    axes[0].bar(x - width/2, original_counts, width, label='Original', color='skyblue')
    axes[0].bar(x + width/2, new_counts, width, label='Reorganized', color='lightcoral')
    axes[0].set_xlabel('Dataset Split')
    axes[0].set_ylabel('Number of Images')
    axes[0].set_title('Total Images Comparison')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(['Train', 'Test', 'Val'])
    axes[0].legend()
    
    # class distribution in new training set
    labels = ['Normal', 'Pneumonia']
    train_counts = [new_stats['train']['NORMAL'], new_stats['train']['PNEUMONIA']]
    
    axes[1].bar(labels, train_counts, color=['lightgreen', 'salmon'])
    axes[1].set_ylabel('Count')
    axes[1].set_title('Class Distribution in New Training Set')
    
    for i, count in enumerate(train_counts):
        axes[1].text(i, count + 50, str(count), ha='center')
    
    plt.tight_layout()
    plt.savefig('pictures/dataset_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nReorganized dataset saved to: {new_base}/")
    print(f"Visualization saved as: dataset_distribution.png")

# 5. MAIN EXECUTION

if __name__ == "__main__":
    print("CHEST X-RAY DATASET REORGANIZATION")
    print("=" * 50)
    
    if not os.path.exists('data/reorganized_chest_xray'):
        # step 1: Analyze original
        original_stats = analyze_dataset_structure()
        
        # step 2: Create stratified split
        X_train, X_val, y_train, y_val, all_images, all_labels = create_stratified_split(val_size=0.15)
        
        # step 3: Create new directory structure
        new_base = create_new_directory_structure(
            base_path='data/chest_xray',
            X_train=X_train,
            X_val=X_val,
            y_train=y_train,
            y_val=y_val
        )
        
        # step 4: Visualize results
        visualize_distribution(original_stats, new_base)
        
        print("\n✅ Data reorganization complete!")
    else:
        print("✅ Reorganized dataset already exists, skipping reorganization...")
