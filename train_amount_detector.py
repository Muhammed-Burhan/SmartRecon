#!/usr/bin/env python3
"""
Train Amount Column Detector for SmartRecon
Train ML model to automatically detect payment amount columns in your reconciliation files
"""

import os
import glob
import sys
from typing import List, Dict

# Import our ML components
from ml_column_detector import MLColumnDetector

def get_our_files(directory: str) -> List[str]:
    """Get list of reconciliation files from directory"""
    files = []
    for ext in ['*.xlsx', '*.xls', '*.csv']:
        files.extend(glob.glob(os.path.join(directory, '**', ext), recursive=True))
    return files

def load_amount_labels() -> Dict[str, str]:
    """Load amount column labels from training file"""
    try:
        # Import the labels
        import amount_training_labels
        return amount_training_labels.AMOUNT_LABELED_COLUMNS
    except ImportError:
        print("❌ amount_training_labels.py not found!")
        print("📝 Please run: python setup_amount_training.py first")
        sys.exit(1)
    except AttributeError:
        print("❌ AMOUNT_LABELED_COLUMNS not found in amount_training_labels.py")
        print("📝 Please check the file format")
        sys.exit(1)

def validate_labels(our_files: List[str], labeled_columns: Dict[str, str]) -> bool:
    """Validate that labels are properly set"""
    valid_labels = 0
    total_files = 0
    
    for file_path in our_files:
        filename = os.path.basename(file_path)
        if filename in labeled_columns:
            total_files += 1
            label = labeled_columns[filename]
            if label and label.strip():
                valid_labels += 1
                print(f"✅ {filename}: '{label}'")
            else:
                print(f"⚠️ {filename}: No amount column specified")
    
    print(f"\n📊 Label Summary:")
    print(f"   Files with labels: {total_files}")
    print(f"   Valid amount labels: {valid_labels}")
    
    if valid_labels == 0:
        print("❌ No valid amount labels found!")
        print("📝 Please edit amount_training_labels.py and specify amount columns")
        return False
    
    if valid_labels < 3:
        print("⚠️ Very few labeled examples. Consider labeling more files for better accuracy.")
    
    return True

def interactive_training():
    """Interactive training process with user guidance"""
    print("💰 SmartRecon Amount Column Detector Training")
    print("=" * 50)
    
    # Check if training labels exist
    if not os.path.exists("amount_training_labels.py"):
        print("❌ amount_training_labels.py not found!")
        print("📝 Please run: python setup_amount_training.py first")
        return
    
    # Load amount labels
    print("📂 Loading amount training labels...")
    labeled_columns = load_amount_labels()
    
    # Get user's reconciliation files directory
    our_files_dir = input("📁 Enter path to your reconciliation files directory (or press Enter for default): ").strip()
    if not our_files_dir:
        our_files_dir = r"C:\Users\moham\Downloads\madfoat_test_data\TEST_AI"
        print(f"Using default directory: {our_files_dir}")
    
    if not os.path.exists(our_files_dir):
        print("❌ Directory not found!")
        return
    
    # Get our files
    our_files = get_our_files(our_files_dir)
    print(f"📁 Found {len(our_files)} reconciliation files")
    
    # Validate labels
    if not validate_labels(our_files, labeled_columns):
        return
    
    # Initialize ML detector
    print("\n🤖 Initializing Amount Column Detector...")
    detector = MLColumnDetector()
    
    # Training parameters
    epochs = int(input("🔢 Number of training epochs (default: 100): ").strip() or "100")
    learning_rate = float(input("📈 Learning rate (default: 0.001): ").strip() or "0.001")
    
    print(f"\n🚀 Starting amount detection training...")
    print(f"   Training files: {len(our_files)}")
    print(f"   Labeled columns: {sum(1 for v in labeled_columns.values() if v.strip())}")
    print(f"   Epochs: {epochs}")
    print(f"   Learning rate: {learning_rate}")
    
    try:
        # Train the amount detector
        detector.train_amount_detector(
            our_files=our_files,
            labeled_amount_columns=labeled_columns,
            epochs=epochs,
            learning_rate=learning_rate
        )
        
        # Save the trained model
        model_path = "enhanced_column_detector.pth"
        detector.save_model(model_path)
        
        print(f"\n🎉 Amount detection training completed!")
        print(f"💾 Model saved to: {model_path}")
        print(f"📊 Model file size: {os.path.getsize(model_path) / 1024:.1f} KB")
        
        # Test the trained model
        print(f"\n🧪 Testing amount detection on training data...")
        test_file = our_files[0]  # Test on first file
        
        import pandas as pd
        if test_file.endswith('.csv'):
            test_df = pd.read_csv(test_file)
        else:
            test_df = pd.read_excel(test_file)
        
        predicted_column, confidence = detector.predict_amount_column(test_df)
        expected_column = labeled_columns.get(os.path.basename(test_file), "")
        
        print(f"📋 Test Results on {os.path.basename(test_file)}:")
        print(f"   Expected: '{expected_column}'")
        print(f"   Predicted: '{predicted_column}'")
        print(f"   Confidence: {confidence:.1f}%")
        
        if predicted_column == expected_column:
            print("   ✅ Perfect match!")
        else:
            print("   ⚠️ Mismatch - may need more training data")
        
        print(f"\n🔧 Integration Instructions:")
        print(f"1. The amount detector is now ready for use")
        print(f"2. It will be automatically loaded by SmartRecon")
        print(f"3. Run reconciliation as usual - amount comparison will be included")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print(f"📝 Please check your training labels and try again")

def quick_training():
    """Quick training with default settings"""
    print("🚀 Quick Amount Detection Training")
    print("=" * 40)
    
    # Load labels
    try:
        labeled_columns = load_amount_labels()
    except SystemExit:
        return
    
    # Use default directory
    our_files_dir = r"C:\Users\moham\Downloads\madfoat_test_data\TEST_AI"
    our_files = get_our_files(our_files_dir)
    
    if not validate_labels(our_files, labeled_columns):
        return
    
    # Quick training
    detector = MLColumnDetector()
    detector.train_amount_detector(our_files, labeled_columns, epochs=100)
    detector.save_model("enhanced_column_detector.pth")
    
    print("✅ Quick training completed!")

def main():
    print("Choose training mode:")
    print("1. Interactive Training (recommended)")
    print("2. Quick Training (default settings)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "2":
        quick_training()
    else:
        interactive_training()

if __name__ == "__main__":
    main() 