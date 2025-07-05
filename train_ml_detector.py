#!/usr/bin/env python3
"""
Training Script for ML-Based Bank Transaction ID Column Detector
Train the model on your reconciliation files to improve accuracy
"""

import os
import glob
import sys
from ml_column_detector import MLColumnDetector, create_training_setup

def main():
    print("🤖 SmartRecon ML Column Detector Training")
    print("=" * 50)
    
    # Check if training labels exist
    if not os.path.exists("training_labels.py"):
        print("❌ training_labels.py not found!")
        print("📝 Please run the setup first:")
        print("   python ml_column_detector.py")
        return
    
    # Import training labels
    try:
        from training_labels import LABELED_COLUMNS
    except ImportError:
        print("❌ Could not import LABELED_COLUMNS from training_labels.py")
        print("📝 Please check the file format")
        return
    
    # Validate labels
    labeled_files = [f for f, col in LABELED_COLUMNS.items() if col.strip()]
    if not labeled_files:
        print("❌ No files labeled in training_labels.py!")
        print("📝 Please edit training_labels.py and specify Bank Trx ID columns")
        return
    
    print(f"📊 Found {len(labeled_files)} labeled files:")
    for filename, column in LABELED_COLUMNS.items():
        if column.strip():
            print(f"   📁 {filename} → '{column}'")
    
    # Find file paths
    our_files_dir = input("\n📁 Enter path to your reconciliation files directory: ").strip()
    if not os.path.exists(our_files_dir):
        print("❌ Directory not found!")
        return
    
    # Get all Excel/CSV files
    our_files = []
    for ext in ['*.xlsx', '*.xls', '*.csv']:
        our_files.extend(glob.glob(os.path.join(our_files_dir, '**', ext), recursive=True))
    
    # Filter to only labeled files
    labeled_file_paths = []
    for file_path in our_files:
        filename = os.path.basename(file_path)
        if filename in LABELED_COLUMNS and LABELED_COLUMNS[filename].strip():
            labeled_file_paths.append(file_path)
    
    if not labeled_file_paths:
        print("❌ No labeled files found in the directory!")
        print("📝 Make sure your files are in the directory and match the names in training_labels.py")
        return
    
    print(f"\n🔍 Found {len(labeled_file_paths)} labeled files to train on")
    
    # Confirm training parameters
    print(f"\n⚙️ Training Configuration:")
    print(f"   📁 Files: {len(labeled_file_paths)}")
    print(f"   🧠 Model: Neural Network (PyTorch)")
    print(f"   📊 Features: 61 enhanced data-learning features per column")
    print(f"   🎯 Task: Bank Trx ID column detection")
    
    confirm = input(f"\n🚀 Start training? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Training cancelled.")
        return
    
    # Initialize and train the model
    try:
        detector = MLColumnDetector()
        
        print(f"\n🚀 Starting training...")
        detector.train(
            our_files=labeled_file_paths,
            labeled_columns=LABELED_COLUMNS,
            epochs=150,
            learning_rate=0.001
        )
        
        # Save the trained model
        model_path = "bank_trx_detector.pth"
        detector.save_model(model_path)
        
        print(f"\n🎉 Training completed successfully!")
        print(f"💾 Model saved as: {model_path}")
        
        # Test the model on one of the training files
        print(f"\n🧪 Testing model on a sample file...")
        test_file = labeled_file_paths[0]
        test_filename = os.path.basename(test_file)
        
        import pandas as pd
        if test_file.endswith('.csv'):
            test_df = pd.read_csv(test_file)
        else:
            test_df = pd.read_excel(test_file)
        
        predicted_column, confidence = detector.predict_column(test_df)
        true_column = LABELED_COLUMNS[test_filename]
        
        print(f"   📁 Test file: {test_filename}")
        print(f"   🎯 True column: '{true_column}'")
        print(f"   🔮 Predicted: '{predicted_column}' (confidence: {confidence:.1f}%)")
        print(f"   ✅ Result: {'CORRECT' if predicted_column == true_column else 'INCORRECT'}")
        
        print(f"\n📈 Next Steps:")
        print(f"   1. The model is now integrated into SmartRecon")
        print(f"   2. It will automatically detect Bank Trx ID columns")
        print(f"   3. Add more files to training_labels.py to improve accuracy")
        print(f"   4. Re-run this script to retrain with new data")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 