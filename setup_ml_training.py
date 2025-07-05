#!/usr/bin/env python3
"""
Setup Script for ML-Based Bank Transaction ID Column Detection
Helps you set up training for your reconciliation files
"""

import os
import sys

def main():
    print("🎯 SmartRecon ML Column Detection Setup")
    print("=" * 45)
    print()
    print("This will help you train a machine learning model to automatically")
    print("detect Bank Transaction ID columns in YOUR reconciliation files.")
    print()
    print("📋 What you need:")
    print("   • Your reconciliation Excel/CSV files")
    print("   • Know which column contains Bank Trx IDs in each file")
    print("   • 5-10 minutes to label the files")
    print()
    
    # Check PyTorch installation
    try:
        import torch
        print("✅ PyTorch is installed")
    except ImportError:
        print("❌ PyTorch not installed!")
        print("📦 Please install with: pip install torch")
        return
    
    try:
        import sklearn
        print("✅ scikit-learn is installed")
    except ImportError:
        print("❌ scikit-learn not installed!")
        print("📦 Please install with: pip install scikit-learn")
        return
    
    print()
    
    # Get reconciliation files directory
    while True:
        files_dir = input("📁 Enter path to your reconciliation files directory: ").strip()
        if os.path.exists(files_dir):
            break
        else:
            print("❌ Directory not found. Please check the path.")
    
    # Create training setup
    print(f"\n🔍 Scanning for Excel/CSV files in {files_dir}...")
    
    from ml_column_detector import create_training_setup
    our_files = create_training_setup(files_dir)
    
    if not our_files:
        print("❌ No Excel/CSV files found!")
        return
    
    print(f"\n📋 Next Steps:")
    print(f"1. ✏️  Edit training_labels.py to specify Bank Trx ID columns")
    print(f"2. 🚀 Run: python train_ml_detector.py")
    print(f"3. 🎉 Your ML model will be integrated into SmartRecon automatically")
    print()
    print(f"💡 Example labels in training_labels.py:")
    print(f'   "settlement_jan_2024.xlsx": "Bank Trx ID",')
    print(f'   "recon_file_feb.xlsx": "Bank Transaction ID",')
    print(f'   "our_data_march.csv": "Payment Reference",')
    print()
    print(f"🎯 Benefits after training:")
    print(f"   • Much more accurate column detection")
    print(f"   • Learns YOUR specific naming patterns")
    print(f"   • Adapts to new file formats automatically")
    print(f"   • Saves time on manual column selection")

if __name__ == "__main__":
    main() 