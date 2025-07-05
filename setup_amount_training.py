#!/usr/bin/env python3
"""
Setup Training Data for Amount Column Detection
Scan your reconciliation files and create labeled training data for ML amount detection
"""

import os
import glob
import pandas as pd
from typing import List, Dict

def scan_reconciliation_files(directory: str) -> List[str]:
    """Scan directory for Excel/CSV files"""
    files = []
    for ext in ['*.xlsx', '*.xls', '*.csv']:
        files.extend(glob.glob(os.path.join(directory, '**', ext), recursive=True))
    return files

def create_amount_training_labels(our_files: List[str]) -> str:
    """Create training labels template for amount columns"""
    
    print(f"🔍 Analyzing {len(our_files)} files for amount column patterns...")
    
    template_content = '''"""
Training Labels for Amount Column Detection - SmartRecon
Edit this file to specify which column contains payment amounts in each of your reconciliation files

Based on your patterns, common amount column names are:
- "Paid Amt" (in your system files)
- "Credit", "Credit دائن", "PAYMENT AMOUNT", "Credit Amount" (in bank files)
- "amount_credited", "Amount", "credit" (variations)
"""

# Dictionary mapping filename -> Amount column name
# Use the EXACT column name as it appears in your Excel files
AMOUNT_LABELED_COLUMNS = {
'''
    
    # Analyze files and suggest amount columns
    for file_path in our_files[:20]:  # Analyze first 20 files
        filename = os.path.basename(file_path)
        
        try:
            # Load file to analyze columns
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path, nrows=5)  # Sample first 5 rows
            else:
                df = pd.read_excel(file_path, nrows=5)  # Sample first 5 rows
            
            columns = list(df.columns)
            
            # Suggest likely amount columns
            amount_candidates = []
            for col in columns:
                col_lower = str(col).lower().strip()
                
                # Check for amount keywords
                if any(keyword in col_lower for keyword in [
                    'paid amt', 'amount', 'amt', 'credit', 'payment', 'value', 
                    'money', 'sum', 'total', 'balance', 'debit', 'دائن'
                ]):
                    # Analyze if column contains numeric data
                    try:
                        sample_data = df[col].dropna().head(3)
                        numeric_count = 0
                        
                        for val in sample_data:
                            try:
                                # Try to convert to number
                                val_str = str(val).replace(',', '').replace('$', '').strip()
                                if val_str and val_str not in ['nan', 'none', 'null']:
                                    float(val_str)
                                    numeric_count += 1
                            except:
                                pass
                        
                        # If mostly numeric, it's a good candidate
                        if numeric_count >= len(sample_data) * 0.7:
                            amount_candidates.append(col)
                            
                    except Exception:
                        pass
            
            # Create template entry
            template_content += f'''
    # File: {filename}
    # Available columns: {columns}
    # Suggested amount columns: {amount_candidates}
    "{filename}": "",  # <- Put the amount column name here (e.g., "Paid Amt")
'''
            
        except Exception as e:
            template_content += f'''
    # File: {filename} - Error loading: {e}
    "{filename}": "",
'''
    
    template_content += '\n}\n'
    
    if len(our_files) > 20:
        template_content += f'\n# Note: Found {len(our_files)} total files (showing first 20)\n# Add more files manually if needed\n'
    
    return template_content

def main():
    print("💰 SmartRecon Amount Column Detection Setup")
    print("=" * 50)
    
    # Get directory containing user's reconciliation files
    our_files_dir = input("📁 Enter path to your reconciliation files directory: ").strip()
    if not our_files_dir:
        our_files_dir = r"C:\Users\moham\Downloads\madfoat_test_data\TEST_AI"  # Default to user's known directory
        print(f"Using default directory: {our_files_dir}")
    
    if not os.path.exists(our_files_dir):
        print("❌ Directory not found!")
        return
    
    # Scan for files
    our_files = scan_reconciliation_files(our_files_dir)
    
    if not our_files:
        print("❌ No Excel/CSV files found in the directory!")
        return
    
    print(f"✅ Found {len(our_files)} reconciliation files")
    
    # Create training labels template
    template_content = create_amount_training_labels(our_files)
    
    # Save training labels template
    with open("amount_training_labels.py", "w", encoding='utf-8') as f:
        f.write(template_content)
    
    print("✅ Created amount_training_labels.py")
    print("\n📋 Next steps:")
    print("1. Edit amount_training_labels.py to specify amount columns")
    print("2. Run: python train_amount_detector.py")
    print("3. The trained amount detector will be integrated into SmartRecon")
    print("\n💡 Tips:")
    print("- Look for columns like 'Paid Amt', 'Credit', 'Amount', 'Payment Amount'")
    print("- Choose columns that contain actual payment amounts (numbers)")
    print("- Use exact column names as they appear in your files")

if __name__ == "__main__":
    main() 