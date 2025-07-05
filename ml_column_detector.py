#!/usr/bin/env python3
"""
Enhanced ML-Based Column Detector for SmartRecon
Learns from YOUR reconciliation files to identify Bank Trx ID and Amount columns accurately
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import logging
from typing import List, Tuple, Dict, Optional
import os
import glob

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedColumnDetector(nn.Module):
    """Neural Network for detecting Bank Transaction ID and Amount columns"""
    
    def __init__(self, input_size=61):
        super(EnhancedColumnDetector, self).__init__()
        # Separate networks for each column type
        self.bank_trx_network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
            # No Sigmoid here - BCEWithLogitsLoss will handle it
        )
        
        self.amount_network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
            # No Sigmoid here - BCEWithLogitsLoss will handle it
        )
    
    def forward_bank_trx(self, x):
        return self.bank_trx_network(x)
    
    def forward_amount(self, x):
        return self.amount_network(x)
    
    def predict_proba_bank_trx(self, x):
        """Get probability output for Bank Trx ID detection"""
        logits = self.bank_trx_network(x)
        return torch.sigmoid(logits)
    
    def predict_proba_amount(self, x):
        """Get probability output for Amount detection"""
        logits = self.amount_network(x)
        return torch.sigmoid(logits)

class MLColumnDetector:
    """Enhanced ML-based column detector that learns from your reconciliation files"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def extract_features(self, df: pd.DataFrame, column_name: str) -> np.ndarray:
        """Extract smart features from a column for ML prediction"""
        features = []
        
        # 1. Column Name Features (25 features)
        col_name = str(column_name).lower().strip()
        col_clean = re.sub(r'[^a-z0-9]', '', col_name)
        
        # Bank Trx ID keywords (6 features)
        trx_keywords = ['bank', 'trx', 'transaction', 'id', 'ref', 'reference']
        for keyword in trx_keywords:
            features.append(1 if keyword in col_name else 0)
        
        # Related keywords (7 features)
        related_keywords = ['payment', 'txn', 'number', 'no', 'code', 'key', 'pay']
        for keyword in related_keywords:
            features.append(1 if keyword in col_name else 0)
        
        # Column name patterns (12 features)
        features.extend([
            len(col_name),  # Name length
            col_name.count('_'),  # Underscore count
            col_name.count(' '),  # Space count
            1 if col_name.isupper() else 0,  # All uppercase
            1 if col_name.islower() else 0,  # All lowercase
            1 if any(c.isdigit() for c in col_name) else 0,  # Contains digits
            1 if 'trx' in col_name and 'id' in col_name else 0,  # Both trx and id
            1 if 'bank' in col_name and ('trx' in col_name or 'id' in col_name) else 0,  # Bank + trx/id
            1 if col_name.startswith('bank') else 0,  # Starts with 'bank'
            1 if col_name.endswith('id') else 0,  # Ends with 'id'
            1 if 'ref' in col_name and ('trx' in col_name or 'transaction' in col_name) else 0,  # Ref + transaction
            1 if re.search(r'trx.*id|id.*trx', col_name) else 0  # Trx and ID nearby
        ])
        # Total so far: 6 + 7 + 12 = 25 features
        
        # 2. Data Characteristics Features (15 features)
        column_data = df[column_name].dropna()
        
        if len(column_data) > 0:
            str_data = column_data.astype(str)
            
            # Basic statistics (3 features)
            unique_count = len(str_data.unique())
            total_count = len(str_data)
            
            features.extend([
                total_count,  # Non-null count
                unique_count,  # Unique count
                unique_count / max(total_count, 1),  # Uniqueness ratio (very important for Trx IDs)
            ])
            
            # String length analysis (4 features)
            lengths = [len(str(val).strip()) for val in str_data.head(100)]  # Sample first 100
            if lengths:
                features.extend([
                    np.mean(lengths),  # Average length
                    np.std(lengths),   # Length consistency (low std = good)
                    min(lengths),      # Min length
                    max(lengths),      # Max length
                ])
            else:
                features.extend([0, 0, 0, 0])
            
            # Character pattern analysis (4 features)
            sample_data = str_data.head(50)  # Sample for speed
            
            # Count different character types
            has_letters = sum(1 for val in sample_data if any(c.isalpha() for c in str(val)))
            has_digits = sum(1 for val in sample_data if any(c.isdigit() for c in str(val)))
            has_special = sum(1 for val in sample_data if any(c in str(val) for c in '-_/\\()[]{}'))
            has_long_values = sum(1 for val in sample_data if len(str(val).strip()) > 10)
            
            sample_count = len(sample_data)
            features.extend([
                has_letters / max(sample_count, 1),    # Letter ratio
                has_digits / max(sample_count, 1),     # Digit ratio  
                has_special / max(sample_count, 1),    # Special char ratio
                has_long_values / max(sample_count, 1) # Long values ratio (Trx IDs often long)
            ])
            
            # Enhanced Bank Transaction ID Pattern Learning (20 features)
            # Learn from actual data values much more deeply!
            
            # 1. Advanced Pattern Recognition (8 features)
            advanced_patterns = [
                r'^[A-Z]{2,5}\d{8,20}$',           # Bank code + numbers (TBPM123...)
                r'^No\d{10,20}$',                  # "No" prefix (No25020503350380)
                r'^Ref\w{8,20}$',                  # "Ref" prefix  
                r'^\d{15,25}$',                    # Pure long numbers
                r'^\d+\.?\d*[Ee][+-]?\d+$',        # Scientific notation
                r'^[A-Z]+\d+[A-Z]*\d*$',          # Mixed letters/numbers
                r'^\w{10,30}$',                    # General alphanumeric IDs
                r'^[A-Z0-9]{8,25}$'                # Uppercase alphanumeric
            ]
            
            for pattern in advanced_patterns:
                matches = sum(1 for val in sample_data if re.match(pattern, str(val).strip()))
                features.append(matches / max(sample_count, 1))
            
            # 2. Character Position Patterns (4 features)
            # Learn where letters vs numbers appear in your data
            first_char_letter = sum(1 for val in sample_data if str(val).strip() and str(val).strip()[0].isalpha())
            first_char_digit = sum(1 for val in sample_data if str(val).strip() and str(val).strip()[0].isdigit())
            last_char_letter = sum(1 for val in sample_data if str(val).strip() and str(val).strip()[-1].isalpha())
            last_char_digit = sum(1 for val in sample_data if str(val).strip() and str(val).strip()[-1].isdigit())
            
            features.extend([
                first_char_letter / max(sample_count, 1),  # Starts with letter
                first_char_digit / max(sample_count, 1),   # Starts with digit
                last_char_letter / max(sample_count, 1),   # Ends with letter
                last_char_digit / max(sample_count, 1)     # Ends with digit
            ])
            
            # 3. N-gram Pattern Learning (4 features)
            # Learn common prefixes and patterns from your actual data
            common_prefixes = ['TB', 'No', 'Re', 'PM']  # Learn from your data
            for prefix in common_prefixes:
                prefix_matches = sum(1 for val in sample_data if str(val).strip().startswith(prefix))
                features.append(prefix_matches / max(sample_count, 1))
            
            # 4. Data Structure Analysis (4 features)
            # Analyze the actual structure of your transaction IDs
            has_consecutive_digits = sum(1 for val in sample_data 
                                       if re.search(r'\d{5,}', str(val).strip()))  # 5+ consecutive digits
            has_mixed_case = sum(1 for val in sample_data 
                               if any(c.isupper() for c in str(val)) and any(c.islower() for c in str(val)))
            has_sequential_pattern = sum(1 for val in sample_data 
                                       if re.search(r'(012|123|234|345|456|567|678|789)', str(val).strip()))
            has_repeated_chars = sum(1 for val in sample_data 
                                   if re.search(r'(.)\1{2,}', str(val).strip()))  # 3+ repeated chars
            
            features.extend([
                has_consecutive_digits / max(sample_count, 1),  # Long digit sequences
                has_mixed_case / max(sample_count, 1),          # Mixed case patterns
                has_sequential_pattern / max(sample_count, 1),   # Sequential numbers
                has_repeated_chars / max(sample_count, 1)        # Repeated characters
            ])
                
        else:
            # Empty column - fill with zeros (exactly 31 features for enhanced data analysis)
            features.extend([0] * 31)
        
        # Total so far: 25 + 31 = 56 features
        
        # 3. Position Features (5 features)
        columns = list(df.columns)
        position = columns.index(column_name) if column_name in columns else -1
        total_cols = len(columns)
        
        features.extend([
            position,  # Absolute position
            position / max(total_cols - 1, 1),  # Relative position (0-1)
            1 if position == 0 else 0,  # Is first column
            1 if position == total_cols - 1 else 0,  # Is last column
            1 if position < total_cols / 2 else 0  # Is in first half
        ])
        
        # Total: 25 + 31 + 5 = 61 features
        return np.array(features, dtype=np.float32)

    def extract_amount_features(self, df: pd.DataFrame, column_name: str) -> np.ndarray:
        """Extract features specifically for amount/payment column detection"""
        features = []
        
        # 1. Amount-Specific Column Name Features (20 features)
        col_name = str(column_name).lower().strip()
        col_clean = re.sub(r'[^a-z0-9]', '', col_name)
        
        # Amount keywords (8 features)
        amount_keywords = ['amount', 'amt', 'paid', 'payment', 'credit', 'debit', 'value', 'sum']
        for keyword in amount_keywords:
            features.append(1 if keyword in col_name else 0)
        
        # Related financial keywords (6 features)
        financial_keywords = ['money', 'cash', 'total', 'balance', 'fee', 'cost']
        for keyword in financial_keywords:
            features.append(1 if keyword in col_name else 0)
        
        # Amount-specific patterns (6 features)
        features.extend([
            1 if 'paid' in col_name and 'amt' in col_name else 0,  # "Paid Amt" pattern
            1 if col_name.startswith('amount') else 0,  # Starts with amount
            1 if col_name.endswith('amount') else 0,    # Ends with amount
            1 if 'credit' in col_name else 0,           # Credit variations
            1 if any(arabic in col_name for arabic in ['دائن', 'مدين']) else 0,  # Arabic amount terms
            1 if re.search(r'amount.*credit|credit.*amount|payment.*amount', col_name) else 0  # Combined patterns
        ])
        
        # 2. Numeric Data Analysis Features (25 features)
        column_data = df[column_name].dropna()
        
        if len(column_data) > 0:
            # Try to convert to numeric, handling various formats
            numeric_values = []
            string_values = []
            
            for val in column_data.head(100):  # Sample first 100
                val_str = str(val).strip()
                string_values.append(val_str)
                
                # Try various numeric conversion approaches
                try:
                    # Handle scientific notation
                    if 'E' in val_str.upper() or 'e' in val_str:
                        numeric_val = float(val_str)
                        numeric_values.append(numeric_val)
                    # Handle regular numbers (with commas, etc.)
                    else:
                        # Remove common formatting
                        clean_val = re.sub(r'[,$\s]', '', val_str)
                        if clean_val and not clean_val.lower() in ['nan', 'none', 'null', '']:
                            numeric_val = float(clean_val)
                            numeric_values.append(numeric_val)
                except (ValueError, TypeError):
                    continue
            
            # Numeric characteristics (10 features)
            if numeric_values:
                features.extend([
                    len(numeric_values) / len(string_values),  # Numeric conversion ratio
                    np.mean(numeric_values),    # Average value
                    np.median(numeric_values),  # Median value
                    np.std(numeric_values),     # Standard deviation
                    min(numeric_values),        # Minimum value
                    max(numeric_values),        # Maximum value
                    np.sum(numeric_values),     # Total sum
                    len([v for v in numeric_values if v > 0]) / len(numeric_values),  # Positive ratio
                    len([v for v in numeric_values if v == 0]) / len(numeric_values), # Zero ratio
                    len([v for v in numeric_values if v < 0]) / len(numeric_values)   # Negative ratio
                ])
            else:
                features.extend([0] * 10)
            
            # String pattern analysis for amounts (10 features)
            sample_count = len(string_values)
            
            # Pattern matching for amount formats
            has_decimal = sum(1 for val in string_values if '.' in val and val.count('.') == 1)
            has_comma = sum(1 for val in string_values if ',' in val)
            has_currency_symbol = sum(1 for val in string_values if any(sym in val for sym in ['$', '€', '£', '¥', '₹']))
            has_scientific = sum(1 for val in string_values if re.search(r'\d+\.?\d*[Ee][+-]?\d+', val))
            has_parentheses = sum(1 for val in string_values if '(' in val and ')' in val)
            has_negative_sign = sum(1 for val in string_values if val.strip().startswith('-'))
            starts_with_digit = sum(1 for val in string_values if val.strip() and val.strip()[0].isdigit())
            reasonable_length = sum(1 for val in string_values if 1 <= len(val.strip()) <= 15)  # Reasonable for amounts
            mostly_digits = sum(1 for val in string_values if len(re.sub(r'[^\d]', '', val)) >= len(val) * 0.7)
            has_zero_values = sum(1 for val in string_values if val.strip() in ['0', '0.0', '0.00'])
            
            features.extend([
                has_decimal / sample_count,        # Decimal point ratio
                has_comma / sample_count,          # Comma ratio (formatting)
                has_currency_symbol / sample_count,# Currency symbol ratio
                has_scientific / sample_count,     # Scientific notation ratio
                has_parentheses / sample_count,    # Parentheses ratio (negative amounts)
                has_negative_sign / sample_count,  # Negative sign ratio
                starts_with_digit / sample_count,  # Starts with digit ratio
                reasonable_length / sample_count,  # Reasonable length ratio
                mostly_digits / sample_count,      # Mostly digits ratio
                has_zero_values / sample_count     # Zero values ratio
            ])
            
            # Value range analysis (5 features)
            if numeric_values:
                sorted_values = sorted(numeric_values)
                n_vals = len(sorted_values)
                
                features.extend([
                    len([v for v in numeric_values if 0 < v <= 100]) / n_vals,      # Small amounts
                    len([v for v in numeric_values if 100 < v <= 10000]) / n_vals,  # Medium amounts  
                    len([v for v in numeric_values if v > 10000]) / n_vals,         # Large amounts
                    (sorted_values[-1] - sorted_values[0]) if n_vals > 1 else 0,    # Range
                    len(set(numeric_values)) / n_vals                               # Uniqueness of values
                ])
            else:
                features.extend([0] * 5)
                
        else:
            # Empty column - fill with zeros (25 features for numeric analysis)
            features.extend([0] * 25)
        
        # 3. Position and Context Features (16 features)
        columns = list(df.columns)
        position = columns.index(column_name) if column_name in columns else -1
        total_cols = len(columns)
        
        # Basic position features (5 features)
        features.extend([
            position,  # Absolute position
            position / max(total_cols - 1, 1),  # Relative position (0-1)
            1 if position == 0 else 0,  # Is first column
            1 if position == total_cols - 1 else 0,  # Is last column
            1 if position < total_cols / 2 else 0  # Is in first half
        ])
        
        # Context features - nearby columns (11 features)
        nearby_has_trx_id = 0
        nearby_has_date = 0
        nearby_has_bank = 0
        nearby_has_name = 0
        nearby_has_reference = 0
        nearby_amount_cols = 0
        is_only_numeric_col = 1
        similar_name_count = 0
        
        # Check surrounding columns (±2 positions)
        for i in range(max(0, position-2), min(total_cols, position+3)):
            if i != position and i < len(columns):
                other_col = str(columns[i]).lower()
                
                if any(keyword in other_col for keyword in ['trx', 'transaction', 'id', 'ref']):
                    nearby_has_trx_id = 1
                if any(keyword in other_col for keyword in ['date', 'time', 'created']):
                    nearby_has_date = 1
                if any(keyword in other_col for keyword in ['bank', 'institution']):
                    nearby_has_bank = 1
                if any(keyword in other_col for keyword in ['name', 'description', 'memo']):
                    nearby_has_name = 1
                if any(keyword in other_col for keyword in ['ref', 'reference', 'number']):
                    nearby_has_reference = 1
                if any(keyword in other_col for keyword in ['amount', 'amt', 'paid', 'credit', 'debit']):
                    nearby_amount_cols += 1
                
                # Check if this is the only numeric column
                try:
                    other_data = df[columns[i]].dropna().head(20)
                    if len(other_data) > 0:
                        numeric_count = 0
                        for val in other_data:
                            try:
                                float(str(val).replace(',', '').replace('$', ''))
                                numeric_count += 1
                            except:
                                pass
                        if numeric_count > len(other_data) * 0.5:  # Other column is also numeric
                            is_only_numeric_col = 0
                except:
                    pass
        
        # Count columns with similar names
        for other_col in columns:
            if other_col != column_name:
                other_col_lower = str(other_col).lower()
                if any(keyword in other_col_lower and keyword in col_name 
                      for keyword in ['amount', 'amt', 'paid', 'credit', 'total']):
                    similar_name_count += 1
        
        features.extend([
            nearby_has_trx_id,           # Has transaction ID nearby
            nearby_has_date,             # Has date column nearby
            nearby_has_bank,             # Has bank column nearby  
            nearby_has_name,             # Has name/description nearby
            nearby_has_reference,        # Has reference nearby
            min(nearby_amount_cols, 5),  # Number of amount columns nearby (capped at 5)
            is_only_numeric_col,         # Is the only numeric column
            min(similar_name_count, 5),  # Similar named columns (capped at 5)
            1 if 'paid' in col_name and position < total_cols / 2 else 0,  # "Paid" in first half
            1 if 'credit' in col_name and position > total_cols / 2 else 0, # "Credit" in second half
            1 if any(keyword in col_name for keyword in ['total', 'sum', 'net']) else 0  # Summary amount indicators
        ])
        
        # Total: 20 + 25 + 16 = 61 features (same as bank trx detection for consistency)
        return np.array(features, dtype=np.float32)
    
    def prepare_training_data(self, our_files: List[str], 
                            labeled_columns: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data from YOUR reconciliation files
        
        Args:
            our_files: List of your reconciliation file paths
            labeled_columns: Dict mapping filename -> Bank Trx ID column name
        """
        X = []
        y = []
        
        logger.info(f"🔍 Processing {len(our_files)} reconciliation files for training...")
        
        for file_path in our_files:
            try:
                # Load your reconciliation file
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_excel(file_path)
                
                filename = os.path.basename(file_path)
                true_bank_trx_column = labeled_columns.get(filename, None)
                
                if true_bank_trx_column is None:
                    logger.warning(f"⚠️ No label for {filename}, skipping...")
                    continue
                
                # Extract features for each column in your file
                for column in df.columns:
                    features = self.extract_features(df, column)
                    X.append(features)
                    
                    # Label: 1 if this is the Bank Trx ID column, 0 otherwise
                    label = 1 if column == true_bank_trx_column else 0
                    y.append(label)
                
                logger.info(f"✅ Processed {filename}: {len(df.columns)} columns, Bank Trx ID: '{true_bank_trx_column}'")
                
            except Exception as e:
                logger.error(f"❌ Error processing {file_path}: {e}")
                continue
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"📊 Training data prepared:")
        logger.info(f"   Total samples: {X.shape[0]}")
        logger.info(f"   Features per sample: {X.shape[1]}")
        logger.info(f"   Bank Trx ID columns (positive): {y.sum()}")
        logger.info(f"   Other columns (negative): {len(y) - y.sum()}")
        
        return X, y
    
    def train(self, our_files: List[str], labeled_columns: Dict[str, str],
              epochs: int = 150, learning_rate: float = 0.001):
        """Train the ML model on your labeled reconciliation files"""
        
        # Prepare training data
        X, y = self.prepare_training_data(our_files, labeled_columns)
        
        if len(X) == 0:
            raise ValueError("❌ No training data available! Please check your file paths and labels.")
        
        if y.sum() == 0:
            raise ValueError("❌ No positive samples! Please check your Bank Trx ID column labels.")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data (stratified to ensure both classes in train/test)
        if y.sum() >= 2 and (len(y) - y.sum()) >= 2:  # Need at least 2 of each class
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
        else:
            # If too few samples, use all for training
            X_train, X_test = X_scaled, X_scaled
            y_train, y_test = y, y
            logger.warning("⚠️ Too few samples for proper train/test split, using all data for training")
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test).reshape(-1, 1)
        
        # Initialize model
        self.model = EnhancedColumnDetector(input_size=X.shape[1])
        
        # Use weighted loss to handle class imbalance
        pos_weight = torch.FloatTensor([(len(y) - y.sum()) / max(y.sum(), 1)])
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        # Training loop
        logger.info(f"🚀 Starting training for {epochs} epochs...")
        
        best_accuracy = 0.0
        patience = 20
        patience_counter = 0
        
        for epoch in range(epochs):
            self.model.train()
            
            # Forward pass - get raw logits for BCEWithLogitsLoss
            logits = self.model(X_train_tensor)
            loss = criterion(logits, y_train_tensor)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Validation every 10 epochs
            if epoch % 10 == 0:
                self.model.eval()
                with torch.no_grad():
                    test_outputs = self.model.predict_proba_bank_trx(X_test_tensor)
                    predicted = (test_outputs > 0.5).float()
                    accuracy = (predicted == y_test_tensor).float().mean().item()
                    
                    # Calculate precision and recall for Bank Trx ID detection
                    true_positives = ((predicted == 1) & (y_test_tensor == 1)).sum().item()
                    predicted_positives = (predicted == 1).sum().item()
                    actual_positives = (y_test_tensor == 1).sum().item()
                    
                    precision = true_positives / max(predicted_positives, 1)
                    recall = true_positives / max(actual_positives, 1)
                    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
                    
                    logger.info(f"Epoch {epoch:3d}: Loss={loss:.4f}, Acc={accuracy:.3f}, Prec={precision:.3f}, Rec={recall:.3f}, F1={f1:.3f}")
                    
                    # Early stopping
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break
        
        self.is_trained = True
        logger.info(f"🎉 Training completed! Best accuracy: {best_accuracy:.3f}")
    
    def prepare_amount_training_data(self, our_files: List[str], 
                                   labeled_amount_columns: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data for amount column detection from YOUR reconciliation files
        
        Args:
            our_files: List of your reconciliation file paths
            labeled_amount_columns: Dict mapping filename -> Amount column name
        """
        X = []
        y = []
        
        logger.info(f"🔍 Processing {len(our_files)} reconciliation files for amount training...")
        
        for file_path in our_files:
            try:
                # Load your reconciliation file
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_excel(file_path)
                
                filename = os.path.basename(file_path)
                true_amount_column = labeled_amount_columns.get(filename, None)
                
                if true_amount_column is None:
                    logger.warning(f"⚠️ No amount label for {filename}, skipping...")
                    continue
                
                # Extract features for each column in your file
                for column in df.columns:
                    features = self.extract_amount_features(df, column)
                    X.append(features)
                    
                    # Label: 1 if this is the Amount column, 0 otherwise
                    label = 1 if column == true_amount_column else 0
                    y.append(label)
                
                logger.info(f"✅ Processed {filename}: {len(df.columns)} columns, Amount: '{true_amount_column}'")
                
            except Exception as e:
                logger.error(f"❌ Error processing {file_path}: {e}")
                continue
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"📊 Amount training data prepared:")
        logger.info(f"   Total samples: {X.shape[0]}")
        logger.info(f"   Features per sample: {X.shape[1]}")
        logger.info(f"   Amount columns (positive): {y.sum()}")
        logger.info(f"   Other columns (negative): {len(y) - y.sum()}")
        
        return X, y

    def train_amount_detector(self, our_files: List[str], labeled_amount_columns: Dict[str, str],
                            epochs: int = 150, learning_rate: float = 0.001):
        """Train the amount detection network on your labeled reconciliation files"""
        
        # Prepare training data
        X, y = self.prepare_amount_training_data(our_files, labeled_amount_columns)
        
        if len(X) == 0:
            raise ValueError("❌ No amount training data available! Please check your file paths and labels.")
        
        if y.sum() == 0:
            raise ValueError("❌ No positive amount samples! Please check your Amount column labels.")
        
        # Scale features (use separate scaler for amount features)
        if not hasattr(self, 'amount_scaler'):
            self.amount_scaler = StandardScaler()
        X_scaled = self.amount_scaler.fit_transform(X)
        
        # Split data (stratified to ensure both classes in train/test)
        if y.sum() >= 2 and (len(y) - y.sum()) >= 2:  # Need at least 2 of each class
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
        else:
            # If too few samples, use all for training
            X_train, X_test = X_scaled, X_scaled
            y_train, y_test = y, y
            logger.warning("⚠️ Too few samples for proper train/test split, using all data for training")
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test).reshape(-1, 1)
        
        # Initialize model if not exists
        if self.model is None:
            self.model = EnhancedColumnDetector(input_size=X.shape[1])
        
        # Use weighted loss to handle class imbalance
        pos_weight = torch.FloatTensor([(len(y) - y.sum()) / max(y.sum(), 1)])
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(self.model.amount_network.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        # Training loop
        logger.info(f"🚀 Starting amount detection training for {epochs} epochs...")
        
        best_accuracy = 0.0
        patience = 20
        patience_counter = 0
        
        for epoch in range(epochs):
            self.model.train()
            
            # Forward pass - get raw logits for BCEWithLogitsLoss
            logits = self.model.forward_amount(X_train_tensor)
            loss = criterion(logits, y_train_tensor)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Validation every 10 epochs
            if epoch % 10 == 0:
                self.model.eval()
                with torch.no_grad():
                    test_outputs = self.model.predict_proba_amount(X_test_tensor)
                    predicted = (test_outputs > 0.5).float()
                    accuracy = (predicted == y_test_tensor).float().mean().item()
                    
                    # Calculate precision and recall for Amount detection
                    true_positives = ((predicted == 1) & (y_test_tensor == 1)).sum().item()
                    predicted_positives = (predicted == 1).sum().item()
                    actual_positives = (y_test_tensor == 1).sum().item()
                    
                    precision = true_positives / max(predicted_positives, 1)
                    recall = true_positives / max(actual_positives, 1)
                    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
                    
                    logger.info(f"Amount Epoch {epoch:3d}: Loss={loss:.4f}, Acc={accuracy:.3f}, Prec={precision:.3f}, Rec={recall:.3f}, F1={f1:.3f}")
                    
                    # Early stopping
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break
        
        self.amount_trained = True
        logger.info(f"🎉 Amount detection training completed! Best accuracy: {best_accuracy:.3f}")

    def predict_column(self, df: pd.DataFrame) -> Tuple[str, float]:
        """Predict the Bank Trx ID column in your reconciliation file"""
        if not self.is_trained:
            raise ValueError("❌ Model not trained! Please train the model first.")
        
        best_column = None
        best_confidence = 0.0
        all_predictions = []
        
        self.model.eval()
        with torch.no_grad():
            for column in df.columns:
                # Extract features
                features = self.extract_features(df, column)
                features_scaled = self.scaler.transform(features.reshape(1, -1))
                features_tensor = torch.FloatTensor(features_scaled)
                
                # Predict confidence using probability output
                confidence = self.model.predict_proba_bank_trx(features_tensor).item()
                all_predictions.append((column, confidence))
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_column = column
        
        # Log top 3 predictions for debugging
        top_predictions = sorted(all_predictions, key=lambda x: x[1], reverse=True)[:3]
        logger.info("🔍 Top 3 Bank Trx ID column predictions:")
        for i, (col, conf) in enumerate(top_predictions, 1):
            logger.info(f"   {i}. '{col}': {conf*100:.1f}%")
        
        return best_column, best_confidence * 100  # Convert to percentage

    def predict_amount_column(self, df: pd.DataFrame) -> Tuple[str, float]:
        """Predict the Amount/Payment column in a file"""
        if not hasattr(self, 'amount_trained') or not self.amount_trained:
            raise ValueError("❌ Amount model not trained! Please train the amount model first.")
        
        best_column = None
        best_confidence = 0.0
        all_predictions = []
        
        self.model.eval()
        with torch.no_grad():
            for column in df.columns:
                # Extract amount-specific features
                features = self.extract_amount_features(df, column)
                features_scaled = self.amount_scaler.transform(features.reshape(1, -1))
                features_tensor = torch.FloatTensor(features_scaled)
                
                # Predict confidence using probability output
                confidence = self.model.predict_proba_amount(features_tensor).item()
                all_predictions.append((column, confidence))
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_column = column
        
        # Log top 3 predictions for debugging
        top_predictions = sorted(all_predictions, key=lambda x: x[1], reverse=True)[:3]
        logger.info("🔍 Top 3 Amount column predictions:")
        for i, (col, conf) in enumerate(top_predictions, 1):
            logger.info(f"   {i}. '{col}': {conf*100:.1f}%")
        
        return best_column, best_confidence * 100  # Convert to percentage
    
    def save_model(self, filepath: str = "enhanced_column_detector.pth"):
        """Save the trained model (both bank trx and amount detectors)"""
        if not self.is_trained and not hasattr(self, 'amount_trained'):
            raise ValueError("❌ No trained model to save")
        
        save_data = {
            'model_state_dict': self.model.state_dict() if self.model else None,
            'scaler': self.scaler,
            'input_size': self.extract_features(pd.DataFrame({'test': [1]}), 'test').shape[0],
            'is_trained': self.is_trained,
        }
        
        # Save amount detection components if available
        if hasattr(self, 'amount_trained'):
            save_data['amount_trained'] = self.amount_trained
            save_data['amount_scaler'] = getattr(self, 'amount_scaler', None)
        
        torch.save(save_data, filepath)
        logger.info(f"💾 Enhanced model saved to {filepath}")
    
    def load_model(self, filepath: str = "enhanced_column_detector.pth"):
        """Load a trained model (supports both old and new formats)"""
        if not os.path.exists(filepath):
            logger.warning(f"⚠️ Model file {filepath} not found")
            return False
            
        try:
            # Fix for PyTorch 2.6+ - allow sklearn objects
            checkpoint = torch.load(filepath, weights_only=False)
            
            # Restore model
            input_size = checkpoint['input_size']
            self.model = EnhancedColumnDetector(input_size=input_size)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Restore bank trx scaler and training status
            self.scaler = checkpoint['scaler']
            self.is_trained = checkpoint.get('is_trained', True)
            
            # Restore amount detection components if available
            if 'amount_trained' in checkpoint:
                self.amount_trained = checkpoint['amount_trained']
                if 'amount_scaler' in checkpoint and checkpoint['amount_scaler'] is not None:
                    self.amount_scaler = checkpoint['amount_scaler']
            
            logger.info(f"📂 Enhanced model loaded from {filepath}")
            if self.is_trained:
                logger.info("   ✅ Bank Transaction ID detector ready")
            if hasattr(self, 'amount_trained') and self.amount_trained:
                logger.info("   ✅ Amount detector ready")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            return False

def create_training_setup(our_files_directory: str):
    """Create training setup for your reconciliation files"""
    
    # Find all Excel/CSV files in the directory
    our_files = []
    for ext in ['*.xlsx', '*.xls', '*.csv']:
        our_files.extend(glob.glob(os.path.join(our_files_directory, '**', ext), recursive=True))
    
    print(f"🔍 Found {len(our_files)} reconciliation files in {our_files_directory}")
    
    # Create training labels template
    template_content = '''"""
Training Labels for Bank Trx ID Column Detection
Edit this file to specify which column contains Bank Transaction IDs in each of your reconciliation files
"""

# Dictionary mapping filename -> Bank Trx ID column name
# Use the EXACT column name as it appears in your Excel files
LABELED_COLUMNS = {
'''
    
    for file_path in our_files[:20]:  # Show first 20 files
        filename = os.path.basename(file_path)
        
        try:
            # Load file to show available columns
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path, nrows=0)  # Just headers
            else:
                df = pd.read_excel(file_path, nrows=0)  # Just headers
            
            columns = list(df.columns)
            template_content += f'''
    # File: {filename}
    # Available columns: {columns}
    "{filename}": "",  # <- Put the Bank Trx ID column name here
'''
            
        except Exception as e:
            template_content += f'''
    # File: {filename} - Error loading: {e}
    "{filename}": "",
'''
    
    template_content += '\n}\n'
    
    if len(our_files) > 20:
        template_content += f'\n# Note: Found {len(our_files)} total files (showing first 20)\n# Add more files manually if needed\n'
    
    # Save training labels template
    with open("training_labels.py", "w", encoding='utf-8') as f:
        f.write(template_content)
    
    print("✅ Created training_labels.py")
    print("📝 Please edit this file to specify Bank Trx ID columns for your files")
    print("📂 Then run the training script")
    
    return our_files

if __name__ == "__main__":
    print("🤖 SmartRecon ML Column Detector Setup")
    print("=" * 50)
    
    # Get directory containing user's reconciliation files
    our_files_dir = input("📁 Enter path to your reconciliation files directory: ").strip()
    
    if not os.path.exists(our_files_dir):
        print("❌ Directory not found!")
        exit(1)
    
    # Create training setup
    our_files = create_training_setup(our_files_dir)
    
    print(f"\n📋 Next steps:")
    print(f"1. Edit training_labels.py to label your Bank Trx ID columns")
    print(f"2. Run: python train_ml_detector.py")
    print(f"3. The trained model will be integrated into SmartRecon automatically") 