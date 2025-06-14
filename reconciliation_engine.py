import pandas as pd
import re
from fuzzywuzzy import fuzz, process
from typing import Dict, List, Tuple, Optional
import json
import chardet
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReconciliationEngine:
    def __init__(self):
        # Standard column names for our reconciliation file
        self.our_standard_columns = [
            'No', 'Biller Name', 'Paying Bank Name', 'Settlement Bank Name',
            'Service Name', 'Billing No', 'Payment Type', 'Payment Status',
            'JOEBPPSTrx', 'Bank Trx ID', 'Corridor', 'Access Channel',
            'Process Date', 'Settlement Date', 'Due Amt', 'Paid Amt'
        ]
        
        # Patterns to identify Bank Trx ID in various formats
        self.bank_trx_patterns = [
            r'[A-Z]{2,4}\d{10,20}',  # Format like TBPM25160112163910187
            r'[A-Z]+\d+[A-Z]*\d*',   # Mixed alphanumeric
            r'\d{10,20}',             # Pure numeric long IDs
            r'[A-Z]{3,5}\d{8,15}',    # Bank code + numbers
        ]
    
    def detect_encoding(self, file_path: str) -> str:
        """Detect file encoding"""
        with open(file_path, 'rb') as file:
            raw_data = file.read()
            result = chardet.detect(raw_data)
            return result['encoding']
    
    def load_file(self, file_path: str) -> pd.DataFrame:
        """Load Excel or CSV file with encoding detection"""
        try:
            encoding = self.detect_encoding(file_path)
            
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path, encoding=encoding)
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            else:
                raise ValueError("Unsupported file format")
            
            logger.info(f"Loaded file with {len(df)} rows and {len(df.columns)} columns")
            return df
            
        except Exception as e:
            logger.error(f"Error loading file: {str(e)}")
            raise
    
    def find_bank_trx_column(self, df: pd.DataFrame) -> Tuple[str, float]:
        """
        Smart detection of Bank Trx ID column using multiple strategies
        Returns: (column_name, confidence_score)
        """
        column_scores = {}
        
        for col in df.columns:
            score = 0
            col_lower = str(col).lower()
            
            # Strategy 1: Column name matching
            if any(keyword in col_lower for keyword in ['trx', 'transaction', 'ref', 'reference', 'id']):
                score += 30
            
            if any(keyword in col_lower for keyword in ['bank', 'payment', 'txn']):
                score += 20
            
            # Strategy 2: Content pattern matching
            sample_values = df[col].dropna().astype(str).head(100)
            if len(sample_values) > 0:
                pattern_matches = 0
                for value in sample_values:
                    for pattern in self.bank_trx_patterns:
                        if re.search(pattern, str(value)):
                            pattern_matches += 1
                            break
                
                pattern_ratio = pattern_matches / len(sample_values)
                score += pattern_ratio * 50
            
            # Strategy 3: Length and format consistency
            if len(sample_values) > 0:
                lengths = [len(str(val)) for val in sample_values]
                avg_length = sum(lengths) / len(lengths)
                if 10 <= avg_length <= 25:  # Typical transaction ID length
                    score += 15
            
            column_scores[col] = score
        
        if not column_scores:
            return None, 0
        
        best_column = max(column_scores, key=column_scores.get)
        best_score = column_scores[best_column]
        
        logger.info(f"Best Bank Trx ID column: '{best_column}' with score: {best_score}")
        return best_column, best_score
    
    def extract_bank_trx_ids(self, df: pd.DataFrame, column: str) -> List[str]:
        """Extract clean Bank Trx IDs from the identified column"""
        trx_ids = []
        
        for value in df[column].dropna():
            value_str = str(value)
            # Try to extract transaction ID using patterns
            for pattern in self.bank_trx_patterns:
                matches = re.findall(pattern, value_str)
                if matches:
                    trx_ids.extend(matches)
                    break
            else:
                # If no pattern matches, use the value as is (cleaned)
                clean_value = re.sub(r'[^\w]', '', value_str)
                if len(clean_value) >= 8:  # Minimum reasonable length
                    trx_ids.append(clean_value)
        
        return list(set(trx_ids))  # Remove duplicates
    
    def identify_bank_from_trx_ids(self, our_df: pd.DataFrame, bank_trx_ids: List[str]) -> Dict[str, List[str]]:
        """
        Identify which bank these transaction IDs belong to by matching with our data
        Returns: {bank_name: [matching_trx_ids]}
        """
        bank_matches = {}
        
        # Get our Bank Trx IDs and corresponding bank names
        our_trx_ids = our_df['Bank Trx ID'].dropna().astype(str).tolist()
        
        for bank_trx_id in bank_trx_ids:
            # Find matching records in our data
            matches = our_df[our_df['Bank Trx ID'].astype(str) == str(bank_trx_id)]
            
            if not matches.empty:
                bank_name = matches['Paying Bank Name'].iloc[0]
                if bank_name not in bank_matches:
                    bank_matches[bank_name] = []
                bank_matches[bank_name].append(bank_trx_id)
        
        return bank_matches
    
    def _convert_to_json_serializable(self, obj):
        """Convert pandas objects to JSON serializable format"""
        if hasattr(obj, 'to_dict'):
            # Convert pandas Series to dict
            obj_dict = obj.to_dict()
        else:
            obj_dict = obj
        
        # Convert any datetime/timestamp objects to strings
        for key, value in obj_dict.items():
            if pd.isna(value):
                obj_dict[key] = None
            elif hasattr(value, 'strftime'):  # datetime-like objects
                obj_dict[key] = str(value)
            elif isinstance(value, pd.Timestamp):
                obj_dict[key] = str(value) if not pd.isna(value) else None
            elif isinstance(value, (int, float)) and pd.isna(value):
                obj_dict[key] = None
            else:
                # Convert any other non-serializable objects to string
                try:
                    # Test if it's JSON serializable
                    import json
                    json.dumps(value)
                except (TypeError, ValueError):
                    obj_dict[key] = str(value)
        
        return obj_dict
    
    def perform_reconciliation(self, our_df: pd.DataFrame, bank_df: pd.DataFrame, 
                             bank_trx_column: str, target_bank: str) -> Dict:
        """
        Perform the actual reconciliation between our data and bank data
        """
        results = {
            'matched': [],
            'missing_in_bank': [],
            'missing_in_our_file': [],
            'summary': {}
        }
        
        # Extract Bank Trx IDs from bank file
        bank_trx_ids = set(self.extract_bank_trx_ids(bank_df, bank_trx_column))
        
        # Filter our data for the target bank
        our_bank_data = our_df[our_df['Paying Bank Name'] == target_bank].copy()
        our_trx_ids = set(our_bank_data['Bank Trx ID'].astype(str).tolist())
        
        # Find matches and mismatches
        matched_ids = our_trx_ids.intersection(bank_trx_ids)
        missing_in_bank = our_trx_ids - bank_trx_ids
        missing_in_our_file = bank_trx_ids - our_trx_ids
        
        # Prepare detailed results
        for trx_id in matched_ids:
            our_record = our_bank_data[our_bank_data['Bank Trx ID'].astype(str) == trx_id].iloc[0]
            bank_record = bank_df[bank_df[bank_trx_column].astype(str).str.contains(trx_id, na=False)].iloc[0]
            
            results['matched'].append({
                'bank_trx_id': trx_id,
                'our_record': self._convert_to_json_serializable(our_record),
                'bank_record': self._convert_to_json_serializable(bank_record)
            })
        
        for trx_id in missing_in_bank:
            our_record = our_bank_data[our_bank_data['Bank Trx ID'].astype(str) == trx_id].iloc[0]
            results['missing_in_bank'].append({
                'bank_trx_id': trx_id,
                'our_record': self._convert_to_json_serializable(our_record)
            })
        
        for trx_id in missing_in_our_file:
            bank_record = bank_df[bank_df[bank_trx_column].astype(str).str.contains(trx_id, na=False)].iloc[0]
            results['missing_in_our_file'].append({
                'bank_trx_id': trx_id,
                'bank_record': self._convert_to_json_serializable(bank_record)
            })
        
        # Summary statistics
        results['summary'] = {
            'total_our_records': len(our_trx_ids),
            'total_bank_records': len(bank_trx_ids),
            'matched_records': len(matched_ids),
            'missing_in_bank_count': len(missing_in_bank),
            'missing_in_our_file_count': len(missing_in_our_file),
            'match_percentage': (len(matched_ids) / max(len(our_trx_ids), 1)) * 100
        }
        
        return results
    
    def generate_report(self, results: Dict, bank_name: str) -> str:
        """Generate a human-readable reconciliation report"""
        summary = results['summary']
        
        report = f"""
RECONCILIATION REPORT - {bank_name}
{'='*50}

Summary:
- Total records in our file: {summary['total_our_records']}
- Total records in bank file: {summary['total_bank_records']}
- Matched records: {summary['matched_records']}
- Missing in bank file: {summary['missing_in_bank_count']}
- Missing in our file: {summary['missing_in_our_file_count']}
- Match percentage: {summary['match_percentage']:.2f}%

"""
        
        if results['missing_in_bank']:
            report += "\nMISSING IN BANK FILE:\n"
            report += "-" * 30 + "\n"
            for item in results['missing_in_bank'][:10]:  # Show first 10
                report += f"Bank Trx ID: {item['bank_trx_id']}\n"
            if len(results['missing_in_bank']) > 10:
                report += f"... and {len(results['missing_in_bank']) - 10} more\n"
        
        if results['missing_in_our_file']:
            report += "\nMISSING IN OUR FILE:\n"
            report += "-" * 30 + "\n"
            for item in results['missing_in_our_file'][:10]:  # Show first 10
                report += f"Bank Trx ID: {item['bank_trx_id']}\n"
            if len(results['missing_in_our_file']) > 10:
                report += f"... and {len(results['missing_in_our_file']) - 10} more\n"
        
        return report 