import pandas as pd
import re
import logging
from typing import List, Tuple, Dict
import chardet

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReconciliationEngine:
    def __init__(self):
        """Initialize the reconciliation engine"""
        # Bank name patterns for identification (generic examples - works with any bank worldwide)
        self.bank_patterns = {
            'HSBC': [r'HSBC', r'hsbc'],
            'Standard Chartered': [r'Standard', r'standard', r'SCB', r'scb'],
            'Citibank': [r'Citi', r'citi'],
            'JPMorgan Chase': [r'Chase', r'chase', r'JPMorgan', r'jpmorgan'],
            'Bank of America': [r'Bank of America', r'BofA', r'bofa'],
            'Wells Fargo': [r'Wells', r'wells'],
            'Deutsche Bank': [r'Deutsche', r'deutsche'],
            'Barclays': [r'Barclays', r'barclays'],
            'BNP Paribas': [r'BNP', r'bnp', r'Paribas', r'paribas'],
            'Credit Suisse': [r'Credit Suisse', r'credit suisse'],
            'ING Bank': [r'ING', r'ing'],
            'Royal Bank': [r'Royal', r'royal', r'RBC', r'rbc'],
            'Commonwealth Bank': [r'Commonwealth', r'commonwealth', r'CBA', r'cba'],
            'ANZ Bank': [r'ANZ', r'anz'],
            'National Bank': [r'National', r'national', r'NBK', r'nbk'],
            'First National': [r'First National', r'first national'],
            'Central Bank': [r'Central', r'central'],
            'Commercial Bank': [r'Commercial', r'commercial'],
            'Investment Bank': [r'Investment', r'investment'],
            'International Bank': [r'International', r'international']
        }
    
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
    
    def find_our_bank_trx_column(self, df: pd.DataFrame) -> Tuple[str, float]:
        """
        Find Bank Trx ID column in OUR reconciliation file
        Returns: (column_name, confidence_score)
        """
        column_scores = {}
        
        # Enhanced patterns for our file column names
        our_file_patterns = [
            'bank trx id', 'banktrxid', 'bank_trx_id', 'bank transaction id',
            'banktransactionid', 'bank_transaction_id', 'trx id', 'trxid',
            'transaction id', 'transactionid', 'bank ref', 'bankref',
            'bank reference', 'bankreference', 'payment ref', 'paymentref'
        ]
        
        for col in df.columns:
            score = 0
            col_lower = str(col).lower().strip().replace(' ', '').replace('_', '').replace('-', '')
            col_original = str(col).lower().strip()
            
            # Strategy 1: Exact matching for our file patterns
            for pattern in our_file_patterns:
                pattern_clean = pattern.replace(' ', '').replace('_', '').replace('-', '')
                if pattern_clean == col_lower:
                    score += 100  # Perfect match
                elif pattern in col_original:
                    score += 80   # Good match
                elif pattern_clean in col_lower:
                    score += 60   # Partial match
            
            # Strategy 2: Keyword matching
            if any(keyword in col_original for keyword in ['bank', 'trx', 'transaction']):
                score += 40
            
            if any(keyword in col_original for keyword in ['id', 'ref', 'reference']):
                score += 20
            
            # Strategy 3: Content pattern matching
            sample_values = df[col].dropna().astype(str).head(100)
            if len(sample_values) > 0:
                # Focus on data characteristics rather than pattern matching
                non_empty_count = len([v for v in sample_values if str(v).strip() and str(v).lower() not in ['nan', 'none', 'null']])
                unique_count = len(set(sample_values))
                
                # Bonus points for columns with mostly unique, non-empty values
                if non_empty_count > 0:
                    uniqueness_ratio = unique_count / non_empty_count
                    if uniqueness_ratio > 0.8:  # High uniqueness suggests transaction IDs
                        score += 25
                    
                    # Bonus for reasonable length values (likely transaction IDs)
                    avg_length = sum(len(str(v).strip()) for v in sample_values) / len(sample_values)
                    if 8 <= avg_length <= 30:  # Reasonable transaction ID length
                        score += 15
            
            column_scores[col] = score
        
        if not column_scores:
            return None, 0
        
        best_column = max(column_scores, key=column_scores.get)
        best_score = column_scores[best_column]
        
        logger.info(f"Best Bank Trx ID column in our file: '{best_column}' with score: {best_score}")
        return best_column, best_score
    
    def find_bank_trx_column(self, df: pd.DataFrame) -> Tuple[str, float]:
        """
        Smart detection of Bank Trx ID column in BANK/PSP file
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
                # Focus on data characteristics rather than pattern matching
                non_empty_count = len([v for v in sample_values if str(v).strip() and str(v).lower() not in ['nan', 'none', 'null']])
                unique_count = len(set(sample_values))
                
                # Bonus points for columns with mostly unique, non-empty values
                if non_empty_count > 0:
                    uniqueness_ratio = unique_count / non_empty_count
                    if uniqueness_ratio > 0.7:  # High uniqueness suggests transaction IDs
                        score += 35
                    
                    # Bonus for reasonable length values (likely transaction IDs)
                    avg_length = sum(len(str(v).strip()) for v in sample_values) / len(sample_values)
                    if 8 <= avg_length <= 30:  # Reasonable transaction ID length
                        score += 20
            
            column_scores[col] = score
        
        if not column_scores:
            return None, 0
        
        best_column = max(column_scores, key=column_scores.get)
        best_score = column_scores[best_column]
        
        logger.info(f"Best Bank Trx ID column in bank file: '{best_column}' with score: {best_score}")
        return best_column, best_score

    def extract_bank_trx_ids(self, df: pd.DataFrame, column: str) -> List[str]:
        """Extract Bank Trx IDs exactly as they appear - NO pattern matching or modification"""
        trx_ids = []
        
        for value in df[column].dropna():
            value_str = str(value).strip()  # Only remove leading/trailing whitespace
            
            # Skip empty values
            if not value_str or value_str.lower() in ['nan', 'none', 'null']:
                continue
                
            # Use the value EXACTLY as it appears - no extraction or modification
            trx_ids.append(value_str)
        
        return list(set(trx_ids))  # Remove duplicates only
    
    def identify_bank_from_trx_ids(self, our_df: pd.DataFrame, bank_trx_ids: List[str], our_trx_column: str = None) -> Dict[str, List[str]]:
        """
        Identify which bank these transaction IDs belong to by matching with our data
        Returns: {bank_name: [matching_trx_ids]}
        """
        bank_matches = {}
        
        # Use provided column or auto-detect
        if our_trx_column is None:
            our_trx_column, confidence = self.find_our_bank_trx_column(our_df)
            if confidence < 50:
                logger.warning(f"Low confidence ({confidence}) for our Bank Trx ID column: '{our_trx_column}'")
        
        # Get our Bank Trx IDs and corresponding bank names
        our_trx_ids = our_df[our_trx_column].dropna().astype(str).tolist()
        
        for bank_trx_id in bank_trx_ids:
            # Use exact matching only
            matches = our_df[our_df[our_trx_column].astype(str) == str(bank_trx_id)]
        
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
                             bank_trx_column: str, target_bank: str, our_trx_column: str = None) -> Dict:
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
        
        # Use provided column or auto-detect
        if our_trx_column is None:
            our_trx_column, confidence = self.find_our_bank_trx_column(our_df)
            if confidence < 50:
                logger.warning(f"Low confidence ({confidence}) for our Bank Trx ID column: '{our_trx_column}'")
        
        # Filter our data for the target bank
        our_bank_data = our_df[our_df['Paying Bank Name'] == target_bank].copy()
        our_trx_ids = set(our_bank_data[our_trx_column].astype(str).tolist())
        
        # Find exact matches
        matched_ids = our_trx_ids.intersection(bank_trx_ids)
        
        # Find missing records  
        missing_in_bank = our_trx_ids - matched_ids
        missing_in_our_file = bank_trx_ids - matched_ids
        
        # Prepare detailed results for exact matches
        for trx_id in matched_ids:
            our_record = our_bank_data[our_bank_data[our_trx_column].astype(str) == trx_id].iloc[0]
            
            # Find bank record more safely
            bank_matches_df = bank_df[bank_df[bank_trx_column].astype(str) == trx_id]
            if bank_matches_df.empty:
                # Try contains if exact match fails
                bank_matches_df = bank_df[bank_df[bank_trx_column].astype(str).str.contains(trx_id, na=False)]
            
            if not bank_matches_df.empty:
                bank_record = bank_matches_df.iloc[0]
                
                results['matched'].append({
                    'bank_trx_id': trx_id,
                    'our_record': self._convert_to_json_serializable(our_record),
                    'bank_record': self._convert_to_json_serializable(bank_record)
                })
        
        # Handle remaining unmatched records
        for trx_id in missing_in_bank:
            our_record = our_bank_data[our_bank_data[our_trx_column].astype(str) == trx_id].iloc[0]
            results['missing_in_bank'].append({
                'bank_trx_id': trx_id,
                'our_record': self._convert_to_json_serializable(our_record)
            })
        
        for trx_id in missing_in_our_file:
            # Find bank record more safely
            bank_matches_df = bank_df[bank_df[bank_trx_column].astype(str) == trx_id]
            if bank_matches_df.empty:
                # Try contains if exact match fails
                bank_matches_df = bank_df[bank_df[bank_trx_column].astype(str).str.contains(trx_id, na=False)]
            
            if not bank_matches_df.empty:
                bank_record = bank_matches_df.iloc[0]
                results['missing_in_our_file'].append({
                    'bank_trx_id': trx_id,
                    'bank_record': self._convert_to_json_serializable(bank_record)
                })
        
        # Summary statistics
        total_matches = len(matched_ids)
        results['summary'] = {
            'total_our_records': len(our_trx_ids),
            'total_bank_records': len(bank_trx_ids),
            'matched_records': total_matches,
            'exact_matches': total_matches,
            'fuzzy_matches': 0, # No fuzzy matches - exact matching only
            'missing_in_bank_count': len(missing_in_bank),
            'missing_in_our_file_count': len(missing_in_our_file),
            'match_percentage': (total_matches / max(len(our_trx_ids), 1)) * 100
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