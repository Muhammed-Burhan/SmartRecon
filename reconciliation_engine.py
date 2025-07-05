import pandas as pd
import re
import logging
from typing import List, Tuple, Dict
import chardet
import os

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
        
        # Initialize keyword manager for amount detection
        self.keyword_manager = None
        self._load_keyword_manager()
        
        # Try to load ML model if available
        self.ml_detector = None
        self._load_ml_detector()
    
    def _load_keyword_manager(self):
        """Load keyword manager for database-driven amount detection"""
        try:
            from database import KeywordManager
            self.keyword_manager = KeywordManager()
            logger.info("📊 Keyword manager loaded successfully")
        except Exception as e:
            logger.warning(f"⚠️ Could not load keyword manager: {e}")
            self.keyword_manager = None
    
    def _load_ml_detector(self):
        """Load ML column detector if available"""
        try:
            from ml_column_detector import MLColumnDetector
            self.ml_detector = MLColumnDetector()
            
            # Try to load enhanced model first, then fallback to old model
            enhanced_model_path = "enhanced_column_detector.pth"
            old_model_path = "bank_trx_detector.pth"
            
            if os.path.exists(enhanced_model_path):
                success = self.ml_detector.load_model(enhanced_model_path)
                if success:
                    logger.info("🤖 Enhanced ML column detector loaded successfully")
                    # Check what capabilities are available
                    if hasattr(self.ml_detector, 'amount_trained') and self.ml_detector.amount_trained:
                        logger.info("   💰 Amount detection available")
                    else:
                        logger.info("   💰 Amount detection not trained")
                else:
                    self.ml_detector = None
            elif os.path.exists(old_model_path):
                success = self.ml_detector.load_model(old_model_path)
                if success:
                    logger.info("🤖 ML column detector loaded successfully (Bank Trx ID only)")
                else:
                    self.ml_detector = None
            else:
                logger.info("📝 ML model not found. Use rule-based detection or train a model first.")
                self.ml_detector = None
                
        except ImportError:
            logger.info("📝 ML column detector not available. Install PyTorch to enable ML features.")
            self.ml_detector = None
        except Exception as e:
            logger.warning(f"⚠️ Could not load ML detector: {e}")
            self.ml_detector = None
    
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
        Uses ML detection if available, otherwise falls back to rule-based detection
        Returns: (column_name, confidence_score)
        """
        
        # Try ML detection first if available
        if self.ml_detector is not None:
            try:
                column, confidence = self.ml_detector.predict_column(df)
                if confidence > 70:  # High confidence threshold for ML
                    logger.info(f"🤖 ML detected Bank Trx ID column: '{column}' (confidence: {confidence:.1f}%)")
                    return column, confidence
                else:
                    logger.info(f"⚠️ ML detection low confidence ({confidence:.1f}%), falling back to rule-based")
            except Exception as e:
                logger.warning(f"⚠️ ML detection failed: {e}, falling back to rule-based")
        
        # Fall back to rule-based detection
        logger.info("🔧 Using rule-based Bank Trx ID column detection")
        return self._rule_based_our_bank_trx_detection(df)
    
    def _rule_based_our_bank_trx_detection(self, df: pd.DataFrame) -> Tuple[str, float]:
        """
        Rule-based Bank Trx ID column detection for OUR reconciliation file
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
        
        logger.info(f"🔧 Rule-based detected Bank Trx ID column: '{best_column}' (score: {best_score})")
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
    
    def find_amount_column(self, df: pd.DataFrame, file_type: str = "unknown") -> Tuple[str, float]:
        """
        Smart detection of Amount/Payment column in reconciliation files
        file_type: "our" for our reconciliation file, "bank" for bank file, "unknown" for auto-detect
        Returns: (column_name, confidence_score)
        """
        
        # Try ML detection first if available
        if (self.ml_detector is not None and 
            hasattr(self.ml_detector, 'amount_trained') and 
            self.ml_detector.amount_trained):
            try:
                column, confidence = self.ml_detector.predict_amount_column(df)
                if confidence > 70:  # High confidence threshold for ML
                    logger.info(f"🤖💰 ML detected Amount column: '{column}' (confidence: {confidence:.1f}%)")
                    return column, confidence
                else:
                    logger.info(f"⚠️💰 ML amount detection low confidence ({confidence:.1f}%), falling back to rule-based")
            except Exception as e:
                logger.warning(f"⚠️💰 ML amount detection failed: {e}, falling back to rule-based")
        
        # Fall back to rule-based detection
        logger.info("🔧💰 Using rule-based Amount column detection")
        return self._rule_based_amount_detection(df, file_type)
    
    def _rule_based_amount_detection(self, df: pd.DataFrame, file_type: str = "unknown") -> Tuple[str, float]:
        """
        Enhanced rule-based Amount column detection using database keywords
        Returns: (column_name, confidence_score)
        """
        column_scores = {}
        
        # Get keywords from database
        db_keywords = []
        if self.keyword_manager:
            try:
                db_keywords = self.keyword_manager.get_keywords(file_type=file_type if file_type != "unknown" else None)
                logger.info(f"📊 Using {len(db_keywords)} database keywords for amount detection")
            except Exception as e:
                logger.warning(f"⚠️ Could not load database keywords: {e}")
        
        # Fallback to hardcoded patterns if database not available
        if not db_keywords:
            logger.info("🔧 Using hardcoded keywords as fallback")
            # Use your exact keywords as fallback
            if file_type == "our":
                fallback_patterns = [
                    {'keyword': 'paid amt', 'priority': 100, 'is_exact_match': True},
                    {'keyword': 'paidamt', 'priority': 95, 'is_exact_match': True},
                    {'keyword': 'paid_amt', 'priority': 95, 'is_exact_match': True},
                    {'keyword': 'payment amount', 'priority': 90, 'is_exact_match': False},
                    {'keyword': 'amount paid', 'priority': 85, 'is_exact_match': False},
                ]
            else:
                fallback_patterns = [
                    {'keyword': 'credit', 'priority': 100, 'is_exact_match': True},
                    {'keyword': 'credit دائن', 'priority': 100, 'is_exact_match': True},
                    {'keyword': 'payment amount', 'priority': 90, 'is_exact_match': False},
                    {'keyword': 'credit amount', 'priority': 85, 'is_exact_match': False},
                    {'keyword': 'amount', 'priority': 75, 'is_exact_match': False},
                    {'keyword': 'value', 'priority': 65, 'is_exact_match': False},
                    {'keyword': 'sum', 'priority': 60, 'is_exact_match': False},
                    {'keyword': 'total', 'priority': 60, 'is_exact_match': False},
                ]
            db_keywords = fallback_patterns
        
        # Score each column
        for col in df.columns:
            score = 0
            col_lower = str(col).lower().strip()
            col_clean = col_lower.replace(' ', '').replace('_', '').replace('-', '')
            
            # Strategy 1: Database-driven keyword matching
            for keyword_data in db_keywords:
                keyword = keyword_data['keyword']
                priority = keyword_data['priority']
                is_exact = keyword_data.get('is_exact_match', False)
                
                # Clean keyword for comparison
                keyword_clean = keyword.replace(' ', '').replace('_', '').replace('-', '')
                
                if is_exact:
                    # Exact match gets full priority score
                    if keyword_clean == col_clean or keyword == col_lower:
                        score += priority
                        logger.info(f"💰 Exact match found: '{col}' = '{keyword}' (score: +{priority})")
                    elif keyword in col_lower:
                        score += priority * 0.8  # 80% for partial exact match
                        logger.info(f"💰 Partial exact match: '{col}' contains '{keyword}' (score: +{priority*0.8:.1f})")
                else:
                    # Flexible match gets scaled priority score
                    if keyword_clean in col_clean or keyword in col_lower:
                        match_score = priority * 0.7  # 70% for flexible match
                        score += match_score
                        logger.info(f"💰 Flexible match: '{col}' contains '{keyword}' (score: +{match_score:.1f})")
            
            # Strategy 2: Content analysis - check if column contains numeric data
            sample_values = df[col].dropna().head(100)
            if len(sample_values) > 0:
                numeric_count = 0
                total_count = len(sample_values)
                
                for val in sample_values:
                    try:
                        # Try to convert to number (handle various formats)
                        val_str = str(val).strip().replace(',', '').replace('$', '').replace('€', '').replace('£', '')
                        
                        # Handle scientific notation
                        if 'E' in val_str.upper() or 'e' in val_str:
                            float(val_str)
                            numeric_count += 1
                        elif val_str and val_str not in ['nan', 'none', 'null', '']:
                            float(val_str)
                            numeric_count += 1
                    except (ValueError, TypeError):
                        pass
                
                # Bonus for high numeric ratio (essential for amount columns)
                if total_count > 0:
                    numeric_ratio = numeric_count / total_count
                    if numeric_ratio > 0.8:  # 80%+ numeric
                        score += 100  # High bonus for numeric content
                    elif numeric_ratio > 0.5:  # 50%+ numeric
                        score += 50   # Medium bonus
                    elif numeric_ratio > 0.3:  # 30%+ numeric
                        score += 25   # Small bonus
                
                # Bonus for reasonable value ranges (amounts are usually positive)
                if numeric_count > 0:
                    try:
                        numeric_values = []
                        for val in sample_values:
                            try:
                                val_str = str(val).replace(',', '').replace('$', '').replace('€', '').replace('£', '').strip()
                                if 'E' in val_str.upper():
                                    numeric_values.append(float(val_str))
                                else:
                                    numeric_values.append(float(val_str))
                            except:
                                pass
                        
                        if numeric_values:
                            positive_ratio = len([v for v in numeric_values if v > 0]) / len(numeric_values)
                            if positive_ratio > 0.7:  # Mostly positive amounts
                                score += 30
                            
                            # Check for reasonable amount ranges
                            avg_value = sum(numeric_values) / len(numeric_values)
                            if 1 <= avg_value <= 1000000:  # Reasonable amount range
                                score += 25
                                
                    except Exception:
                        pass
            
            column_scores[col] = score
        
        if not column_scores:
            logger.warning("⚠️ No amount columns found in dataframe")
            return None, 0
        
        best_column = max(column_scores, key=column_scores.get)
        best_score = column_scores[best_column]
        
        # Log detailed results
        logger.info(f"🔧💰 Enhanced rule-based amount detection results:")
        logger.info(f"   Best column: '{best_column}' (score: {best_score:.1f})")
        
        # Log top 3 candidates for debugging
        sorted_columns = sorted(column_scores.items(), key=lambda x: x[1], reverse=True)
        for i, (col, score) in enumerate(sorted_columns[:3]):
            logger.info(f"   #{i+1}: '{col}' (score: {score:.1f})")
        
        return best_column, best_score

    def extract_amounts(self, df: pd.DataFrame, column: str) -> List[float]:
        """Extract numeric amounts from amount column, handling various formats"""
        amounts = []
        
        for value in df[column].dropna():
            try:
                value_str = str(value).strip()
                
                # Skip empty values
                if not value_str or value_str.lower() in ['nan', 'none', 'null']:
                    continue
                
                # Handle scientific notation
                if 'E' in value_str.upper() or 'e' in value_str:
                    amount = float(value_str)
                    amounts.append(amount)
                else:
                    # Remove common formatting characters
                    clean_value = value_str.replace(',', '').replace('$', '').replace('€', '').replace('£', '').strip()
                    
                    if clean_value:
                        amount = float(clean_value)
                        amounts.append(amount)
                        
            except (ValueError, TypeError):
                # Skip non-numeric values
                continue
        
        return amounts

    def compare_amounts(self, our_df: pd.DataFrame, bank_df: pd.DataFrame, 
                       our_trx_column: str, bank_trx_column: str,
                       our_amount_column: str, bank_amount_column: str,
                       target_bank: str) -> Dict:
        """
        Compare amounts between our file and bank file for matched transactions
        Returns detailed comparison results including discrepancies
        """
        results = {
            'our_total': 0.0,
            'bank_total': 0.0,
            'matched_total_our': 0.0,
            'matched_total_bank': 0.0,
            'discrepancies': [],
            'summary': {}
        }
        
        # Filter our data for the target bank
        our_bank_data = our_df[our_df['Paying Bank Name'] == target_bank].copy()
        
        # Get our transaction amounts
        our_amounts = {}
        for _, row in our_bank_data.iterrows():
            trx_id = str(row[our_trx_column])
            try:
                amount_str = str(row[our_amount_column]).strip()
                
                # Skip empty values
                if not amount_str or amount_str.lower() in ['nan', 'none', 'null', '']:
                    our_amounts[trx_id] = 0.0
                    continue
                
                # Handle scientific notation
                if 'E' in amount_str.upper() or 'e' in amount_str:
                    amount = float(amount_str)
                else:
                    # Remove common formatting characters (consistent with extract_amounts)
                    clean_amount = amount_str.replace(',', '').replace('$', '').replace('€', '').replace('£', '').strip()
                    amount = float(clean_amount) if clean_amount else 0.0
                
                our_amounts[trx_id] = amount
            except (ValueError, TypeError):
                our_amounts[trx_id] = 0.0
        
        # Get bank transaction amounts  
        bank_amounts = {}
        for _, row in bank_df.iterrows():
            trx_id = str(row[bank_trx_column])
            try:
                amount_str = str(row[bank_amount_column]).strip()
                
                # Skip empty values
                if not amount_str or amount_str.lower() in ['nan', 'none', 'null', '']:
                    bank_amounts[trx_id] = 0.0
                    continue
                
                # Handle scientific notation
                if 'E' in amount_str.upper() or 'e' in amount_str:
                    amount = float(amount_str)
                else:
                    # Remove common formatting characters (consistent with extract_amounts)
                    clean_amount = amount_str.replace(',', '').replace('$', '').replace('€', '').replace('£', '').strip()
                    amount = float(clean_amount) if clean_amount else 0.0
                
                bank_amounts[trx_id] = amount
            except (ValueError, TypeError):
                bank_amounts[trx_id] = 0.0
        
        # Calculate totals
        results['our_total'] = sum(our_amounts.values())
        results['bank_total'] = sum(bank_amounts.values())
        
        # Find matched transactions and compare amounts
        matched_our_total = 0.0
        matched_bank_total = 0.0
        
        for trx_id in our_amounts:
            if trx_id in bank_amounts:
                our_amount = our_amounts[trx_id]
                bank_amount = bank_amounts[trx_id]
                
                matched_our_total += our_amount
                matched_bank_total += bank_amount
                
                # Check for amount discrepancies (with small tolerance for floating point)
                if abs(our_amount - bank_amount) > 0.01:  # 1 cent tolerance
                    results['discrepancies'].append({
                        'bank_trx_id': trx_id,
                        'our_amount': our_amount,
                        'bank_amount': bank_amount,
                        'difference': our_amount - bank_amount,
                        'percentage_diff': ((our_amount - bank_amount) / max(our_amount, 0.01)) * 100
                    })
        
        results['matched_total_our'] = matched_our_total
        results['matched_total_bank'] = matched_bank_total
        
        # Summary
        results['summary'] = {
            'total_discrepancies': len(results['discrepancies']),
            'total_difference': matched_our_total - matched_bank_total,
            'amounts_match': abs(matched_our_total - matched_bank_total) < 0.01,
            'our_amount_column': our_amount_column,
            'bank_amount_column': bank_amount_column
        }
        
        logger.info(f"💰 Amount Comparison Results:")
        logger.info(f"   Our total (matched): {matched_our_total:,.2f}")
        logger.info(f"   Bank total (matched): {matched_bank_total:,.2f}")
        logger.info(f"   Difference: {matched_our_total - matched_bank_total:,.2f}")
        logger.info(f"   Discrepancies: {len(results['discrepancies'])}")
        
        return results

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
        Now includes amount comparison functionality
        """
        results = {
            'matched': [],
            'missing_in_bank': [],
            'missing_in_our_file': [],
            'amount_comparison': {},
            'summary': {}
        }
        
        # Extract Bank Trx IDs from bank file
        bank_trx_ids = set(self.extract_bank_trx_ids(bank_df, bank_trx_column))
        
        # Use provided column or auto-detect Bank Trx ID
        if our_trx_column is None:
            our_trx_column, confidence = self.find_our_bank_trx_column(our_df)
            if confidence < 50:
                logger.warning(f"Low confidence ({confidence}) for our Bank Trx ID column: '{our_trx_column}'")
        
        # Detect amount columns
        logger.info("💰 Detecting amount columns...")
        our_amount_column, our_amount_confidence = self.find_amount_column(our_df, "our")
        bank_amount_column, bank_amount_confidence = self.find_amount_column(bank_df, "bank")
        
        logger.info(f"💰 Amount column detection results:")
        logger.info(f"   Our file: '{our_amount_column}' (confidence: {our_amount_confidence:.1f}%)")
        logger.info(f"   Bank file: '{bank_amount_column}' (confidence: {bank_amount_confidence:.1f}%)")
        
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
        
        # Perform amount comparison if amount columns were detected
        if (our_amount_column and bank_amount_column and 
            our_amount_confidence > 30 and bank_amount_confidence > 30):
            
            logger.info("💰 Performing amount comparison...")
            amount_results = self.compare_amounts(
                our_df, bank_df, 
                our_trx_column, bank_trx_column,
                our_amount_column, bank_amount_column,
                target_bank
            )
            # Add the column detection information to the results
            amount_results['our_amount_column'] = our_amount_column
            amount_results['bank_amount_column'] = bank_amount_column
            amount_results['our_amount_confidence'] = our_amount_confidence
            amount_results['bank_amount_confidence'] = bank_amount_confidence
            amount_results['comparison_performed'] = True
            
            results['amount_comparison'] = amount_results
        else:
            logger.warning("⚠️💰 Amount comparison skipped - insufficient confidence in amount column detection")
            logger.warning(f"⚠️💰 Details: our_column='{our_amount_column}' ({our_amount_confidence:.1f}%), bank_column='{bank_amount_column}' ({bank_amount_confidence:.1f}%)")
            results['amount_comparison'] = {
                'our_amount_column': our_amount_column,
                'bank_amount_column': bank_amount_column,
                'our_amount_confidence': our_amount_confidence,
                'bank_amount_confidence': bank_amount_confidence,
                'comparison_performed': False,
                'reason': f'Low confidence in amount column detection (our: {our_amount_confidence:.1f}%, bank: {bank_amount_confidence:.1f}%, threshold: 30%)'
            }
        
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
            'match_percentage': (total_matches / max(len(our_trx_ids), 1)) * 100,
            # Add amount comparison summary
            'amount_comparison_available': our_amount_column is not None and bank_amount_column is not None,
            'our_amount_column': our_amount_column,
            'bank_amount_column': bank_amount_column,
            'our_amount_confidence': our_amount_confidence,
            'bank_amount_confidence': bank_amount_confidence
        }
        
        return results
    
    def generate_report(self, results: Dict, bank_name: str) -> str:
        """Generate a human-readable reconciliation report with amount comparison"""
        summary = results['summary']
        amount_comp = results.get('amount_comparison', {})
        
        report = f"""
RECONCILIATION REPORT - {bank_name}
{'='*50}

TRANSACTION MATCHING SUMMARY:
- Total records in our file: {summary['total_our_records']}
- Total records in bank file: {summary['total_bank_records']}
- Matched records: {summary['matched_records']}
- Missing in bank file: {summary['missing_in_bank_count']}
- Missing in our file: {summary['missing_in_our_file_count']}
- Match percentage: {summary['match_percentage']:.2f}%

AMOUNT COMPARISON SUMMARY:
"""
        
        if amount_comp.get('comparison_performed', False):
            report += f"""- Our amount column: '{amount_comp['our_amount_column']}'
- Bank amount column: '{amount_comp['bank_amount_column']}'
- Our total (all transactions): {amount_comp['our_total']:,.2f}
- Bank total (all transactions): {amount_comp['bank_total']:,.2f}
- Matched total (our): {amount_comp['matched_total_our']:,.2f}
- Matched total (bank): {amount_comp['matched_total_bank']:,.2f}
- Difference: {amount_comp['summary']['total_difference']:,.2f}
- Amounts match: {'✅ YES' if amount_comp['summary']['amounts_match'] else '❌ NO'}
- Transaction discrepancies: {amount_comp['summary']['total_discrepancies']}

"""
        else:
            reason = amount_comp.get('reason', 'Unknown')
            report += f"""- Amount comparison: ⚠️ Not performed
- Reason: {reason}
- Our amount column: '{amount_comp.get('our_amount_column', 'Not detected')}'
- Bank amount column: '{amount_comp.get('bank_amount_column', 'Not detected')}'

"""
        
        # Add amount discrepancies section
        if amount_comp.get('discrepancies'):
            report += "AMOUNT DISCREPANCIES:\n"
            report += "-" * 30 + "\n"
            for i, disc in enumerate(amount_comp['discrepancies'][:20], 1):  # Show first 20
                report += f"{i}. Transaction ID: {disc['bank_trx_id']}\n"
                report += f"   Our amount: {disc['our_amount']:,.2f}\n"
                report += f"   Bank amount: {disc['bank_amount']:,.2f}\n"
                report += f"   Difference: {disc['difference']:,.2f} ({disc['percentage_diff']:+.1f}%)\n\n"
            
            if len(amount_comp['discrepancies']) > 20:
                report += f"... and {len(amount_comp['discrepancies']) - 20} more discrepancies\n\n"
        
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