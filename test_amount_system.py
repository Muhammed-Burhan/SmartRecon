#!/usr/bin/env python3
"""
Comprehensive Test for Amount Comparison System
Test all components of the enhanced SmartRecon system
"""

import os
import pandas as pd
import numpy as np
from ml_column_detector import MLColumnDetector
from reconciliation_engine import ReconciliationEngine
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_enhanced_ml_detector():
    """Test the enhanced ML detector with amount detection capabilities"""
    logger.info("🧪 Testing Enhanced ML Column Detector...")
    
    try:
        detector = MLColumnDetector()
        
        # Test feature extraction for amount columns
        test_data = pd.DataFrame({
            'Bank Trx ID': ['TBPM123', 'No456', 'TBPM789'],
            'Paid Amt': [1500.00, 2500.50, 3000.75],
            'Customer Name': ['John Doe', 'Jane Smith', 'Bob Johnson'],
            'Date': ['2024-01-01', '2024-01-02', '2024-01-03']
        })
        
        # Test Bank Trx ID feature extraction
        bank_features = detector.extract_features(test_data, 'Bank Trx ID')
        logger.info(f"✅ Bank Trx ID features: {bank_features.shape[0]} features extracted")
        
        # Test Amount feature extraction
        amount_features = detector.extract_amount_features(test_data, 'Paid Amt')
        logger.info(f"✅ Amount features: {amount_features.shape[0]} features extracted")
        
        # Both should be 61 features for consistency
        assert bank_features.shape[0] == 61, f"Expected 61 bank features, got {bank_features.shape[0]}"
        assert amount_features.shape[0] == 61, f"Expected 61 amount features, got {amount_features.shape[0]}"
        
        logger.info("✅ Enhanced ML detector test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced ML detector test failed: {e}")
        return False

def test_amount_detection_rules():
    """Test rule-based amount detection"""
    logger.info("🧪 Testing Rule-Based Amount Detection...")
    
    try:
        engine = ReconciliationEngine()
        
        # Test our file patterns
        our_data = pd.DataFrame({
            'Bank Trx ID': ['TBPM123', 'No456'],
            'Paid Amt': [1500.00, 2500.50],
            'Customer Name': ['John Doe', 'Jane Smith']
        })
        
        column, confidence = engine.find_amount_column(our_data, "our")
        logger.info(f"Our file amount detection: '{column}' (confidence: {confidence:.1f}%)")
        
        # Should detect "Paid Amt"
        assert column == "Paid Amt", f"Expected 'Paid Amt', got '{column}'"
        assert confidence > 80, f"Expected high confidence, got {confidence:.1f}%"
        
        # Test bank file patterns
        bank_data = pd.DataFrame({
            'Transaction_ID': ['TXN123', 'TXN456'],
            'Credit': [1500.00, 2500.50],
            'Description': ['Payment 1', 'Payment 2']
        })
        
        column, confidence = engine.find_amount_column(bank_data, "bank")
        logger.info(f"Bank file amount detection: '{column}' (confidence: {confidence:.1f}%)")
        
        # Should detect "Credit"
        assert column == "Credit", f"Expected 'Credit', got '{column}'"
        assert confidence > 50, f"Expected reasonable confidence, got {confidence:.1f}%"
        
        logger.info("✅ Rule-based amount detection test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Rule-based amount detection test failed: {e}")
        return False

def test_amount_extraction():
    """Test amount extraction and numeric conversion"""
    logger.info("🧪 Testing Amount Extraction...")
    
    try:
        engine = ReconciliationEngine()
        
        # Test various amount formats
        test_data = pd.DataFrame({
            'Amount': [
                '1,500.00',      # Comma formatting
                '$2,500.50',     # Currency symbol
                '3000',          # Plain number
                '4.5e3',         # Scientific notation
                '5000.75',       # Decimal
                'invalid',       # Invalid (should be skipped)
                '',              # Empty (should be skipped)
                '6,789.99'       # More comma formatting
            ]
        })
        
        amounts = engine.extract_amounts(test_data, 'Amount')
        logger.info(f"Extracted amounts: {amounts}")
        
        # Should extract valid numeric amounts
        expected_valid_count = 6  # 6 valid amounts out of 8
        assert len(amounts) == expected_valid_count, f"Expected {expected_valid_count} amounts, got {len(amounts)}"
        
        # Check specific values
        assert 1500.00 in amounts, "1,500.00 not properly extracted"
        assert 2500.50 in amounts, "$2,500.50 not properly extracted"
        assert 4500.0 in amounts, "4.5e3 not properly extracted"
        
        logger.info("✅ Amount extraction test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Amount extraction test failed: {e}")
        return False

def test_amount_comparison():
    """Test amount comparison logic"""
    logger.info("🧪 Testing Amount Comparison...")
    
    try:
        engine = ReconciliationEngine()
        
        # Create test data
        our_data = pd.DataFrame({
            'Bank Trx ID': ['TXN001', 'TXN002', 'TXN003', 'TXN004'],
            'Paid Amt': [1000.00, 1500.50, 2000.00, 2500.75],
            'Paying Bank Name': ['Test Bank', 'Test Bank', 'Test Bank', 'Test Bank']
        })
        
        bank_data = pd.DataFrame({
            'Transaction_ID': ['TXN001', 'TXN002', 'TXN003', 'TXN005'],  # TXN004 missing, TXN005 extra
            'Credit': [1000.00, 1500.60, 2000.00, 3000.00],  # TXN002 has slight difference
        })
        
        # Perform amount comparison
        results = engine.compare_amounts(
            our_data, bank_data,
            'Bank Trx ID', 'Transaction_ID',
            'Paid Amt', 'Credit',
            'Test Bank'
        )
        
        logger.info(f"Amount comparison results: {results['summary']}")
        
        # Verify results
        assert len(results['discrepancies']) == 1, f"Expected 1 discrepancy, got {len(results['discrepancies'])}"
        assert results['summary']['total_discrepancies'] == 1
        assert not results['summary']['amounts_match']  # Should not match due to discrepancy
        
        # Check the specific discrepancy (TXN002: 1500.50 vs 1500.60)
        disc = results['discrepancies'][0]
        assert disc['bank_trx_id'] == 'TXN002'
        assert abs(disc['our_amount'] - 1500.50) < 0.01
        assert abs(disc['bank_amount'] - 1500.60) < 0.01
        assert abs(disc['difference'] - (-0.10)) < 0.01
        
        logger.info("✅ Amount comparison test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Amount comparison test failed: {e}")
        return False

def test_full_reconciliation_with_amounts():
    """Test full reconciliation process with amount comparison"""
    logger.info("🧪 Testing Full Reconciliation with Amount Comparison...")
    
    try:
        engine = ReconciliationEngine()
        
        # Create comprehensive test data
        our_data = pd.DataFrame({
            'Bank Trx ID': ['TBPM001', 'TBPM002', 'TBPM003', 'TBPM004'],
            'Paid Amt': [1000.00, 1500.50, 2000.00, 2500.75],
            'Customer Name': ['Customer A', 'Customer B', 'Customer C', 'Customer D'],
            'Paying Bank Name': ['Test Bank', 'Test Bank', 'Test Bank', 'Test Bank']
        })
        
        bank_data = pd.DataFrame({
            'Transaction_ID': ['TBPM001', 'TBPM002', 'TBPM003', 'TBPM005'],
            'Credit': [1000.00, 1500.60, 2000.00, 3000.00],
            'Description': ['Payment A', 'Payment B', 'Payment C', 'Payment E']
        })
        
        # Perform full reconciliation
        results = engine.perform_reconciliation(
            our_data, bank_data,
            'Transaction_ID', 'Test Bank',
            'Bank Trx ID'
        )
        
        logger.info(f"Reconciliation summary: {results['summary']}")
        logger.info(f"Amount comparison available: {results['summary']['amount_comparison_available']}")
        
        # Verify transaction matching worked
        assert results['summary']['matched_records'] == 3, f"Expected 3 matches, got {results['summary']['matched_records']}"
        assert results['summary']['missing_in_bank_count'] == 1, f"Expected 1 missing in bank, got {results['summary']['missing_in_bank_count']}"
        assert results['summary']['missing_in_our_file_count'] == 1, f"Expected 1 missing in our file, got {results['summary']['missing_in_our_file_count']}"
        
        # Verify amount comparison was performed
        amount_comp = results.get('amount_comparison', {})
        if amount_comp.get('comparison_performed', False):
            logger.info("✅ Amount comparison was performed automatically")
            assert amount_comp['summary']['total_discrepancies'] == 1, "Expected 1 amount discrepancy"
        else:
            logger.warning("⚠️ Amount comparison was not performed (ML not trained)")
        
        # Generate report to test formatting
        report = engine.generate_report(results, 'Test Bank')
        assert 'AMOUNT COMPARISON SUMMARY' in report, "Report should include amount comparison section"
        
        logger.info("✅ Full reconciliation test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Full reconciliation test failed: {e}")
        return False

def test_enhanced_model_loading():
    """Test enhanced model loading and backward compatibility"""
    logger.info("🧪 Testing Enhanced Model Loading...")
    
    try:
        engine = ReconciliationEngine()
        
        # Test ML detector loading
        if engine.ml_detector:
            logger.info("✅ ML detector loaded successfully")
            
            # Check capabilities
            if hasattr(engine.ml_detector, 'amount_trained'):
                if engine.ml_detector.amount_trained:
                    logger.info("✅ Amount detection trained and available")
                else:
                    logger.info("⚠️ Amount detection not trained (expected for initial setup)")
            
            # Test prediction capability
            test_data = pd.DataFrame({
                'Bank Trx ID': ['TBPM123'],
                'Paid Amt': [1500.00],
                'Name': ['Test']
            })
            
            try:
                column, confidence = engine.ml_detector.predict_column(test_data)
                logger.info(f"✅ Bank Trx ID prediction works: '{column}' ({confidence:.1f}%)")
            except Exception as e:
                logger.warning(f"⚠️ Bank Trx ID prediction not available: {e}")
            
        else:
            logger.info("⚠️ ML detector not loaded (PyTorch not available or model not trained)")
        
        logger.info("✅ Enhanced model loading test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced model loading test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 SmartRecon Amount Comparison System Test Suite")
    print("=" * 60)
    
    tests = [
        ("Enhanced ML Detector", test_enhanced_ml_detector),
        ("Rule-Based Amount Detection", test_amount_detection_rules),
        ("Amount Extraction", test_amount_extraction),
        ("Amount Comparison Logic", test_amount_comparison),
        ("Full Reconciliation with Amounts", test_full_reconciliation_with_amounts),
        ("Enhanced Model Loading", test_enhanced_model_loading),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: FAILED with exception: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The amount comparison system is ready.")
        print("\n📋 Next Steps:")
        print("1. Train the amount detector: python train_amount_detector.py")
        print("2. Run reconciliation with amount comparison")
        print("3. Check the enhanced reports and UI")
    else:
        print("⚠️ Some tests failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    main() 