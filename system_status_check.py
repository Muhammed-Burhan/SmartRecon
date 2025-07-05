#!/usr/bin/env python3
"""
SmartRecon System Status Check
Comprehensive verification of all components
"""

import pandas as pd
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    print("🔍 SmartRecon System Status Check")
    print("=" * 50)
    
    results = {
        'database': False,
        'keywords': False,
        'ml_model': False,
        'bank_trx_ml': False,
        'amount_ml': False,
        'bank_trx_fallback': False,
        'amount_fallback': False,
        'reconciliation': False,
        'api': False
    }
    
    # 1. Database Check
    print("\n1️⃣ Database System Check...")
    try:
        from database import KeywordManager, create_database
        create_database()
        km = KeywordManager()
        keywords = km.get_keywords()
        print(f"   ✅ Database connection: OK")
        print(f"   ✅ Keywords available: {len(keywords)}")
        results['database'] = True
        results['keywords'] = len(keywords) > 0
    except Exception as e:
        print(f"   ❌ Database error: {e}")
    
    # 2. ML Model Check
    print("\n2️⃣ ML Model Check...")
    try:
        from ml_column_detector import MLColumnDetector
        detector = MLColumnDetector()
        
        # Check if enhanced model exists
        if os.path.exists('enhanced_column_detector.pth'):
            success = detector.load_model('enhanced_column_detector.pth')
            if success:
                print("   ✅ Enhanced ML model loaded")
                results['ml_model'] = True
                
                # Check capabilities
                if hasattr(detector, 'amount_trained') and detector.amount_trained:
                    print("   ✅ Amount detection trained")
                    results['amount_ml'] = True
                else:
                    print("   ⚠️ Amount detection not trained")
                
                if hasattr(detector, 'bank_trx_trained') and detector.bank_trx_trained:
                    print("   ✅ Bank Trx ID detection trained")
                    results['bank_trx_ml'] = True
                else:
                    print("   ⚠️ Bank Trx ID detection not trained")
            else:
                print("   ❌ Could not load enhanced model")
        else:
            print("   ⚠️ Enhanced model not found")
    except Exception as e:
        print(f"   ❌ ML model error: {e}")
    
    # 3. Reconciliation Engine Check
    print("\n3️⃣ Reconciliation Engine Check...")
    try:
        from reconciliation_engine import ReconciliationEngine
        engine = ReconciliationEngine()
        print("   ✅ Reconciliation engine initialized")
        
        # Test Bank Trx ID detection
        print("\n   📊 Testing Bank Trx ID Detection...")
        our_test_df = pd.DataFrame({
            'Bank Trx ID': ['TBPM123456', 'TBPM789012'],
            'Paid Amt': [1500.00, 2500.50],
            'Customer Name': ['John Doe', 'Jane Smith']
        })
        
        # Test our file Bank Trx ID
        col, conf = engine.find_our_bank_trx_column(our_test_df)
        print(f"   ✅ Our file Bank Trx ID: '{col}' (confidence: {conf:.1f}%)")
        if conf > 80:
            results['bank_trx_fallback'] = True
        
        # Test bank file Bank Trx ID
        bank_test_df = pd.DataFrame({
            'Transaction_ID': ['TBPM123456', 'TBPM789012'],
            'Credit': [1500.00, 2500.50],
            'Description': ['Payment 1', 'Payment 2']
        })
        
        col, conf = engine.find_bank_trx_column(bank_test_df)
        print(f"   ✅ Bank file Bank Trx ID: '{col}' (confidence: {conf:.1f}%)")
        
        # Test Amount detection
        print("\n   💰 Testing Amount Detection...")
        
        # Test our file amount
        col, conf = engine.find_amount_column(our_test_df, "our")
        print(f"   ✅ Our file Amount: '{col}' (confidence: {conf:.1f}%)")
        if conf > 80:
            results['amount_fallback'] = True
        
        # Test bank file amount
        col, conf = engine.find_amount_column(bank_test_df, "bank")
        print(f"   ✅ Bank file Amount: '{col}' (confidence: {conf:.1f}%)")
        
        results['reconciliation'] = True
        
    except Exception as e:
        print(f"   ❌ Reconciliation engine error: {e}")
    
    # 4. API Check
    print("\n4️⃣ API Check...")
    try:
        # Just try to import the API
        from api import app
        print("   ✅ API module loaded")
        results['api'] = True
    except Exception as e:
        print(f"   ❌ API error: {e}")
    
    # 5. Summary
    print("\n📋 SYSTEM STATUS SUMMARY")
    print("=" * 50)
    
    components = [
        ("Database Connection", results['database']),
        ("Keywords Database", results['keywords']),
        ("ML Model Available", results['ml_model']),
        ("Bank Trx ID ML", results['bank_trx_ml']),
        ("Amount ML", results['amount_ml']),
        ("Bank Trx ID Fallback", results['bank_trx_fallback']),
        ("Amount Fallback", results['amount_fallback']),
        ("Reconciliation Engine", results['reconciliation']),
        ("API Module", results['api'])
    ]
    
    for component, status in components:
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {component}")
    
    # Overall status
    critical_components = [
        results['database'],
        results['keywords'],
        results['bank_trx_fallback'],
        results['amount_fallback'],
        results['reconciliation']
    ]
    
    all_critical_ok = all(critical_components)
    ml_ready = results['ml_model'] and results['bank_trx_ml'] and results['amount_ml']
    
    print(f"\n🎯 OVERALL STATUS:")
    if all_critical_ok:
        print("✅ SYSTEM READY FOR PRODUCTION")
        print("✅ All fallback systems working")
        if ml_ready:
            print("✅ ML systems fully operational")
        else:
            print("⚠️ ML systems partially ready (fallback will handle)")
    else:
        print("❌ SYSTEM NEEDS ATTENTION")
        print("❌ Some critical components not working")
    
    print(f"\n🔧 CAPABILITIES:")
    print(f"   📊 Bank Trx ID Detection: {'ML + Fallback' if results['bank_trx_ml'] else 'Fallback Only'}")
    print(f"   💰 Amount Detection: {'ML + Fallback' if results['amount_ml'] else 'Fallback Only'}")
    print(f"   📈 Reconciliation: {'Full' if results['reconciliation'] else 'Limited'}")
    print(f"   🗄️ Database: {'Available' if results['database'] else 'Unavailable'}")

if __name__ == "__main__":
    main() 