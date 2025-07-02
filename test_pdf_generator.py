#!/usr/bin/env python3
"""
Test script for PDF Report Generator
"""

from pdf_report_generator import PDFReportGenerator
from datetime import datetime

def test_pdf_generation():
    """Test the PDF report generation functionality"""
    
    # Sample job data
    job_data = {
        'id': 1,
        'job_name': 'Test Reconciliation Job',
        'bank_name': 'Test Bank',
        'our_file_name': 'our_reconciliation.xlsx',
        'bank_file_name': 'bank_statement.csv',
        'created_at': datetime.now().isoformat(),
        'status': 'completed',
        'total_our_records': 1000,
        'total_bank_records': 950,
        'matched_records': 900,
        'unmatched_our_records': 100,
        'unmatched_bank_records': 50
    }
    
    # Sample results data
    results_data = [
        {
            'bank_trx_id': 'TBPM25160112163910187',
            'status': 'matched',
            'paying_bank_name': 'Test Bank',
            'amount': 1500.00,
            'created_at': datetime.now().isoformat()
        },
        {
            'bank_trx_id': 'TBPM25160112163910188',
            'status': 'missing_in_bank',
            'paying_bank_name': 'Test Bank',
            'amount': 2500.00,
            'created_at': datetime.now().isoformat()
        },
        {
            'bank_trx_id': 'TBPM25160112163910189',
            'status': 'missing_in_our_file',
            'paying_bank_name': 'Test Bank',
            'amount': 0.00,
            'created_at': datetime.now().isoformat()
        }
    ]
    
    # Generate PDF report
    pdf_generator = PDFReportGenerator()
    output_path = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    
    try:
        pdf_generator.generate_reconciliation_report(job_data, results_data, output_path)
        print(f"✅ PDF report generated successfully: {output_path}")
        print(f"📄 File size: {len(open(output_path, 'rb').read())} bytes")
        return True
    except Exception as e:
        print(f"❌ Error generating PDF: {str(e)}")
        return False

if __name__ == "__main__":
    print("🧪 Testing PDF Report Generator...")
    success = test_pdf_generation()
    if success:
        print("🎉 PDF generation test passed!")
    else:
        print("💥 PDF generation test failed!") 