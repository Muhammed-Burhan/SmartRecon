"""
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

    # File: 1.xlsx
    # Available columns: ['No', 'Biller Name', 'Paying Bank Name', 'Settlement Bank Name', 'Service Name', 'Billing No', 'Payment Type', 'Payment Status', 'JOEBPPSTrx', 'Bank Trx ID', 'Corridor', 'Access Channel', 'Procces Date', 'Settlement Date', 'Due Amt', 'Paid Amt']
    # Suggested amount columns: ['Due Amt', 'Paid Amt']
    "1.xlsx": "Paid Amt",  # <- Put the amount column name here (e.g., "Paid Amt")

    # File: 10.xlsx
    # Available columns: ['No', 'Biller Name', 'Paying Bank Name', 'Settlement Bank Name', 'Service Name', 'Billing No', 'Payment Type', 'Payment Status', 'JOEBPPSTrx', 'Bank Trx ID', 'Corridor', 'Access Channel', 'Procces Date', 'Settlement Date', 'Due Amt', 'Paid Amt']
    # Suggested amount columns: ['Paid Amt']
    "10.xlsx": "Paid Amt",  # <- Put the amount column name here (e.g., "Paid Amt")

    # File: 2.xlsx
    # Available columns: ['No', 'Biller Name', 'Paying Bank Name', 'Settlement Bank Name', 'Service Name', 'Billing No', 'Payment Type', 'Payment Status', 'JOEBPPSTrx', 'Bank Trx ID', 'Corridor', 'Access Channel', 'Procces Date', 'Settlement Date', 'Due Amt', 'Paid Amt']
    # Suggested amount columns: ['Due Amt', 'Paid Amt']
    "2.xlsx": "Paid Amt",  # <- Put the amount column name here (e.g., "Paid Amt")

    # File: 3.xlsx
    # Available columns: ['Unnamed: 0', 'Unnamed: 1', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7', 'Unnamed: 8', 'Unnamed: 9', 'Unnamed: 10', 'Unnamed: 11', 'Unnamed: 12', 'Unnamed: 13', 'Unnamed: 14']
    # Suggested amount columns: []
    "3.xlsx": "Paid Amt",  # <- Put the amount column name here (e.g., "Paid Amt")

    # File: 4.xlsx
    # Available columns: ['No', 'Biller Name', 'Paying Bank Name', 'Settlement Bank Name', 'Service Name', 'Billing No', 'Payment Type', 'Payment Status', 'JOEBPPSTrx', 'Bank Trx ID', 'Corridor', 'Access Channel', 'Procces Date', 'Settlement Date', 'Due Amt', 'Paid Amt']
    # Suggested amount columns: ['Due Amt', 'Paid Amt']
    "4.xlsx": "Paid Amt",  # <- Put the amount column name here (e.g., "Paid Amt")

    # File: 5.xlsx
    # Available columns: ['No', 'Biller Name', 'Paying Bank Name', 'Settlement Bank Name', 'Service Name', 'Billing No', 'Payment Type', 'Payment Status', 'JOEBPPSTrx', 'Bank Trx ID', 'Corridor', 'Access Channel', 'Procces Date', 'Settlement Date', 'Due Amt', 'Paid Amt']
    # Suggested amount columns: ['Due Amt', 'Paid Amt']
    "5.xlsx": "Paid Amt",  # <- Put the amount column name here (e.g., "Paid Amt")

    # File: 6.xlsx
    # Available columns: ['No', 'Biller Name', 'Paying Bank Name', 'Settlement Bank Name', 'Service Name', 'Billing No', 'Payment Type', 'Payment Status', 'JOEBPPSTrx', 'Bank Trx ID', 'Corridor', 'Access Channel', 'Procces Date', 'Settlement Date', 'Due Amt', 'Paid Amt']
    # Suggested amount columns: ['Due Amt', 'Paid Amt']
    "6.xlsx": "Paid Amt",  # <- Put the amount column name here (e.g., "Paid Amt")

    # File: 7.xlsx
    # Available columns: ['No', 'Biller Name', 'Paying Bank Name', 'Settlement Bank Name', 'PSPName', 'Service Name', 'Billing No', 'Payment Type', 'Payment Status', 'JOEBPPSTrx', 'Bank Trx ID', 'Corridor', 'Access Channel', 'Procces Date', 'Settlement Date', 'Due Amt', 'Paid Amt']
    # Suggested amount columns: ['Due Amt', 'Paid Amt']
    "7.xlsx": "Paid Amt",  # <- Put the amount column name here (e.g., "Paid Amt")

    # File: 8.xlsx
    # Available columns: ['No', 'Biller Name', 'Paying Bank Name', 'Settlement Bank Name', 'Service Name', 'Billing No', 'Payment Type', 'Payment Status', 'JOEBPPSTrx', 'Bank Trx ID', 'Corridor', 'Access Channel', 'Procces Date', 'Settlement Date', 'Due Amt', 'Paid Amt']
    # Suggested amount columns: ['Paid Amt']
    "8.xlsx": "Paid Amt",  # <- Put the amount column name here (e.g., "Paid Amt")

    # File: 9.xlsx
    # Available columns: ['No', 'Biller Name', 'Paying Bank Name', 'Settlement Bank Name', 'Service Name', 'Billing No', 'Payment Type', 'Payment Status', 'JOEBPPSTrx', 'Bank Trx ID', 'Corridor', 'Access Channel', 'Procces Date', 'Settlement Date', 'Due Amt', 'Paid Amt']
    # Suggested amount columns: ['Paid Amt']
    "9.xlsx": "Paid Amt",  # <- Put the amount column name here (e.g., "Paid Amt")

    # File: Credit دائن.xlsx
    # Available columns: ['Date التاريخ', 'Value Date تاريخ الإستحقاق', 'Description التفاصيل', 'Credit دائن', 'Unnamed: 4', 'Description التفاصيل.1', 'Description']
    # Suggested amount columns: ['Credit دائن']
    "Credit دائن.xlsx": "Credit دائن",  # <- Put the amount column name here (e.g., "Paid Amt")

    # File: Credit.xlsx
    # Available columns: ['Transaction Date', 'Unnamed: 1', 'Unnamed: 2', 'Description', 'Unnamed: 4', 'Debit', 'Unnamed: 6', 'Credit', 'Unnamed: 8', 'Description.1']
    # Suggested amount columns: ['Debit', 'Credit']
    "Credit.xlsx": "Credit",  # <- Put the amount column name here (e.g., "Paid Amt")

    # File: Credit_1.xlsx
    # Available columns: ['Transaction Date', 'Narration', 'Reference', 'Amount Tag', 'Value Date', 'Unnamed: 5', 'Debit', 'Credit', 'Balance']
    # Suggested amount columns: ['Debit', 'Credit', 'Balance']
    "Credit_1.xlsx": "Credit",  # <- Put the amount column name here (e.g., "Paid Amt")

}
