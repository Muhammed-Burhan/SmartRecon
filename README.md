# 🏦 Bank Reconciliation System

A sophisticated web-based system for reconciling financial transactions between your settlement files and bank/PSP files. The system uses intelligent column detection to automatically identify Bank Transaction IDs and perform comprehensive reconciliation analysis.

## ✨ Features

- **Smart Column Detection**: Automatically identifies Bank Trx ID columns in bank/PSP files
- **Multi-Format Support**: Handles both Excel (.xlsx, .xls) and CSV files
- **Bank Identification**: Automatically determines which bank the transactions belong to
- **Comprehensive Reconciliation**: Finds matched, missing, and extra transactions
- **Web Interface**: User-friendly Streamlit frontend with file upload and reporting
- **API Backend**: FastAPI server for processing and data management
- **Database Storage**: SQLite database for storing reconciliation history
- **Large File Support**: Efficiently handles files with 250k+ rows

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the API Server
```bash
python start_api.py
```
The API will be available at `http://localhost:8000`
API documentation at `http://localhost:8000/docs`

### 3. Start the Web Interface
```bash
python start_streamlit.py
```
The web app will be available at `http://localhost:8501`

## 📋 How It Works

### File Structure Expected

**Your Reconciliation File** should contain these columns:
- `No` - Record number
- `Biller Name` - Name of the biller
- `Paying Bank Name` - Name of the paying bank
- `Settlement Bank Name` - Settlement bank name
- `Service Name` - Service type
- `Billing No` - Billing reference
- `Payment Type` - Type of payment
- `Payment Status` - Status of payment
- `JOEBPPSTrx` - Internal transaction reference
- `Bank Trx ID` - **Key field for matching**
- `Corridor` - Payment corridor
- `Access Channel` - Channel used
- `Process Date` - Processing date
- `Settlement Date` - Settlement date
- `Due Amt` - Due amount
- `Paid Amt` - Paid amount

**Bank/PSP File** can have any structure. The system will automatically:
1. Analyze all columns to find the one containing Bank Trx IDs
2. Use pattern matching to identify transaction ID formats
3. Match against your reconciliation file

### Processing Flow

1. **Upload Files**: Upload your reconciliation file + bank/PSP file
2. **Column Detection**: System identifies Bank Trx ID column in bank file
3. **Bank Identification**: Determines which bank these transactions belong to
4. **Reconciliation**: Compares transactions and finds:
   - ✅ **Matched**: Transactions present in both files
   - ❌ **Missing in Bank**: Transactions in your file but not in bank file
   - ⚠️ **Missing in Our File**: Transactions in bank file but not in your file
5. **Reporting**: Generates detailed reconciliation report

## 🔧 System Architecture

```
Frontend (Streamlit)
     ↓
FastAPI Backend
     ↓
ReconciliationEngine (Pandas)
     ↓
SQLite Database
```

### Core Components

- **`reconciliation_engine.py`**: Core logic for file processing and reconciliation
- **`api.py`**: FastAPI server with REST endpoints
- **`streamlit_app.py`**: Web interface for file uploads and reporting
- **`database.py`**: Database models and connection handling

## 📊 Supported Transaction ID Formats

The system recognizes various Bank Trx ID patterns:
- `TBPM25160112163910187` (Alphanumeric with bank code)
- `1234567890123456789` (Pure numeric)
- `ABC123456789DEF` (Mixed alphanumeric)
- Custom patterns (system learns from your data)

## 💡 Usage Examples

### Example 1: Standard Reconciliation
1. Upload your settlement file with standard column structure
2. Upload bank file (any format)
3. System identifies "Description" column contains Bank Trx IDs
4. Matches against "ABC Bank" transactions
5. Generates report showing 95% match rate

### Example 2: Multiple Banks
1. Upload file containing transactions from multiple banks
2. System identifies transactions belong to "XYZ Bank" and "ABC Bank"
3. Processes the bank with most matching transactions
4. Provides detailed breakdown by bank

## 🔍 API Endpoints

- `POST /upload-files/` - Upload and process reconciliation files
- `GET /jobs/` - List all reconciliation jobs
- `GET /jobs/{job_id}/` - Get specific job details
- `GET /jobs/{job_id}/report/` - Generate detailed report

## 📁 File Structure

```
.
├── requirements.txt          # Python dependencies
├── database.py              # Database models
├── reconciliation_engine.py # Core reconciliation logic
├── api.py                   # FastAPI server
├── streamlit_app.py         # Web interface
├── start_api.py             # API server starter
├── start_streamlit.py       # Streamlit starter
├── README.md                # This file
└── reconciliation.db        # SQLite database (created automatically)
```

## 🎯 Performance

- **Large Files**: Efficiently processes 250k+ row files
- **Memory Management**: Uses chunked processing for very large files
- **Fast Matching**: Optimized pandas operations for quick reconciliation
- **Encoding Detection**: Automatically handles different file encodings

## 🛠️ Troubleshooting

### Common Issues

**"Could not identify Bank Trx ID column"**
- Check if bank file contains transaction IDs
- Verify transaction ID format matches expected patterns
- Try manual column specification

**"No matching Bank Trx IDs found"**
- Ensure both files contain the same transaction ID format
- Check date ranges - files should cover same period
- Verify Bank Trx ID column contains actual transaction IDs

**"Cannot connect to API server"**
- Ensure API server is running on port 8000
- Check firewall settings
- Verify `python start_api.py` completed successfully

## 🔮 Future Enhancements

- [ ] Manual column mapping interface
- [ ] Multiple bank reconciliation in single job
- [ ] Excel report export
- [ ] Email notifications
- [ ] Advanced matching algorithms
- [ ] Integration with cloud storage

## 🤝 Contributing

This system is designed to be extensible. Key areas for enhancement:
- Additional transaction ID patterns
- New file format support
- Advanced matching algorithms
- Enhanced reporting features

---

**Built with**: Python 3.13, FastAPI, Streamlit, Pandas, SQLite 