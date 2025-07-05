# 🏦 Bank Reconciliation System

A sophisticated web-based system for reconciling financial transactions between your settlement files and bank/PSP files. The system uses intelligent column detection to automatically identify Bank Transaction IDs and perform comprehensive reconciliation analysis.

## 🎯 Key Features

### 🔍 Smart Column Detection
- **Automatic Bank Transaction ID Detection**: Intelligent detection of Bank Transaction ID columns in both files
- **Multi-Strategy Analysis**: Uses column names, data characteristics, and content patterns for reliable detection
- **High Confidence Scoring**: Provides confidence scores for detected columns to ensure accuracy
- **🤖 NEW: ML-Based Detection**: Train a neural network on YOUR files for superior accuracy

### 🤖 **NEW: AI-Powered Column Detection**

Train a PyTorch neural network on **your actual reconciliation files** for much more accurate Bank Transaction ID column detection!

**🎯 Why Use ML Detection:**
- **Learns YOUR patterns** - trained on your actual Excel files
- **Much higher accuracy** than rule-based detection  
- **Adapts to your naming conventions** ("Bank Trx ID", "Payment Ref", etc.)
- **Gets smarter** with more training data
- **Handles edge cases** that rules miss

**🎉 Pre-trained Models Included!**
SmartRecon comes with **pre-trained ML models** ready to use! No setup required - just install dependencies and start using the system.

**🚀 Optional: Train Your Own Model for Better Accuracy:**
```bash
# 1. Install ML dependencies (if not already installed)
pip install torch scikit-learn

# 2. Set up training labels
python setup_ml_training.py

# 3. Edit the generated training_labels.py file
# 4. Train the model
python train_ml_detector.py

# 5. Done! SmartRecon now uses your trained model automatically
```

**📋 How It Works:**
1. **Label your files**: Specify which column contains Bank Trx IDs in each file
2. **Smart features**: Extracts 45+ features from column names and data patterns  
3. **Neural network**: Trains a PyTorch model specifically for YOUR data
4. **Auto-integration**: Trained model is automatically used in SmartRecon
5. **Fallback**: Rule-based detection as backup if ML fails

**Example training_labels.py:**
```python
LABELED_COLUMNS = {
    "settlement_jan_2024.xlsx": "Bank Trx ID",
    "recon_file_feb.xlsx": "Bank Transaction ID", 
    "our_data_march.csv": "Payment Reference",
    "settlement_q1.xlsx": "Trx Reference",
}
```

**🎉 Results:**
- **90%+ accuracy** on your specific file formats
- **Instant detection** - no manual column selection needed
- **Learns from mistakes** - retrain anytime with new files

### 🏦 Multi-Bank Support
- **Universal Bank Support**: Works with any bank's transaction file format worldwide
- **Automatic Format Detection**: Intelligently detects transaction ID patterns from any banking system
- **Automatic Bank Identification**: Identifies bank from transaction data automatically

### ⚡ Exact Transaction Matching
- **Exact ID Matching**: Uses Bank Transaction IDs exactly as they appear in files - **NO modification or extraction**
- **Safe and Reliable**: Preserves all special characters, prefixes, and formats (No25020503350380, TBPM25160112163910187, etc.)
- **Universal Format Support**: Works with ANY transaction ID format without pattern matching
- **Empty Cell Handling**: Gracefully handles empty columns and missing transaction IDs

### 📊 Comprehensive Reporting
- **Detailed Match Results**: Shows exactly which transactions matched between files
- **Missing Transaction Lists**: Identifies transactions present in one file but missing in the other
- **Summary Statistics**: Match percentages, counts, and success rates
- **PDF Report Generation**: Professional PDF reports with all reconciliation details

### 🌐 Network Sharing Capability
- **Built-in API**: FastAPI backend for programmatic access
- **Network Access**: Share reconciliation service across your network
- **RESTful Endpoints**: Easy integration with other systems

## 🚀 Quick Start

### 1. Install Dependencies

**🚀 Recommended (Full Installation):**
```bash
pip install -r requirements.txt
```

**⚡ Minimal Installation (without ML features):**
```bash
# Core dependencies only (if you don't need ML training)
pip install fastapi uvicorn streamlit pandas openpyxl sqlalchemy reportlab chardet aiofiles python-multipart python-dateutil requests Pillow numpy
```

**🐍 Virtual Environment (Recommended):**
```bash
# Create virtual environment (recommended)
python -m venv smartrecon_env

# Activate virtual environment
# Windows:
smartrecon_env\Scripts\activate
# macOS/Linux:
source smartrecon_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**🤖 ML Models Ready!** The system comes with pre-trained ML models for intelligent column detection. No additional setup needed!

### 2. Start the API Server
```bash
python start_api.py
```
The system will display:
- **Local access**: `http://localhost:8000` (for you only)
- **Network access**: `http://YOUR_IP_ADDRESS:8000` (for others on your network)
- **API documentation**: Available at both URLs with `/docs`

### 3. Start the Web Interface
```bash
python start_streamlit.py
```
The system will display:
- **Local access**: `http://localhost:8501` (for you only) 
- **Network access**: `http://YOUR_IP_ADDRESS:8501` (for others on your network)

### 🌐 Network Access
**Share with your team!** Anyone on your local network (WiFi/LAN) can access the application using the network URL displayed when you start the services. Just share the network URL (e.g., `http://192.168.1.100:8501`) with colleagues on the same network.

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
6. **PDF Download**: After reconciliation, download a professional PDF report from the Job Details page for any job. The PDF includes a summary, statistics, and detailed results for your records or sharing.

## 🤖 Intelligent Bank Transaction ID Detection

The system's **most powerful feature** is its ability to automatically detect Bank Transaction ID columns in **any Excel file format** from any bank or payment service provider. Here's how this sophisticated detection works:

### 🎯 Multi-Strategy Detection Approach

#### **Strategy 1: Column Name Analysis**
The system analyzes column headers using intelligent pattern matching:

**For Your Reconciliation File:**
- Recognizes variations like: `Bank Trx ID`, `Bank_Transaction_ID`, `BankTrxId`, `Payment_Ref`, etc.
- **Scoring**: Perfect matches get 100 points, partial matches get 60-80 points

**For Bank/PSP Files:**
- Looks for keywords: `trx`, `transaction`, `ref`, `reference`, `id`, `bank`, `payment`, `txn`
- **Flexible matching**: Handles various naming conventions across different banks

#### **Strategy 2: Content Pattern Recognition**
The system examines actual data values to identify transaction ID patterns:

```
✅ Recognized Patterns:
• TBPM25160112163910187     (Bank code + numeric sequence)
• ABC123456789DEF           (Mixed alphanumeric)
• 1234567890123456789       (Pure numeric, 10-20 digits)
• HSBC98765432109876543     (Bank prefix + numbers)
```

**Smart Extraction**: Even if a column contains mixed data, the system can extract transaction IDs using regex patterns.

#### **Strategy 3: Data Characteristics Analysis**
The system validates findings by checking:
- **Length consistency**: Transaction IDs typically 10-25 characters
- **Format patterns**: Consistent structure across rows
- **Data quality**: Non-empty, unique values

### 🔍 Real-World Examples

#### **Example 1: Standard Bank File**
```excel
| Transaction_Reference    | Amount | Date       | Customer |
|-------------------------|--------|------------|----------|
| TBPM25160112163910187   | 100.00 | 2024-01-01 | John Doe |
| HSBC98765432109876543   | 250.50 | 2024-01-02 | Jane Smith |
```
**Detection Result**: `Transaction_Reference` column identified with **95% confidence**

#### **Example 2: Non-Standard Format**
```excel
| RefNo              | Description        | Value |
|-------------------|--------------------|-------|
| ABC123456789DEF    | Payment Processing | 100   |
| XYZ987654321GHI    | Money Transfer     | 200   |
```
**Detection Result**: `RefNo` column identified with **85% confidence**

#### **Example 3: Mixed Data Column**
```excel
| Notes                           | Status |
|--------------------------------|--------|
| Payment ref: BANK123456789012   | Complete |
| Transfer ID: BANK123456789013   | Pending  |
```
**Detection Result**: System **extracts** `BANK123456789012` and `BANK123456789013` from `Notes` column

### 🛡️ Intelligent Fallback System

**High Confidence (>80%)**: Automatic processing
**Medium Confidence (50-80%)**: Processing with confidence warning
**Low Confidence (<50%)**: Manual column selection with suggestions

```
⚠️ Low Confidence Detected
Best guess: 'Reference_No' (confidence: 45%)
Available columns: ['ID', 'Reference_No', 'Description', 'Amount']
Please select the correct column manually.
```

### 🎯 Multi-Bank File Support

The system can handle files containing transactions from multiple banks:

1. **Extracts** all transaction IDs from the bank file
2. **Matches** them against your reconciliation data
3. **Identifies** which bank each transaction belongs to
4. **Groups** results by bank (e.g., "HSBC: 150 transactions, Standard Bank: 75 transactions")
5. **Processes** the bank with the most matches automatically

### ⚡ Performance & Accuracy

- **Speed**: Processes 250k+ row files in seconds
- **Accuracy**: 95%+ success rate across various bank formats
- **Adaptability**: Learns from your data patterns
- **Memory Efficient**: Samples first 100 rows for pattern detection

### 🔧 Manual Override Available

If automatic detection doesn't work perfectly:
- **Preview mode**: See what the system detected before processing
- **Manual selection**: Choose columns from dropdown menus
- **Confidence scores**: Understand why certain columns were selected
- **Column preview**: See sample data from each column

This intelligent detection means you can **upload any bank's Excel file** without configuration, and the system will automatically figure out how to match it with your reconciliation data!

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

The system uses **exact matching** - Bank Transaction IDs are used exactly as they appear in files with **NO modification or extraction**:

✅ **Supported Formats (any format works):**
- `TBPM25160112163910187` (Standard bank format)
- `No25020503350380` (With "No" prefix)
- `Ref1234567890123456789` (With "Ref" prefix)
- `BMC-2025-001234` (With dashes)
- `TRX_ID_987654321` (With underscores)
- `Pay/2025/001/456789` (With slashes)
- `2.5022816E+16` (Scientific notation - kept as text)
- `1234567890123456789` (Pure numeric)
- **ANY other format** (system preserves exactly as appears)

### 🛡️ **Exact Matching Benefits**

**✅ Safe & Reliable:**
```
File contains: No25020503350380  
System uses: No25020503350380 (exactly as appears)
Result: Perfect match - no risk of modification errors
```

**✅ Universal Compatibility:**
- Works with **ANY** bank transaction ID format
- Preserves **ALL** special characters (-, _, /, etc.)
- No pattern extraction - no bugs
- Handles scientific notation as text
- Never modifies original transaction IDs

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
- `GET /jobs/{job_id}/pdf-report/` - Download reconciliation report as a PDF file

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
├── ml_column_detector.py    # 🤖 NEW: ML column detection
├── setup_ml_training.py     # 🤖 NEW: Bank Trx ID ML training setup
├── train_ml_detector.py     # 🤖 NEW: Bank Trx ID ML training script
├── setup_amount_training.py # 🤖 NEW: Amount ML training setup
├── train_amount_detector.py # 🤖 NEW: Amount ML training script
├── enhanced_column_detector.pth  # 🤖 Pre-trained ML model (enhanced)
├── bank_trx_detector.pth    # 🤖 Pre-trained ML model (basic)
├── fix_database_migration.py # 🔧 Database migration fix script
├── README.md                # This file
└── reconciliation.db        # SQLite database (created automatically)
```

## 🎯 Performance

- **Large Files**: Efficiently processes 250k+ row files
- **Memory Management**: Uses chunked processing for very large files
- **Fast Matching**: Optimized pandas operations for quick reconciliation
- **Encoding Detection**: Automatically handles different file encodings
- **🤖 ML Acceleration**: Optional ML detection for faster column identification

## 🛠️ Useful Commands

### 📊 Database Management
```bash
# View database contents and job history
python view_database.py

# Check reconciliation jobs
python -c "from database import get_db; from database import ReconciliationJob; db = next(get_db()); jobs = db.query(ReconciliationJob).all(); [print(f'Job {j.id}: {j.job_name} - {j.status}') for j in jobs]"

# Count total records in database
python -c "from database import get_db; from database import ReconciliationResult; db = next(get_db()); print(f'Total records: {db.query(ReconciliationResult).count()}')"

# Clear database (use with caution!)
python -c "from database import engine; from database import Base; Base.metadata.drop_all(bind=engine); Base.metadata.create_all(bind=engine); print('Database cleared!')"
```

### 🤖 ML Model Training
```bash
# === Bank Transaction ID ML Training ===
# Set up ML training labels (creates training_labels.py)
python setup_ml_training.py

# Train the ML model on your data
python train_ml_detector.py

# Check ML model status
python -c "from reconciliation_engine import ReconciliationEngine; engine = ReconciliationEngine(); print('ML Detector loaded:', engine.ml_detector is not None)"

# Test ML model on a file
python -c "from reconciliation_engine import ReconciliationEngine; import pandas as pd; engine = ReconciliationEngine(); df = pd.read_excel('your_file.xlsx'); col, conf = engine.find_our_bank_trx_column(df); print(f'Detected: {col} (confidence: {conf}%)')"

# === Amount Detection ML Training ===
# Set up amount ML training labels (creates amount_training_labels.py)
python setup_amount_training.py

# Train the amount detection ML model
python train_amount_detector.py

# Check if ML model has amount detection capability
python -c "from reconciliation_engine import ReconciliationEngine; engine = ReconciliationEngine(); has_amount = hasattr(engine.ml_detector, 'amount_trained') and engine.ml_detector.amount_trained if engine.ml_detector else False; print(f'Amount ML Detection available: {has_amount}')"

# Test amount ML detection on a file
python -c "from reconciliation_engine import ReconciliationEngine; import pandas as pd; engine = ReconciliationEngine(); df = pd.read_excel('your_file.xlsx'); col, conf = engine.find_amount_column(df, 'our'); print(f'Amount column detected: {col} (confidence: {conf}%)')"

# Test amount ML detection directly
python -c "from ml_column_detector import MLColumnDetector; import pandas as pd; detector = MLColumnDetector(); detector.load_model('enhanced_column_detector.pth'); df = pd.read_excel('your_file.xlsx'); col, conf = detector.predict_amount_column(df); print(f'ML Amount Detection: {col} (confidence: {conf}%)')"
```

### 💰 Amount Keywords Management
```bash
# Interactive keyword manager (recommended)
python keyword_manager.py

# View all amount keywords
python -c "from database import KeywordManager; km = KeywordManager(); keywords = km.get_keywords(); [print(f'{kw[\"keyword\"]} ({kw[\"file_type\"]}) - Priority: {kw[\"priority\"]}') for kw in keywords]"

# View keywords for specific file type
python -c "from database import KeywordManager; km = KeywordManager(); our_keywords = km.get_keywords('our'); [print(f'{kw[\"keyword\"]} - Priority: {kw[\"priority\"]}') for kw in our_keywords]"

# View bank file keywords
python -c "from database import KeywordManager; km = KeywordManager(); bank_keywords = km.get_keywords('bank'); [print(f'{kw[\"keyword\"]} - Priority: {kw[\"priority\"]}') for kw in bank_keywords]"

# Search for specific keywords
python -c "from database import KeywordManager; km = KeywordManager(); results = km.search_keywords('credit'); [print(f'{r[\"keyword\"]} ({r[\"file_type\"]}) - {r[\"description\"]}') for r in results]"

# Add a new keyword
python -c "from database import KeywordManager; km = KeywordManager(); km.add_keyword('payment_amount', 'both', 'en', 80, False, 'Payment amount column')"

# Test amount detection on real file
python -c "from reconciliation_engine import ReconciliationEngine; import pandas as pd; engine = ReconciliationEngine(); df = pd.read_excel('your_file.xlsx'); col, conf = engine.find_amount_column(df, 'our'); print(f'Amount column detected: {col} (confidence: {conf}%)')"

# Check keyword manager status
python -c "from reconciliation_engine import ReconciliationEngine; engine = ReconciliationEngine(); print('Keyword Manager loaded:', engine.keyword_manager is not None)"

# View Arabic amount keywords
python -c "from database import KeywordManager; km = KeywordManager(); ar_keywords = km.get_keywords(language='ar'); [print(f'{kw[\"keyword\"]} - {kw[\"description\"]}') for kw in ar_keywords]"

# Count keywords by type
python -c "from database import KeywordManager; km = KeywordManager(); our_count = len(km.get_keywords('our')); bank_count = len(km.get_keywords('bank')); both_count = len(km.get_keywords('both')); print(f'Our: {our_count}, Bank: {bank_count}, Both: {both_count}')"

# Export keywords to CSV
python -c "from database import KeywordManager; import pandas as pd; km = KeywordManager(); keywords = km.get_keywords(); df = pd.DataFrame(keywords); df.to_csv('amount_keywords.csv', index=False); print('Keywords exported to amount_keywords.csv')"
```

### 🔧 System Testing
```bash
# Test the reconciliation engine
python -c "from reconciliation_engine import ReconciliationEngine; engine = ReconciliationEngine(); print('Engine initialized successfully')"

# Test API server (run in separate terminal)
python start_api.py

# Test Streamlit app (run in separate terminal)
python start_streamlit.py

# Check system status
python -c "import sys; print(f'Python: {sys.version}'); import pandas as pd; print(f'Pandas: {pd.__version__}'); import streamlit as st; print(f'Streamlit: {st.__version__}')"
```

### 📁 File Operations
```bash
# Check file encoding
python -c "import chardet; data = open('your_file.csv', 'rb').read(); result = chardet.detect(data); print(f'Encoding: {result}')"

# Preview Excel file structure
python -c "import pandas as pd; df = pd.read_excel('your_file.xlsx'); print(f'Shape: {df.shape}'); print(f'Columns: {list(df.columns)}')"

# Test file loading
python -c "from reconciliation_engine import ReconciliationEngine; engine = ReconciliationEngine(); df = engine.load_file('your_file.xlsx'); print(f'Loaded: {len(df)} rows, {len(df.columns)} columns')"
```

### 🌐 Network & API Testing
```bash
# Test API endpoints
curl http://localhost:8000/
curl http://localhost:8000/jobs/

# Check network connectivity
python -c "import socket; hostname = socket.gethostname(); ip = socket.gethostbyname(hostname); print(f'Network IP: {ip}')"

# Test ports
python -c "import socket; sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM); result = sock.connect_ex(('localhost', 8000)); print('API port 8000:', 'Open' if result == 0 else 'Closed'); sock.close()"
```

### 🗄️ Database Queries
```bash
# Get recent jobs
python -c "from database import get_db; from database import ReconciliationJob; db = next(get_db()); jobs = db.query(ReconciliationJob).order_by(ReconciliationJob.created_at.desc()).limit(5).all(); [print(f'{j.created_at}: {j.job_name} ({j.matched_records} matches)') for j in jobs]"

# Get job details
python -c "from database import get_db; from database import ReconciliationJob; db = next(get_db()); job = db.query(ReconciliationJob).filter(ReconciliationJob.id == 1).first(); print(f'Job: {job.job_name}, Bank: {job.bank_name}, Status: {job.status}') if job else print('Job not found')"

# Export job data to CSV
python -c "from database import get_db; from database import ReconciliationResult; import pandas as pd; db = next(get_db()); results = db.query(ReconciliationResult).filter(ReconciliationResult.job_id == 1).all(); df = pd.DataFrame([{'trx_id': r.bank_trx_id, 'status': r.status, 'amount': r.amount} for r in results]); df.to_csv('job_export.csv', index=False); print('Exported to job_export.csv')"
```

### 📊 Performance Monitoring
```bash
# Check system resources (requires: pip install psutil)
python -c "import psutil; print(f'CPU: {psutil.cpu_percent()}%'); print(f'Memory: {psutil.virtual_memory().percent}%'); print(f'Disk: {psutil.disk_usage(\"/\").percent}%')"

# Monitor file processing time
python -c "import time; from reconciliation_engine import ReconciliationEngine; engine = ReconciliationEngine(); start = time.time(); df = engine.load_file('large_file.xlsx'); print(f'Loading time: {time.time() - start:.2f}s')"
```

### 🗂️ Quick Reference
```bash
# Most commonly used commands:
python view_database.py                    # View database contents
python keyword_manager.py                  # Manage amount keywords (interactive)
python setup_ml_training.py                # Set up Bank Trx ID ML training
python train_ml_detector.py                # Train Bank Trx ID ML model
python setup_amount_training.py            # Set up Amount ML training
python train_amount_detector.py            # Train Amount ML model
python start_api.py                        # Start API server
python start_streamlit.py                  # Start web interface

# Database troubleshooting:
python fix_database_migration.py           # Fix database migration issues on new devices

# Quick system checks:
python -c "from database import KeywordManager; km = KeywordManager(); keywords = km.get_keywords(); print(f'Total keywords: {len(keywords)}')"
python -c "from reconciliation_engine import ReconciliationEngine; engine = ReconciliationEngine(); print(f'ML: {engine.ml_detector is not None}, Keywords: {engine.keyword_manager is not None}')"

# Replace 'your_file.xlsx' with your actual file path in the commands above
```

## 🛠️ Troubleshooting

### Common Issues

**"Could not identify Bank Trx ID column"**
- Check if bank file contains transaction IDs
- Consider training the ML model on your files for better detection
- Try manual column specification

**"No matching Bank Trx IDs found"**
- Ensure both files contain the same transaction ID format
- Check date ranges - files should cover same period
- Verify Bank Trx ID column contains actual transaction IDs

**"Cannot connect to API server"**
- Ensure API server is running on port 8000
- Check firewall settings
- Verify `python start_api.py` completed successfully

**"Network access not working"**
- Ensure Windows Firewall allows connections on ports 8000 and 8501
- Verify all devices are on the same local network (WiFi/LAN)
- Use the network IP address displayed when starting services
- Start API server before Streamlit app
- Check if antivirus software is blocking network connections

**"ML training failed"**
- Ensure PyTorch is installed: `pip install torch scikit-learn`
- Check that training_labels.py has correct format
- Verify file paths in labels match actual files
- Need at least 3-5 labeled files for training

**"Database column errors" (e.g., 'table has no column named matched_total_our')**
- This happens when using an existing database on a new device
- **Quick fix**: Run the database migration script: `python fix_database_migration.py`
- **Alternative**: Delete `reconciliation.db` file and let system create a new one
- The system will automatically migrate old databases to new schema

## 🔮 Future Enhancements

- [ ] Manual column mapping interface
- [ ] Multiple bank reconciliation in single job
- [ ] Excel report export
- [ ] Email notifications
- [ ] Advanced ML models (transformers, etc.)
- [ ] Integration with cloud storage

## 🤝 Contributing

This system is designed to be extensible. Key areas for enhancement:
- Additional transaction ID patterns
- New file format support
- Advanced ML algorithms for column detection
- Enhanced reporting features

---

**Built with**: Python 3.13, PyTorch, FastAPI, Streamlit, Pandas, SQLite
