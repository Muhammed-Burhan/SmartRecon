import sqlite3
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Text, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database setup
DATABASE_URL = "sqlite:///./reconciliation.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class ReconciliationJob(Base):
    __tablename__ = "reconciliation_jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    job_name = Column(String, index=True)
    our_file_name = Column(String)
    bank_file_name = Column(String)
    bank_name = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default="processing")
    total_our_records = Column(Integer)
    total_bank_records = Column(Integer)
    matched_records = Column(Integer)
    unmatched_our_records = Column(Integer)
    unmatched_bank_records = Column(Integer)
    
    # Amount comparison fields
    our_amount_column = Column(String)
    bank_amount_column = Column(String)
    our_amount_confidence = Column(Float)
    bank_amount_confidence = Column(Float)
    amount_comparison_performed = Column(Boolean, default=False)
    our_total_amount = Column(Float)
    bank_total_amount = Column(Float)
    matched_total_our = Column(Float)
    matched_total_bank = Column(Float)
    total_amount_difference = Column(Float)
    amounts_match = Column(Boolean)
    total_discrepancies = Column(Integer, default=0)

class ReconciliationResult(Base):
    __tablename__ = "reconciliation_results"
    
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, index=True)
    bank_trx_id = Column(String, index=True)
    status = Column(String)  # 'matched', 'missing_in_bank', 'missing_in_our_file'
    our_record_data = Column(Text)  # JSON string of our record
    bank_record_data = Column(Text)  # JSON string of bank record
    paying_bank_name = Column(String)
    amount = Column(Float)  # Legacy field for backward compatibility
    
    # Enhanced amount fields
    our_amount = Column(Float)
    bank_amount = Column(Float)
    amount_difference = Column(Float)
    amount_match = Column(Boolean)
    
    created_at = Column(DateTime, default=datetime.utcnow)

class AmountDiscrepancy(Base):
    __tablename__ = "amount_discrepancies"
    
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, index=True)
    bank_trx_id = Column(String, index=True)
    our_amount = Column(Float)
    bank_amount = Column(Float)
    difference = Column(Float)
    percentage_diff = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

class AmountKeyword(Base):
    __tablename__ = 'amount_keywords'
    
    id = Column(Integer, primary_key=True)
    keyword = Column(String(100), nullable=False)
    file_type = Column(String(20), nullable=False)  # 'our', 'bank', 'both'
    language = Column(String(10), default='en')  # 'en', 'ar', 'both'
    priority = Column(Integer, default=50)  # Higher = more important
    is_exact_match = Column(Boolean, default=False)  # True for exact matches
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(String(100), default='system')
    description = Column(String(255))

def migrate_database():
    """Migrate existing database to include amount comparison fields"""
    import sqlite3
    from sqlalchemy import text
    
    # Connect to database
    conn = sqlite3.connect('reconciliation.db')
    cursor = conn.cursor()
    
    try:
        # Check if new columns exist in reconciliation_jobs table
        cursor.execute("PRAGMA table_info(reconciliation_jobs)")
        columns = [column[1] for column in cursor.fetchall()]
        
        # Add amount comparison columns if they don't exist
        new_columns = [
            ('our_amount_column', 'TEXT'),
            ('bank_amount_column', 'TEXT'),
            ('our_amount_confidence', 'REAL'),
            ('bank_amount_confidence', 'REAL'),
            ('amount_comparison_performed', 'BOOLEAN DEFAULT FALSE'),
            ('our_total_amount', 'REAL'),
            ('bank_total_amount', 'REAL'),
            ('matched_total_our', 'REAL'),
            ('matched_total_bank', 'REAL'),
            ('total_amount_difference', 'REAL'),
            ('amounts_match', 'BOOLEAN'),
            ('total_discrepancies', 'INTEGER DEFAULT 0')
        ]
        
        for column_name, column_type in new_columns:
            if column_name not in columns:
                logger.info(f"Adding column {column_name} to reconciliation_jobs")
                cursor.execute(f"ALTER TABLE reconciliation_jobs ADD COLUMN {column_name} {column_type}")
        
        # Check if new columns exist in reconciliation_results table
        cursor.execute("PRAGMA table_info(reconciliation_results)")
        columns = [column[1] for column in cursor.fetchall()]
        
        # Add amount columns if they don't exist
        new_result_columns = [
            ('our_amount', 'REAL'),
            ('bank_amount', 'REAL'),
            ('amount_difference', 'REAL'),
            ('amount_match', 'BOOLEAN')
        ]
        
        for column_name, column_type in new_result_columns:
            if column_name not in columns:
                logger.info(f"Adding column {column_name} to reconciliation_results")
                cursor.execute(f"ALTER TABLE reconciliation_results ADD COLUMN {column_name} {column_type}")
        
        # Check if amount_discrepancies table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='amount_discrepancies'")
        if not cursor.fetchone():
            logger.info("📊 Creating amount_discrepancies table...")
            cursor.execute("""
                CREATE TABLE amount_discrepancies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    reconciliation_job_id INTEGER,
                    bank_trx_id TEXT,
                    our_amount REAL,
                    bank_amount REAL,
                    difference REAL,
                    percentage_diff REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (reconciliation_job_id) REFERENCES reconciliation_jobs (id)
                )
            """)
            cursor.execute("CREATE INDEX ix_amount_discrepancies_reconciliation_job_id ON amount_discrepancies (reconciliation_job_id)")
            cursor.execute("CREATE INDEX ix_amount_discrepancies_bank_trx_id ON amount_discrepancies (bank_trx_id)")
        
        # Check if amount_keywords table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='amount_keywords'")
        if not cursor.fetchone():
            logger.info("📊 Creating amount_keywords table...")
            cursor.execute("""
                CREATE TABLE amount_keywords (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    keyword TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    language TEXT DEFAULT 'en',
                    priority INTEGER DEFAULT 50,
                    is_exact_match BOOLEAN DEFAULT FALSE,
                    is_active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_by TEXT DEFAULT 'system',
                    description TEXT
                )
            """)
            cursor.execute("CREATE INDEX ix_amount_keywords_keyword ON amount_keywords (keyword)")
            cursor.execute("CREATE INDEX ix_amount_keywords_file_type ON amount_keywords (file_type)")
            cursor.execute("CREATE INDEX ix_amount_keywords_language ON amount_keywords (language)")
            cursor.execute("CREATE INDEX ix_amount_keywords_priority ON amount_keywords (priority)")
            cursor.execute("CREATE INDEX ix_amount_keywords_is_exact_match ON amount_keywords (is_exact_match)")
            cursor.execute("CREATE INDEX ix_amount_keywords_is_active ON amount_keywords (is_active)")
            cursor.execute("CREATE INDEX ix_amount_keywords_created_by ON amount_keywords (created_by)")
            
            # Populate initial keywords
            populate_initial_keywords(cursor)
        
        conn.commit()
        conn.close()
        
        logger.info("✅ Database migration completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Database migration failed: {e}")
        conn.rollback()
        conn.close()
        raise

def populate_initial_keywords(cursor):
    """Populate initial amount detection keywords based on user requirements"""
    logger.info("📝 Populating initial amount keywords...")
    
    # Define initial keywords based on user's patterns
    initial_keywords = [
        # User's internal file patterns (high priority)
        ('paid amt', 'our', 'en', 100, True, 'Primary amount column for internal files'),
        ('paidamt', 'our', 'en', 95, True, 'Alternative format for paid amount'),
        ('paid_amt', 'our', 'en', 95, True, 'Underscore format for paid amount'),
        ('payment amount', 'our', 'en', 90, False, 'Common payment amount column'),
        ('amount paid', 'our', 'en', 85, False, 'Alternative paid amount format'),
        
        # Bank file patterns (high priority)
        ('credit', 'bank', 'en', 100, True, 'Primary credit column for bank files'),
        ('credit دائن', 'bank', 'ar', 100, True, 'Arabic credit column for bank files'),
        ('payment amount', 'bank', 'en', 90, False, 'Payment amount in bank files'),
        ('credit amount', 'bank', 'en', 85, False, 'Credit amount column'),
        ('amount_credited', 'bank', 'en', 80, False, 'Amount credited column'),
        
        # General amount keywords (medium priority)
        ('amount', 'both', 'en', 75, False, 'General amount column'),
        ('amt', 'both', 'en', 70, False, 'Abbreviated amount'),
        ('value', 'both', 'en', 65, False, 'Value amount column'),
        ('sum', 'both', 'en', 60, False, 'Sum amount column'),
        ('total', 'both', 'en', 60, False, 'Total amount column'),
        ('money', 'both', 'en', 55, False, 'Money amount column'),
        ('cash', 'both', 'en', 55, False, 'Cash amount column'),
        ('balance', 'both', 'en', 50, False, 'Balance amount column'),
        
        # Debit patterns (lower priority)
        ('debit', 'both', 'en', 45, False, 'Debit amount column'),
        ('debit amount', 'both', 'en', 40, False, 'Debit amount column'),
        
        # Arabic patterns
        ('دائن', 'bank', 'ar', 90, False, 'Arabic credit term'),
        ('مدين', 'bank', 'ar', 45, False, 'Arabic debit term'),
        
        # Currency-specific patterns
        ('dollar amount', 'both', 'en', 55, False, 'Dollar amount column'),
        ('euro amount', 'both', 'en', 55, False, 'Euro amount column'),
        ('pound amount', 'both', 'en', 55, False, 'Pound amount column'),
        
        # Transaction-specific patterns
        ('transaction amount', 'both', 'en', 70, False, 'Transaction amount'),
        ('txn amount', 'both', 'en', 65, False, 'Transaction amount abbreviated'),
        ('trx amount', 'both', 'en', 65, False, 'Transaction amount abbreviated'),
        ('net amount', 'both', 'en', 60, False, 'Net amount column'),
        ('gross amount', 'both', 'en', 60, False, 'Gross amount column'),
    ]
    
    # Insert initial keywords
    for keyword, file_type, language, priority, is_exact, description in initial_keywords:
        cursor.execute("""
            INSERT INTO amount_keywords 
            (keyword, file_type, language, priority, is_exact_match, description, created_by)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (keyword, file_type, language, priority, is_exact, description, 'system_init'))
    
    logger.info(f"✅ Populated {len(initial_keywords)} initial amount keywords")

class KeywordManager:
    """Manage amount detection keywords with database storage"""
    
    def __init__(self, db_path='reconciliation.db'):
        self.db_path = db_path
        
    def get_keywords(self, file_type=None, language=None, active_only=True):
        """Get keywords from database with optional filtering"""
        import sqlite3
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT keyword, file_type, language, priority, is_exact_match, description FROM amount_keywords WHERE 1=1"
        params = []
        
        if file_type:
            query += " AND (file_type = ? OR file_type = 'both')"
            params.append(file_type)
        
        if language:
            query += " AND (language = ? OR language = 'both')"
            params.append(language)
        
        if active_only:
            query += " AND is_active = 1"
        
        query += " ORDER BY priority DESC, keyword"
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()
        
        return [{'keyword': row[0], 'file_type': row[1], 'language': row[2], 
                'priority': row[3], 'is_exact_match': row[4], 'description': row[5]} 
                for row in results]
    
    def add_keyword(self, keyword, file_type, language='en', priority=50, 
                   is_exact_match=False, description=None, created_by='user'):
        """Add a new keyword to the database"""
        import sqlite3
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO amount_keywords 
            (keyword, file_type, language, priority, is_exact_match, description, created_by)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (keyword.lower().strip(), file_type, language, priority, is_exact_match, description, created_by))
        
        conn.commit()
        conn.close()
        
        logger.info(f"✅ Added keyword: '{keyword}' (file_type: {file_type}, priority: {priority})")
        
    def update_keyword(self, keyword_id, **kwargs):
        """Update an existing keyword"""
        import sqlite3
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Build update query dynamically
        update_fields = []
        params = []
        
        for field, value in kwargs.items():
            if field in ['keyword', 'file_type', 'language', 'priority', 'is_exact_match', 'description', 'is_active']:
                update_fields.append(f"{field} = ?")
                params.append(value)
        
        if update_fields:
            params.append(keyword_id)
            query = f"UPDATE amount_keywords SET {', '.join(update_fields)}, updated_at = CURRENT_TIMESTAMP WHERE id = ?"
            cursor.execute(query, params)
            conn.commit()
        
        conn.close()
        
    def delete_keyword(self, keyword_id):
        """Delete a keyword (soft delete by setting is_active=False)"""
        import sqlite3
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("UPDATE amount_keywords SET is_active = 0, updated_at = CURRENT_TIMESTAMP WHERE id = ?", (keyword_id,))
        conn.commit()
        conn.close()
        
        logger.info(f"✅ Deactivated keyword ID: {keyword_id}")
        
    def search_keywords(self, search_term):
        """Search for keywords containing the search term"""
        import sqlite3
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, keyword, file_type, language, priority, is_exact_match, description 
            FROM amount_keywords 
            WHERE keyword LIKE ? AND is_active = 1
            ORDER BY priority DESC, keyword
        """, (f'%{search_term}%',))
        
        results = cursor.fetchall()
        conn.close()
        
        return [{'id': row[0], 'keyword': row[1], 'file_type': row[2], 'language': row[3], 
                'priority': row[4], 'is_exact_match': row[5], 'description': row[6]} 
                for row in results]

def create_tables():
    """Create database tables"""
    try:
        # First, run migration to handle existing databases
        migrate_database()
        
        # Then create any missing tables
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database tables created/verified successfully")
    except Exception as e:
        logger.error(f"❌ Error creating database tables: {e}")
        raise

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def reset_database():
    """Reset database for testing purposes - USE WITH CAUTION"""
    db_path = "./reconciliation.db"
    if os.path.exists(db_path):
        os.remove(db_path)
        logger.info("🗑️ Database reset - removed existing database")
    create_tables()
    logger.info("🆕 Database recreated with fresh schema")

# Initialize database
logger.info("🚀 Initializing SmartRecon database...")
create_tables() 