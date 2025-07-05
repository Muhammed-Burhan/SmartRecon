#!/usr/bin/env python3
"""
Database Migration Fix Script
Run this if you get database column errors on a new device
"""

import sqlite3
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def fix_database_migration():
    """Fix database migration issues by adding missing columns"""
    db_path = 'reconciliation.db'
    
    if not os.path.exists(db_path):
        logger.info("✅ No existing database found - migration not needed")
        return
    
    logger.info("🔧 Fixing database migration...")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Check existing columns in reconciliation_jobs table
        cursor.execute("PRAGMA table_info(reconciliation_jobs)")
        existing_columns = [column[1] for column in cursor.fetchall()]
        logger.info(f"📊 Found {len(existing_columns)} existing columns in reconciliation_jobs")
        
        # Complete list of required columns for reconciliation_jobs
        required_columns = [
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
        
        added_columns = 0
        for column_name, column_type in required_columns:
            if column_name not in existing_columns:
                logger.info(f"➕ Adding missing column: {column_name}")
                cursor.execute(f"ALTER TABLE reconciliation_jobs ADD COLUMN {column_name} {column_type}")
                added_columns += 1
        
        # Check existing columns in reconciliation_results table
        cursor.execute("PRAGMA table_info(reconciliation_results)")
        existing_result_columns = [column[1] for column in cursor.fetchall()]
        
        # Required columns for reconciliation_results
        required_result_columns = [
            ('our_amount', 'REAL'),
            ('bank_amount', 'REAL'),
            ('amount_difference', 'REAL'),
            ('amount_match', 'BOOLEAN')
        ]
        
        for column_name, column_type in required_result_columns:
            if column_name not in existing_result_columns:
                logger.info(f"➕ Adding missing column to results: {column_name}")
                cursor.execute(f"ALTER TABLE reconciliation_results ADD COLUMN {column_name} {column_type}")
                added_columns += 1
        
        # Check if amount_discrepancies table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='amount_discrepancies'")
        if not cursor.fetchone():
            logger.info("📊 Creating amount_discrepancies table...")
            cursor.execute("""
                CREATE TABLE amount_discrepancies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id INTEGER,
                    bank_trx_id TEXT,
                    our_amount REAL,
                    bank_amount REAL,
                    difference REAL,
                    percentage_diff REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (job_id) REFERENCES reconciliation_jobs (id)
                )
            """)
            cursor.execute("CREATE INDEX ix_amount_discrepancies_job_id ON amount_discrepancies (job_id)")
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
            
            # Add indexes
            cursor.execute("CREATE INDEX ix_amount_keywords_keyword ON amount_keywords (keyword)")
            cursor.execute("CREATE INDEX ix_amount_keywords_file_type ON amount_keywords (file_type)")
            cursor.execute("CREATE INDEX ix_amount_keywords_language ON amount_keywords (language)")
            cursor.execute("CREATE INDEX ix_amount_keywords_priority ON amount_keywords (priority)")
            cursor.execute("CREATE INDEX ix_amount_keywords_is_exact_match ON amount_keywords (is_exact_match)")
            cursor.execute("CREATE INDEX ix_amount_keywords_is_active ON amount_keywords (is_active)")
            cursor.execute("CREATE INDEX ix_amount_keywords_created_by ON amount_keywords (created_by)")
            
            # Populate initial keywords
            populate_keywords(cursor)
        
        conn.commit()
        
        if added_columns > 0:
            logger.info(f"✅ Database migration completed! Added {added_columns} missing columns")
        else:
            logger.info("✅ Database is already up to date!")
            
    except Exception as e:
        logger.error(f"❌ Database migration failed: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()

def populate_keywords(cursor):
    """Populate initial amount keywords"""
    logger.info("📝 Populating initial amount keywords...")
    
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
        
        # Arabic patterns
        ('دائن', 'bank', 'ar', 90, False, 'Arabic credit term'),
        ('مدين', 'bank', 'ar', 45, False, 'Arabic debit term'),
        
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

def main():
    """Main function"""
    print("🔧 SmartRecon Database Migration Fix")
    print("=" * 50)
    
    try:
        fix_database_migration()
        print("\n🎉 Database migration completed successfully!")
        print("You can now run the SmartRecon system normally.")
        
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        print("\n💡 Solutions:")
        print("1. Delete the existing reconciliation.db file and let the system create a new one")
        print("2. Contact support with the error details")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 