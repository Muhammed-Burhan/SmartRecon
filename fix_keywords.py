#!/usr/bin/env python3
"""
Fix keywords in the database
"""

import sqlite3
from database import populate_initial_keywords

def main():
    print("🔧 Fixing keywords in database...")
    
    # Connect to database
    conn = sqlite3.connect('reconciliation.db')
    cursor = conn.cursor()
    
    # Check current keyword count
    cursor.execute('SELECT COUNT(*) FROM amount_keywords')
    count = cursor.fetchone()[0]
    print(f"Current keyword count: {count}")
    
    if count == 0:
        print("📝 Populating initial keywords...")
        populate_initial_keywords(cursor)
        conn.commit()
        
        # Check new count
        cursor.execute('SELECT COUNT(*) FROM amount_keywords')
        new_count = cursor.fetchone()[0]
        print(f"✅ Added {new_count} keywords")
        
        # Show some samples
        cursor.execute('SELECT keyword, file_type, priority FROM amount_keywords ORDER BY priority DESC LIMIT 10')
        samples = cursor.fetchall()
        print("\n📊 Sample keywords:")
        for keyword, file_type, priority in samples:
            print(f"  - {keyword} ({file_type}, priority: {priority})")
    else:
        print("✅ Keywords already exist in database")
    
    conn.close()

if __name__ == "__main__":
    main() 