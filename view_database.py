#!/usr/bin/env python3
"""
Database viewer for the Bank Reconciliation System
"""

import sqlite3
import pandas as pd
from pathlib import Path

def connect_to_database():
    """Connect to the SQLite database"""
    db_path = "reconciliation.db"
    
    if not Path(db_path).exists():
        print(f"❌ Database file '{db_path}' not found!")
        print("Run the system first to create the database.")
        return None
    
    return sqlite3.connect(db_path)

def show_tables(conn):
    """Show all tables in the database"""
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    print("📊 Available Tables:")
    for table in tables:
        print(f"  - {table[0]}")
    print()

def show_jobs(conn):
    """Show all reconciliation jobs"""
    df = pd.read_sql_query("SELECT * FROM reconciliation_jobs ORDER BY created_at DESC", conn)
    
    if df.empty:
        print("No reconciliation jobs found.")
        return
    
    print("🏦 Reconciliation Jobs:")
    print(df.to_string(index=False))
    print()

def show_results(conn, job_id=None):
    """Show reconciliation results"""
    if job_id:
        query = f"SELECT * FROM reconciliation_results WHERE job_id = {job_id} ORDER BY created_at DESC"
        print(f"📋 Results for Job {job_id}:")
    else:
        query = "SELECT * FROM reconciliation_results ORDER BY created_at DESC LIMIT 50"
        print("📋 Recent Reconciliation Results (Last 50):")
    
    df = pd.read_sql_query(query, conn)
    
    if df.empty:
        print("No results found.")
        return
    
    # Show summary columns
    summary_df = df[['job_id', 'bank_trx_id', 'status', 'paying_bank_name', 'amount', 'created_at']]
    print(summary_df.to_string(index=False))
    print()

def export_to_excel(conn, filename="reconciliation_export.xlsx"):
    """Export database content to Excel"""
    try:
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Export jobs
            jobs_df = pd.read_sql_query("SELECT * FROM reconciliation_jobs ORDER BY created_at DESC", conn)
            jobs_df.to_excel(writer, sheet_name='Jobs', index=False)
            
            # Export results
            results_df = pd.read_sql_query("SELECT * FROM reconciliation_results ORDER BY created_at DESC", conn)
            results_df.to_excel(writer, sheet_name='Results', index=False)
            
        print(f"✅ Data exported to '{filename}'")
    except Exception as e:
        print(f"❌ Export failed: {str(e)}")

def main():
    print("🏦 Bank Reconciliation Database Viewer")
    print("=" * 50)
    
    conn = connect_to_database()
    if not conn:
        return
    
    try:
        while True:
            print("\nOptions:")
            print("1. Show tables")
            print("2. Show all jobs")
            print("3. Show results for specific job")
            print("4. Show recent results")
            print("5. Export to Excel")
            print("6. Exit")
            
            choice = input("\nEnter your choice (1-6): ").strip()
            
            if choice == '1':
                show_tables(conn)
            elif choice == '2':
                show_jobs(conn)
            elif choice == '3':
                job_id = input("Enter Job ID: ").strip()
                if job_id.isdigit():
                    show_results(conn, int(job_id))
                else:
                    print("Invalid Job ID")
            elif choice == '4':
                show_results(conn)
            elif choice == '5':
                filename = input("Enter filename (default: reconciliation_export.xlsx): ").strip()
                if not filename:
                    filename = "reconciliation_export.xlsx"
                export_to_excel(conn, filename)
            elif choice == '6':
                break
            else:
                print("Invalid choice")
                
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    finally:
        conn.close()

if __name__ == "__main__":
    main() 