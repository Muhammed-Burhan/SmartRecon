#!/usr/bin/env python3
"""
SmartRecon Keyword Manager
Interactive tool for managing amount detection keywords
"""

import os
import sys
from typing import List, Dict
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main interactive keyword management interface"""
    print("💰 SmartRecon Keyword Manager")
    print("=" * 50)
    
    # Import database components
    try:
        from database import KeywordManager, migrate_database
        
        # Ensure database is migrated
        migrate_database()
        
        # Initialize keyword manager
        km = KeywordManager()
        
    except Exception as e:
        print(f"❌ Error initializing keyword manager: {e}")
        return
    
    while True:
        print("\n📋 Available Commands:")
        print("1. View keywords")
        print("2. Add keyword")
        print("3. Search keywords")
        print("4. Update keyword")
        print("5. Delete keyword")
        print("6. Import keywords from file")
        print("7. Export keywords to file")
        print("8. Test keyword detection")
        print("9. Exit")
        
        choice = input("\n🔢 Enter your choice (1-9): ").strip()
        
        if choice == '1':
            view_keywords(km)
        elif choice == '2':
            add_keyword(km)
        elif choice == '3':
            search_keywords(km)
        elif choice == '4':
            update_keyword(km)
        elif choice == '5':
            delete_keyword(km)
        elif choice == '6':
            import_keywords(km)
        elif choice == '7':
            export_keywords(km)
        elif choice == '8':
            test_detection(km)
        elif choice == '9':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please try again.")

def view_keywords(km):
    """View all keywords"""
    print("\n📊 Current Keywords:")
    print("-" * 80)
    
    file_type = input("Filter by file type (our/bank/both or Enter for all): ").strip()
    if not file_type:
        file_type = None
    
    keywords = km.get_keywords(file_type=file_type)
    
    if not keywords:
        print("❌ No keywords found.")
        return
    
    print(f"{'ID':<4} {'Keyword':<20} {'Type':<8} {'Lang':<6} {'Priority':<8} {'Exact':<6} {'Description':<25}")
    print("-" * 80)
    
    # Get IDs for display
    search_results = km.search_keywords('')  # Get all with IDs
    keyword_ids = {k['keyword']: k['id'] for k in search_results}
    
    for kw in keywords:
        keyword_id = keyword_ids.get(kw['keyword'], 'N/A')
        print(f"{keyword_id:<4} {kw['keyword']:<20} {kw['file_type']:<8} {kw['language']:<6} "
              f"{kw['priority']:<8} {kw['is_exact_match']:<6} {kw['description']:<25}")

def add_keyword(km):
    """Add a new keyword"""
    print("\n➕ Add New Keyword")
    print("-" * 30)
    
    keyword = input("💰 Enter keyword: ").strip()
    if not keyword:
        print("❌ Keyword cannot be empty.")
        return
    
    print("📂 File type options:")
    print("  our  - For your internal reconciliation files")
    print("  bank - For bank/PSP files")
    print("  both - For both file types")
    
    file_type = input("📂 File type (our/bank/both): ").strip().lower()
    if file_type not in ['our', 'bank', 'both']:
        print("❌ Invalid file type.")
        return
    
    language = input("🌐 Language (en/ar/both) [default: en]: ").strip().lower()
    if not language:
        language = 'en'
    
    priority_str = input("⭐ Priority (1-100) [default: 50]: ").strip()
    try:
        priority = int(priority_str) if priority_str else 50
        if priority < 1 or priority > 100:
            print("❌ Priority must be between 1 and 100.")
            return
    except ValueError:
        print("❌ Priority must be a number.")
        return
    
    is_exact = input("🎯 Exact match only? (y/n) [default: n]: ").strip().lower()
    is_exact_match = is_exact in ['y', 'yes', 'true']
    
    description = input("📝 Description (optional): ").strip()
    
    try:
        km.add_keyword(keyword, file_type, language, priority, is_exact_match, description)
        print(f"✅ Added keyword: '{keyword}'")
    except Exception as e:
        print(f"❌ Error adding keyword: {e}")

def search_keywords(km):
    """Search for keywords"""
    print("\n🔍 Search Keywords")
    print("-" * 30)
    
    search_term = input("🔍 Enter search term: ").strip()
    if not search_term:
        print("❌ Search term cannot be empty.")
        return
    
    results = km.search_keywords(search_term)
    
    if not results:
        print(f"❌ No keywords found containing '{search_term}'.")
        return
    
    print(f"\n📊 Found {len(results)} keywords:")
    print(f"{'ID':<4} {'Keyword':<20} {'Type':<8} {'Lang':<6} {'Priority':<8} {'Exact':<6} {'Description':<25}")
    print("-" * 80)
    
    for kw in results:
        print(f"{kw['id']:<4} {kw['keyword']:<20} {kw['file_type']:<8} {kw['language']:<6} "
              f"{kw['priority']:<8} {kw['is_exact_match']:<6} {kw['description']:<25}")

def update_keyword(km: 'KeywordManager'):
    """Update an existing keyword"""
    print("\n✏️ Update Keyword")
    print("-" * 30)
    
    keyword_id_str = input("🔢 Enter keyword ID to update: ").strip()
    try:
        keyword_id = int(keyword_id_str)
    except ValueError:
        print("❌ Invalid keyword ID.")
        return
    
    # Get current keyword details
    results = km.search_keywords('')  # Get all
    current_keyword = None
    for kw in results:
        if kw['id'] == keyword_id:
            current_keyword = kw
            break
    
    if not current_keyword:
        print(f"❌ Keyword with ID {keyword_id} not found.")
        return
    
    print(f"📊 Current keyword: {current_keyword['keyword']}")
    print("💡 Leave blank to keep current value")
    
    # Get updates
    updates = {}
    
    new_keyword = input(f"💰 Keyword [{current_keyword['keyword']}]: ").strip()
    if new_keyword:
        updates['keyword'] = new_keyword
    
    new_file_type = input(f"📂 File type [{current_keyword['file_type']}]: ").strip()
    if new_file_type and new_file_type in ['our', 'bank', 'both']:
        updates['file_type'] = new_file_type
    
    new_priority = input(f"⭐ Priority [{current_keyword['priority']}]: ").strip()
    if new_priority:
        try:
            priority = int(new_priority)
            if 1 <= priority <= 100:
                updates['priority'] = priority
        except ValueError:
            pass
    
    new_exact = input(f"🎯 Exact match [{current_keyword['is_exact_match']}] (y/n): ").strip().lower()
    if new_exact in ['y', 'yes', 'true']:
        updates['is_exact_match'] = True
    elif new_exact in ['n', 'no', 'false']:
        updates['is_exact_match'] = False
    
    new_description = input(f"📝 Description [{current_keyword['description']}]: ").strip()
    if new_description:
        updates['description'] = new_description
    
    if updates:
        try:
            km.update_keyword(keyword_id, **updates)
            print(f"✅ Updated keyword ID {keyword_id}")
        except Exception as e:
            print(f"❌ Error updating keyword: {e}")
    else:
        print("ℹ️ No changes made.")

def delete_keyword(km: 'KeywordManager'):
    """Delete a keyword"""
    print("\n🗑️ Delete Keyword")
    print("-" * 30)
    
    keyword_id_str = input("🔢 Enter keyword ID to delete: ").strip()
    try:
        keyword_id = int(keyword_id_str)
    except ValueError:
        print("❌ Invalid keyword ID.")
        return
    
    confirm = input(f"⚠️ Are you sure you want to delete keyword ID {keyword_id}? (y/n): ").strip().lower()
    if confirm in ['y', 'yes']:
        try:
            km.delete_keyword(keyword_id)
            print(f"✅ Deleted keyword ID {keyword_id}")
        except Exception as e:
            print(f"❌ Error deleting keyword: {e}")
    else:
        print("ℹ️ Deletion cancelled.")

def import_keywords(km: 'KeywordManager'):
    """Import keywords from a file"""
    print("\n📥 Import Keywords")
    print("-" * 30)
    
    file_path = input("📁 Enter path to keyword file: ").strip()
    if not os.path.exists(file_path):
        print("❌ File not found.")
        return
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        imported = 0
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            try:
                # Format: keyword,file_type,language,priority,is_exact,description
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 2:
                    keyword = parts[0]
                    file_type = parts[1]
                    language = parts[2] if len(parts) > 2 else 'en'
                    priority = int(parts[3]) if len(parts) > 3 else 50
                    is_exact = parts[4].lower() in ['true', '1', 'yes'] if len(parts) > 4 else False
                    description = parts[5] if len(parts) > 5 else ''
                    
                    km.add_keyword(keyword, file_type, language, priority, is_exact, description, 'import')
                    imported += 1
                    
            except Exception as e:
                print(f"⚠️ Error on line {line_num}: {e}")
        
        print(f"✅ Imported {imported} keywords")
        
    except Exception as e:
        print(f"❌ Error importing keywords: {e}")

def export_keywords(km: 'KeywordManager'):
    """Export keywords to a file"""
    print("\n📤 Export Keywords")
    print("-" * 30)
    
    file_path = input("📁 Enter export file path: ").strip()
    if not file_path:
        print("❌ File path cannot be empty.")
        return
    
    try:
        keywords = km.get_keywords()
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("# SmartRecon Keyword Export\n")
            f.write("# Format: keyword,file_type,language,priority,is_exact,description\n")
            f.write("#\n")
            
            for kw in keywords:
                f.write(f"{kw['keyword']},{kw['file_type']},{kw['language']},{kw['priority']},{kw['is_exact_match']},{kw['description']}\n")
        
        print(f"✅ Exported {len(keywords)} keywords to {file_path}")
        
    except Exception as e:
        print(f"❌ Error exporting keywords: {e}")

def test_detection(km: 'KeywordManager'):
    """Test keyword detection on sample data"""
    print("\n🧪 Test Keyword Detection")
    print("-" * 30)
    
    # Create sample data
    sample_columns = [
        'Paid Amt', 'Credit', 'Credit دائن', 'Amount', 'Total Amount',
        'Customer Name', 'Bank Trx ID', 'Date', 'Reference'
    ]
    
    print("📊 Sample columns:")
    for i, col in enumerate(sample_columns, 1):
        print(f"  {i}. {col}")
    
    file_type = input("\n📂 Test for file type (our/bank): ").strip().lower()
    if file_type not in ['our', 'bank']:
        print("❌ Invalid file type.")
        return
    
    # Get keywords for this file type
    keywords = km.get_keywords(file_type=file_type)
    
    print(f"\n🔍 Testing with {len(keywords)} keywords...")
    
    # Score each column
    for col in sample_columns:
        score = 0
        matches = []
        
        col_lower = col.lower().strip()
        col_clean = col_lower.replace(' ', '').replace('_', '').replace('-', '')
        
        for keyword_data in keywords:
            keyword = keyword_data['keyword']
            priority = keyword_data['priority']
            is_exact = keyword_data['is_exact_match']
            
            keyword_clean = keyword.replace(' ', '').replace('_', '').replace('-', '')
            
            if is_exact:
                if keyword_clean == col_clean or keyword == col_lower:
                    score += priority
                    matches.append(f"Exact: '{keyword}' (+{priority})")
                elif keyword in col_lower:
                    match_score = priority * 0.8
                    score += match_score
                    matches.append(f"Partial: '{keyword}' (+{match_score:.1f})")
            else:
                if keyword_clean in col_clean or keyword in col_lower:
                    match_score = priority * 0.7
                    score += match_score
                    matches.append(f"Flexible: '{keyword}' (+{match_score:.1f})")
        
        print(f"\n📊 Column: '{col}' - Score: {score:.1f}")
        for match in matches:
            print(f"    {match}")
    
    print(f"\n✅ Detection test completed for {file_type} files")

if __name__ == "__main__":
    main() 