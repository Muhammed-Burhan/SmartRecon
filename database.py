import sqlite3
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

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

class ReconciliationResult(Base):
    __tablename__ = "reconciliation_results"
    
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, index=True)
    bank_trx_id = Column(String, index=True)
    status = Column(String)  # 'matched', 'missing_in_bank', 'missing_in_our_file'
    our_record_data = Column(Text)  # JSON string of our record
    bank_record_data = Column(Text)  # JSON string of bank record
    paying_bank_name = Column(String)
    amount = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

def create_tables():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Initialize database
create_tables() 