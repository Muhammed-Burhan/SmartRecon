from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, Form
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
import tempfile
import os
import json
from typing import List, Optional
from datetime import datetime

from database import get_db, ReconciliationJob, ReconciliationResult
from reconciliation_engine import ReconciliationEngine

app = FastAPI(title="Bank Reconciliation System", version="1.0.0")

# Initialize reconciliation engine
engine = ReconciliationEngine()

@app.post("/upload-files/")
async def upload_files(
    our_file: UploadFile = File(...),
    bank_file: UploadFile = File(...),
    job_name: str = Form(...),
    db: Session = Depends(get_db)
):
    """
    Upload both reconciliation files and start processing
    """
    try:
        # Create temporary files
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(our_file.filename)[1]) as temp_our:
            temp_our.write(await our_file.read())
            our_file_path = temp_our.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(bank_file.filename)[1]) as temp_bank:
            temp_bank.write(await bank_file.read())
            bank_file_path = temp_bank.name
        
        # Load files
        our_df = engine.load_file(our_file_path)
        bank_df = engine.load_file(bank_file_path)
        
        # Find Bank Trx ID column in bank file
        bank_trx_column, confidence = engine.find_bank_trx_column(bank_df)
        
        if confidence < 20:  # Minimum confidence threshold
            raise HTTPException(
                status_code=400,
                detail=f"Could not identify Bank Trx ID column with sufficient confidence. Best guess: '{bank_trx_column}' (confidence: {confidence:.1f})"
            )
        
        # Extract Bank Trx IDs from bank file
        bank_trx_ids = engine.extract_bank_trx_ids(bank_df, bank_trx_column)
        
        # Identify which bank(s) these transactions belong to
        bank_matches = engine.identify_bank_from_trx_ids(our_df, bank_trx_ids)
        
        if not bank_matches:
            raise HTTPException(
                status_code=400,
                detail="No matching Bank Trx IDs found between files. Please check your data."
            )
        
        # For now, process the bank with the most matches
        target_bank = max(bank_matches.keys(), key=lambda k: len(bank_matches[k]))
        
        # Perform reconciliation
        results = engine.perform_reconciliation(our_df, bank_df, bank_trx_column, target_bank)
        
        # Save job to database
        job = ReconciliationJob(
            job_name=job_name,
            our_file_name=our_file.filename,
            bank_file_name=bank_file.filename,
            bank_name=target_bank,
            status="completed",
            total_our_records=results['summary']['total_our_records'],
            total_bank_records=results['summary']['total_bank_records'],
            matched_records=results['summary']['matched_records'],
            unmatched_our_records=results['summary']['missing_in_bank_count'],
            unmatched_bank_records=results['summary']['missing_in_our_file_count']
        )
        
        db.add(job)
        db.commit()
        db.refresh(job)
        
        # Save detailed results
        for item in results['matched']:
            try:
                paid_amt = item['our_record'].get('Paid Amt', 0)
                amount = float(paid_amt) if paid_amt is not None and str(paid_amt).replace('.','').replace('-','').isdigit() else 0.0
            except (ValueError, TypeError):
                amount = 0.0
                
            result = ReconciliationResult(
                job_id=job.id,
                bank_trx_id=item['bank_trx_id'],
                status='matched',
                our_record_data=json.dumps(item['our_record']),
                bank_record_data=json.dumps(item['bank_record']),
                paying_bank_name=target_bank,
                amount=amount
            )
            db.add(result)
        
        for item in results['missing_in_bank']:
            try:
                paid_amt = item['our_record'].get('Paid Amt', 0)
                amount = float(paid_amt) if paid_amt is not None and str(paid_amt).replace('.','').replace('-','').isdigit() else 0.0
            except (ValueError, TypeError):
                amount = 0.0
                
            result = ReconciliationResult(
                job_id=job.id,
                bank_trx_id=item['bank_trx_id'],
                status='missing_in_bank',
                our_record_data=json.dumps(item['our_record']),
                bank_record_data=None,
                paying_bank_name=target_bank,
                amount=amount
            )
            db.add(result)
        
        for item in results['missing_in_our_file']:
            result = ReconciliationResult(
                job_id=job.id,
                bank_trx_id=item['bank_trx_id'],
                status='missing_in_our_file',
                our_record_data=None,
                bank_record_data=json.dumps(item['bank_record']),
                paying_bank_name=target_bank,
                amount=0
            )
            db.add(result)
        
        db.commit()
        
        # Generate report
        report = engine.generate_report(results, target_bank)
        
        # Cleanup temporary files
        os.unlink(our_file_path)
        os.unlink(bank_file_path)
        
        return {
            "job_id": job.id,
            "status": "success",
            "bank_name": target_bank,
            "bank_trx_column": bank_trx_column,
            "confidence": confidence,
            "summary": results['summary'],
            "report": report,
            "bank_matches": bank_matches
        }
        
    except Exception as e:
        # Cleanup temporary files if they exist
        try:
            if 'our_file_path' in locals():
                os.unlink(our_file_path)
            if 'bank_file_path' in locals():
                os.unlink(bank_file_path)
        except:
            pass
        
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/jobs/")
async def get_jobs(db: Session = Depends(get_db)):
    """Get all reconciliation jobs"""
    jobs = db.query(ReconciliationJob).order_by(ReconciliationJob.created_at.desc()).all()
    return jobs

@app.get("/jobs/{job_id}/")
async def get_job_details(job_id: int, db: Session = Depends(get_db)):
    """Get detailed results for a specific job"""
    job = db.query(ReconciliationJob).filter(ReconciliationJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    results = db.query(ReconciliationResult).filter(ReconciliationResult.job_id == job_id).all()
    
    return {
        "job": job,
        "results": results
    }

@app.get("/jobs/{job_id}/report/")
async def get_job_report(job_id: int, db: Session = Depends(get_db)):
    """Generate and return a reconciliation report for a job"""
    job = db.query(ReconciliationJob).filter(ReconciliationJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    results = db.query(ReconciliationResult).filter(ReconciliationResult.job_id == job_id).all()
    
    # Reconstruct results dictionary for report generation
    results_dict = {
        'matched': [],
        'missing_in_bank': [],
        'missing_in_our_file': [],
        'summary': {
            'total_our_records': job.total_our_records,
            'total_bank_records': job.total_bank_records,
            'matched_records': job.matched_records,
            'missing_in_bank_count': job.unmatched_our_records,
            'missing_in_our_file_count': job.unmatched_bank_records,
            'match_percentage': (job.matched_records / max(job.total_our_records, 1)) * 100
        }
    }
    
    for result in results:
        item = {'bank_trx_id': result.bank_trx_id}
        if result.status == 'matched':
            item['our_record'] = json.loads(result.our_record_data)
            item['bank_record'] = json.loads(result.bank_record_data)
            results_dict['matched'].append(item)
        elif result.status == 'missing_in_bank':
            item['our_record'] = json.loads(result.our_record_data)
            results_dict['missing_in_bank'].append(item)
        elif result.status == 'missing_in_our_file':
            item['bank_record'] = json.loads(result.bank_record_data)
            results_dict['missing_in_our_file'].append(item)
    
    report = engine.generate_report(results_dict, job.bank_name)
    
    return {"report": report}

@app.get("/")
async def root():
    return {"message": "Bank Reconciliation System API", "version": "1.0.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 