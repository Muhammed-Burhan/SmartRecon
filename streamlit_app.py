import streamlit as st
import requests
import pandas as pd
import json
from datetime import datetime
import time

# Configure the page
st.set_page_config(
    page_title="Bank Reconciliation System",
    page_icon="🏦",
    layout="wide"
)

# API base URL
API_BASE_URL = "http://localhost:8000"

def main():
    st.title("🏦 Bank Reconciliation System")
    st.markdown("---")
    
    # Initialize session state for page navigation
    if 'page' not in st.session_state:
        st.session_state.page = "Upload Files"
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Upload Files", "View Jobs", "Job Details"], 
                           index=["Upload Files", "View Jobs", "Job Details"].index(st.session_state.page))
    
    # Update session state when radio button changes
    if page != st.session_state.page:
        st.session_state.page = page
    
    if st.session_state.page == "Upload Files":
        upload_files_page()
    elif st.session_state.page == "View Jobs":
        view_jobs_page()
    elif st.session_state.page == "Job Details":
        job_details_page()

def upload_files_page():
    st.header("📁 Upload Reconciliation Files")
    
    # File upload section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Our Reconciliation File")
        our_file = st.file_uploader(
            "Upload your reconciliation file (Excel/CSV)",
            type=['xlsx', 'xls', 'csv'],
            key="our_file"
        )
        
        if our_file:
            st.success(f"✅ File uploaded: {our_file.name}")
            # Show file preview
            try:
                if our_file.name.endswith('.csv'):
                    df = pd.read_csv(our_file)
                else:
                    df = pd.read_excel(our_file)
                
                st.write("**File Preview:**")
                st.dataframe(df.head(3))
                st.write(f"**Rows:** {len(df)}, **Columns:** {len(df.columns)}")
                
                # Reset file pointer
                our_file.seek(0)
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    
    with col2:
        st.subheader("Bank/PSP File")
        bank_file = st.file_uploader(
            "Upload bank/PSP file (Excel/CSV)",
            type=['xlsx', 'xls', 'csv'],
            key="bank_file"
        )
        
        if bank_file:
            st.success(f"✅ File uploaded: {bank_file.name}")
            # Show file preview
            try:
                if bank_file.name.endswith('.csv'):
                    df = pd.read_csv(bank_file)
                else:
                    df = pd.read_excel(bank_file)
                
                st.write("**File Preview:**")
                st.dataframe(df.head(3))
                st.write(f"**Rows:** {len(df)}, **Columns:** {len(df.columns)}")
                
                # Show column names to help identify Bank Trx ID
                st.write("**Columns:** ", ", ".join(df.columns.tolist()))
                
                # Reset file pointer
                bank_file.seek(0)
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    
    # Job configuration
    st.subheader("Job Configuration")
    job_name = st.text_input("Job Name", value=f"Reconciliation_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # Process button
    if st.button("🚀 Start Reconciliation", type="primary"):
        if our_file and bank_file and job_name:
            process_reconciliation(our_file, bank_file, job_name)
        else:
            st.error("Please upload both files and provide a job name.")

def process_reconciliation(our_file, bank_file, job_name):
    """Process the reconciliation via API"""
    
    with st.spinner("Processing reconciliation... This may take a few minutes for large files."):
        try:
            # Prepare files for upload
            files = {
                'our_file': (our_file.name, our_file.getvalue(), our_file.type),
                'bank_file': (bank_file.name, bank_file.getvalue(), bank_file.type)
            }
            
            data = {'job_name': job_name}
            
            # Make API call
            response = requests.post(f"{API_BASE_URL}/upload-files/", files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                
                st.success("✅ Reconciliation completed successfully!")
                
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Identified Bank", result['bank_name'])
                    st.metric("Bank Trx Column", result['bank_trx_column'])
                    st.metric("Detection Confidence", f"{result['confidence']:.1f}%")
                
                with col2:
                    summary = result['summary']
                    st.metric("Total Our Records", summary['total_our_records'])
                    st.metric("Total Bank Records", summary['total_bank_records'])
                    st.metric("Matched Records", summary['matched_records'])
                
                with col3:
                    st.metric("Missing in Bank", summary['missing_in_bank_count'])
                    st.metric("Missing in Our File", summary['missing_in_our_file_count'])
                    st.metric("Match Percentage", f"{summary['match_percentage']:.1f}%")
                
                # Display bank matches
                if result['bank_matches']:
                    st.subheader("Bank Identification Results")
                    for bank, trx_ids in result['bank_matches'].items():
                        st.write(f"**{bank}:** {len(trx_ids)} matching transactions")
                
                # Display report
                st.subheader("Detailed Report")
                st.text(result['report'])
                
                # Save job ID in session state for easy access
                st.session_state.last_job_id = result['job_id']
                
            else:
                st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
                
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to the API server. Please make sure the FastAPI server is running on port 8000.")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

def view_jobs_page():
    st.header("📊 Reconciliation Jobs")
    
    try:
        response = requests.get(f"{API_BASE_URL}/jobs/")
        if response.status_code == 200:
            jobs = response.json()
            
            if jobs:
                # Create a dataframe for better display
                jobs_df = pd.DataFrame([
                    {
                        'Job ID': job['id'],
                        'Job Name': job['job_name'],
                        'Bank Name': job['bank_name'],
                        'Our File': job['our_file_name'],
                        'Bank File': job['bank_file_name'],
                        'Status': job['status'],
                        'Created': job['created_at'][:19].replace('T', ' '),
                        'Matched': job['matched_records'],
                        'Our Records': job['total_our_records'],
                        'Bank Records': job['total_bank_records'],
                        'Match %': f"{(job['matched_records'] / max(job['total_our_records'], 1) * 100):.1f}%"
                    }
                    for job in jobs
                ])
                
                st.dataframe(jobs_df, use_container_width=True)
                
                # Job selection for details
                st.subheader("Select Job for Details")
                selected_job_id = st.selectbox(
                    "Choose a job:",
                    options=[job['id'] for job in jobs],
                    format_func=lambda x: f"Job {x}: {next(job['job_name'] for job in jobs if job['id'] == x)}"
                )
                
                if st.button("View Job Details"):
                    st.session_state.selected_job_id = selected_job_id
                    st.session_state.page = "Job Details"
                    st.rerun()
                    
            else:
                st.info("No reconciliation jobs found. Upload some files to get started!")
                
        else:
            st.error("Failed to fetch jobs from API")
            
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to the API server. Please make sure the FastAPI server is running on port 8000.")
    except Exception as e:
        st.error(f"Error: {str(e)}")

def job_details_page():
    st.header("🔍 Job Details")
    
    # Get job ID from session state or input
    job_id = st.session_state.get('selected_job_id') or st.session_state.get('last_job_id')
    
    if not job_id:
        job_id = st.number_input("Enter Job ID:", min_value=1, step=1)
    
    if job_id:
        try:
            # Get job details
            response = requests.get(f"{API_BASE_URL}/jobs/{job_id}/")
            if response.status_code == 200:
                data = response.json()
                job = data['job']
                results = data['results']
                
                # Job summary
                st.subheader("Job Summary")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Job Name:** {job['job_name']}")
                    st.write(f"**Bank Name:** {job['bank_name']}")
                    st.write(f"**Status:** {job['status']}")
                    st.write(f"**Created:** {job['created_at'][:19].replace('T', ' ')}")
                
                with col2:
                    st.write(f"**Our File:** {job['our_file_name']}")
                    st.write(f"**Bank File:** {job['bank_file_name']}")
                    st.write(f"**Total Records (Ours):** {job['total_our_records']}")
                    st.write(f"**Total Records (Bank):** {job['total_bank_records']}")
                
                # Statistics
                st.subheader("Statistics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Matched", job['matched_records'])
                with col2:
                    st.metric("Missing in Bank", job['unmatched_our_records'])
                with col3:
                    st.metric("Missing in Our File", job['unmatched_bank_records'])
                with col4:
                    match_pct = (job['matched_records'] / max(job['total_our_records'], 1)) * 100
                    st.metric("Match %", f"{match_pct:.1f}%")
                
                # Detailed results
                st.subheader("Detailed Results")
                
                # Filter options
                status_filter = st.selectbox(
                    "Filter by status:",
                    ["All", "matched", "missing_in_bank", "missing_in_our_file"]
                )
                
                # Filter results
                if status_filter != "All":
                    filtered_results = [r for r in results if r['status'] == status_filter]
                else:
                    filtered_results = results
                
                if filtered_results:
                    # Create display dataframe
                    display_data = []
                    for result in filtered_results[:100]:  # Limit to first 100 for performance
                        row = {
                            'Bank Trx ID': result['bank_trx_id'],
                            'Status': result['status'],
                            'Bank Name': result['paying_bank_name'],
                            'Amount': result['amount'],
                            'Created': result['created_at'][:19].replace('T', ' ')
                        }
                        display_data.append(row)
                    
                    df = pd.DataFrame(display_data)
                    st.dataframe(df, use_container_width=True)
                    
                    if len(filtered_results) > 100:
                        st.info(f"Showing first 100 out of {len(filtered_results)} results")
                
                # Generate report
                if st.button("📋 Generate Full Report"):
                    report_response = requests.get(f"{API_BASE_URL}/jobs/{job_id}/report/")
                    if report_response.status_code == 200:
                        report_data = report_response.json()
                        st.subheader("Full Report")
                        st.text(report_data['report'])
                    else:
                        st.error("Failed to generate report")
                        
            elif response.status_code == 404:
                st.error("Job not found")
            else:
                st.error("Failed to fetch job details")
                
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to the API server. Please make sure the FastAPI server is running on port 8000.")
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 