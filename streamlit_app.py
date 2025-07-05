import streamlit as st
import requests
import pandas as pd
import json
from datetime import datetime
import time
import socket
import os

# Configure the page
st.set_page_config(
    page_title="Bank Reconciliation System",
    page_icon="🏦",
    layout="wide"
)

def get_local_ip():
    """Get the local IP address of this machine"""
    try:
        # Connect to a remote address to determine local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"

# API base URL - use environment variable or auto-detect IP
API_HOST = os.getenv('API_HOST', get_local_ip())
API_PORT = os.getenv('API_PORT', '8000')
API_BASE_URL = f"http://{API_HOST}:{API_PORT}"

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
    
    # Initialize default job name once using session state
    if 'default_job_name' not in st.session_state:
        st.session_state.default_job_name = f"Reconciliation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    col1, col2 = st.columns([4, 1])
    with col1:
        job_name = st.text_input("Job Name", value=st.session_state.default_job_name)
    with col2:
        if st.button("🔄 New Name", help="Generate a new timestamp-based job name"):
            st.session_state.default_job_name = f"Reconciliation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            st.rerun()
    
    # Column selection options
    our_trx_column = None
    bank_trx_column = None
    
    if our_file and bank_file:
        st.subheader("🔧 Advanced Options (Optional)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Our File - Bank Trx ID Column:**")
            try:
                if our_file.name.endswith('.csv'):
                    our_df_preview = pd.read_csv(our_file)
                else:
                    our_df_preview = pd.read_excel(our_file)
                
                our_columns = list(our_df_preview.columns)
                our_trx_column = st.selectbox(
                    "Select Bank Trx ID column (leave as 'Auto-detect' for automatic detection):",
                    ["Auto-detect"] + our_columns,
                    key="our_trx_col"
                )
                if our_trx_column == "Auto-detect":
                    our_trx_column = None
                
                # Reset file pointer
                our_file.seek(0)
            except Exception as e:
                st.warning(f"Could not read our file for column selection: {str(e)}")
        
        with col2:
            st.write("**Bank File - Bank Trx ID Column:**")
            try:
                if bank_file.name.endswith('.csv'):
                    bank_df_preview = pd.read_csv(bank_file)
                else:
                    bank_df_preview = pd.read_excel(bank_file)
                
                bank_columns = list(bank_df_preview.columns)
                bank_trx_column = st.selectbox(
                    "Select Bank Trx ID column (leave as 'Auto-detect' for automatic detection):",
                    ["Auto-detect"] + bank_columns,
                    key="bank_trx_col"
                )
                if bank_trx_column == "Auto-detect":
                    bank_trx_column = None
                
                # Reset file pointer
                bank_file.seek(0)
            except Exception as e:
                st.warning(f"Could not read bank file for column selection: {str(e)}")
    
    # Process button
    if st.button("🚀 Start Reconciliation", type="primary"):
        if our_file and bank_file and job_name:
            process_reconciliation(our_file, bank_file, job_name, our_trx_column, bank_trx_column)
        else:
            st.error("Please upload both files and provide a job name.")

def process_reconciliation(our_file, bank_file, job_name, our_trx_column=None, bank_trx_column=None):
    """Process the reconciliation via API"""
    
    with st.spinner("Processing reconciliation... This may take a few minutes for large files."):
        try:
            # Prepare files for upload
            files = {
                'our_file': (our_file.name, our_file.getvalue(), our_file.type),
                'bank_file': (bank_file.name, bank_file.getvalue(), bank_file.type)
            }
            
            data = {'job_name': job_name}
            if our_trx_column:
                data['our_trx_column'] = our_trx_column
            if bank_trx_column:
                data['bank_trx_column_manual'] = bank_trx_column
            
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
                
                # Display amount comparison results
                amount_comp = result.get('amount_comparison', {})
                if amount_comp.get('comparison_performed', False):
                    st.subheader("💰 Amount Comparison Results")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Our Amount Column", amount_comp['our_amount_column'])
                        st.metric("Our Total (All)", f"{amount_comp['our_total']:,.2f}")
                    
                    with col2:
                        st.metric("Bank Amount Column", amount_comp['bank_amount_column'])
                        st.metric("Bank Total (All)", f"{amount_comp['bank_total']:,.2f}")
                    
                    with col3:
                        # Calculate difference between FULL totals, not just matched totals
                        difference = amount_comp['our_total'] - amount_comp['bank_total']
                        st.metric("Total Difference", f"{difference:,.2f}", delta=f"{difference:,.2f}")
                        amounts_match = "✅ YES" if amount_comp['summary']['amounts_match'] else "❌ NO"
                        st.metric("Amounts Match", amounts_match)
                    
                    with col4:
                        discrepancies = amount_comp['summary']['total_discrepancies']
                        st.metric("Discrepancies", discrepancies)
                        if discrepancies > 0:
                            st.warning(f"⚠️ {discrepancies} transactions have amount differences")
                        else:
                            st.success("✅ All amounts match perfectly")
                    
                    # Display amount discrepancies if any
                    if amount_comp.get('discrepancies'):
                        with st.expander(f"💰 View {len(amount_comp['discrepancies'])} Amount Discrepancies"):
                            discrepancy_data = []
                            for i, disc in enumerate(amount_comp['discrepancies'][:50], 1):  # Show first 50
                                discrepancy_data.append({
                                    '#': i,
                                    'Transaction ID': disc.get('bank_trx_id_display', disc['bank_trx_id']),
                                    'Our Amount': f"{disc['our_amount']:,.2f}",
                                    'Bank Amount': f"{disc['bank_amount']:,.2f}",
                                    'Difference': f"{disc['difference']:,.2f}",
                                    'Percentage': f"{disc['percentage_diff']:+.1f}%"
                                })
                            
                            df_discrepancies = pd.DataFrame(discrepancy_data)
                            st.dataframe(df_discrepancies, use_container_width=True)
                            
                            if len(amount_comp['discrepancies']) > 50:
                                st.info(f"Showing first 50 out of {len(amount_comp['discrepancies'])} discrepancies")
                
                elif amount_comp.get('our_amount_column') and amount_comp.get('bank_amount_column'):
                    st.subheader("💰 Amount Detection Results")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.info(f"Our Amount Column: '{amount_comp['our_amount_column']}' (Confidence: {amount_comp.get('our_amount_confidence', 0):.1f}%)")
                    
                    with col2:
                        st.info(f"Bank Amount Column: '{amount_comp['bank_amount_column']}' (Confidence: {amount_comp.get('bank_amount_confidence', 0):.1f}%)")
                    
                    reason = amount_comp.get('reason', 'Unknown reason')
                    st.warning(f"⚠️ Amount comparison not performed: {reason}")
                
                else:
                    st.subheader("💰 Amount Detection")
                    st.warning("⚠️ Amount columns could not be detected automatically. Consider training the amount detector or manual selection.")
                
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
                
                # Add navigation buttons
                st.subheader("Next Steps")
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("📊 View Jobs List", key="nav_to_jobs"):
                        st.session_state.page = "View Jobs"
                        st.rerun()
                
                with col2:
                    if st.button("🔍 View Job Details", key="nav_to_details"):
                        st.session_state.selected_job_id = result['job_id']
                        st.session_state.page = "Job Details"
                        st.rerun()
                
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
    
    # Add back navigation button
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("← Back to Jobs", key="back_to_jobs"):
            st.session_state.page = "View Jobs"
            st.rerun()
    
    # Get job ID from session state or input
    job_id = st.session_state.get('selected_job_id') or st.session_state.get('last_job_id')
    
    if not job_id:
        st.info("💡 No job selected. Please select a job from the list below or enter a Job ID.")
        
        # Show recent jobs for quick selection
        try:
            response = requests.get(f"{API_BASE_URL}/jobs/")
            if response.status_code == 200:
                jobs = response.json()
                if jobs:
                    st.subheader("Recent Jobs")
                    for job in jobs[:5]:  # Show last 5 jobs
                        if st.button(f"Job {job['id']}: {job['job_name']}", key=f"quick_select_{job['id']}"):
                            st.session_state.selected_job_id = job['id']
                            st.rerun()
        except Exception:
            pass
            
        job_id = st.number_input("Or enter Job ID manually:", min_value=1, step=1, key="manual_job_id")
    
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
                            'Bank Trx ID': result.get('bank_trx_id_display', result['bank_trx_id']),
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
                
                # Generate report, PDF download, and Excel export
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("📋 Generate Full Report", key=f"generate_report_{job_id}"):
                        report_response = requests.get(f"{API_BASE_URL}/jobs/{job_id}/report/")
                        if report_response.status_code == 200:
                            report_data = report_response.json()
                            st.subheader("Full Report")
                            st.text(report_data['report'])
                        else:
                            st.error("Failed to generate report")
                
                with col2:
                    # Generate PDF and provide download button
                    try:
                        pdf_response = requests.get(f"{API_BASE_URL}/jobs/{job_id}/pdf-report/")
                        if pdf_response.status_code == 200:
                            # Get filename from response headers
                            filename = f"reconciliation_report_job_{job_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                            
                            # Create download button (this will trigger the download immediately)
                            st.download_button(
                                label="📄 Download PDF Report",
                                data=pdf_response.content,
                                file_name=filename,
                                mime="application/pdf",
                                help="Click to download the PDF report",
                                key=f"download_pdf_{job_id}"
                            )
                        else:
                            st.error("Failed to generate PDF report")
                            if st.button("🔄 Retry PDF Generation", key=f"retry_pdf_error_{job_id}"):
                                st.rerun()
                    except Exception as e:
                        st.error(f"Error generating PDF: {str(e)}")
                        if st.button("🔄 Retry PDF Generation", key=f"retry_pdf_exception_{job_id}"):
                            st.rerun()
                
                with col3:
                    # Excel Export with reconciliation status
                    if st.button("📊 Export to Excel", key=f"export_excel_{job_id}", help="Export reconciliation results to Excel with status columns"):
                        try:
                            with st.spinner("Generating Excel export..."):
                                export_response = requests.get(f"{API_BASE_URL}/jobs/{job_id}/export/")
                                
                                if export_response.status_code == 200:
                                    # Get filename from response headers or create one
                                    filename = f"reconciliation_export_job_{job_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                                    
                                    # Create download button
                                    st.download_button(
                                        label="📥 Download Excel Export",
                                        data=export_response.content,
                                        file_name=filename,
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                        help="Click to download the Excel file with reconciliation status",
                                        key=f"download_excel_{job_id}"
                                    )
                                    
                                    st.success("✅ Excel export ready! Click the download button above.")
                                    st.info("📋 The Excel file contains:\n• Sheet 1: 'Cash Collection' (your data)\n• Sheet 2: Bank data\n• Both sheets include 'Reconciliation Result' column")
                                    
                                else:
                                    st.error("Failed to generate Excel export")
                                    if st.button("🔄 Retry Excel Export", key=f"retry_excel_error_{job_id}"):
                                        st.rerun()
                        except Exception as e:
                            st.error(f"Error generating Excel export: {str(e)}")
                            if st.button("🔄 Retry Excel Export", key=f"retry_excel_exception_{job_id}"):
                                st.rerun()
                        
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
