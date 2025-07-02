#!/usr/bin/env python3
"""
PDF Report Generator for Bank Reconciliation System
"""

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from datetime import datetime

class PDFReportGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        )
        self.subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceAfter=20,
            textColor=colors.darkblue
        )
        self.header_style = ParagraphStyle(
            'CustomHeader',
            parent=self.styles['Heading3'],
            fontSize=14,
            spaceAfter=12,
            textColor=colors.darkblue
        )
        self.normal_style = ParagraphStyle(
            'CustomNormal',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=6
        )
        self.summary_style = ParagraphStyle(
            'SummaryBox',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=8,
            leftIndent=20,
            rightIndent=20,
            backColor=colors.lightgrey
        )

    def generate_reconciliation_report(self, job_data, results_data, output_path):
        doc = SimpleDocTemplate(output_path, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
        story = []
        story.extend(self._create_title_page(job_data))
        story.append(PageBreak())
        story.extend(self._create_summary_page(job_data, results_data))
        story.append(PageBreak())
        story.extend(self._create_detailed_results(results_data))
        doc.build(story)
        return output_path

    def _create_title_page(self, job_data):
        elements = []
        title = Paragraph("Bank Reconciliation Report", self.title_style)
        elements.append(title)
        elements.append(Spacer(1, 40))
        job_info = [
            ["Job Name:", job_data.get('job_name', 'N/A')],
            ["Bank Name:", job_data.get('bank_name', 'N/A')],
            ["Our File:", job_data.get('our_file_name', 'N/A')],
            ["Bank File:", job_data.get('bank_file_name', 'N/A')],
            ["Created Date:", job_data.get('created_at', 'N/A')[:19].replace('T', ' ') if job_data.get('created_at') else 'N/A'],
            ["Status:", job_data.get('status', 'N/A')]
        ]
        job_table = Table(job_info, colWidths=[2*inch, 4*inch])
        job_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ]))
        elements.append(job_table)
        elements.append(Spacer(1, 30))
        report_info = Paragraph(
            f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            self.normal_style
        )
        elements.append(report_info)
        return elements

    def _create_summary_page(self, job_data, results_data):
        elements = []
        summary_title = Paragraph("Reconciliation Summary", self.subtitle_style)
        elements.append(summary_title)
        elements.append(Spacer(1, 20))
        stats_data = [
            ["Metric", "Count", "Percentage"],
            ["Total Our Records", str(job_data.get('total_our_records', 0)), "100%"],
            ["Total Bank Records", str(job_data.get('total_bank_records', 0)), "100%"],
            ["Matched Records", str(job_data.get('matched_records', 0)), f"{(job_data.get('matched_records', 0) / max(job_data.get('total_our_records', 1), 1) * 100):.1f}%"],
            ["Missing in Bank", str(job_data.get('unmatched_our_records', 0)), f"{(job_data.get('unmatched_our_records', 0) / max(job_data.get('total_our_records', 1), 1) * 100):.1f}%"],
            ["Missing in Our File", str(job_data.get('unmatched_bank_records', 0)), f"{(job_data.get('unmatched_bank_records', 0) / max(job_data.get('total_our_records', 1), 1) * 100):.1f}%"]
        ]
        stats_table = Table(stats_data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
        stats_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('BACKGROUND', (0, 2), (-1, 2), colors.lightgreen),
            ('BACKGROUND', (0, 4), (-1, 4), colors.lightyellow),
            ('BACKGROUND', (0, 5), (-1, 5), colors.lightcoral),
        ]))
        elements.append(stats_table)
        elements.append(Spacer(1, 30))
        match_percentage = (job_data.get('matched_records', 0) / max(job_data.get('total_our_records', 1), 1)) * 100
        summary_text = f"""
        <b>Reconciliation Summary:</b><br/>
        This reconciliation processed {job_data.get('total_our_records', 0)} records from your file and \
        {job_data.get('total_bank_records', 0)} records from the bank file. \
        {job_data.get('matched_records', 0)} records were successfully matched, representing a \
        {match_percentage:.1f}% match rate.
        """
        summary_para = Paragraph(summary_text, self.summary_style)
        elements.append(summary_para)
        return elements

    def _create_detailed_results(self, results_data):
        elements = []
        details_title = Paragraph("Detailed Results", self.subtitle_style)
        elements.append(details_title)
        elements.append(Spacer(1, 20))
        status_groups = {
            'matched': [],
            'missing_in_bank': [],
            'missing_in_our_file': []
        }
        for result in results_data:
            status = result.get('status', 'unknown')
            if status in status_groups:
                status_groups[status].append(result)
        for status, results in status_groups.items():
            if results:
                status_title = self._get_status_title(status)
                status_header = Paragraph(status_title, self.header_style)
                elements.append(status_header)
                elements.append(Spacer(1, 10))
                table_data = [["Bank Trx ID", "Bank Name", "Amount", "Created Date"]]
                for result in results[:50]:
                    try:
                        amount = result.get('amount', 0)
                        if amount is None:
                            amount = 0
                        amount_str = f"{float(amount):,.2f}" if amount != 0 else "N/A"
                    except:
                        amount_str = "N/A"
                    created_date = result.get('created_at', 'N/A')
                    if created_date and created_date != 'N/A':
                        created_date = created_date[:19].replace('T', ' ')
                    table_data.append([
                        result.get('bank_trx_id', 'N/A'),
                        result.get('paying_bank_name', 'N/A'),
                        amount_str,
                        created_date
                    ])
                if len(results) > 50:
                    table_data.append([f"... and {len(results) - 50} more records", "", "", ""])
                table = Table(table_data, colWidths=[2*inch, 1.5*inch, 1*inch, 1.5*inch])
                table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ]))
                elements.append(table)
                elements.append(Spacer(1, 20))
        return elements

    def _get_status_title(self, status):
        titles = {
            'matched': '✅ Matched Transactions',
            'missing_in_bank': '❌ Missing in Bank File',
            'missing_in_our_file': '⚠️ Missing in Our File'
        }
        return titles.get(status, status.replace('_', ' ').title()) 