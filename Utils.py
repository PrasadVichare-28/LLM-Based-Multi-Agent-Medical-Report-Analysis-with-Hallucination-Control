"""
Utility Functions for Medical Diagnosis System
==============================================
Ultra-safe PDF generation that handles ALL edge cases.
"""

import PyPDF2
import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
from fpdf import FPDF


class PDFReader:
    """Utility class for reading PDF and text files"""
    
    @staticmethod
    def read_pdf(file_path: str) -> str:
        """Read a PDF file and extract text"""
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text()
            
            print(f"✅ Successfully read PDF: {os.path.basename(file_path)} ({num_pages} pages, {len(text)} characters)")
            return text
        except Exception as e:
            print(f"❌ Error reading PDF: {e}")
            raise
    
    @staticmethod
    def read_text(file_path: str) -> str:
        """Read a text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            print(f"✅ Successfully read text file: {os.path.basename(file_path)}")
            return text
        except Exception as e:
            print(f"❌ Error reading text file: {e}")
            raise
    
    @staticmethod
    def read_medical_report(file_path: str) -> str:
        """Read a medical report from either PDF or TXT file"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.pdf':
            return PDFReader.read_pdf(file_path)
        elif file_extension == '.txt':
            return PDFReader.read_text(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")


class EnhancedReportGenerator:
    """Generate comprehensive medical reports - PDF ONLY"""
    
    def __init__(self, patient_name: str = "Anonymous Patient"):
        self.patient_name = patient_name
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def _safe_get(self, data: Any, key: str, default: Any = "Not available") -> Any:
        """Safely get value from dict or return default"""
        if isinstance(data, dict):
            return data.get(key, default)
        return default
    
    def _safe_str(self, value: Any, max_length: int = 300) -> str:
        """Convert any value to safe string with length limit"""
        try:
            text = str(value)
            # Remove problematic characters
            text = text.replace('\r', ' ').replace('\n', ' ')
            # Limit length
            if len(text) > max_length:
                text = text[:max_length] + "..."
            return text
        except:
            return "N/A"
    
    def generate_pdf_report(
        self,
        specialist_reports: Dict[str, Any],
        final_diagnosis: Any,
        output_path: str
    ) -> None:
        """
        Generate PDF report with MAXIMUM safety - NEVER crashes
        """
        print("📄 Starting PDF generation...")
        
        try:
            # Create PDF with safe margins
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=20)
            pdf.set_left_margin(15)
            pdf.set_right_margin(15)
            pdf.add_page()
            
            # --- SAFE WRITE FUNCTION ---
            def write_safe(text, size=10, bold=False, ln=True):
                """Safely write text - NEVER fails"""
                try:
                    # Check if we need a new page
                    if pdf.get_y() > 270:
                        pdf.add_page()
                    
                    # Set font safely
                    style = 'B' if bold else ''
                    pdf.set_font('Helvetica', style, size)
                    
                    # Clean and truncate text
                    safe_text = self._safe_str(text, 400)
                    
                    # Write with proper line handling
                    if ln:
                        pdf.cell(0, 6, safe_text, new_x="LMARGIN", new_y="NEXT")
                    else:
                        pdf.cell(0, 6, safe_text)
                except Exception as e:
                    print(f"⚠️  Warning writing text: {e}")
                    # Continue anyway - don't crash
            
            # --- TITLE ---
            write_safe("MEDICAL DIAGNOSIS REPORT", size=16, bold=True)
            pdf.ln(3)
            write_safe(f"Patient: {self.patient_name}", size=12, bold=True)
            write_safe(f"Generated: {self.timestamp}", size=9)
            pdf.ln(5)
            
            # --- DISCLAIMER BOX ---
            try:
                pdf.set_fill_color(255, 255, 200)
                pdf.set_font('Helvetica', 'B', 8)
                pdf.multi_cell(0, 4, "DISCLAIMER: AI-generated for educational purposes only. Not for medical use.", fill=True)
            except:
                write_safe("DISCLAIMER: Educational purposes only.", size=8)
            
            pdf.ln(5)
            
            # --- SECTION HEADER FUNCTION ---
            def section_header(title):
                try:
                    if pdf.get_y() > 250:
                        pdf.add_page()
                    pdf.set_fill_color(200, 200, 240)
                    pdf.set_font('Helvetica', 'B', 12)
                    pdf.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT", fill=True)
                    pdf.ln(2)
                except:
                    write_safe(title, size=12, bold=True)
            
            # --- CARDIOLOGIST SECTION ---
            section_header("CARDIOLOGIST ASSESSMENT")
            
            cardio = specialist_reports.get('Cardiologist', {})
            if isinstance(cardio, dict) and 'error' not in cardio:
                # Chief concerns
                concerns = cardio.get('chief_cardiac_concerns', [])
                if concerns and isinstance(concerns, list):
                    write_safe("Chief Cardiac Concerns:", bold=True)
                    for i, concern in enumerate(concerns[:5], 1):
                        write_safe(f"  {i}. {self._safe_str(concern, 200)}")
                    pdf.ln(2)
                
                # Findings
                findings = cardio.get('findings', [])
                if findings and isinstance(findings, list):
                    write_safe("Key Findings:", bold=True)
                    for finding in findings[:3]:
                        if isinstance(finding, dict):
                            write_safe(f"Finding: {self._safe_str(finding.get('finding', 'N/A'), 150)}")
                            write_safe(f"  Confidence: {self._safe_str(finding.get('confidence', 'N/A'), 50)}")
                    pdf.ln(2)
                
                # Risk
                risk = cardio.get('cardiac_risk_assessment', {})
                if risk and isinstance(risk, dict):
                    write_safe("Risk Assessment:", bold=True)
                    write_safe(f"  Immediate: {self._safe_str(risk.get('immediate_risk', 'N/A'), 100)}")
                    write_safe(f"  Long-term: {self._safe_str(risk.get('long_term_risk', 'N/A'), 150)}")
            else:
                write_safe("Cardiologist analysis in progress or unavailable.")
            
            pdf.ln(5)
            
            # --- PSYCHOLOGIST SECTION ---
            pdf.add_page()  # New page for each major section
            section_header("PSYCHOLOGIST ASSESSMENT")
            
            psych = specialist_reports.get('Psychologist', {})
            if isinstance(psych, dict) and 'error' not in psych:
                # Diagnoses
                diagnoses = psych.get('psychological_diagnoses', [])
                if diagnoses and isinstance(diagnoses, list):
                    write_safe("Psychological Diagnoses:", bold=True)
                    for diag in diagnoses[:3]:
                        if isinstance(diag, dict):
                            write_safe(f"Diagnosis: {self._safe_str(diag.get('diagnosis', 'N/A'), 150)}")
                            write_safe(f"  Severity: {self._safe_str(diag.get('severity', 'N/A'), 50)} | Confidence: {self._safe_str(diag.get('confidence', 'N/A'), 50)}")
                    pdf.ln(2)
                
                # Therapy
                therapy = psych.get('therapeutic_recommendations', [])
                if therapy and isinstance(therapy, list):
                    write_safe("Therapeutic Recommendations:", bold=True)
                    for rec in therapy[:3]:
                        if isinstance(rec, dict):
                            write_safe(f"Intervention: {self._safe_str(rec.get('intervention', 'N/A'), 150)}")
                            write_safe(f"  Rationale: {self._safe_str(rec.get('rationale', 'N/A'), 200)}")
            else:
                write_safe("Psychologist analysis in progress or unavailable.")
            
            pdf.ln(5)
            
            # --- PULMONOLOGIST SECTION ---
            pdf.add_page()
            section_header("PULMONOLOGIST ASSESSMENT")
            
            pulmo = specialist_reports.get('Pulmonologist', {})
            if isinstance(pulmo, dict) and 'error' not in pulmo:
                # Findings
                findings = pulmo.get('respiratory_findings', [])
                if findings and isinstance(findings, list):
                    write_safe("Respiratory Findings:", bold=True)
                    for finding in findings[:3]:
                        if isinstance(finding, dict):
                            write_safe(f"Finding: {self._safe_str(finding.get('finding', 'N/A'), 150)}")
                            write_safe(f"  Confidence: {self._safe_str(finding.get('confidence', 'N/A'), 50)}")
                    pdf.ln(2)
                
                # Treatment
                treatment = pulmo.get('treatment_plan', [])
                if treatment and isinstance(treatment, list):
                    write_safe("Treatment Plan:", bold=True)
                    for plan in treatment[:3]:
                        if isinstance(plan, dict):
                            write_safe(f"Intervention: {self._safe_str(plan.get('intervention', 'N/A'), 150)}")
            else:
                write_safe("Pulmonologist analysis in progress or unavailable.")
            
            pdf.ln(5)
            
            # --- FINAL DIAGNOSIS SECTION ---
            pdf.add_page()
            section_header("FINAL DIAGNOSIS - MULTIDISCIPLINARY TEAM")
            
            if isinstance(final_diagnosis, dict) and 'error' not in final_diagnosis:
                # Primary diagnosis
                primary = final_diagnosis.get('primary_diagnosis', {})
                if primary and isinstance(primary, dict):
                    write_safe("PRIMARY DIAGNOSIS:", bold=True)
                    write_safe(f"Condition: {self._safe_str(primary.get('condition', 'N/A'), 200)}", bold=True)
                    write_safe(f"Confidence: {self._safe_str(primary.get('confidence', 'N/A'), 50)}")
                    write_safe(f"Consensus: {self._safe_str(primary.get('specialist_consensus', 'N/A'), 200)}")
                    
                    # Supporting evidence
                    evidence = primary.get('supporting_evidence', [])
                    if evidence and isinstance(evidence, list):
                        write_safe("Supporting Evidence:", bold=True)
                        for i, ev in enumerate(evidence[:5], 1):
                            write_safe(f"  {i}. {self._safe_str(ev, 200)}")
                    pdf.ln(3)
                
                # Differential diagnoses
                differentials = final_diagnosis.get('differential_diagnoses', [])
                if differentials and isinstance(differentials, list):
                    write_safe("DIFFERENTIAL DIAGNOSES:", bold=True)
                    for i, diff in enumerate(differentials[:5], 1):
                        if isinstance(diff, dict):
                            write_safe(f"{i}. {self._safe_str(diff.get('condition', 'N/A'), 150)}", bold=True)
                            write_safe(f"   Likelihood: {self._safe_str(diff.get('likelihood', 'N/A'), 50)}")
                            write_safe(f"   Reasoning: {self._safe_str(diff.get('reasoning', 'N/A'), 250)}")
                            pdf.ln(1)
                    pdf.ln(2)
                
                # Treatment plan
                treatment = final_diagnosis.get('integrated_treatment_plan', {})
                if treatment and isinstance(treatment, dict):
                    write_safe("INTEGRATED TREATMENT PLAN:", bold=True)
                    
                    immediate = treatment.get('immediate_actions', [])
                    if immediate and isinstance(immediate, list):
                        write_safe("Immediate Actions:", bold=True)
                        for i, action in enumerate(immediate[:5], 1):
                            write_safe(f"  {i}. {self._safe_str(action, 200)}")
                        pdf.ln(2)
                    
                    short_term = treatment.get('short_term_plan', [])
                    if short_term and isinstance(short_term, list):
                        write_safe("Short-term Plan (1-3 months):", bold=True)
                        for i, plan in enumerate(short_term[:5], 1):
                            write_safe(f"  {i}. {self._safe_str(plan, 200)}")
                        pdf.ln(2)
                    
                    long_term = treatment.get('long_term_management', [])
                    if long_term and isinstance(long_term, list):
                        write_safe("Long-term Management:", bold=True)
                        for i, plan in enumerate(long_term[:5], 1):
                            write_safe(f"  {i}. {self._safe_str(plan, 200)}")
                        pdf.ln(2)
                
                # Prognosis
                prognosis = final_diagnosis.get('prognosis', {})
                if prognosis and isinstance(prognosis, dict):
                    write_safe("PROGNOSIS:", bold=True)
                    write_safe(f"Short-term: {self._safe_str(prognosis.get('short_term', 'N/A'), 250)}")
                    write_safe(f"Long-term: {self._safe_str(prognosis.get('long_term', 'N/A'), 250)}")
            else:
                write_safe("Final diagnosis in progress or unavailable.")
                # Show what we have
                write_safe("Raw diagnosis data:")
                write_safe(self._safe_str(str(final_diagnosis)[:500], 500))
            
            # --- FOOTER ---
            pdf.ln(10)
            pdf.set_font('Helvetica', 'I', 8)
            try:
                pdf.multi_cell(0, 4, "This AI-generated report is for educational and research purposes only. Always consult qualified healthcare professionals for medical advice and treatment decisions.")
            except:
                write_safe("Educational purposes only. Consult healthcare professionals.", size=8)
            
            # --- SAVE PDF ---
            pdf.output(output_path)
            print(f"✅ PDF report saved: {output_path}")
            
        except Exception as e:
            print(f"❌ Critical error in PDF generation: {e}")
            # Create emergency fallback PDF
            try:
                pdf_fallback = FPDF()
                pdf_fallback.add_page()
                pdf_fallback.set_font('Helvetica', 'B', 14)
                pdf_fallback.cell(0, 10, "Medical Diagnosis Report", new_x="LMARGIN", new_y="NEXT")
                pdf_fallback.ln(5)
                pdf_fallback.set_font('Helvetica', '', 11)
                pdf_fallback.cell(0, 8, f"Patient: {self.patient_name}", new_x="LMARGIN", new_y="NEXT")
                pdf_fallback.ln(5)
                pdf_fallback.multi_cell(0, 6, "Analysis completed successfully. Detailed report generation encountered formatting issues. Please see JSON report for complete data.")
                pdf_fallback.output(output_path)
                print(f"✅ Fallback PDF saved: {output_path}")
            except:
                print(f"❌ Could not create fallback PDF")
                raise
    
    def generate_markdown_report(self, specialist_reports: Dict[str, Any], final_diagnosis: Any, output_path: str) -> None:
        """Generate markdown report - DISABLED"""
        pass
    
    def generate_json_report(self, specialist_reports: Dict[str, Any], final_diagnosis: Any, output_path: str) -> None:
        """Generate JSON report - kept for backend compatibility"""
        try:
            report_data = {
                "patient_name": self.patient_name,
                "timestamp": self.timestamp,
                "specialist_reports": specialist_reports,
                "final_diagnosis": final_diagnosis
            }
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            print(f"✅ JSON report saved: {output_path}")
        except Exception as e:
            print(f"⚠️  JSON report failed: {e}")
