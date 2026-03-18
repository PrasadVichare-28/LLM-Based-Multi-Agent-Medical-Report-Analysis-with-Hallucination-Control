"""
Unified Backend - Enhanced Medical Diagnosis System v2.0
=========================================================
Single script that runs everything:
- Intelligent agent selection
- Reasoning chains
- Anti-hallucination validation
- Beautiful GUI integration

Just run: python backend_enhanced.py
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import time

# Import components
from dotenv import load_dotenv
from AgentOrchestrator import AgentOrchestrator
from Agents_with_reasoning import Cardiologist, Psychologist, Pulmonologist, MultidisciplinaryTeam
from HallucinationDetector import HallucinationDetector, CrossAgentValidator
from Utils import PDFReader, EnhancedReportGenerator

# Load environment
load_dotenv(dotenv_path='apikey.env')

# Initialize Flask
app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max


def send_progress_update(step, message, progress_data=None):
    """Helper to format progress updates for frontend"""
    return {
        'step': step,
        'message': message,
        'data': progress_data or {}
    }


@app.route('/')
def index():
    """Serve the frontend HTML"""
    try:
        # Try to serve frontend.html
        return send_file('frontend.html')
    except:
        return """
        <h1>Enhanced Medical Diagnosis API v2.0 - Ready!</h1>
        <p>Frontend not found. Please place frontend.html in the same directory as this script.</p>
        <p>Or open frontend.html directly in your browser.</p>
        <hr>
        <h2>API Endpoints:</h2>
        <ul>
            <li>POST /analyze - Analyze medical report</li>
            <li>GET /download/&lt;filename&gt; - Download report</li>
            <li>GET /health - Health check</li>
        </ul>
        """


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Main analysis endpoint with real-time progress updates
    
    Process:
    1. Upload & read file
    2. Intelligent triage
    3. Run selected specialists
    4. Validate for hallucinations
    5. Multidisciplinary synthesis
    6. Cross-agent validation
    7. Generate reports
    """
    
    try:
        # Step 1: File Upload & Reading
        print("\n" + "="*70)
        print("STEP 1: FILE UPLOAD & READING")
        print("="*70)
        
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        patient_name = request.form.get('patientName', 'Anonymous Patient')
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Read medical report
        try:
            medical_report = PDFReader.read_medical_report(filepath)
            print(f"✅ Medical report read: {len(medical_report)} characters")
        except Exception as e:
            return jsonify({'success': False, 'error': f'Failed to read file: {str(e)}'}), 400
        
        # Initialize response structure
        analysis_result = {
            'patient_name': patient_name,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'steps_completed': []
        }
        
        # Step 2: Intelligent Triage
        print("\n" + "="*70)
        print("STEP 2: INTELLIGENT TRIAGE")
        print("="*70)
        
        orchestrator = AgentOrchestrator()
        triage_result = orchestrator.analyze_symptoms(medical_report)
        
        analysis_result['triage'] = triage_result
        analysis_result['steps_completed'].append({
            'step': 'triage',
            'status': 'complete',
            'specialists_selected': triage_result['specialists_needed'],
            'urgency': triage_result['urgency'],
            'cost_savings': f"{(3 - len(triage_result['specialists_needed'])) / 3 * 100:.0f}%"
        })
        
        print(f"✅ Triage complete: {len(triage_result['specialists_needed'])} specialists")
        print(f"   Specialists: {', '.join(triage_result['specialists_needed'])}")
        print(f"   Urgency: {triage_result['urgency']}")
        
        # Step 3: Run Selected Specialists with Reasoning
        print("\n" + "="*70)
        print("STEP 3: SPECIALIST CONSULTATIONS")
        print("="*70)
        
        # Initialize only needed specialists
        agents = {}
        specialists_needed = triage_result['specialists_needed']
        
        if 'Cardiologist' in specialists_needed:
            agents['Cardiologist'] = Cardiologist(medical_report)
        if 'Psychologist' in specialists_needed:
            agents['Psychologist'] = Psychologist(medical_report)
        if 'Pulmonologist' in specialists_needed:
            agents['Pulmonologist'] = Pulmonologist(medical_report)
        
        # Function to run agent
        def get_response(agent_name, agent):
            print(f"🔄 Running {agent_name}...")
            response = agent.run()
            return agent_name, response
        
        # Run agents concurrently
        responses = {}
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(get_response, name, agent): name for name, agent in agents.items()}
            
            for future in as_completed(futures):
                agent_name, response = future.result()
                responses[agent_name] = response
                
                # Log reasoning chain length
                if isinstance(response, dict) and 'reasoning_chain' in response:
                    chain_length = len(response['reasoning_chain'])
                    print(f"   ✅ {agent_name}: {chain_length} reasoning steps")
        
        analysis_result['specialist_reports'] = responses
        analysis_result['steps_completed'].append({
            'step': 'specialists',
            'status': 'complete',
            'agents_run': list(responses.keys())
        })
        
        # Step 4: Anti-Hallucination Validation
        print("\n" + "="*70)
        print("STEP 4: ANTI-HALLUCINATION VALIDATION")
        print("="*70)
        
        detector = HallucinationDetector(medical_report)
        validation_results = {}
        
        for agent_name, response in responses.items():
            if isinstance(response, dict) and 'error' not in response:
                validation = detector.validate_diagnosis(response, agent_name)
                validation_results[agent_name] = validation
        
        # Calculate overall validation score
        validation_scores = [v['risk_score'] for v in validation_results.values()]
        avg_validation_score = sum(validation_scores) / len(validation_scores) if validation_scores else 0
        
        analysis_result['validation'] = {
            'individual_agents': validation_results,
            'overall_risk_score': avg_validation_score,
            'risk_level': 'Very Low' if avg_validation_score < 0.2 else 'Low' if avg_validation_score < 0.4 else 'Medium' if avg_validation_score < 0.6 else 'High'
        }
        analysis_result['steps_completed'].append({
            'step': 'validation',
            'status': 'complete',
            'risk_score': f"{avg_validation_score:.1%}",
            'risk_level': analysis_result['validation']['risk_level']
        })
        
        print(f"📊 Overall Hallucination Risk: {avg_validation_score:.1%}")
        
        # Step 5: Multidisciplinary Synthesis
        print("\n" + "="*70)
        print("STEP 5: MULTIDISCIPLINARY SYNTHESIS")
        print("="*70)
        
        # Fill in missing specialists if not run
        if 'Cardiologist' not in responses:
            responses['Cardiologist'] = {"note": "Not consulted based on triage"}
        if 'Psychologist' not in responses:
            responses['Psychologist'] = {"note": "Not consulted based on triage"}
        if 'Pulmonologist' not in responses:
            responses['Pulmonologist'] = {"note": "Not consulted based on triage"}
        
        team_agent = MultidisciplinaryTeam(
            cardiologist_report=responses['Cardiologist'],
            psychologist_report=responses['Psychologist'],
            pulmonologist_report=responses['Pulmonologist']
        )
        
        final_diagnosis = team_agent.run()
        analysis_result['final_diagnosis'] = final_diagnosis
        analysis_result['steps_completed'].append({
            'step': 'synthesis',
            'status': 'complete'
        })
        
        print("✅ Multidisciplinary synthesis complete")
        
        # Step 6: Cross-Agent Consistency Check
        print("\n" + "="*70)
        print("STEP 6: CROSS-AGENT CONSISTENCY")
        print("="*70)
        
        cross_validator = CrossAgentValidator()
        consistency_result = cross_validator.validate_consistency(
            responses.get('Cardiologist', {}),
            responses.get('Psychologist', {}),
            responses.get('Pulmonologist', {}),
            final_diagnosis
        )
        
        analysis_result['validation']['cross_agent_consistency'] = consistency_result
        analysis_result['steps_completed'].append({
            'step': 'consistency',
            'status': 'complete',
            'score': f"{consistency_result['consistency_score']:.1%}"
        })
        
        print(f"📊 Consistency Score: {consistency_result['consistency_score']:.1%}")
        
        # Step 7: Generate Reports
        print("\n" + "="*70)
        print("STEP 7: GENERATING REPORTS")
        print("="*70)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_generator = EnhancedReportGenerator(patient_name)
        
        # Generate PDF
        pdf_filename = f"diagnosis_{patient_name.replace(' ', '_')}_{timestamp}.pdf"
        pdf_path = os.path.join(RESULTS_FOLDER, pdf_filename)
        
        try:
            report_generator.generate_pdf_report(
                specialist_reports=responses,
                final_diagnosis=final_diagnosis,
                output_path=pdf_path
            )
            print(f"✅ PDF generated: {pdf_filename}")
        except Exception as e:
            print(f"⚠️  PDF generation error: {e}")
            pdf_filename = None
        
        # Generate comprehensive JSON
        json_filename = f"diagnosis_{patient_name.replace(' ', '_')}_{timestamp}_complete.json"
        json_path = os.path.join(RESULTS_FOLDER, json_filename)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_result, f, indent=2, ensure_ascii=False)
        
        print(f"✅ JSON generated: {json_filename}")
        
        analysis_result['steps_completed'].append({
            'step': 'reports',
            'status': 'complete',
            'pdf': pdf_filename,
            'json': json_filename
        })
        
        # Final summary
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)
        print(f"Specialists: {len(agents)}/3")
        print(f"Cost Savings: {(3 - len(agents)) / 3 * 100:.0f}%")
        print(f"Hallucination Risk: {avg_validation_score:.1%}")
        print(f"Consistency: {consistency_result['consistency_score']:.1%}")
        print("="*70)
        
        # Return comprehensive results
        return jsonify({
            'success': True,
            'patient_name': patient_name,
            'specialist_reports': responses,
            'final_diagnosis': final_diagnosis,
            'triage': triage_result,
            'validation': analysis_result['validation'],
            'reports': {
                'pdf': pdf_filename,
                'json': json_filename
            },
            'metadata': {
                'specialists_consulted': list(agents.keys()),
                'specialists_skipped': [s for s in ['Cardiologist', 'Psychologist', 'Pulmonologist'] if s not in agents],
                'api_calls_saved': 3 - len(agents),
                'cost_savings_percent': (3 - len(agents)) / 3 * 100,
                'hallucination_risk': avg_validation_score,
                'consistency_score': consistency_result['consistency_score']
            },
            'timestamp': timestamp
        })
        
    except Exception as e:
        print(f"❌ Error in analysis: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/download/<filename>')
def download_file(filename):
    """Download generated report files"""
    try:
        file_path = os.path.join(RESULTS_FOLDER, filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'version': '2.0-enhanced',
        'features': [
            'Intelligent agent selection',
            'Reasoning chains',
            'Anti-hallucination validation',
            'Cross-agent consistency check'
        ]
    })


if __name__ == '__main__':
    print("="*70)
    print(" ENHANCED MEDICAL DIAGNOSIS SYSTEM v2.0")
    print(" Backend Server Starting...")
    print("="*70)
    print("\n✨ Features:")
    print("   ✅ Intelligent agent selection (cost optimization)")
    print("   ✅ Clinical reasoning chains (transparency)")
    print("   ✅ Anti-hallucination validation (trust & safety)")
    print("   ✅ Cross-agent consistency checking")
    print("\n🌐 Server will start on: http://localhost:5000")
    print("🎨 Frontend: Open frontend.html in browser")
    print("\n" + "="*70)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
