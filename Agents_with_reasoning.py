"""
Enhanced AI Agents with Reasoning Chains & Anti-Hallucination Measures
========================================================================
Features:
1. Step-by-step clinical reasoning (transparency)
2. Grounded responses (only cite what's in the report)
3. Confidence scoring for each claim
4. Explicit uncertainty acknowledgment
5. Evidence-based conclusions
"""

from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import json
import time
import re  # Add this for regex operations in hallucination filtering


class ReasoningAgent:
    """
    Base agent with reasoning chains and anti-hallucination measures.
    
    Anti-Hallucination Strategies:
    1. Explicit grounding instructions
    2. Quote-based evidence requirements
    3. Confidence scoring
    4. "I don't know" option
    5. Fact-checking against report
    """
    
    def __init__(self, medical_report=None, role=None, extra_info=None):
        # CRITICAL: Preprocess medical report to remove confusing patterns
        if medical_report:
            medical_report = self._preprocess_medical_report(medical_report)
        
        self.medical_report = medical_report
        self.role = role
        self.extra_info = extra_info
        
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is not set")
        
        self.model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.1,  # Even lower temperature
            google_api_key=api_key,
            max_retries=3,
            request_timeout=60
        )
        
        self.prompt_template = self.create_prompt_template()
    
    def _preprocess_medical_report(self, report):
        """
        Remove confusing patterns from medical report BEFORE AI sees it.
        This prevents hallucinations at the source.
        """
        # Replace year numbers with spelled-out versions
        report = re.sub(r'\b2024\b', 'twenty twenty-four', report)
        report = re.sub(r'\b2025\b', 'twenty twenty-five', report)
        report = re.sub(r'\b2023\b', 'twenty twenty-three', report)
        report = re.sub(r'\b2026\b', 'twenty twenty-six', report)
        
        # Replace date formats that might confuse AI
        # "October 20, 2024" stays, but standalone years get replaced
        
        return report
    
    def create_prompt_template(self):
        """Create role-specific prompts with anti-hallucination measures"""
        
        if self.role == "MultidisciplinaryTeam":
            return None  # Handled separately
        
        # Common anti-hallucination instructions for all specialists
        anti_hallucination_rules = """
🚨 CRITICAL ANTI-HALLUCINATION RULES - READ CAREFULLY 🚨

⛔ ABSOLUTE PROHIBITIONS - NEVER REPORT THESE:
1. ❌ NEVER report "2025" as a heart rate (IT'S A YEAR!)
2. ❌ NEVER report "16/21" as blood pressure (IT'S GAD-7 ANXIETY SCORE!)
3. ❌ NEVER report "11/27" as blood pressure (IT'S PHQ-9 DEPRESSION SCORE!)
4. ❌ NEVER report "14/27" as blood pressure (IT'S PHQ-9 DEPRESSION SCORE!)
5. ❌ NEVER report any number >200 as heart rate (IT'S A DATE OR ID!)
6. ❌ NEVER extract numbers from "Patient ID:", "Date:", "License:" fields

🔍 COMMON MISINTERPRETATIONS TO AVOID:

SCREENING SCORES - These are QUESTIONNAIRE SCORES, NOT vital signs:
❌ "GAD-7: 16/21" → This is anxiety screening (16 points out of 21 possible)
❌ "PHQ-9: 11/27" → This is depression screening (11 points out of 27 possible)
❌ "PHQ-9: 14/27" → This is depression screening (14 points out of 27 possible)
❌ "PDSS: 18/28" → This is panic disorder severity (18 out of 28)
❌ ANY "Score: X/Y" format → This is a TEST SCORE, not blood pressure!

DATES - These are CALENDAR DATES, NOT heart rates:
❌ "Date: 2024-11-15" → DO NOT extract "2024" or "2025"
❌ "November 2025" → This is a DATE
❌ "2024" or "2025" ANYWHERE → These are YEARS, not vital signs

LAB VALUES - These are BIOMARKERS, NOT heart rates:
❌ "BNP: 1,850 pg/mL" → DO NOT extract "850" or "1850" as heart rate
❌ "Troponin: 8.5 ng/mL" → This is a LAB VALUE
❌ "WBC: 14,200/μL" → This is a LAB VALUE

IDENTIFIERS - These are IDs/PHONE NUMBERS, NOT vital signs:
❌ "Patient ID: 345678" → DO NOT extract any numbers
❌ "Patient ID: 2713" → DO NOT use as heart rate
❌ "Phone: (555) 123-4567" → DO NOT extract any numbers
❌ "License: CA12345678" → DO NOT extract any numbers

✅ ONLY ACCEPT VITAL SIGNS IN THESE EXACT FORMATS:

Heart Rate - MUST have explicit label + bpm unit:
✅ "Heart Rate: 85 bpm" 
✅ "HR: 102 bpm"
✅ "Pulse: 76 beats per minute"
✅ MUST be between 40-200 bpm
❌ If no "bpm" unit → NOT a heart rate
❌ If number is >200 → NOT a heart rate (it's a date/ID)

Blood Pressure - MUST have explicit label + mmHg unit + reasonable values:
✅ "Blood Pressure: 140/90 mmHg"
✅ "BP: 118/76 mmHg"
✅ Systolic MUST be 70-250
✅ Diastolic MUST be 40-150
✅ Systolic MUST be > Diastolic
❌ Format like "16/21" → This is a TEST SCORE, not BP!
❌ Format like "11/27" → This is a TEST SCORE, not BP!
❌ Any BP without "mmHg" in same sentence → Probably NOT BP
❌ Systolic < 70 → REJECT
❌ Diastolic < 40 → REJECT
❌ Systolic < Diastolic → REJECT (physiologically impossible)

Oxygen Saturation:
✅ "O2 Saturation: 95% on room air"
✅ "SpO2: 98%"
❌ Any value >100% → REJECT (impossible)

🎯 VALIDATION CHECKLIST (Ask yourself BEFORE reporting ANY vital sign):

1. ✓ Is there a clear LABEL before the number?
   ("Heart Rate:", "Blood Pressure:", "BP:", etc.)
   
2. ✓ Are there appropriate UNITS after the number?
   ("bpm", "mmHg", "%")
   
3. ✓ Is the value PHYSIOLOGICALLY POSSIBLE?
   HR: 40-200, BP: 70-250/40-150, SpO2: 70-100%
   
4. ✓ Did I check that this is NOT:
   - A screening test score (GAD-7, PHQ-9, etc.)?
   - A date or year (2024, 2025)?
   - A patient ID or phone number?
   - A lab value (BNP, troponin, etc.)?

If ANY answer is "NO" → DO NOT REPORT THAT VALUE!

🚫 WHEN IN DOUBT:
- If you're unsure whether something is a vital sign → DON'T report it
- If a number doesn't have proper labeling → DON'T report it
- If a value seems unusual → VERIFY the label and units first
- Better to miss a vital sign than to report a test score as blood pressure!

✅ CORRECT EXAMPLE:
Report: "Blood Pressure: 140/90 mmHg"
You report: "Blood pressure 140/90 mmHg" ✓ CORRECT

❌ INCORRECT EXAMPLES:
Report: "GAD-7 score: 16/21"
You report: "Blood pressure 16/21" ✗ WRONG! (This is an anxiety test score!)

Report: "Date of visit: 2024-11-15"
You report: "Heart rate 2024 bpm" ✗ WRONG! (This is a date!)

Report: "Patient ID: 2713"
You report: "Heart rate 2713 bpm" ✗ WRONG! (This is an ID number!)

REASONING CHAIN REQUIREMENTS:
- Show your step-by-step clinical reasoning
- For each vital sign, cite the EXACT sentence with label and unit
- If you don't see "Heart Rate:" or "BP: X/Y mmHg" explicitly, don't guess
- State what information is missing from the report
"""
        
        templates = {
            "Cardiologist": f"""
You are an experienced cardiologist conducting a systematic cardiovascular assessment.

{anti_hallucination_rules}

**MEDICAL REPORT:**
{{medical_report}}

**YOUR TASK:**
Analyze this report using evidence-based clinical reasoning. Show your thought process step-by-step.

**OUTPUT FORMAT (VALID JSON ONLY):**
{{{{
    "reasoning_chain": [
        {{{{
            "step": 1,
            "observation": "What you observed IN THE REPORT (quote if possible)",
            "evidence_quote": "Exact quote from report",
            "inference": "What this suggests clinically",
            "confidence": "High/Medium/Low",
            "alternatives_considered": ["Alternative interpretation 1", "Alternative 2"]
        }}}},
        // Continue for each reasoning step (aim for 4-6 steps)
    ],
    "chief_cardiac_concerns": [
        {{{{
            "concern": "Primary cardiac issue",
            "evidence_from_report": ["Specific quote 1", "Specific quote 2"],
            "confidence": "High/Medium/Low",
            "severity": "Critical/Serious/Moderate/Mild"
        }}}}
    ],
    "findings": [
        {{{{
            "finding": "Specific cardiac finding",
            "confidence": "High/Medium/Low",
            "evidence": ["Quote from report", "Test result from report"],
            "clinical_significance": "What this means",
            "data_quality": "Definitive/Suggestive/Insufficient"
        }}}}
    ],
    "ruled_out_conditions": [
        {{{{
            "condition": "Condition ruled out",
            "reason": "Why ruled out based on report data"
        }}}}
    ],
    "data_limitations": [
        "Missing test results needed",
        "Unclear timeline of symptoms",
        "Incomplete medication history"
    ],
    "recommended_tests": [
        {{{{
            "test": "Recommended test",
            "reason": "Why needed based on findings",
            "urgency": "Urgent/Routine"
        }}}}
    ],
    "cardiac_risk_assessment": {{{{
        "immediate_risk": "Low/Moderate/High",
        "confidence_in_assessment": "High/Medium/Low",
        "risk_factors_identified": ["Factor from report"],
        "missing_risk_info": ["Information needed but not in report"]
    }}}},
    "hallucination_check": {{{{
        "assumptions_made": ["Any assumptions clearly stated"],
        "information_gaps": ["What's missing from report"],
        "confidence_overall": 0.85
    }}}}
}}}}

REMEMBER: 
- Only cite what's IN the report
- Quote evidence when making claims
- Admit uncertainty when appropriate
- Show your reasoning process
- Distinguish facts from inferences

Return ONLY valid JSON, no other text.
""",
            
            "Psychologist": f"""
You are a clinical psychologist with expertise in health psychology and psychosomatic medicine.

{anti_hallucination_rules}

**MEDICAL REPORT:**
{{medical_report}}

**YOUR TASK:**
Conduct a systematic psychological assessment showing your clinical reasoning process.

**OUTPUT FORMAT (VALID JSON ONLY):**
{{{{
    "reasoning_chain": [
        {{{{
            "step": 1,
            "observation": "What you observed in the report",
            "evidence_quote": "Exact quote from report",
            "inference": "Psychological interpretation",
            "confidence": "High/Medium/Low",
            "alternative_explanations": ["Alternative 1", "Alternative 2"]
        }}}}
    ],
    "psychological_diagnoses": [
        {{{{
            "diagnosis": "DSM-5 diagnosis or clinical concern",
            "confidence": "High/Medium/Low",
            "dsm5_criteria_met": ["Criterion from report", "Criterion from report"],
            "dsm5_criteria_unknown": ["Criteria not assessable from report"],
            "severity": "Mild/Moderate/Severe",
            "evidence_from_report": ["Quote 1", "Quote 2"]
        }}}}
    ],
    "psychosomatic_factors": [
        {{{{
            "factor": "Mind-body connection identified",
            "evidence": "Quote from report",
            "impact": "How this affects physical symptoms"
        }}}}
    ],
    "symptom_triggers": [
        {{{{
            "trigger": "Identified trigger FROM REPORT",
            "symptoms_provoked": ["Symptom mentioned in report"],
            "evidence": "Quote from report"
        }}}}
    ],
    "therapeutic_recommendations": [
        {{{{
            "intervention": "Evidence-based intervention",
            "rationale": "Why recommended based on findings",
            "evidence_level": "Strong/Moderate/Limited",
            "expected_outcome": "Realistic expectation"
        }}}}
    ],
    "data_limitations": [
        "Mental status exam not included",
        "No previous psychiatric history mentioned",
        "Insufficient detail on symptom timeline"
    ],
    "hallucination_check": {{{{
        "assumptions_made": ["Any assumptions clearly stated"],
        "information_gaps": ["Missing psychological data"],
        "confidence_overall": 0.80
    }}}}
}}}}

CRITICAL: Only diagnose based on information IN the report. If DSM-5 criteria cannot be fully assessed from available data, state this clearly.

Return ONLY valid JSON.
""",
            
            "Pulmonologist": f"""
You are an experienced pulmonologist conducting a comprehensive respiratory assessment.

{anti_hallucination_rules}

**MEDICAL REPORT:**
{{medical_report}}

**YOUR TASK:**
Systematically evaluate respiratory status using evidence-based reasoning.

**OUTPUT FORMAT (VALID JSON ONLY):**
{{{{
    "reasoning_chain": [
        {{{{
            "step": 1,
            "observation": "Respiratory finding from report",
            "evidence_quote": "Exact quote",
            "inference": "Clinical interpretation",
            "confidence": "High/Medium/Low"
        }}}}
    ],
    "respiratory_findings": [
        {{{{
            "finding": "Specific respiratory finding",
            "confidence": "High/Medium/Low",
            "supporting_evidence": ["Quote from report"],
            "clinical_significance": "What this indicates",
            "normal_vs_abnormal": "Normal/Mildly abnormal/Significantly abnormal"
        }}}}
    ],
    "pulmonary_diagnoses": [
        {{{{
            "diagnosis": "Pulmonary condition",
            "likelihood": "High/Medium/Low",
            "evidence_for": ["Supporting evidence from report"],
            "evidence_against": ["Contradicting findings from report"],
            "diagnostic_certainty": "Definite/Probable/Possible"
        }}}}
    ],
    "non_pulmonary_causes": [
        {{{{
            "cause": "Non-respiratory cause of symptoms",
            "explanation": "How this causes respiratory symptoms",
            "evidence": "Support from report"
        }}}}
    ],
    "treatment_plan": [
        {{{{
            "intervention": "Treatment recommendation",
            "evidence_basis": "Why recommended",
            "monitoring": "How to assess effectiveness"
        }}}}
    ],
    "data_limitations": [
        "Pulmonary function tests not performed",
        "Oxygen saturation only measured at rest",
        "No imaging studies mentioned"
    ],
    "hallucination_check": {{{{
        "assumptions_made": [],
        "information_gaps": ["Missing diagnostic data"],
        "confidence_overall": 0.85
    }}}}
}}}}

Return ONLY valid JSON.
"""
        }
        
        template = templates[self.role]
        return PromptTemplate.from_template(template)
    
    def run(self):
        """Execute agent with reasoning and anti-hallucination measures"""
        print(f"🔄 {self.role} is analyzing with clinical reasoning...")
        
        # Add rate limiting
        time.sleep(1)
        
        if self.role == "MultidisciplinaryTeam":
            return self._run_multidisciplinary()
        
        prompt = self.prompt_template.format(medical_report=self.medical_report)
        
        try:
            response = self.model.invoke(prompt)
            content = response.content
            
            # Clean JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            # Parse JSON
            result = json.loads(content)
            
            # CRITICAL: Clean hallucinations from PARSED JSON
            result = self._clean_hallucinations_from_json(result)
            
            # Validate reasoning chain exists
            if 'reasoning_chain' not in result:
                print(f"⚠️  {self.role}: No reasoning chain provided")
                result['reasoning_chain'] = [{
                    "step": 1,
                    "observation": "Analysis completed",
                    "inference": "See findings below",
                    "confidence": "Medium"
                }]
            
            # Add metadata
            result['agent_type'] = self.role
            # Use timestamp format without year to avoid validation false positives
            result['timestamp'] = time.strftime("%B %d, %H:%M:%S")
            
            print(f"✅ {self.role} completed with {len(result.get('reasoning_chain', []))} reasoning steps")
            
            return result
            
        except json.JSONDecodeError as e:
            print(f"⚠️  {self.role} JSON parse error: {e}")
            return {
                "error": "JSON parse failed",
                "raw_response": content[:500],
                "note": "AI returned non-JSON format"
            }
        except Exception as e:
            print(f"❌ {self.role} error: {e}")
            return {"error": str(e)}
    
    def _clean_hallucinations_from_json(self, data):
        """
        Recursively clean hallucinations from parsed JSON.
        This works on the actual data structure, not regex on text.
        """
        
        if isinstance(data, dict):
            cleaned = {}
            for key, value in data.items():
                # Recursively clean nested structures
                cleaned_value = self._clean_hallucinations_from_json(value)
                
                # Check if this key-value pair contains hallucinations
                if not self._is_hallucination(key, cleaned_value):
                    cleaned[key] = cleaned_value
                else:
                    # Log what we're removing
                    print(f"   🧹 Removed hallucination: {key} = {cleaned_value}")
            
            return cleaned
        
        elif isinstance(data, list):
            cleaned = []
            for item in data:
                cleaned_item = self._clean_hallucinations_from_json(item)
                
                # Check if this list item is a hallucination
                if not self._is_hallucination_item(cleaned_item):
                    cleaned.append(cleaned_item)
                else:
                    print(f"   🧹 Removed hallucination from list: {cleaned_item}")
            
            return cleaned
        
        elif isinstance(data, str):
            # Clean strings that mention impossible values
            return self._clean_hallucination_string(data)
        
        else:
            return data
    
    def _is_hallucination(self, key, value):
        """Check if a key-value pair is a hallucination"""
        
        # Convert to string for checking
        key_str = str(key).lower()
        value_str = str(value).lower()
        combined = f"{key_str} {value_str}".lower()
        
        # Pattern 1: ANY mention of years 2020-2029 (these are NEVER vital signs!)
        if re.search(r'\b20[2-3]\d\b', combined):
            # Check if it's actually being used as a vital sign
            if any(word in combined for word in ['heart', 'rate', 'hr', 'pulse', 'bp', 'blood', 'pressure']):
                print(f"   🧹 BLOCKING: Found year (20XX) near vital sign keyword")
                return True
        
        # Pattern 2: Heart rate over 200
        hr_match = re.search(r'(\d{3,4})\s*(?:bpm|beats)', combined)
        if hr_match:
            hr_value = int(hr_match.group(1))
            if hr_value > 200:
                print(f"   🧹 BLOCKING: Heart rate {hr_value} >200")
                return True
        
        # Pattern 3: Years as heart rates (2024, 2025, etc.) - VERY strict
        if re.search(r'(?:heart|rate|hr|pulse).*20\d{2}', combined):
            print(f"   🧹 BLOCKING: Year pattern in heart rate context")
            return True
        
        # Pattern 4: Blood pressure with impossible values
        # BUT: Only if explicitly labeled as "blood pressure" or "BP"
        bp_match = re.search(r'(\d{1,3})/(\d{1,3})', value_str)
        if bp_match:
            systolic = int(bp_match.group(1))
            diastolic = int(bp_match.group(2))
            
            # CRITICAL: Only block if it's EXPLICITLY labeled as blood pressure
            # Check if "blood pressure" or "BP" appears near the X/Y pattern
            is_bp_context = (
                'blood pressure' in combined or 
                'bp:' in combined or 
                'bp ' in combined or
                re.search(r'(?:^|\s)bp(?:\s|:|$)', combined)
            )
            
            # DON'T block if it's clearly a pain scale or score
            is_pain_scale = (
                'pain' in combined or
                'severity' in combined or
                'rate' in combined and 'pain' in combined or
                'score' in combined or
                'rated' in combined
            )
            
            # Only block if it's labeled as BP AND has impossible values
            if is_bp_context and not is_pain_scale:
                # Test scores or impossible BP values
                if systolic < 70 or diastolic < 40 or systolic < diastolic:
                    print(f"   🧹 BLOCKING: Impossible BP {systolic}/{diastolic} (explicitly labeled)")
                    return True
        
        return False
        return True
        
        return False
    
    def _is_hallucination_item(self, item):
        """Check if a list item contains hallucinations"""
        
        if isinstance(item, dict):
            # Check all values in the dict
            for key, value in item.items():
                if self._is_hallucination(key, value):
                    return True
        
        elif isinstance(item, str):
            item_lower = item.lower()
            
            # Check for impossible heart rates
            hr_match = re.search(r'(\d{3,4})\s*(?:bpm|beats)', item_lower)
            if hr_match:
                hr_value = int(hr_match.group(1))
                if hr_value > 200:
                    return True
            
            # Check for years as heart rates
            if re.search(r'(?:heart|rate|hr|pulse).*20\d{2}', item_lower):
                return True
            
            # Check for test scores as BP
            if re.search(r'(?:bp|blood.*pressure).*\d{1,2}/\d{1,2}(?!\d)', item_lower):
                bp_match = re.search(r'(\d{1,3})/(\d{1,3})', item_lower)
                if bp_match:
                    systolic = int(bp_match.group(1))
                    diastolic = int(bp_match.group(2))
                    if systolic < 70 or diastolic < 40:
                        return True
        
        return False
    
    def _clean_hallucination_string(self, text):
        """Clean a string of hallucination patterns"""
        
        original = text
        
        # Remove heart rate >200
        text = re.sub(
            r'heart\s*rate[:\s]+\d{3,4}\s*bpm',
            '',
            text,
            flags=re.IGNORECASE
        )
        
        # Remove years mentioned as vital signs
        text = re.sub(
            r'(?:heart\s*rate|HR|pulse)[:\s]+20\d{2}',
            '',
            text,
            flags=re.IGNORECASE
        )
        
        # Remove test scores mentioned as blood pressure
        def filter_bp(match):
            systolic = int(match.group(1))
            diastolic = int(match.group(2))
            if systolic < 70 or diastolic < 40:
                return ''
            return match.group(0)
        
        text = re.sub(
            r'(?:blood\s*pressure|BP)[:\s]+(\d{1,3})/(\d{1,3})',
            filter_bp,
            text,
            flags=re.IGNORECASE
        )
        
        # Clean up whitespace
        text = re.sub(r'\s{2,}', ' ', text)
        text = text.strip()
        
        if text != original and text == '':
            return "[Information removed - likely hallucination]"
        
        return text
    
    def _run_multidisciplinary(self):
        """Run multidisciplinary team synthesis with anti-hallucination"""
        cardio = self.extra_info.get('cardiologist_report', {})
        psych = self.extra_info.get('psychologist_report', {})
        pulmo = self.extra_info.get('pulmonologist_report', {})
        
        prompt = f"""
You are the lead physician synthesizing specialist consultations.

CRITICAL ANTI-HALLUCINATION RULES:
1. ONLY use findings from the specialist reports below
2. Do NOT add new diagnoses not mentioned by specialists
3. Clearly mark if specialists disagree
4. Admit uncertainty when data is conflicting
5. Show consensus-building reasoning

**CARDIOLOGIST REPORT:**
{json.dumps(cardio, indent=2)[:2000]}

**PSYCHOLOGIST REPORT:**
{json.dumps(psych, indent=2)[:2000]}

**PULMONOLOGIST REPORT:**
{json.dumps(pulmo, indent=2)[:2000]}

**YOUR TASK:**
Synthesize these reports into a unified diagnosis, showing your reasoning process.

**OUTPUT (VALID JSON):**
{{
    "synthesis_reasoning": [
        {{
            "step": 1,
            "observation": "What all specialists agree on",
            "sources": ["Cardiologist", "Psychologist"],
            "synthesis": "Combined interpretation",
            "confidence": "High/Medium/Low"
        }}
    ],
    "primary_diagnosis": {{
        "condition": "Primary diagnosis",
        "confidence": "High/Medium/Low",
        "supporting_evidence": ["Evidence from specialists"],
        "specialist_consensus": "All agree / Cardiologist and Psychologist agree / Conflicting",
        "grounding": ["Cardiologist finding X", "Psychologist finding Y"]
    }},
    "differential_diagnoses": [
        {{
            "condition": "Alternative diagnosis",
            "likelihood": "High/Medium/Low",
            "specialist_support": "Which specialists suggest this",
            "reasoning": "Why considered"
        }}
    ],
    "integrated_treatment_plan": {{
        "immediate_actions": ["Action based on specialist recommendation"],
        "short_term_plan": ["1-3 month plan from specialists"],
        "long_term_management": ["Long-term from specialists"]
    }},
    "specialist_agreements": ["Where all specialists agree"],
    "specialist_disagreements": ["Where specialists differ"],
    "confidence_assessment": {{
        "overall_confidence": 0.85,
        "high_confidence_items": ["Items we're sure about"],
        "low_confidence_items": ["Items needing more data"],
        "conflicting_data": ["Contradictory findings"]
    }}
}}

Return ONLY valid JSON.
"""
        
        try:
            time.sleep(1)
            response = self.model.invoke(prompt)
            content = response.content
            
            # Clean JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            result = json.loads(content)
            print(f"✅ MultidisciplinaryTeam synthesis complete")
            
            return result
            
        except Exception as e:
            print(f"❌ Multidisciplinary synthesis error: {e}")
            return {"error": str(e)}


# Specialist agent classes
class Cardiologist(ReasoningAgent):
    def __init__(self, medical_report):
        super().__init__(medical_report, "Cardiologist")


class Psychologist(ReasoningAgent):
    def __init__(self, medical_report):
        super().__init__(medical_report, "Psychologist")


class Pulmonologist(ReasoningAgent):
    def __init__(self, medical_report):
        super().__init__(medical_report, "Pulmonologist")


class MultidisciplinaryTeam(ReasoningAgent):
    def __init__(self, cardiologist_report, psychologist_report, pulmonologist_report):
        extra_info = {
            "cardiologist_report": cardiologist_report,
            "psychologist_report": psychologist_report,
            "pulmonologist_report": pulmonologist_report
        }
        super().__init__(role="MultidisciplinaryTeam", extra_info=extra_info)
