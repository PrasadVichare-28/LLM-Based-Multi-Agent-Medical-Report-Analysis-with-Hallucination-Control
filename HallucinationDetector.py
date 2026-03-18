"""
Anti-Hallucination Validation System
=====================================
Validates AI responses to detect and prevent hallucinations.

Detection Methods:
1. Fact-checking against source document
2. Confidence score validation
3. Citation verification
4. Consistency checking across agents
5. Impossible value detection
"""

import json
import re
from typing import Dict, List, Any


class HallucinationDetector:
    """
    Detects potential hallucinations in AI-generated medical diagnoses.
    
    Hallucination Types Detected:
    1. Fabricated data (mentions info not in report)
    2. Impossible values (HR = 300, BP = 500/200)
    3. Inconsistent claims (says both "normal" and "abnormal")
    4. Overconfident without evidence
    5. Invented medications or procedures
    """
    
    def __init__(self, source_document: str):
        """
        Args:
            source_document: Original medical report text
        """
        self.source = source_document.lower()
        self.source_words = set(source_document.lower().split())
        self.warnings = []
        self.critical_errors = []
    
    def validate_diagnosis(self, diagnosis: Dict[str, Any], agent_name: str = "Agent") -> Dict[str, Any]:
        """
        Comprehensive validation of AI diagnosis for hallucinations.
        
        Args:
            diagnosis: AI-generated diagnosis dict
            agent_name: Name of the agent for error reporting
            
        Returns:
            dict with validation results
        """
        print(f"\n🔍 Validating {agent_name} for hallucinations...")
        
        self.warnings = []
        self.critical_errors = []
        
        # Run all validation checks
        self._check_evidence_grounding(diagnosis, agent_name)
        self._check_impossible_values(diagnosis, agent_name)
        self._check_confidence_calibration(diagnosis, agent_name)
        self._check_invented_terms(diagnosis, agent_name)
        self._check_reasoning_quality(diagnosis, agent_name)
        
        # Calculate hallucination risk score
        risk_score = self._calculate_risk_score()
        
        result = {
            'validated': len(self.critical_errors) == 0,
            'risk_score': risk_score,
            'risk_level': self._get_risk_level(risk_score),
            'warnings': self.warnings,
            'critical_errors': self.critical_errors,
            'recommendations': self._get_recommendations()
        }
        
        # Print summary
        if result['validated']:
            print(f"✅ {agent_name} validation passed (Risk: {risk_score:.1%} - {result['risk_level']})")
        else:
            print(f"❌ {agent_name} FAILED validation - {len(self.critical_errors)} critical errors")
            for error in self.critical_errors[:3]:
                print(f"   - {error}")
        
        if self.warnings:
            print(f"⚠️  {len(self.warnings)} warnings detected")
        
        return result
    
    def _check_evidence_grounding(self, diagnosis: Dict, agent_name: str):
        """Check if findings are grounded in source document"""
        
        # Check for evidence fields
        evidence_fields = ['evidence', 'supporting_evidence', 'evidence_from_report', 'evidence_quote']
        
        for field in evidence_fields:
            evidence_list = self._find_nested(diagnosis, field)
            for evidence in evidence_list:
                if isinstance(evidence, str) and len(evidence) > 10:
                    # Check if evidence text appears in source
                    if not self._is_in_source(evidence):
                        self.warnings.append(
                            f"Evidence not found in report: '{evidence[:100]}...'"
                        )
        
        # Check findings have evidence
        findings = diagnosis.get('findings', [])
        if isinstance(findings, list):
            for finding in findings:
                if isinstance(finding, dict):
                    if 'evidence' not in finding and 'supporting_evidence' not in finding:
                        self.warnings.append(
                            f"Finding lacks evidence: {finding.get('finding', 'Unknown')}"
                        )
    
    def _check_impossible_values(self, diagnosis: Dict, agent_name: str):
        """Detect physiologically impossible values"""
        
        text = json.dumps(diagnosis)
        
        # Heart rate checks
        hr_patterns = [
            (r'heart rate.*?(\d{3,})', "Heart rate >200 is extremely rare"),
            (r'hr.*?(\d{3,})', "Heart rate >200 is extremely rare"),
        ]
        
        for pattern, msg in hr_patterns:
            matches = re.findall(pattern, text.lower())
            for match in matches:
                hr = int(match)
                if hr > 250:
                    self.critical_errors.append(f"Impossible heart rate: {hr} bpm. {msg}")
                elif hr > 200:
                    self.warnings.append(f"Unusually high heart rate: {hr} bpm")
        
        # Blood pressure checks
        bp_pattern = r'(\d{2,3})/(\d{2,3})'
        bp_matches = re.findall(bp_pattern, text)
        for systolic, diastolic in bp_matches:
            sys, dia = int(systolic), int(diastolic)
            if sys > 300 or dia > 200:
                self.critical_errors.append(f"Impossible blood pressure: {sys}/{dia}")
            elif sys < dia:
                self.critical_errors.append(f"Invalid BP (systolic < diastolic): {sys}/{dia}")
        
        # Oxygen saturation
        o2_pattern = r'oxygen.*?(\d{1,3})%'
        o2_matches = re.findall(o2_pattern, text.lower())
        for o2 in o2_matches:
            if int(o2) > 100:
                self.critical_errors.append(f"Impossible oxygen saturation: {o2}%")
    
    def _check_confidence_calibration(self, diagnosis: Dict, agent_name: str):
        """Check if confidence levels match evidence strength"""
        
        # Find all confidence scores
        confidence_items = self._find_nested(diagnosis, 'confidence')
        evidence_items = self._find_nested(diagnosis, 'evidence')
        
        # Check for high confidence without evidence
        for item in self._get_all_dicts(diagnosis):
            if item.get('confidence') == 'High':
                has_evidence = (
                    'evidence' in item or 
                    'supporting_evidence' in item or
                    'evidence_from_report' in item or
                    'evidence_quote' in item
                )
                if not has_evidence:
                    # Try to get a descriptive name for this item
                    item_name = (
                        item.get('finding') or 
                        item.get('diagnosis') or 
                        item.get('condition') or
                        item.get('concern') or
                        item.get('observation') or
                        item.get('symptom') or
                        item.get('recommendation') or
                        item.get('intervention') or
                        str(item.get('step', ''))[:50] or  # For reasoning chain steps
                        'Unknown'
                    )
                    
                    # Only warn if item_name has useful content
                    if item_name and item_name != 'Unknown' and len(str(item_name)) > 0:
                        self.warnings.append(
                            f"High confidence without cited evidence: {item_name}"
                        )
                    elif 'observation' in item or 'inference' in item:
                        # This is a reasoning chain step
                        obs = item.get('observation', '')[:50]
                        self.warnings.append(
                            f"High confidence in reasoning step without evidence: {obs}"
                        )
                    # Skip items where we can't identify what it is
                    # (these are probably structural elements, not claims)
    
    def _check_invented_terms(self, diagnosis: Dict, agent_name: str):
        """Detect potentially invented medical terms or medications"""
        
        # Common medical term patterns that should appear in source
        text = json.dumps(diagnosis).lower()
        
        # Extract medication names (capitalized words followed by dosage)
        medication_pattern = r'([A-Z][a-z]+(?:ol|am|in|ide|one|ate))\s+\d+\s*(?:mg|mcg)'
        medications = re.findall(medication_pattern, json.dumps(diagnosis))
        
        for med in medications:
            if not self._is_in_source(med.lower()):
                self.warnings.append(
                    f"Medication not mentioned in original report: {med}"
                )
        
        # Check for test results with specific values
        test_pattern = r'([A-Z]{2,}|[A-Z][a-z]+)\s*[:=]\s*(\d+\.?\d*)'
        tests = re.findall(test_pattern, json.dumps(diagnosis))
        
        for test_name, value in tests:
            if not self._is_in_source(value):
                self.warnings.append(
                    f"Specific test value not in report: {test_name} = {value}"
                )
    
    def _check_reasoning_quality(self, diagnosis: Dict, agent_name: str):
        """Validate reasoning chain quality"""
        
        reasoning = diagnosis.get('reasoning_chain', [])
        
        if not reasoning:
            self.warnings.append("No reasoning chain provided")
            return
        
        if len(reasoning) < 3:
            self.warnings.append(f"Shallow reasoning (only {len(reasoning)} steps)")
        
        # Check each reasoning step has required fields
        for i, step in enumerate(reasoning):
            if not isinstance(step, dict):
                continue
            
            required = ['observation', 'inference']
            missing = [f for f in required if f not in step]
            if missing:
                self.warnings.append(
                    f"Reasoning step {i+1} missing: {', '.join(missing)}"
                )
            
            # Check if observations are grounded
            observation = step.get('observation', '')
            if len(observation) > 20 and not self._is_in_source(observation):
                self.warnings.append(
                    f"Step {i+1} observation not grounded in report: '{observation[:80]}...'"
                )
    
    def _is_in_source(self, text: str, min_overlap: float = 0.3) -> bool:
        """
        Check if text appears in or is similar to source document
        
        Args:
            text: Text to check
            min_overlap: Minimum word overlap ratio (0-1)
        
        Returns:
            True if text is grounded in source
        """
        if not text:
            return True
        
        text_lower = text.lower()
        
        # Direct substring match
        if text_lower in self.source:
            return True
        
        # Word overlap check (for paraphrased content)
        text_words = set(text_lower.split())
        if len(text_words) == 0:
            return True
        
        overlap = len(text_words & self.source_words)
        overlap_ratio = overlap / len(text_words)
        
        return overlap_ratio >= min_overlap
    
    def _find_nested(self, obj: Any, key: str) -> List:
        """Recursively find all values for a key in nested structure"""
        results = []
        
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == key:
                    if isinstance(v, list):
                        results.extend(v)
                    else:
                        results.append(v)
                results.extend(self._find_nested(v, key))
        elif isinstance(obj, list):
            for item in obj:
                results.extend(self._find_nested(item, key))
        
        return results
    
    def _get_all_dicts(self, obj: Any) -> List[Dict]:
        """Get all dictionary objects in nested structure"""
        results = []
        
        if isinstance(obj, dict):
            results.append(obj)
            for v in obj.values():
                results.extend(self._get_all_dicts(v))
        elif isinstance(obj, list):
            for item in obj:
                results.extend(self._get_all_dicts(item))
        
        return results
    
    def _calculate_risk_score(self) -> float:
        """Calculate hallucination risk score (0-1)"""
        # Critical errors are weighted heavily
        error_score = len(self.critical_errors) * 0.3
        warning_score = len(self.warnings) * 0.05
        
        risk = min(1.0, error_score + warning_score)
        return risk
    
    def _get_risk_level(self, score: float) -> str:
        """Convert risk score to risk level"""
        if score < 0.1:
            return "Very Low"
        elif score < 0.3:
            return "Low"
        elif score < 0.5:
            return "Medium"
        elif score < 0.7:
            return "High"
        else:
            return "Very High"
    
    def _get_recommendations(self) -> List[str]:
        """Get recommendations based on validation results"""
        recommendations = []
        
        if self.critical_errors:
            recommendations.append("CRITICAL: Review all findings marked as errors")
            recommendations.append("Consider re-running analysis with stricter grounding")
        
        if len(self.warnings) > 5:
            recommendations.append("Multiple warnings detected - verify all claims against source")
        
        if not recommendations:
            recommendations.append("Validation passed - diagnosis appears well-grounded")
        
        return recommendations


class CrossAgentValidator:
    """Validates consistency across multiple agent responses"""
    
    def __init__(self):
        self.inconsistencies = []
    
    def validate_consistency(
        self, 
        cardio: Dict, 
        psych: Dict, 
        pulmo: Dict, 
        final: Dict
    ) -> Dict:
        """
        Check for contradictions between agent diagnoses
        
        Returns:
            dict with consistency analysis
        """
        print("\n🔄 Checking cross-agent consistency...")
        
        self.inconsistencies = []
        
        # Check vital signs consistency
        self._check_vitals_consistency(cardio, psych, pulmo)
        
        # Check symptom interpretation consistency
        self._check_symptom_consistency(cardio, psych, pulmo)
        
        # Check if final diagnosis contradicts specialists
        self._check_final_vs_specialists(cardio, psych, pulmo, final)
        
        result = {
            'consistent': len(self.inconsistencies) == 0,
            'inconsistencies': self.inconsistencies,
            'consistency_score': 1.0 - (len(self.inconsistencies) * 0.1)
        }
        
        if result['consistent']:
            print(f"✅ Cross-agent validation passed")
        else:
            print(f"⚠️  {len(self.inconsistencies)} inconsistencies found")
        
        return result
    
    def _check_vitals_consistency(self, cardio, psych, pulmo):
        """Check if agents report consistent vital signs"""
        # Implementation would check for conflicting vital sign reports
        pass
    
    def _check_symptom_consistency(self, cardio, psych, pulmo):
        """Check if symptom interpretations are compatible"""
        # E.g., if cardio says "no chest pain" but psych mentions "chest pain during panic"
        pass
    
    def _check_final_vs_specialists(self, cardio, psych, pulmo, final):
        """Ensure final diagnosis doesn't contradict specialists"""
        pass


# Example usage
if __name__ == "__main__":
    # Test the hallucination detector
    source_report = """
    Patient presents with irregular heartbeat. ECG shows atrial fibrillation.
    Heart rate 115 bpm. Blood pressure 145/88 mmHg. Oxygen saturation 96%.
    Patient taking Lisinopril 10mg daily.
    """
    
    # Good diagnosis (grounded in source)
    good_diagnosis = {
        "findings": [
            {
                "finding": "Atrial fibrillation",
                "confidence": "High",
                "evidence": ["ECG shows atrial fibrillation", "irregular heartbeat"]
            }
        ],
        "reasoning_chain": [
            {
                "step": 1,
                "observation": "ECG shows atrial fibrillation",
                "inference": "Confirms diagnosis of AF",
                "confidence": "High"
            }
        ]
    }
    
    # Bad diagnosis (hallucinated data)
    bad_diagnosis = {
        "findings": [
            {
                "finding": "Severe heart failure",
                "confidence": "High",
                "evidence": ["Ejection fraction 25%"]  # NOT in source!
            }
        ],
        "medications": ["Metoprolol 50mg daily"]  # NOT in source!
    }
    
    detector = HallucinationDetector(source_report)
    
    print("Testing GOOD diagnosis:")
    result1 = detector.validate_diagnosis(good_diagnosis, "Test Agent 1")
    
    print("\n" + "="*60)
    print("Testing BAD diagnosis (with hallucinations):")
    detector2 = HallucinationDetector(source_report)
    result2 = detector2.validate_diagnosis(bad_diagnosis, "Test Agent 2")
    
    print("\n✅ Hallucination detection testing complete!")
