from typing import Dict, List, Set, Tuple, Optional
import re
from dataclasses import dataclass
from enum import Enum

class NamespaceType(Enum):
    LEGAL = "legal"
    PROJECT = "project"
    GENERAL = "general"

class RegulationType(Enum):
    GDPR = "GDPR"
    AI_ACT = "AI Act"
    DATA_ACT = "Data Act"

@dataclass
class ProjectInfo:
    name: str
    patterns: List[str]
    phonetic_variations: List[str]
    common_misrecognitions: List[str]

@dataclass
class NamespaceMatch:
    namespace: NamespaceType
    confidence: float
    matched_patterns: List[str]
    project_id: Optional[str] = None
    project_name: Optional[str] = None
    regulation_type: Optional[RegulationType] = None

class NamespaceConfig:
    def __init__(self):
        # Project definitie
        self.projects = {
            "skylla": ProjectInfo(
                name="Skylla",
                patterns=[
                    r'\bskylla\b',
                    r'\bproject\s+skylla\b',
                    r'\bskylla\s+project\b'
                ],
                phonetic_variations=[
                    r'\bskilla\b',
                    r'\bskyla\b',
                    r'\bskyler\b',
                    r'\bskill\w*\b'
                ],
                common_misrecognitions=[
                    r'\bscilla\b',
                    r'\bstiller\b',
                    r'\bschiller\b',
                    r'\bskill\s+a\b',
                    r'\bskill\s+ah\b',
                    r'\bcilla\b'
                ]
            )
        }

        # Legal patterns - AI specifiek
        self.legal_patterns = {
            RegulationType.GDPR: [
                (re.compile(r'\b(gdpr|g\s*d\s*p\s*r|privacy|avg)\w*\b', re.IGNORECASE), 1.0),
                (re.compile(r'\b(persoon\w*|data\w*|gegevens)\b', re.IGNORECASE), 0.8),
                (re.compile(r'\b(verwerking|verwerker|verwerkings\w*)\b', re.IGNORECASE), 0.9),
                (re.compile(r'\b(toestemming|consent|grondslag)\b', re.IGNORECASE), 0.8),
                (re.compile(r'\b(dpia|data\s*protection|impact\s*analysis)\b', re.IGNORECASE), 0.9)
            ],
            RegulationType.AI_ACT: [
                (re.compile(r'\b(ai\s*act|ai\s*wet|ai\s*regulation)\b', re.IGNORECASE), 1.0),
                (re.compile(r'\b(hoog\s*risico|high\s*risk)\b', re.IGNORECASE), 0.9),
                (re.compile(r'\b(ai\s*systeem|ai\s*system)\b', re.IGNORECASE), 0.9),
                (re.compile(r'\b(ai\s*compliance|conformiteit)\b', re.IGNORECASE), 0.8),
                (re.compile(r'\b(classificatie|classification)\b', re.IGNORECASE), 0.7),
                # Verhoogde gewichten voor sandbox gerelateerde patronen
                (re.compile(r'\b(regulatory\s*sandbox|sandbox|test\s*omgeving)\b', re.IGNORECASE), 1.0),
                (re.compile(r'\b(innovatie\s*hub|innovation\s*hub)\b', re.IGNORECASE), 0.9),
                (re.compile(r'\b(test\s*faciliteit|testing\s*facility)\b', re.IGNORECASE), 0.9),
                (re.compile(r'\b(regulatory|regulation|regulering|regelgeving)\b', re.IGNORECASE), 0.8),
                # Nieuwe patronen voor consumer/benefits context
                (re.compile(r'\b(consumer|consument|gebruiker)\b', re.IGNORECASE), 0.8),
                (re.compile(r'\b(benefit|voordeel|advantage)\b', re.IGNORECASE), 0.8)
            ],
            RegulationType.DATA_ACT: [
                (re.compile(r'\b(data\s*act|data\s*wet)\b', re.IGNORECASE), 1.0),
                (re.compile(r'\b(data\s*delen|data\s*sharing)\b', re.IGNORECASE), 0.9),
                (re.compile(r'\b(data\s*access|toegang)\b', re.IGNORECASE), 0.8)
            ]
        }

        # AI context patterns
        self.ai_context_patterns = [
            (re.compile(r'\b(ai|a\s*i|kunstmatige\s*intelligentie)\b', re.IGNORECASE), 0.9),  # Verhoogd gewicht
            (re.compile(r'\b(machine\s*learning|m\s*l)\b', re.IGNORECASE), 0.8),
            (re.compile(r'\b(model|training|dataset)\b', re.IGNORECASE), 0.7),
            (re.compile(r'\b(neural|deep\s*learning)\b', re.IGNORECASE), 0.7),
            (re.compile(r'\b(algoritme|algorithm)\b', re.IGNORECASE), 0.7),
            # Nieuwe patronen voor regulatory context
            (re.compile(r'\b(regulatory|regulation|regulering)\b', re.IGNORECASE), 0.8),
            (re.compile(r'\b(compliance|conformiteit)\b', re.IGNORECASE), 0.8)
        ]

        # Compile project patterns
        self.project_patterns: List[Tuple[re.Pattern, float, str]] = []
        for project_id, info in self.projects.items():
            for pattern in info.patterns:
                self.project_patterns.append(
                    (re.compile(pattern, re.IGNORECASE), 1.0, project_id)
                )
            for var in info.phonetic_variations:
                self.project_patterns.append(
                    (re.compile(var, re.IGNORECASE), 0.9, project_id)
                )
            for misrec in info.common_misrecognitions:
                self.project_patterns.append(
                    (re.compile(misrec, re.IGNORECASE), 0.8, project_id)
                )

    def normalize_query(self, query: str) -> str:
        """Normaliseert spraak-naar-tekst output"""
        query = re.sub(r'\s+', ' ', query)
        query = query.lower().strip()
        query = re.sub(r'[,.!?]', '', query)
        return query

    def detect_ai_context(self, query: str) -> float:
        """Detecteert of de query AI-gerelateerd is"""
        max_confidence = 0.0
        matched_patterns = []
        for pattern, weight in self.ai_context_patterns:
            if pattern.search(query):
                max_confidence = max(max_confidence, weight)
                matched_patterns.append(f"AI pattern: {pattern.pattern} (confidence: {weight})")
        
        print(f"AI Context - Query: {query}")
        print(f"AI Context - Max Confidence: {max_confidence}")
        print(f"AI Context - Matched Patterns: {matched_patterns}")
        return max_confidence

    def detect_regulation(self, query: str) -> Tuple[Optional[RegulationType], float]:
        """Detecteert specifieke regulering in de query"""
        best_match = (None, 0.0)
        matched_patterns = []
        
        for reg_type, patterns in self.legal_patterns.items():
            for pattern, weight in patterns:
                if pattern.search(query):
                    matched_patterns.append(f"Regulation pattern: {pattern.pattern} (type: {reg_type}, confidence: {weight})")
                    if weight > best_match[1]:
                        best_match = (reg_type, weight)
        
        print(f"Regulation - Query: {query}")
        print(f"Regulation - Best Match: {best_match}")
        print(f"Regulation - Matched Patterns: {matched_patterns}")
        return best_match

    def detect_project(self, query: str) -> Tuple[Optional[str], float, List[str]]:
        """Detecteert project met voice-specifieke matching"""
        matches = []
        best_confidence = 0.0
        project_id = None
        
        for pattern, confidence, pid in self.project_patterns:
            if match := pattern.search(query):
                matches.append(match.group())
                if confidence > best_confidence:
                    best_confidence = confidence
                    project_id = pid
        
        return (project_id, best_confidence, matches)

    def detect_namespace(self, query: str) -> NamespaceMatch:
        """Hoofdfunctie voor namespace detectie"""
        normalized_query = self.normalize_query(query)
        print(f"\nNamespace Detection - Original Query: {query}")
        print(f"Namespace Detection - Normalized Query: {normalized_query}")
        
        # Check project eerst
        project_id, project_confidence, project_matches = self.detect_project(normalized_query)
        print(f"Project Detection - ID: {project_id}, Confidence: {project_confidence}")
        
        # Check regulering en AI context parallel
        reg_type, reg_confidence = self.detect_regulation(normalized_query)
        ai_confidence = self.detect_ai_context(normalized_query)
        print(f"Combined Detection - AI Confidence: {ai_confidence}, Regulation Confidence: {reg_confidence}")
        
        # Als er een sterke regulering match is OF een combinatie van AI context en regulering
        is_legal = reg_confidence > 0.7 or (ai_confidence > 0.6 and reg_confidence > 0.6)
        print(f"Legal Namespace Check - Is Legal: {is_legal}")
        
        if is_legal:
            confidence = max(reg_confidence, ai_confidence)
            matched_patterns = []
            if ai_confidence > 0.6:
                matched_patterns.append("AI Context")
            if reg_type:
                matched_patterns.append(str(reg_type))
            
            print(f"Selected Legal Namespace - Confidence: {confidence}, Patterns: {matched_patterns}")
            return NamespaceMatch(
                namespace=NamespaceType.LEGAL,
                confidence=confidence,
                matched_patterns=matched_patterns,
                regulation_type=reg_type
            )

        # Als project gevonden, return project namespace
        if project_id and project_confidence > 0.7:
            print(f"Selected Project Namespace - Project: {project_id}, Confidence: {project_confidence}")
            return NamespaceMatch(
                namespace=NamespaceType.PROJECT,
                confidence=project_confidence,
                matched_patterns=project_matches,
                project_id=project_id,
                project_name=self.projects[project_id].name
            )

        # Fallback naar general
        print("Fallback to General Namespace")
        return NamespaceMatch(
            namespace=NamespaceType.GENERAL,
            confidence=0.5,
            matched_patterns=[]
        )

# Test functie
def main():
    config = NamespaceConfig()
    
    test_queries = [
        "Wat zijn de GDPR vereisten voor het AI model van Skylla?",
        "Voldoet project skill a aan de AI Act?",
        "Kunnen we deze training data delen volgens de Data Act?",
        "G D P R compliance voor het Schiller project",
        "Is ons machine learning model compliant met de nieuwe AI wet?",
        "Wat zijn de deadlines voor project Skylla?",
        "Hoe zit het met data protection voor de AI in Scilla?"
    ]
    
    for query in test_queries:
        result = config.detect_namespace(query)
        print(f"\nQuery: {query}")
        print(f"Normalized: {config.normalize_query(query)}")
        print(f"Namespace: {result.namespace.value}")
        print(f"Confidence: {result.confidence:.2f}")
        if result.project_name:
            print(f"Project: {result.project_name}")
        if result.regulation_type:
            print(f"Regulation: {result.regulation_type.value}")
        print(f"Matched Patterns: {result.matched_patterns}")

if __name__ == "__main__":
    main()
