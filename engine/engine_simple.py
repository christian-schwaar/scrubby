#!/usr/bin/env python3
"""
Tauri Sanitizer Engine - Simplified Version
Funktioniert ohne problematische Dependencies (numpy, spacy, opencv)
Verwendet nur Standard-Bibliotheken und einfache Regex-Patterns
"""

import argparse
import sys
import os
import json
import logging
import shutil
import time
import re
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
from pdfminer.high_level import extract_text, extract_pages
from pdfminer.layout import LTTextContainer, LTTextLine, LTChar
import fitz  # PyMuPDF

class SimpleSanitizerEngine:
    """Vereinfachte Pseudonymisierungs-Engine ohne externe ML-Dependencies"""
    
    def __init__(self):
        self.setup_logging()
        self.setup_patterns()
        self.entity_counters = {}
        self.chunk_size = 2 * 1024 * 1024  # 2MB chunks
        
    def setup_logging(self):
        """Logging konfigurieren"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(sys.stderr)]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_patterns(self):
        """Regex-Patterns für verschiedene Entitäten (sprachspezifisch)."""
        self.patterns_de = {
            'PERSON': [
                # Titel + Nachname(+ Zweitname)
                r'\b(?:Herr|Frau|Dr\.?|Prof\.?)\s+[A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)?\b',
                # Kontextbasiert: "für <Vorname Nachname>" (Name in Gruppe 1)
                r'(?i)\bfür\s+([A-ZÄÖÜ][a-zäöüß]+\s+[A-ZÄÖÜ][a-zäöüß]+)\b',
                # Mehrteilig, inkl. Zusätze (von/zu/van/de/...) und Bindestriche
                r'\b(?:[A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+){0,2}\s+)?(?:von|zu|zum|zur|vom|van|de|der|den|di|du|del|della|da|la|le)?(?:\s+(?:der|den|de|la|le))?\s*[A-ZÄÖÜ][a-zäöüß]+(?:-[A-ZÄÖÜ][a-zäöüß]+)?\b',
            ],
            'DOB': [
                # "geb. 26.02.2005" oder "Geburtsdatum: 26.02.2005" (Datum in Gruppe 1)
                r'(?i)\bgeb\.?\s*:?\s*(\d{1,2}[.]\d{1,2}[.]\d{2,4})\b',
                r'(?i)\bgeburtsdatum\s*:?\s*(\d{1,2}[.]\d{1,2}[.]\d{2,4})\b',
            ],
        }
        self.patterns_en = {
            'PERSON': [
                r'\b(?:Mr\.?|Ms\.?|Mrs\.?|Dr\.?|Prof\.?)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b',
                r'(?i)\bfor\s+([A-Z][a-z]+\s+[A-Z][a-z]+)\b',
            ],
            'DOB': [
                r'(?i)\b(?:dob|date\s+of\s+birth|born)\s*:?\s*(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})\b',
            ],
        }

    def get_patterns(self, language: str) -> Dict[str, List[str]]:
        return self.patterns_de if (language or 'de').lower().startswith('de') else self.patterns_en
    
    def detect_type(self, file_path: Path) -> str:
        """Robuste Dateityp-Erkennung"""
        suffix = file_path.suffix.lower()
        
        if suffix == '.pdf':
            return 'pdf'
        elif suffix in ['.png', '.jpg', '.jpeg']:
            return 'image'
        elif suffix == '.txt':
            return 'text'
        elif suffix == '.md':
            return 'markdown'
        elif suffix == '.json':
            return 'json'
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
    
    def safe_mkdirs(self, path: Path) -> None:
        """Sichere Verzeichniserstellung"""
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise RuntimeError(f"Failed to create directory {path}: {e}")
    
    def wipe_dir(self, path: Path) -> None:
        """Verzeichnis komplett leeren"""
        if path.exists():
            try:
                shutil.rmtree(path)
                self.safe_mkdirs(path)
            except Exception as e:
                self.logger.warning(f"Could not wipe directory {path}: {e}")
        else:
            self.safe_mkdirs(path)
    
    def get_timestamp_dir(self, base_path: Path) -> Path:
        """Timestamp-basiertes Verzeichnis erstellen"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return base_path / timestamp
    
    def reset_entity_counters(self):
        """Entity-Zähler zurücksetzen"""
        self.entity_counters = {}
    
    def get_entity_replacement(self, entity_type: str) -> str:
        """Erstelle TYPE_### Ersatz"""
        if entity_type not in self.entity_counters:
            self.entity_counters[entity_type] = 0
        self.entity_counters[entity_type] += 1
        return f"{entity_type}_{self.entity_counters[entity_type]:03d}"
    
    def find_entities(self, text: str, language: str) -> List[Dict[str, Any]]:
        """Finde Entitäten mit Regex-Patterns"""
        results = []
        patterns_map = self.get_patterns(language)
        for entity_type, patterns in patterns_map.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    # Falls wir via Kontext arbeiten (Gruppen), extrahiere die Gruppe 1
                    matched_text = match.group(1) if match.lastindex else match.group()
                    results.append({
                        'entity_type': entity_type,
                        'text': matched_text,
                        'start': match.start(),
                        'end': match.end(),
                        'score': 0.95  # Engere Patterns -> höhere Zuverlässigkeit
                    })
        
        # Duplikate entfernen (überlappende Matches)
        results = self.remove_overlapping_matches(results)
        return results
    
    def remove_overlapping_matches(self, matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Entferne überlappende Matches (behalte den mit höchster Score)"""
        if not matches:
            return matches
        
        # Sortiere nach Start-Position
        matches.sort(key=lambda x: x['start'])
        
        filtered = [matches[0]]
        for match in matches[1:]:
            last_match = filtered[-1]
            
            # Prüfe auf Überlappung
            if match['start'] < last_match['end']:
                # Überlappung gefunden, behalte den mit höherer Score
                if match['score'] > last_match['score']:
                    filtered[-1] = match
            else:
                filtered.append(match)
        
        return filtered
    
    def anonymize_text(self, text: str, entities: List[Dict[str, Any]]) -> str:
        """Text mit gefundenen Entitäten anonymisieren"""
        if not entities:
            return text
        
        # Sortiere nach Position (rückwärts, um Indizes nicht zu verschieben)
        entities.sort(key=lambda x: x['start'], reverse=True)
        
        anonymized_text = text
        for entity in entities:
            entity_type = entity['entity_type']
            replacement = self.get_entity_replacement(entity_type)
            
            anonymized_text = (
                anonymized_text[:entity['start']] + 
                replacement + 
                anonymized_text[entity['end']:]
            )
        
        return anonymized_text
    
    def process_text(self, input_path: Path, output_path: Path, language: str) -> Dict[str, Any]:
        """Textdatei-Verarbeitung mit Chunking"""
        start_time = time.time()
        entities_found = {}
        
        try:
            # Datei lesen
            with open(input_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Chunking für große Dateien
            chunks = self.chunk_text(content)
            processed_chunks = []
            
            for chunk in chunks:
                # Entitäten finden
                entities = self.find_entities(chunk, language)
                
                # Zähler aktualisieren
                for entity in entities:
                    entity_type = entity['entity_type']
                    if entity_type not in entities_found:
                        entities_found[entity_type] = 0
                    entities_found[entity_type] += 1
                
                # Anonymisieren
                anonymized_chunk = self.anonymize_text(chunk, entities)
                processed_chunks.append(anonymized_chunk)
            
            # Zusammensetzen und speichern
            processed_content = ''.join(processed_chunks)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(processed_content)
            
            processing_time = int((time.time() - start_time) * 1000)
            
            return {
                "ok": True,
                "output": str(output_path.absolute()),
                "summary": {
                    "entities": entities_found,
                    "type": "text",
                    "timeMs": processing_time,
                    "chunks": len(chunks)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Text processing failed: {e}")
            return {"ok": False, "error": f"Text processing failed: {str(e)}"}
    
    def process_json(self, input_path: Path, output_path: Path, language: str) -> Dict[str, Any]:
        """JSON-Verarbeitung mit rekursiver Textfeld-Behandlung"""
        start_time = time.time()
        entities_found = {}
        
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:
                    return {"ok": False, "error": "Empty JSON file"}
                data = json.loads(content)
            
            # Rekursiv alle String-Felder verarbeiten
            processed_data = self.process_json_recursive(data, language, entities_found)
            
            # Speichern
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=2)
            
            processing_time = int((time.time() - start_time) * 1000)
            
            return {
                "ok": True,
                "output": str(output_path.absolute()),
                "summary": {
                    "entities": entities_found,
                    "type": "json",
                    "timeMs": processing_time
                }
            }
            
        except Exception as e:
            self.logger.error(f"JSON processing failed: {e}")
            return {"ok": False, "error": f"JSON processing failed: {str(e)}"}
    
    def process_json_recursive(self, obj: Any, language: str, entities_found: Dict[str, int]) -> Any:
        """Rekursive JSON-Verarbeitung"""
        if isinstance(obj, dict):
            return {key: self.process_json_recursive(value, language, entities_found) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.process_json_recursive(item, language, entities_found) for item in obj]
        elif isinstance(obj, str):
            # String-Felder mit Regex verarbeiten
            entities = self.find_entities(obj, language)
            
            # Zähler aktualisieren
            for entity in entities:
                entity_type = entity['entity_type']
                if entity_type not in entities_found:
                    entities_found[entity_type] = 0
                entities_found[entity_type] += 1
            
            return self.anonymize_text(obj, entities)
        else:
            return obj
    
    def chunk_text(self, text: str) -> List[str]:
        """Text in Chunks aufteilen"""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Versuche bei Wortgrenze zu teilen
            if end < len(text):
                last_space = text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space
            
            chunks.append(text[start:end])
            start = end
        
        return chunks
    
    def process_pdf(self, input_path: Path, output_path: Path, language: str, ocr_enabled: bool) -> Dict[str, Any]:
        """PDF-Verarbeitung: Text mit pdfminer.six analysieren und PII hart schwärzen (Redaction)."""
        start_time = time.time()
        entities_found: Dict[str, int] = {}
        try:
            # 1) Analyse: Text + Zeichen-Bounding-Boxes pro Seite sammeln
            redactions_per_page: Dict[int, List[Dict[str, Any]]] = {}

            for page_index, page_layout in enumerate(extract_pages(str(input_path))):
                for element in page_layout:
                    if not isinstance(element, LTTextContainer):
                        continue

                    text_to_anonymize = element.get_text() or ""
                    if not text_to_anonymize.strip():
                        continue

                    # Zeichenliste in Anzeige-Reihenfolge holen
                    characters: List[LTChar] = []
                    for text_line in filter(lambda t: isinstance(t, LTTextLine), element):
                        for ch in filter(lambda t: isinstance(t, LTChar), text_line):
                            characters.append(ch)

                    # 2) Entitäten im Container-Text finden (Regex-basiert)
                    entities = self.find_entities(text_to_anonymize, language)
                    for ent in entities:
                        ent_type = ent['entity_type']
                        start = ent['start']
                        end = ent['end']

                        # Zähler aktualisieren
                        if ent_type not in entities_found:
                            entities_found[ent_type] = 0
                        entities_found[ent_type] += 1

                        # Zeichenbereich in Bounding-Box umwandeln
                        if start < 0 or end <= start or end > len(characters):
                            # Fallback: überspringen falls Mapping nicht möglich
                            continue

                        # Suche die Text-Instanzen direkt im PDF (robuster)
                        # PyMuPDF-Koordinaten werden später verwendet; hier nur Marker
                        x0 = y0 = 0.0
                        x1 = y1 = 0.0

                        if page_index not in redactions_per_page:
                            redactions_per_page[page_index] = []
                        redactions_per_page[page_index].append({
                            'text': ent['text'],
                            'type': ent_type,
                        })

            # 3) Redactions mit PyMuPDF anwenden (harte Schwärzung)
            doc = fitz.open(str(input_path))
            for page_index, items in redactions_per_page.items():
                if page_index >= len(doc):
                    continue
                page = doc[page_index]
                for ann in items:
                    text = ann.get('text', '').strip()
                    if not text:
                        continue
                    # Alle Vorkommen des Textes auf der Seite finden
                    for inst in page.search_for(text):
                        page.add_redact_annot(inst, fill=(0, 0, 0))
                # Anwenden der Redactions entfernt Inhalte endgültig
                page.apply_redactions()

            # Speichern als neues PDF
            doc.save(str(output_path), garbage=4, deflate=True, clean=True)
            doc.close()

            processing_time = int((time.time() - start_time) * 1000)
            return {
                "ok": True,
                "output": str(output_path.absolute()),
                "summary": {
                    "entities": entities_found,
                    "type": "pdf_highlight",
                    "timeMs": processing_time
                }
            }
        except Exception as e:
            self.logger.error(f"PDF processing failed: {e}")
            return {"ok": False, "error": f"PDF processing failed: {str(e)}"}
    
    def process_image(self, input_path: Path, output_path: Path, language: str, ocr_enabled: bool) -> Dict[str, Any]:
        """Bildverarbeitung - vereinfachte Version"""
        return {
            "ok": False,
            "error": "Image processing requires OpenCV and Tesseract. Please install the full version with: pip install -r requirements.txt"
        }
    
    def process_file(self, input_path: str, output_dir: str, mode: str = "pseudo", 
                    language: str = "de", ocr_enabled: bool = False) -> Dict[str, Any]:
        """Hauptverarbeitungsfunktion"""
        try:
            input_file = Path(input_path)
            output_dir_path = Path(output_dir)
            
            # Validierung
            if not input_file.exists():
                return {"ok": False, "error": f"Input file not found: {input_path}"}
            
            # Verzeichnisse vorbereiten
            self.wipe_dir(output_dir_path)
            timestamp_dir = self.get_timestamp_dir(output_dir_path)
            self.safe_mkdirs(timestamp_dir)
            
            # Input nach data/input kopieren
            input_timestamp_dir = self.get_timestamp_dir(Path("data/input"))
            self.safe_mkdirs(input_timestamp_dir)
            input_copy_path = input_timestamp_dir / input_file.name
            shutil.copy2(input_file, input_copy_path)
            
            # Dateityp erkennen
            file_type = self.detect_type(input_file)
            
            # Output-Pfad generieren
            output_filename = f"{input_file.stem}_pseudo{input_file.suffix}"
            output_path = timestamp_dir / output_filename
            
            # Entity-Zähler zurücksetzen
            self.reset_entity_counters()
            
            # Verarbeitung basierend auf Dateityp
            if file_type == "pdf":
                result = self.process_pdf(input_copy_path, output_path, language, ocr_enabled)
            elif file_type == "image":
                result = self.process_image(input_copy_path, output_path, language, ocr_enabled)
            elif file_type in ["text", "markdown"]:
                result = self.process_text(input_copy_path, output_path, language)
            elif file_type == "json":
                result = self.process_json(input_copy_path, output_path, language)
            else:
                return {"ok": False, "error": f"Unsupported file type: {file_type}"}
            
            # Original nach erfolgreicher Verarbeitung löschen
            if result.get("ok"):
                try:
                    input_copy_path.unlink()
                    self.logger.info(f"Original file deleted: {input_copy_path}")
                except Exception as e:
                    self.logger.warning(f"Could not delete original file: {e}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Processing error: {str(e)}")
            return {"ok": False, "error": str(e)}

def main():
    """CLI Entry Point"""
    parser = argparse.ArgumentParser(description="Tauri Sanitizer Engine (Simplified)")
    parser.add_argument("--input", required=True, help="Input file path")
    parser.add_argument("--mode", default="pseudo", choices=["pseudo", "anon"], help="Processing mode")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--language", default="de", choices=["de", "en"], help="Language for analysis")
    parser.add_argument("--ocr", action="store_true", help="Enable OCR for images")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # OCR aus Umgebungsvariable
    ocr_enabled = args.ocr or os.getenv("OCR", "0") == "1"
    
    # Logging Level setzen
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Engine initialisieren und verarbeiten
        engine = SimpleSanitizerEngine()
        result = engine.process_file(
            input_path=args.input,
            output_dir=args.outdir,
            mode=args.mode,
            language=args.language,
            ocr_enabled=ocr_enabled
        )
        
        # Ergebnis als JSON ausgeben
        print(json.dumps(result, ensure_ascii=False), flush=True)
        
        # Exit Code basierend auf Status
        sys.exit(0 if result.get("ok") else 1)
        
    except Exception as e:
        error_result = {"ok": False, "error": str(e)}
        print(json.dumps(error_result, ensure_ascii=False), flush=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
