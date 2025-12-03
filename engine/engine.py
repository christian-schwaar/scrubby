#!/usr/bin/env python3
"""
Tauri Sanitizer Engine
Produktionsnahe Python-Engine für Presidio-basierte Pseudonymisierung
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
from typing import Optional, Dict, Any, List, Tuple, Union
from datetime import datetime
import tempfile

# Core dependencies
try:
    from presidio_analyzer import (
        AnalyzerEngine,
        RecognizerResult,
        Pattern,
        PatternRecognizer,
        RecognizerRegistry,
    )
    from presidio_analyzer.predefined_recognizers.email_recognizer import EmailRecognizer
    from presidio_analyzer.predefined_recognizers.phone_recognizer import PhoneRecognizer
    from presidio_analyzer.predefined_recognizers.iban_recognizer import IbanRecognizer
    from presidio_analyzer.predefined_recognizers.ip_recognizer import IpRecognizer
    from presidio_analyzer.nlp_engine import NlpArtifacts, NlpEngine
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig
    import spacy
    import fitz  # PyMuPDF
    import pytesseract
    from PIL import Image, ExifTags
    import piexif
    import cv2
    import numpy as np
    import regex
    from tqdm import tqdm
except ImportError as e:
    print(f"Missing dependency: {e}", file=sys.stderr)
    print("Please install: pip install -r requirements.txt", file=sys.stderr)
    sys.exit(1)

class SanitizerEngine:
    """Produktionsnahe Pseudonymisierungs-Engine"""
    
    def __init__(self):
        self.setup_logging()
        self.setup_presidio()
        self.entity_counters = {}
        self.chunk_size = 2 * 1024 * 1024  # 2MB chunks
        self.max_file_size_bytes = 100 * 1024 * 1024  # 100 MB limit
        
    def setup_logging(self):
        """Logging konfigurieren"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(sys.stderr)]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_presidio(self):
        """Presidio-Engines initialisieren"""
        try:
            # Anonymizer mit TYPE_### Zählern
            self.anonymizer = AnonymizerEngine()
            
            # Regex-Recognizer für zusätzliche Entitäten
            self.setup_regex_recognizers()
            
            # Eigene Registry mit PatternRecognizers (spaCy-frei)
            registry = RecognizerRegistry()
            # Built-ins: EMAIL_ADDRESS, PHONE_NUMBER, IBAN_CODE, IP_ADDRESS
            registry.add_recognizer(EmailRecognizer(supported_language="en", supported_entity="EMAIL_ADDRESS"))
            registry.add_recognizer(EmailRecognizer(supported_language="de", supported_entity="EMAIL_ADDRESS"))
            registry.add_recognizer(PhoneRecognizer(supported_language="en"))
            registry.add_recognizer(PhoneRecognizer(supported_language="de"))
            registry.add_recognizer(IbanRecognizer(supported_language="en", supported_entity="IBAN_CODE"))
            registry.add_recognizer(IbanRecognizer(supported_language="de", supported_entity="IBAN_CODE"))
            registry.add_recognizer(IpRecognizer(supported_language="en", supported_entity="IP_ADDRESS"))
            registry.add_recognizer(IpRecognizer(supported_language="de", supported_entity="IP_ADDRESS"))

            # Custom regex-only for ADDRESS
            def add_pattern(entity: str, pattern_regex: str, score: float = 0.8):
                registry.add_recognizer(PatternRecognizer(
                    supported_entity=entity,
                    patterns=[Pattern(name=f"{entity.lower()}_de", regex=pattern_regex, score=score)],
                    supported_language="de",
                ))
                registry.add_recognizer(PatternRecognizer(
                    supported_entity=entity,
                    patterns=[Pattern(name=f"{entity.lower()}_en", regex=pattern_regex, score=score)],
                    supported_language="en",
                ))

            # ADDRESS immer als Regex-Pattern
            add_pattern("ADDRESS", self.regex_patterns['ADDRESS'], 0.8)
            
            # spaCy NLP Engine (de/en) + SpacyRecognizer für PERSON etc.
            from presidio_analyzer.nlp_engine import NlpEngineProvider
            from presidio_analyzer.predefined_recognizers.spacy_recognizer import SpacyRecognizer
            
            # Noop NLP Engine (Fallback), um Presidio-Defaults zu verhindern
            class NoopNlpEngine(NlpEngine):
                engine_name = "noop"
                @property
                def is_available(self):
                    return True
                def __init__(self, *args, **kwargs):
                    pass
                def process_text(self, text: str, language: str) -> NlpArtifacts:  # type: ignore[override]
                    return NlpArtifacts(entities=[], tokens=[], tokens_indices=[], lemmas=[], nlp_engine=None, language=language)
                def process_batch(self, texts, language: str, **kwargs):  # type: ignore[override]
                    for t in texts:
                        yield t, self.process_text(t, language)
                def is_stopword(self, word: str, language: str) -> bool:  # type: ignore[override]
                    return False
                def is_punct(self, word: str, language: str) -> bool:  # type: ignore[override]
                    return False
            nlp_conf = {
                "nlp_engine_name": "spacy",
                "models": [
                    {"lang_code": "de", "model_name": "de_core_news_sm"},
                    {"lang_code": "en", "model_name": "en_core_web_sm"},
                ],
            }
            # spaCy muss vorhanden sein – andernfalls abbrechen
            nlp_engine = NlpEngineProvider(nlp_configuration=nlp_conf).create_engine()
            registry.add_recognizer(SpacyRecognizer(supported_language="de"))
            registry.add_recognizer(SpacyRecognizer(supported_language="en"))
            self.has_spacy = True

            self.analyzer = AnalyzerEngine(
                registry=registry,
                nlp_engine=nlp_engine,
                supported_languages=["de", "en"],
                default_score_threshold=0.6,
            )

            # Filter aus ENV lesen (JSON), Standard: alles an
            self.filters = {
                'names': True,
                'addresses': True,
                'phoneNumbers': True,
                'emails': True,
                'iban': True,
            }
            try:
                env_filters = os.getenv('FILTERS')
                if env_filters:
                    parsed = json.loads(env_filters)
                    if isinstance(parsed, dict):
                        self.filters.update({
                            'names': bool(parsed.get('names', self.filters['names'])),
                            'addresses': bool(parsed.get('addresses', self.filters['addresses'])),
                            'phoneNumbers': bool(parsed.get('phoneNumbers', self.filters['phoneNumbers'])),
                            'emails': bool(parsed.get('emails', self.filters['emails'])),
                            'iban': bool(parsed.get('iban', self.filters['iban'])),
                        })
            except Exception:
                pass
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Presidio: {e}")
            raise
    
    def setup_regex_recognizers(self):
        """Regex-basierte Recognizer für IBAN, EMAIL, PHONE, IP hinzufügen"""
        # Diese werden in process_text verwendet
        self.regex_patterns = {
            # IBAN (DE, EU allgemein) robust, erlaubt Leerzeichen
            'IBAN': r'\b(?:(?:[A-Z]{2}\d{2})\s?(?:[A-Z0-9]{4}\s?){2,7}[A-Z0-9]{0,2})\b',
            # Emails
            'EMAIL': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',
            # Telefonnummern: mindestens 8 Ziffern gesamt, erlaubt Trennzeichen
            'PHONE': r'\b(?:\+?\d[\d\s()./\-]{6,}\d)\b',
            # IPv4
            'IP': r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
            # Einfache deutsche Adresse: Straße/Hausnummer + PLZ Ort
            'ADDRESS': r'\b([A-ZÄÖÜ][a-zäöüß]+(?:strasse|straße|weg|platz|allee|gasse)\s+\d+[a-zA-Z]?)(?:,?\s+)?(\d{5})\s+([A-ZÄÖÜ][a-zäöüß]+)\b',
        }

        # Person-Name Pattern (Deutsch/Allgemein): Vorname(n) + optionaler Namenszusatz (z.B. "von", "zu") + Nachname
        # - Erlaubt 1-3 Vornamen
        # - Erlaubt Zusätze wie: von|zu|zum|zur|vom|van|de|der|den|di|du|del|della|da|la|le (inkl. Kombinationen wie "von der")
        # - Erlaubt Bindestrich im Nachnamen
        self.person_name_pattern = (
            r"\b"
            r"[A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+){0,2}\s+"
            r"(?:(?:von|zu|zum|zur|vom|van|de|der|den|di|du|del|della|da|la|le)\s+)?"
            r"[A-ZÄÖÜ][a-zäöüß]+(?:-[A-ZÄÖÜ][a-zäöüß]+)?"
            r"\b"
        )
    
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

    def map_lang_for_tesseract(self, language: str) -> str:
        """Map 'de'/'en'/auto to Tesseract language codes."""
        if language == "de":
            return "deu"
        if language == "en":
            return "eng"
        return "deu+eng"
    
    def strip_metadata_pdf(self, doc_path: Path) -> None:
        """PDF-Metadaten entfernen"""
        try:
            doc = fitz.open(doc_path)
            doc.set_metadata({})
            doc.save(doc_path, garbage=4, deflate=True, clean=True)
            doc.close()
        except Exception as e:
            self.logger.warning(f"Could not strip PDF metadata: {e}")
    
    def strip_metadata_image(self, image_path: Path) -> None:
        """EXIF-Metadaten aus Bildern entfernen"""
        try:
            image = Image.open(image_path)
            if hasattr(image, '_getexif') and image._getexif() is not None:
                # EXIF entfernen
                data = list(image.getdata())
                image_without_exif = Image.new(image.mode, image.size)
                image_without_exif.putdata(data)
                image_without_exif.save(image_path)
        except Exception as e:
            self.logger.warning(f"Could not strip image metadata: {e}")
    
    def detect_language_from_text(self, text: str) -> str:
        """Very lightweight de/en heuristic for offline auto-detection.
        Prioritizes speed and zero extra dependencies.
        """
        sample = text[:5000].lower()
        german_hits = 0
        english_hits = 0

        # Umlauts and sharp s strongly hint German
        if any(ch in sample for ch in ["ä", "ö", "ü", "ß"]):
            german_hits += 3

        # Common stopwords (lightweight, not exhaustive)
        de_words = [" und ", " der ", " die ", " das ", " ist ", " nicht ", " mit ", " für "]
        en_words = [" and ", " the ", " is ", " not ", " with ", " for ", " to ", " of "]
        german_hits += sum(sample.count(w) for w in de_words)
        english_hits += sum(sample.count(w) for w in en_words)

        return "de" if german_hits >= english_hits else "en"

    def detect_language_for_file(self, input_path: Path, file_type: str, ocr_enabled: bool) -> str:
        """Detect de/en from file content. Falls back to 'de' if inconclusive."""
        try:
            if file_type == "pdf":
                doc = fitz.open(input_path)
                text = ""
                try:
                    for i in range(min(len(doc), 3)):
                        text += doc[i].get_text() or ""
                finally:
                    doc.close()
                if not text.strip() and ocr_enabled:
                    # Best-effort: rasterize first page and run quick OCR with both languages
                    try:
                        doc = fitz.open(input_path)
                        if len(doc) > 0:
                            pix = doc[0].get_pixmap()
                            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                            text = pytesseract.image_to_string(img, lang="deu+eng")
                        doc.close()
                    except Exception:
                        pass
                if text.strip():
                    return self.detect_language_from_text(text)
                return "de"
            elif file_type in ["text", "markdown", "json"]:
                # Read up to a small sample
                with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read(5000)
                if text.strip():
                    return self.detect_language_from_text(text)
                return "de"
            elif file_type == "image":
                if ocr_enabled:
                    try:
                        image = cv2.imread(str(input_path))
                        if image is not None:
                            text = pytesseract.image_to_string(image, lang="deu+eng")
                            if text.strip():
                                return self.detect_language_from_text(text)
                    except Exception:
                        pass
                return "de"
        except Exception:
            return "de"

    def process_pdf(self, input_path: Path, output_path: Path, language: str, ocr_enabled: bool) -> Dict[str, Any]:
        """PDF-Verarbeitung mit PyMuPDF und Presidio"""
        start_time = time.time()
        entities_found = {}
        
        try:
            doc = fitz.open(input_path)
            total_pages = len(doc)
            
            # Alle Redactions sammeln
            all_redactions = []
            
            for page_num in tqdm(range(total_pages), desc="Processing PDF pages"):
                page = doc[page_num]
                
                # Text extrahieren; bei leerem Text optional OCR-Fallback
                text = page.get_text()
                if not text.strip() and ocr_enabled:
                    try:
                        pix = page.get_pixmap()
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        # OCR with both languages when auto detection is expected
                        tesseract_lang = "deu+eng" if language == "auto" else ("deu" if language == "de" else "eng")
                        text = pytesseract.image_to_string(img, lang=tesseract_lang)
                    except Exception:
                        pass
                if not text.strip():
                    continue
                
                # Presidio-Analyse
                entities = self.collect_requested_entities()
                results = self.analyzer.analyze(
                    text=text,
                    language=language,
                    entities=entities
                )
                
                # Nur Presidio-Results verwenden (PatternRecognizers)
                
                # Redactions für diese Seite erstellen
                for result in results:
                    entity_type = result.entity_type
                    if entity_type not in entities_found:
                        entities_found[entity_type] = 0
                    entities_found[entity_type] += 1
                    
                    # Bounding Box für Text finden
                    try:
                        entity_text = text[result.start:result.end]
                    except Exception:
                        entity_text = None
                    rects = page.search_for(entity_text) if entity_text else []
                    for rect in rects:
                        all_redactions.append({
                            'page': page_num,
                            'rect': rect,
                            'entity_type': entity_type
                        })
            
            # Alle Redactions anwenden
            pages_to_apply = set()
            for redaction in all_redactions:
                page = doc[redaction['page']]
                page.add_redact_annot(redaction['rect'], fill=(0, 0, 0))
                pages_to_apply.add(redaction['page'])

            # Redactions tatsächlich anwenden (pro Seite)
            for page_index in pages_to_apply:
                page = doc[page_index]
                page.apply_redactions()
            
            # Metadaten entfernen
            self.strip_metadata_pdf(input_path)
            
            # Speichern
            doc.save(output_path, garbage=4, deflate=True, clean=True)
            doc.close()
            
            processing_time = int((time.time() - start_time) * 1000)
            
            return {
                "ok": True,
                "output": str(output_path.absolute()),
                "summary": {
                    "entities": entities_found,
                    "type": "pdf",
                    "timeMs": processing_time,
                    "pages": total_pages
                }
            }
            
        except Exception as e:
            self.logger.error(f"PDF processing failed: {e}")
            return {"ok": False, "error": f"PDF processing failed: {str(e)}"}
    
    def process_image(self, input_path: Path, output_path: Path, language: str, ocr_enabled: bool) -> Dict[str, Any]:
        """Bildverarbeitung mit OCR und Redaction-Overlay"""
        start_time = time.time()
        entities_found = {}
        
        try:
            # EXIF-Metadaten entfernen
            self.strip_metadata_image(input_path)
            
            # Bild laden
            image = cv2.imread(str(input_path))
            if image is None:
                return {"ok": False, "error": "Could not load image"}
            
            # OCR falls aktiviert
            if ocr_enabled:
                # Tesseract OCR
                tesseract_lang = self.map_lang_for_tesseract(language)
                text = pytesseract.image_to_string(image, lang=tesseract_lang)
                
                if text.strip():
                    # Presidio-Analyse
                    entities = self.collect_requested_entities()
                    results = self.analyzer.analyze(
                        text=text,
                        language=language,
                        entities=entities
                    )
                    
                    # Nur Presidio-Results verwenden
                    
                    # Bounding Boxes für erkannte Entitäten finden und abdecken
                    for result in results:
                        entity_type = result.entity_type
                        if entity_type not in entities_found:
                            entities_found[entity_type] = 0
                        entities_found[entity_type] += 1
                        
                    # Bounding Box für Text finden
                    boxes = pytesseract.image_to_boxes(image, lang=tesseract_lang)
                    try:
                        entity_text = text[result.start:result.end]
                    except Exception:
                        entity_text = ""
                    for box in boxes.splitlines():
                            box_data = box.split(' ')
                            if len(box_data) >= 6:
                                char = box_data[0]
                            if entity_text and char in entity_text:
                                    x, y, w, h = int(box_data[1]), int(box_data[2]), int(box_data[3]), int(box_data[4])
                                    # Schwarze Box über Text zeichnen
                                    cv2.rectangle(image, (x, image.shape[0] - y), (w, image.shape[0] - h), (0, 0, 0), -1)
            
            # Bild speichern
            cv2.imwrite(str(output_path), image)
            
            processing_time = int((time.time() - start_time) * 1000)
            
            return {
                "ok": True,
                "output": str(output_path.absolute()),
                "summary": {
                    "entities": entities_found,
                    "type": "image",
                    "timeMs": processing_time,
                    "ocr_enabled": ocr_enabled
                }
            }
            
        except Exception as e:
            self.logger.error(f"Image processing failed: {e}")
            return {"ok": False, "error": f"Image processing failed: {str(e)}"}
    
    def process_text(self, input_path: Path, output_path: Path, language: str) -> Dict[str, Any]:
        """Textdatei-Verarbeitung mit Chunking"""
        start_time = time.time()
        entities_found = {}
        
        try:
            # Datei in Chunks lesen
            with open(input_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Chunking für große Dateien
            chunks = self.chunk_text(content)
            processed_chunks = []
            
            for chunk in tqdm(chunks, desc="Processing text chunks"):
                # Presidio-Analyse
                entities = self.collect_requested_entities()
                results = self.analyzer.analyze(
                    text=chunk,
                    language=language if language != "auto" else self.detect_language_from_text(chunk),
                    entities=entities
                )
                
                # Nur Presidio-Results verwenden
                
                # Anonymisierung
                anonymized_chunk = self.anonymize_text(chunk, results, entities_found)
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
                data = json.load(f)
            
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
            # String-Felder mit Presidio verarbeiten
            # If auto-language, detect per string sample so mixed-language JSON works reasonably
            infer_lang = language if language != "auto" else self.detect_language_from_text(obj)
            entities = self.collect_requested_entities()
            results = self.analyzer.analyze(
                text=obj,
                language=infer_lang,
                entities=entities
            )
            
            # Nur Presidio-Results verwenden
            
            return self.anonymize_text(obj, results, entities_found)
        else:
            return obj
    
    def collect_requested_entities(self) -> List[str]:
        """Ermittle die zu analysierenden Entitäten basierend auf Filtern."""
        entities = []
        if self.filters.get('names', True):
            entities.append('PERSON')
            if self.has_spacy:
                entities.extend(['ORG', 'LOCATION'])
        if self.filters.get('emails', True):
            entities.append('EMAIL_ADDRESS')
        if self.filters.get('phoneNumbers', True):
            entities.append('PHONE_NUMBER')
        if self.filters.get('iban', True):
            entities.append('IBAN_CODE')
        # IP und LOCATION/ORGANIZATION nur wenn Adressen gewählt sind
        if self.filters.get('addresses', True):
            entities.extend(['IP_ADDRESS', 'ADDRESS'])
        return entities

    def find_regex_entities(self, text: str) -> List[RecognizerResult]:
        """Regex-basierte Entitätserkennung"""
        results = []
        
        for entity_type, pattern in self.regex_patterns.items():
            # Filter anwenden
            if entity_type == 'EMAIL' and not self.filters.get('emails', True):
                continue
            if entity_type == 'PHONE' and not self.filters.get('phoneNumbers', True):
                continue
            if entity_type == 'IBAN' and not self.filters.get('iban', True):
                continue
            if entity_type in ['IP', 'ADDRESS'] and not self.filters.get('addresses', True):
                continue
            matches = regex.finditer(pattern, text, overlapped=True)
            for match in matches:
                results.append(RecognizerResult(
                    entity_type=entity_type,
                    start=match.start(),
                    end=match.end(),
                    score=0.9
                ))
        
        return results

    def find_person_names(self, text: str) -> List[RecognizerResult]:
        """Finde Personennamen (Vorname Nachname) per Regex als Ergänzung zu spaCy/Presidio."""
        results: List[RecognizerResult] = []
        try:
            for m in regex.finditer(self.person_name_pattern, text, overlapped=True):
                full = m.group(0)
                results.append(RecognizerResult(
                    entity_type='PERSON',
                    start=m.start(),
                    end=m.end(),
                    score=0.85
                ))
        except Exception:
            pass
        return results
    
    def anonymize_text(self, text: str, results: List[RecognizerResult], entities_found: Dict[str, int]) -> str:
        """Text mit Presidio anonymisieren"""
        if not results:
            return text
        
        # Sortiere nach Position (rückwärts, um Indizes nicht zu verschieben)
        results.sort(key=lambda x: x.start, reverse=True)
        
        anonymized_text = text
        for result in results:
            entity_type = result.entity_type
            if entity_type not in entities_found:
                entities_found[entity_type] = 0
            entities_found[entity_type] += 1
            
            replacement = self.get_entity_replacement(entity_type)
            anonymized_text = (
                anonymized_text[:result.start] + 
                replacement + 
                anonymized_text[result.end:]
            )
        
        return anonymized_text
    
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
                # Suche nach dem letzten Leerzeichen vor der Chunk-Grenze
                last_space = text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space
            
            chunks.append(text[start:end])
            start = end
        
        return chunks
    
    def process_file(self, input_path: str, output_dir: str, mode: str = "pseudo", 
                    language: str = "de", ocr_enabled: bool = False) -> Dict[str, Any]:
        """Hauptverarbeitungsfunktion"""
        try:
            input_file = Path(input_path)
            output_dir_path = Path(output_dir)
            
            # Validierung
            if not input_file.exists():
                return {"ok": False, "error": f"Input file not found: {input_path}"}
            
            # Size validation
            try:
                size_bytes = input_file.stat().st_size
                if size_bytes > self.max_file_size_bytes:
                    return {"ok": False, "error": f"File too large: {size_bytes} bytes (limit {self.max_file_size_bytes})"}
            except Exception as e:
                self.logger.warning(f"Could not read file size: {e}")

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
            
            # Auto-language detection if requested
            if language == "auto":
                language = self.detect_language_for_file(input_copy_path, file_type, ocr_enabled)

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
    parser = argparse.ArgumentParser(description="Tauri Sanitizer Engine")
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
        engine = SanitizerEngine()
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
