#!/usr/bin/env python3
"""
Tauri Sanitizer Engine (Presidio-only, fixed, explicit recognizers)
- 100% Microsoft Presidio (Analyzer + Anonymizer)
- Explizit registriert: Email, Phone, IBAN, IP, Date, URL, CreditCard + Spacy (de/en)
- Schnelle PDF-Redactions via Word-Level Mapping (ohne search_for pro Entity)
- Auto-OCR: PDFs nur wenn kein Text; Images immer via OCR
- Language: auto (de/en) per lightweight Heuristik
- Run-Lifecycle: arbeitet auf einer KOPIE im data/input/<RUN_ID>/; löscht nur die KOPIE bei Erfolg
"""

import argparse
import sys
import os
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime

# ---- Dependencies (fail fast) ----
try:
    from presidio_analyzer import AnalyzerEngine, RecognizerResult, RecognizerRegistry
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    from presidio_analyzer.predefined_recognizers.spacy_recognizer import SpacyRecognizer

    # explizit gewünschte Presidio-Recognizers
    from presidio_analyzer.predefined_recognizers.email_recognizer import EmailRecognizer
    from presidio_analyzer.predefined_recognizers.phone_recognizer import PhoneRecognizer
    from presidio_analyzer.predefined_recognizers.iban_recognizer import IbanRecognizer
    from presidio_analyzer.predefined_recognizers.ip_recognizer import IpRecognizer
    from presidio_analyzer.predefined_recognizers.date_recognizer import DateRecognizer
    from presidio_analyzer.predefined_recognizers.url_recognizer import UrlRecognizer
    from presidio_analyzer.predefined_recognizers.credit_card_recognizer import CreditCardRecognizer

    from presidio_anonymizer import AnonymizerEngine
    import fitz  # PyMuPDF
    import pytesseract
    from PIL import Image
    import cv2
except ImportError as e:
    print(f"Missing dependency: {e}", file=sys.stderr)
    print("Please install: pip install -r requirements.txt", file=sys.stderr)
    sys.exit(1)


class SanitizerEngine:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._setup_logging()
        self._setup_presidio()
        self.entity_counters: Dict[str, int] = {}
        self.chunk_size = 2 * 1024 * 1024             # 2MB
        self.max_file_size_bytes = 100 * 1024 * 1024  # 100MB

    # ---------- setup ----------
    def _setup_logging(self):
        level = logging.DEBUG if self.verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler(sys.stderr)],
        )
        self.log = logging.getLogger("engine")

    def _setup_presidio(self):
        """
        Presidio-only:
        - Eigene RecognizerRegistry, NICHT Default – wir tragen explizit ein:
          SpacyRecognizer (de/en) + Email/Phone/IBAN/IP/Date/URL/CreditCard
        """
        # spaCy Pipelines
        nlp_conf = {
            "nlp_engine_name": "spacy",
            "models": [
                {"lang_code": "de", "model_name": "de_core_news_lg"},
                {"lang_code": "en", "model_name": "en_core_web_sm"},
            ],
        }
        nlp_engine = NlpEngineProvider(nlp_configuration=nlp_conf).create_engine()

        # Eigene Registry aufbauen
        registry = RecognizerRegistry()

        # SpacyRecognizer (liefert PERSON/ORG/LOC etc.)
        registry.add_recognizer(SpacyRecognizer(supported_language="de"))
        #registry.add_recognizer(SpacyRecognizer(supported_language="en"))

        # Explizite Pattern/Checksum Recognizers (sprachabhängig registrieren, wo sinnvoll)
        for lang in ("de", "en"):
            registry.add_recognizer(EmailRecognizer(supported_language=lang))
            registry.add_recognizer(PhoneRecognizer(supported_language=lang))
            registry.add_recognizer(IbanRecognizer(supported_language=lang))
            registry.add_recognizer(IpRecognizer(supported_language=lang))
            registry.add_recognizer(DateRecognizer(supported_language=lang))
            registry.add_recognizer(UrlRecognizer(supported_language=lang))
            registry.add_recognizer(CreditCardRecognizer(supported_language=lang))

        # AnalyzerEngine mit unserer Registry
        self.analyzer = AnalyzerEngine(
            nlp_engine=nlp_engine,
            registry=registry,
            supported_languages=["de", "en"],
            default_score_threshold=0.68,
        )

        self.anonymizer = AnonymizerEngine()

    # ---------- utils ----------
    @staticmethod
    def _safe_mkdirs(path: Path):
        path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _wipe_dir(path: Path):
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _ts_dir(base: Path) -> Path:
        return base / datetime.now().strftime("%Y%m%d_%H%M%S")

    def _is_subpath(self, child: Path, parent: Path) -> bool:
        """True, wenn child innerhalb von parent liegt (resolvt, robust)."""
        try:
            child.resolve().relative_to(parent.resolve())
            return True
        except Exception:
            return False

    def _reset_counters(self):
        self.entity_counters.clear()

    def _replacement(self, entity_type: str) -> str:
        self.entity_counters[entity_type] = self.entity_counters.get(entity_type, 0) + 1
        return f"{entity_type}_{self.entity_counters[entity_type]:03d}"

    @staticmethod
    def _detect_lang_from_text(text: str) -> str:
        """Sehr leichte de/en-Heuristik, zero extra deps."""
        s = text[:5000].lower()
        de = 0
        en = 0
        if any(ch in s for ch in ("ä", "ö", "ü", "ß")):
            de += 3
        for w in (" und ", " der ", " die ", " das ", " ist ", " nicht ", " mit ", " für "):
            de += s.count(w)
        for w in (" and ", " the ", " is ", " not ", " with ", " for ", " to ", " of "):
            en += s.count(w)
        return "de" if de >= en else "en"

    def _detect_language_for_file(self, path: Path, ftype: str, ocr_enabled: bool) -> str:
        try:
            if ftype == "pdf":
                doc = fitz.open(path)
                text = ""
                try:
                    for i in range(min(len(doc), 3)):
                        text += doc[i].get_text() or ""
                finally:
                    doc.close()
                if (not text.strip()) and ocr_enabled:
                    doc = fitz.open(path)
                    if len(doc) > 0:
                        pix = doc[0].get_pixmap(matrix=fitz.Matrix(2, 2))
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        text = pytesseract.image_to_string(img, lang="deu+eng")
                    doc.close()
                return self._detect_lang_from_text(text) if text.strip() else "de"
            elif ftype in ("text", "markdown", "json"):
                s = path.read_text(encoding="utf-8", errors="ignore")[:5000]
                return self._detect_lang_from_text(s) if s.strip() else "de"
            else:  # image
                if ocr_enabled:
                    img = cv2.imread(str(path))
                    if img is not None:
                        t = pytesseract.image_to_string(img, lang="deu+eng")
                        return self._detect_lang_from_text(t) if t.strip() else "de"
                return "de"
        except Exception:
            return "de"

    @staticmethod
    def _detect_type(path: Path) -> str:
        s = path.suffix.lower()
        if s == ".pdf":
            return "pdf"
        if s in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
            return "image"
        if s in (".txt", ".log", ".csv", ".md"):
            return "text" if s != ".md" else "markdown"
        if s == ".json":
            return "json"
        raise ValueError(f"Unsupported file type: {s}")

    # ---------- text/json ----------
    def _anonymize_text_presidio(self, text: str, lang: str, entities_found: Dict[str, int], score_threshold: float) -> str:
        results: List[RecognizerResult] = self.analyzer.analyze(
            text=text,
            language=lang,
            score_threshold=score_threshold,
        )
        out = text
        for r in sorted(results, key=lambda x: x.start, reverse=True):
            entities_found[r.entity_type] = entities_found.get(r.entity_type, 0) + 1
            out = out[:r.start] + self._replacement(r.entity_type) + out[r.end:]
        return out

    def _process_text_file(self, inp: Path, outp: Path, lang: str, score_threshold: float) -> Dict[str, Any]:
        start = time.time()
        entities_found: Dict[str, int] = {}
        content = inp.read_text(encoding="utf-8", errors="ignore")

        if len(content) <= self.chunk_size:
            processed = self._anonymize_text_presidio(content, lang, entities_found, score_threshold)
        else:
            chunks: List[str] = []
            i = 0
            L = len(content)
            while i < L:
                j = min(i + self.chunk_size, L)
                if j < L:
                    k = content.rfind(" ", i, j)
                    if k > i:
                        j = k
                chunk = content[i:j]
                chunks.append(self._anonymize_text_presidio(chunk, lang, entities_found, score_threshold))
                i = j
            processed = "".join(chunks)

        outp.write_text(processed, encoding="utf-8")
        return {
            "ok": True,
            "output": str(outp.absolute()),
            "summary": {"entities": entities_found, "type": "text", "timeMs": int((time.time() - start) * 1000)},
        }

    def _process_json_file(self, inp: Path, outp: Path, lang: str, score_threshold: float) -> Dict[str, Any]:
        start = time.time()
        entities_found: Dict[str, int] = {}

        def walk(obj):
            if isinstance(obj, dict):
                return {k: walk(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [walk(v) for v in obj]
            if isinstance(obj, str):
                return self._anonymize_text_presidio(obj, lang, entities_found, score_threshold)
            return obj

        data = json.loads(inp.read_text(encoding="utf-8", errors="ignore"))
        data = walk(data)
        outp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return {
            "ok": True,
            "output": str(outp.absolute()),
            "summary": {"entities": entities_found, "type": "json", "timeMs": int((time.time() - start) * 1000)},
        }

    # ---------- pdf ----------
    @staticmethod
    def _words_with_offsets(page: "fitz.Page") -> Tuple[str, List[Tuple[int, int, fitz.Rect]]]:
        """
        Erzeuge einen Seitentext, indem Wörter mit einem einzelnen Space verbunden werden,
        und sammle (start, end, rect) je Wort – um Presidio-Char-Spans -> Wort-Rects zu mappen.
        """
        words = page.get_text("words")  # (x0,y0,x1,y1,"word", block, line, word_no)
        words.sort(key=lambda w: (w[5], w[6], w[7]))  # reading order
        text_parts: List[str] = []
        offsets: List[Tuple[int, int, fitz.Rect]] = []
        pos = 0
        first = True
        for x0, y0, x1, y1, wtext, *_ in words:
            if not wtext:
                continue
            if not first:
                text_parts.append(" "); pos += 1
            first = False
            start = pos
            text_parts.append(wtext)
            pos += len(wtext)
            offsets.append((start, pos, fitz.Rect(x0, y0, x1, y1)))
        return "".join(text_parts), offsets

    @staticmethod
    def _rects_for_span(span_start: int, span_end: int, offsets: List[Tuple[int, int, fitz.Rect]]) -> List[fitz.Rect]:
        rects: List[fitz.Rect] = []
        for w_start, w_end, rect in offsets:
            if w_end <= span_start:
                continue
            if w_start >= span_end:
                break
            rects.append(rect)
        return rects

    def _process_pdf_file(self, inp: Path, outp: Path, lang: str, ocr_enabled: bool, score_threshold: float) -> Dict[str, Any]:
        start = time.time()
        entities_found: Dict[str, int] = {}
        try:
            doc = fitz.open(inp)
            pages = len(doc)

            for pno in range(pages):
                page = doc[pno]
                page_text, offsets = self._words_with_offsets(page)
                use_ocr = (not page_text.strip()) and ocr_enabled

                if not use_ocr:
                    if not page_text.strip():
                        continue
                    pres = self.analyzer.analyze(
                        text=page_text,
                        language=lang,
                        score_threshold=score_threshold,
                    )
                    rects: List[fitz.Rect] = []
                    for r in pres:
                        entities_found[r.entity_type] = entities_found.get(r.entity_type, 0) + 1
                        rects.extend(self._rects_for_span(r.start, r.end, offsets))
                    for rc in rects:
                        page.add_redact_annot(rc, fill=(0, 0, 0))
                    page.apply_redactions()
                else:
                    # OCR-Path (wenn kein Text extrahierbar)
                    mat = fitz.Matrix(2, 2)
                    pix = page.get_pixmap(matrix=mat, alpha=False)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                    data = pytesseract.image_to_data(img, lang="deu+eng", output_type=pytesseract.Output.DICT)
                    # OCR Wortliste
                    ocr_words: List[Tuple[str, Tuple[int, int, int, int]]] = []
                    for i in range(len(data["text"])):
                        t = (data["text"][i] or "").strip()
                        if t:
                            ocr_words.append((t, (data["left"][i], data["top"][i], data["width"][i], data["height"][i])))

                    # OCR-Text + Offsets aufbauen
                    text_parts: List[str] = []
                    ocr_offsets: List[Tuple[int, int, Tuple[int, int, int, int]]] = []
                    pos = 0
                    first = True
                    for w, box in ocr_words:
                        if not first:
                            text_parts.append(" "); pos += 1
                        first = False
                        start = pos
                        text_parts.append(w)
                        pos += len(w)
                        ocr_offsets.append((start, pos, box))
                    ocr_text = "".join(text_parts)
                    if not ocr_text.strip():
                        continue

                    pres = self.analyzer.analyze(
                        text=ocr_text,
                        language=lang,
                        score_threshold=score_threshold,
                    )

                    page_rect = page.rect
                    rects: List[fitz.Rect] = []
                    for r in pres:
                        entities_found[r.entity_type] = entities_found.get(r.entity_type, 0) + 1
                        for w_start, w_end, (x, y, w, h) in ocr_offsets:
                            if w_end <= r.start:
                                continue
                            if w_start >= r.end:
                                break
                            # OCR (pixel) -> page coords
                            px = x / pix.width * page_rect.width + page_rect.x0
                            py = y / pix.height * page_rect.height + page_rect.y0
                            pw = w / pix.width * page_rect.width
                            ph = h / pix.height * page_rect.height
                            rects.append(fitz.Rect(px, py, px + pw, py + ph))
                    for rc in rects:
                        page.add_redact_annot(rc, fill=(0, 0, 0))
                    page.apply_redactions()

            # Metadaten leeren & sauber speichern
            doc.set_metadata({})
            doc.save(outp, garbage=4, deflate=True, clean=True)
            doc.close()

            return {
                "ok": True,
                "output": str(outp.absolute()),
                "summary": {
                    "entities": entities_found,
                    "type": "pdf",
                    "pages": pages,
                    "timeMs": int((time.time() - start) * 1000),
                },
            }
        except Exception as e:
            self.log.error(f"PDF processing failed: {e}")
            return {"ok": False, "error": f"PDF processing failed: {e}"}

    # ---------- images ----------
    def _process_image_file(self, inp: Path, outp: Path, lang: str, score_threshold: float) -> Dict[str, Any]:
        start = time.time()
        entities_found: Dict[str, int] = {}
        try:
            img = cv2.imread(str(inp))
            if img is None:
                return {"ok": False, "error": "Could not load image"}

            data = pytesseract.image_to_data(img, lang="deu+eng", output_type=pytesseract.Output.DICT)
            words: List[Tuple[str, Tuple[int, int, int, int]]] = []
            for i in range(len(data["text"])):
                t = (data["text"][i] or "").strip()
                if t:
                    words.append((t, (data["left"][i], data["top"][i], data["width"][i], data["height"][i])))

            # Text + Offsets aufbauen
            text_parts: List[str] = []
            offsets: List[Tuple[int, int, Tuple[int, int, int, int]]] = []
            pos = 0
            first = True
            for w, box in words:
                if not first:
                    text_parts.append(" "); pos += 1
                first = False
                start = pos
                text_parts.append(w)
                pos += len(w)
                offsets.append((start, pos, box))
            full_text = "".join(text_parts)

            if full_text.strip():
                pres = self.analyzer.analyze(
                    text=full_text,
                    language=lang,
                    score_threshold=score_threshold,
                )
                for r in pres:
                    entities_found[r.entity_type] = entities_found.get(r.entity_type, 0) + 1
                    for s, e, (x, y, w, h) in offsets:
                        if e <= r.start:
                            continue
                        if s >= r.end:
                            break
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), thickness=-1)

            cv2.imwrite(str(outp), img)
            return {
                "ok": True,
                "output": str(outp.absolute()),
                "summary": {"entities": entities_found, "type": "image", "timeMs": int((time.time() - start) * 1000)},
            }
        except Exception as e:
            self.log.error(f"Image processing failed: {e}")
            return {"ok": False, "error": f"Image processing failed: {e}"}

    # ---------- orchestration ----------
    def process_file(self, input_path: str, output_dir: str, mode: str = "pseudo",
                     language: str = "auto", ocr_enabled: bool = True,
                     score_threshold: float | None = None) -> Dict[str, Any]:
        """
        Robuste Orchestrierung:
        - Wenn die Quelle bereits unter data/input/ liegt, NIEMALS data/input vorab wipen.
        - Immer eine frische Run-Struktur: data/input/<RUN_ID>/ und data/output/<RUN_ID>/
        - Von der Quelle in den aktuellen input-Run kopieren und NUR diese Kopie bei Erfolg löschen.
        """
        src = Path(input_path).expanduser().resolve()
        outdir = Path(output_dir).expanduser()

        if not src.exists():
            return {"ok": False, "error": f"Input not found: {src}"}
        try:
            if src.stat().st_size > self.max_file_size_bytes:
                return {"ok": False, "error": f"File too large (> {self.max_file_size_bytes} bytes)"}
        except Exception:
            pass

        # Output-Run: Basisverzeichnis nur sicherstellen – alte Runs NICHT mehr global löschen,
        # damit frühere PDFs für den Verlauf/Preview erhalten bleiben.
        self._safe_mkdirs(outdir)
        run_out = self._ts_dir(outdir)
        self._safe_mkdirs(run_out)

        # Input-Run: nur wipen, wenn Quelle NICHT bereits unter data/input liegt
        input_base = Path("data/input").expanduser()
        if not self._is_subpath(src, input_base):
            self._wipe_dir(input_base)
        else:
            self._safe_mkdirs(input_base)

        run_in = self._ts_dir(input_base)
        self._safe_mkdirs(run_in)

        # Kopie der Quelle für diesen Run
        inp_copy = run_in / src.name
        try:
            shutil.copy2(src, inp_copy)
        except FileNotFoundError as e:
            return {"ok": False, "error": f"Failed to copy input into run folder: {e}"}

        # Typ & Output
        try:
            ftype = self._detect_type(src)
        except Exception as e:
            return {"ok": False, "error": f"Unsupported file type: {e}"}

        out_name = f"{src.stem}_pseudo{src.suffix}"
        out_path = run_out / out_name

        # Language auto
        if language == "auto":
            language = self._detect_language_for_file(inp_copy, ftype, ocr_enabled)

        # Score threshold (Accuracy)
        if score_threshold is None:
            score_threshold = 0.6

        # Zähler reset
        self._reset_counters()

        # Dispatch
        if ftype == "pdf":
            res = self._process_pdf_file(inp_copy, out_path, language, ocr_enabled, score_threshold)
        elif ftype == "image":
            res = self._process_image_file(inp_copy, out_path, language, score_threshold)
        elif ftype in ("text", "markdown"):
            res = self._process_text_file(inp_copy, out_path, language, score_threshold)
        elif ftype == "json":
            res = self._process_json_file(inp_copy, out_path, language, score_threshold)
        else:
            res = {"ok": False, "error": f"Unsupported file type: {ftype}"}

        # Nur die KOPIE löschen, und nur bei Erfolg
        if res.get("ok"):
            try:
                inp_copy.unlink(missing_ok=True)
            except Exception:
                pass

        return res


# ---------- CLI ----------
def main():
    p = argparse.ArgumentParser(description="Tauri Sanitizer Engine (Presidio-only, explicit recognizers)")
    p.add_argument("--input", required=True)
    p.add_argument("--mode", default="pseudo", choices=["pseudo", "anon"])
    p.add_argument("--outdir", required=True)
    p.add_argument("--language", default="auto", choices=["de", "en", "auto"])
    p.add_argument("--threshold", type=float, default=0.6, help="Presidio score_threshold (accuracy tradeoff)")
    p.add_argument("--ocr", action="store_true", help="Force OCR (images always OCR; PDFs only if no text)")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()

    # OCR: per Spec standardmäßig AN (PDF: nur wenn Text fehlt)
    ocr_enabled = True if os.getenv("OCR", "1") == "1" or args.ocr else True

    eng = SanitizerEngine(verbose=args.verbose)
    res = eng.process_file(
        input_path=args.input,
        output_dir=args.outdir,
        mode=args.mode,
        language=args.language,
        ocr_enabled=ocr_enabled,
        score_threshold=args.threshold,
    )
    print(json.dumps(res, ensure_ascii=False), flush=True)
    sys.exit(0 if res.get("ok") else 1)


if __name__ == "__main__":
    main()