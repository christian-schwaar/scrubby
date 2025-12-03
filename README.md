## Scrubby (Tauri Sanitizer)

Scrubby ist eine Desktop‑App auf Basis von **Tauri 2 + React + Python**, die Dokumente mit Microsoft Presidio anonymisiert.
PDFs, Bilder und Textdateien können per Drag & Drop geladen und mit unterschiedlichen Accuracy‑Einstellungen bereinigt werden.

### Features

- **Multiplattform‑Desktop‑App** via Tauri (Rust Backend, React Frontend)
- **Presidio‑Engine (Python)** mit spaCy‑NER, Pattern‑Recognizers (E‑Mail, Telefon, IBAN, URL, Kreditkarten usw.)
- **PDF‑, Bild‑ und Text‑Support**
  - PDFs: Redaction direkt im PDF via PyMuPDF
  - Bilder: OCR + Schwärzung per OpenCV
  - Text / JSON: String‑basierte Anonymisierung
- **Tabs / Verlauf** in der UI  
  - Seitenleiste im macOS‑Finder‑Look
  - Tabs können hinzugefügt / entfernt werden (mind. ein Tab bleibt immer)
  - Verlauf inkl. Pfade, Dateinamen und Accuracy wird in `localStorage` persistiert
- **Preview‑Pane** für Input & Output
  - PDF‑Preview via `iframe`
  - Image‑Preview (`png/jpg/jpeg/gif/webp/bmp`) via `<img>`
  - Text/Markdown/JSON/CSV/Log via `iframe`
- **Accuracy‑Schalter (0.60 / 0.85)**  
  - 0.60: höherer Recall, etwas mehr False Positives  
  - 0.85: konservativer, weniger False Positives

---

## Projektstruktur (vereinfacht)

- `src/`
  - `App.tsx` – Hauptoberfläche (Sidebar, Tabs, Previews, Accuracy‑Toggle)
  - `components/` – UI‑Bausteine (`FileTile`, `PdfEditor`, Radix‑UI Wrapper usw.)
  - `index.css` – Tailwind 4 Konfiguration + Theme‑Tokens
- `src-tauri/`
  - `src/main.rs` – Tauri‑Commands (`run_engine`, File‑Dialoge, Finder‑Öffnen)
  - `tauri.conf.json` – App‑Konfiguration
- `engine/`
  - `engineV2.py` – Presidio‑Engine (PDF/Image/Text/JSON Orchestrierung)
- `package.json` – Node/Tauri‑Scripts

---

## Voraussetzungen

- **Node.js** ≥ 20
- **Rust toolchain** (für Tauri):  
  siehe Tauri‑Docs (`cargo`, `rustup`, passende Targets)
- **Python 3.11** (für `engineV2.py`)
- System‑Dependencies für:
  - PyMuPDF (`fitz`)
  - Tesseract OCR (Binary + `deu` + `eng` Sprachpakete)

---

## Installation & Setup

1. **Repository klonen**

```bash
git clone <dein-repo>
cd tauri-sanitizer
```

2. **Node‑Abhängigkeiten installieren**

```bash
npm install
```

3. **Python‑Umgebung & Engine‑Deps**

- Empfohlen: virtuelles Env im Projektroot (`venv311`):

```bash
python3.11 -m venv venv311
source venv311/bin/activate
pip install -r engine/requirements.txt  # falls vorhanden
```

Stelle sicher, dass `engine/engineV2.py` alle benötigten Pakete (Presidio, spaCy, PyMuPDF, Tesseract‑Bindings etc.) installieren kann.

4. **Tauri‑CLI installieren** (falls noch nicht vorhanden)

```bash
npm install -g @tauri-apps/cli
```

---

## Entwicklung starten

```bash
npm run dev
```

Das öffnet:
- Vite‑Devserver für das React‑Frontend
- Tauri‑Shell für die Desktop‑App

Hot‑Reload funktioniert für das Frontend; Änderungen an der Engine werden beim nächsten Run des Commands `run_engine` aktiv.

---

## Build

### Desktop‑App (Tauri)

```bash
npm run build
```

Das baut die Tauri‑App für die aktuelle Plattform (Konfiguration siehe `src-tauri/tauri.conf.json`).

### (Optional) Engine als Binary

In `package.json` sind Skripte angelegt, um eine eigenständige Engine zu bauen (ältere Variante, aktuell primär `engineV2.py` im direkten Python‑Aufruf im Einsatz):

```bash
npm run engine:setup      # Modelle/Abhängigkeiten vorbereiten (sofern Script vorhanden)
npm run engine:build      # PyInstaller-Build (alte Engine)
```

---

## Funktionsweise: Frontend ↔ Tauri ↔ Engine

1. **Frontend (`App.tsx`)**
   - Datei via Drag & Drop oder OS‑File‑Drop wählen.
   - Tab‑Session speichert Input‑Pfad, Preview‑URLs, Output‑Pfad und Accuracy.
   - Beim Klick auf **Start**:
     - wenn nötig, wird eine temporäre Datei geschrieben (`write_temp_file`).
     - `invoke("run_engine", { input, mode: "pseudo", outputDir: "data/output", language: "de", ocr: true, threshold, filters })`.

2. **Tauri‑Command (`run_engine` in `main.rs`)**
   - Sucht ein passendes Python aus `venv311`/`venv`.
   - Startet `engine/engineV2.py` mit `--input`, `--outdir`, `--language`, `--threshold` usw.
   - Gibt das JSON‑Ergebnis (inkl. Output‑Pfad) ans Frontend zurück.

3. **Engine (`engineV2.py`)**
   - Erzeugt Run‑Verzeichnisse `data/input/<RUN_ID>/` und `data/output/<RUN_ID>/`.
   - Ermittelt Dateityp (pdf/image/text/json).
   - Wendet Presidio‑Anonymisierung an:
     - `score_threshold` = Accuracy (0.6 oder 0.85)
     - PDF: Mapping von Presidio‑Spans auf Wort‑Rects → Redact‑Annotations.
     - Image: OCR‑Wortlisten → Black‑Rectangles via OpenCV.
     - Text/JSON: String‑Replacement.
   - Liefert `{"ok": true, "output": "<absoluter_pfad>", "summary": {...}}`.

4. **Preview**
   - Frontend baut mit `convertFileSrc(outputPath)` eine Tauri‑Asset‑URL.
   - Je nach Filetyp:
     - PDF: `iframe`
     - Image: `<img>`
     - Text: `iframe` / Fallback `FileTile`.

---

## Accuracy-Einstellung

Im Header kann pro Tab eine Accuracy gewählt werden:

- `0.60` → `threshold = 0.6`
- `0.85` → `threshold = 0.85`

Diese Accuracy wird:
- im Tab‑State (`sessions[tabId].accuracy`) und in `localStorage` gespeichert,
- beim Start an Tauri (`threshold`) und von dort an `engineV2.py` (`--threshold`) übergeben,
- in allen Presidio‑`analyze`‑Aufrufen als `score_threshold` verwendet.

---

## Bekannte Einschränkungen

- Die Engine erwartet lauffähige spaCy‑Modelle (`de_core_news_lg`, ggf. englische Modelle) – diese müssen extern installiert werden.
- Sehr große Dateien (>100 MB) werden serverseitig abgelehnt (Limit in `engineV2.py`).
- Der Verlauf ist aktuell **lokal pro Gerät** (Browser‑`localStorage` im Tauri‑WebView).

---

## Lizenz

Dieses Projekt kombiniert eigene Logik mit Drittbibliotheken wie Presidio, spaCy, PyMuPDF, Tesseract u.a.
Bitte beachte deren jeweilige Lizenzen, falls du Scrubby weiterverbreitest oder kommerziell nutzt.

# Tauri Sanitizer

Eine Tauri v2 App für die Pseudonymisierung von Dokumenten mit Presidio.

## Features

- **Frontend**: React + Vite + Tailwind v4 + shadcn/ui
- **Backend**: Python-Sidecar mit Presidio für Pseudonymisierung
- **Unterstützte Formate**: PDF, Bilder (PNG/JPG/JPEG), TXT, MD, JSON
- **Drag & Drop**: Intuitive Datei-Upload-Funktionalität
- **OCR**: Optional für Bildverarbeitung
- **Mehrsprachig**: Deutsch und Englisch
- **Offline**: Keine Telemetrie, vollständig offline

## Zwei Engine-Versionen

### 🚀 Vollversion (Empfohlen)
- **Presidio**: ML-basierte Entitätserkennung
- **PyMuPDF**: Echte PDF-Redaction
- **OCR**: Tesseract für Bildverarbeitung
- **Alle Features**: Vollständige Funktionalität

### ⚡ Einfache Version (Fallback)
- **Regex-basiert**: Schnelle, einfache Erkennung
- **Nur Text/JSON**: Keine PDF/Image-Verarbeitung
- **Keine Dependencies**: Funktioniert sofort
- **Schnell**: Minimaler Overhead

## Installation

### Voraussetzungen

- Node.js 18+
- Python 3.8-3.13 (3.14+ hat Kompatibilitätsprobleme)
- Rust (für Tauri)

### Option 1: Vollversion (Empfohlen)

```bash
# 1. Dependencies installieren
npm install

# 2. Python-Setup (mit Fallback)
npm run engine:setup

# 3. Engine bauen
npm run engine:build

# 4. App starten
npm run dev
```

### Option 2: Einfache Version (Bei Problemen)

```bash
# 1. Dependencies installieren
npm install

# 2. Einfache Engine-Setup
npm run engine:setup:simple

# 3. Einfache Engine bauen
npm run engine:build:simple

# 4. App starten
npm run dev
```

## Verwendung

1. **Datei auswählen**: Drag & Drop oder Datei-Picker
2. **Einstellungen**: Sprache (DE/EN) und OCR (nur Vollversion)
3. **Verarbeitung starten**: Button klicken
4. **Ergebnis**: Pseudonymisierte Datei in `data/output/`
5. **Cleanup**: Original wird nach Erfolg gelöscht

## Projektstruktur

```
src/                # React UI
├── components/ui/  # shadcn/ui Komponenten
├── App.tsx        # Hauptkomponente
└── index.css      # Tailwind v4 Styles

engine/             # Python Backend
├── engine.py      # Vollversion (Presidio)
└── engine_simple.py # Einfache Version (Regex)

data/               # Runtime-Verzeichnisse
├── input/         # Eingabedateien
└── output/        # Pseudonymisierte Dateien

src-tauri/          # Tauri Konfiguration
└── tauri.conf.json
```

## Troubleshooting

### Python 3.14+ Kompatibilitätsprobleme

```bash
# Verwende Python 3.11 oder 3.12
pyenv install 3.11.7
pyenv local 3.11.7

# Oder verwende die einfache Version
npm run engine:setup:simple
npm run engine:build:simple
```

### Dependencies-Probleme

```bash
# Vollversion mit spezifischen Versionen
pip install -r requirements-minimal.txt

# Oder einfache Version ohne externe Dependencies
npm run engine:setup:simple
```

## Build

```bash
# Vollversion
npm run build

# Einfache Version
npm run engine:build:simple
npm run build
```

## Features-Übersicht

| Feature | Vollversion | Einfache Version |
|---------|-------------|------------------|
| Text/JSON | ✅ ML-basiert | ✅ Regex-basiert |
| PDF | ✅ PyMuPDF | ❌ Nicht unterstützt |
| Images | ✅ OCR + Redaction | ❌ Nicht unterstützt |
| Dependencies | Viele | Minimal |
| Setup-Zeit | 5-10 min | 1-2 min |
| Genauigkeit | Hoch | Mittel |