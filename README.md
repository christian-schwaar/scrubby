# Automatic PII Reduction App

Scrubby ist eine Desktop‑App auf Basis von **Tauri 2 + React + Python**, die Dokumente mit komplett lokal anonymisiert.
PDFs, Bilder und Textdateien können geladen und mit unterschiedlichen Accuracy‑Einstellungen geschwärzt werden.

![Preview](public/preview.png)

### Features

- **Multiplattform‑Desktop‑App** via Tauri (Rust Backend, React Frontend)
- **Presidio‑Engine (Python)** mit spaCy‑NER, Pattern‑Recognizers (E‑Mail, Telefon, IBAN, URL, Kreditkarten usw.)
- **PDF‑, Bild‑ und Text‑Support**
  - PDFs: Redaction direkt im PDF via PyMuPDF
  - Bilder: OCR + Schwärzung per OpenCV
  - Text / JSON: String‑basierte Anonymisierung
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
cd scrubby
```

2. **Node‑Abhängigkeiten installieren**

```bash
npm install
```

3. **Python‑Umgebung & Engine‑Deps**

- Empfohlen: virtuelles Env im Projektroot (`venv311`):

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r engine/requirements.txt
```

Stelle sicher, dass `engine/engineV2.py` alle benötigten Pakete (Presidio, spaCy, PyMuPDF, Tesseract‑Bindings etc.) installieren kann.

4. **Tauri‑CLI installieren** (falls noch nicht vorhanden)

```bash
npm install -g @tauri-apps/cli
```

---

## Entwicklung starten

```bash
npm run tauri dev
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

## Features

- **Frontend**: React + Vite + Tailwind v4 + shadcn/ui
- **Backend**: Python-Sidecar mit Presidio für Pseudonymisierung
- **Unterstützte Formate**: PDF, Bilder (PNG/JPG/JPEG), TXT, MD, JSON
- **Drag & Drop**: Intuitive Datei-Upload-Funktionalität
- **OCR**: Optional für Bildverarbeitung
- **Mehrsprachig**: Deutsch und Englisch
- **Offline**: Keine Telemetrie, vollständig offline

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
