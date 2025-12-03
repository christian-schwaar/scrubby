#!/usr/bin/env python3
import argparse
import json
import os
import sys

try:
    import fitz  # PyMuPDF
except Exception as e:
    print(json.dumps({"ok": False, "error": f"PyMuPDF missing: {e}"}))
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--redactions", required=True, help="JSON: [{page:int, rects:[[x0,y0,x1,y1],...]},...]")
    args = parser.parse_args()

    src_path = os.path.abspath(args.input)
    if not os.path.isfile(src_path):
        print(json.dumps({"ok": False, "error": f"Input not found: {src_path}"}))
        return 1

    try:
        redactions = json.loads(args.redactions)
    except Exception as e:
        print(json.dumps({"ok": False, "error": f"Invalid redactions JSON: {e}"}))
        return 1

    # Build output path
    base, ext = os.path.splitext(src_path)
    out_path = f"{base}-edited{ext or '.pdf'}"

    try:
        doc = fitz.open(src_path)
        # expect page numbers 1-based coming from UI
        for item in redactions:
            page_num = int(item.get("page", 0))
            if page_num <= 0 or page_num > len(doc):
                continue
            page = doc[page_num - 1]
            rects = item.get("rects", [])
            for r in rects:
                try:
                    x0, y0, x1, y1 = [float(v) for v in r]
                except Exception:
                    continue
                rect = fitz.Rect(x0, y0, x1, y1)
                page.add_redact_annot(rect, fill=(0, 0, 0))
            # apply per page to avoid annotation accumulation issues
            page.apply_redactions()
        doc.save(out_path)
        doc.close()
        print(json.dumps({"ok": True, "output": out_path}))
        return 0
    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e)}))
        return 1


if __name__ == "__main__":
    sys.exit(main())


