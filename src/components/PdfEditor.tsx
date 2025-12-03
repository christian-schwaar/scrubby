import React, { useEffect, useMemo, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { ZoomIn, ZoomOut, RotateCcw } from "lucide-react";
// pdfjs-dist setup for Vite/ESM
// @ts-ignore
import * as pdfjsLib from "pdfjs-dist";
// @ts-ignore
import pdfjsWorker from "pdfjs-dist/build/pdf.worker.min.mjs?worker";
import "pdfjs-dist/web/pdf_viewer.css";
// @ts-ignore
import { EventBus, PDFPageView } from "pdfjs-dist/web/pdf_viewer";
import { invoke } from "@tauri-apps/api/core";
import { readFile } from "@tauri-apps/plugin-fs";

type RedactionRect = [number, number, number, number]; // [x0, y0, x1, y1] in PDF points
type PageRedactions = { page: number; rects: RedactionRect[] };

interface PdfEditorProps {
  fileUrl: string;
  filePath?: string | null;
  onCancel: () => void;
  onApply: (redactions: PageRedactions[]) => void;
}

export const PdfEditor: React.FC<PdfEditorProps> = ({ fileUrl, filePath, onCancel, onApply }) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [scale, setScale] = useState(1.0);
  const eventBus = useMemo(() => new EventBus(), []);
  const pagesMeta = useRef<{ height: number; width: number }[]>([]);
  const pdfDocRef = useRef<any>(null);
  const resizeTimeoutRef = useRef<any>(null);

  const [redactions, setRedactions] = useState<PageRedactions[]>([]);
  const [isDrawing, setIsDrawing] = useState(false);
  const [startPoint, setStartPoint] = useState<{ x: number; y: number; pageIndex: number } | null>(null);
  const [currentRect, setCurrentRect] = useState<{ x: number; y: number; w: number; h: number } | null>(null);

  useEffect(() => {
    // @ts-ignore
    pdfjsLib.GlobalWorkerOptions.workerPort = new pdfjsWorker();

    let cancelled = false;
    (async () => {
      try {
        setLoading(true);
        setError(null);
        
        async function loadBytes(): Promise<Uint8Array | null> {
          let abs: string | undefined;
          if (filePath) abs = filePath;
          else if (fileUrl) {
            let encoded: string | null = null;
            if (fileUrl.startsWith("asset://")) encoded = fileUrl.replace(/^asset:\/\/localhost\//, "").replace(/^asset:\/\//, "");
            else if (fileUrl.includes("asset.localhost")) {
               try { encoded = new URL(fileUrl).pathname; } catch { encoded = fileUrl.replace(/^https?:\/\/asset\.localhost\//, ""); }
            }
            if (encoded) {
               try { abs = decodeURIComponent(encoded); } catch { abs = encoded; }
               if (abs && !abs.startsWith("/") && !abs.match(/^[a-zA-Z]:/)) abs = "/" + abs;
            }
          }
          
          console.log("[PdfEditor] Trying to load:", abs);

          if (abs) {
            // 1. Try custom command (no scope check by Tauri, but OS permissions apply)
            try {
              console.log("[PdfEditor] Invoking read_file_base64...");
              const b64 = await invoke<string>("read_file_base64", { path: abs });
              console.log("[PdfEditor] read_file_base64 success, decoding...");
              const bin = atob(b64);
              const arr = new Uint8Array(bin.length);
              for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
              return arr;
            } catch (e1) {
              console.warn("[PdfEditor] read_file_base64 failed:", e1);
              // 2. Try plugin-fs (subject to scope)
              try {
                console.log("[PdfEditor] Trying plugin-fs readFile...");
                const bytes = await readFile(abs);
                return bytes as unknown as Uint8Array;
              } catch (e2) {
                 console.warn("[PdfEditor] plugin-fs readFile failed:", e2);
              }
            }
          }
          
          // 3. Fallback fetch
          if (fileUrl && /^https?:\/\//i.test(fileUrl)) {
             try {
               console.log("[PdfEditor] Fallback fetch:", fileUrl);
               const res = await fetch(fileUrl);
               if (res.ok) return new Uint8Array(await res.arrayBuffer());
             } catch (e3) {
               console.warn("[PdfEditor] Fallback fetch failed:", e3);
             }
          }
          return null;
        }

        const data = await loadBytes();
        if (!data) throw new Error("PDF konnte nicht geladen werden (siehe Konsole für Details).");
        
        const pdf = await pdfjsLib.getDocument({ data }).promise;
        if (!cancelled) {
          pdfDocRef.current = pdf;
          await renderPages(pdf);
          setLoading(false);
        }
      } catch (e: any) {
        console.error("PdfEditor fatal error:", e);
        if (!cancelled) {
          setError(String(e?.message || e));
          setLoading(false);
        }
      }
    })();

    return () => { cancelled = true; };
  }, [fileUrl, filePath]);

  const renderPages = async (pdf: any) => {
    if (!containerRef.current) return;
    const container = containerRef.current;
    container.innerHTML = "";
    pagesMeta.current = [];
    
    // Fester, dokumentbasierter Scale: Seiten behalten ihre Original-Proportion (i.d.R. A4).
    // Vergrößerung/Verkleinerung erfolgt ausschließlich über die Zoom-Controls.
    const currentScale = scale;

    for (let i = 1; i <= pdf.numPages; i++) {
      const page = await pdf.getPage(i);
      // Unskalierter Viewport für PDF-Koordinaten (immer 1.0)
      const baseViewport = page.getViewport({ scale: 1.0 });
      pagesMeta.current.push({ width: baseViewport.width, height: baseViewport.height });
      // Skaliertes Viewport für Darstellung im UI
      const viewport = page.getViewport({ scale: currentScale });
      
      const pageHost = document.createElement("div");
      pageHost.className = "pdf-page relative shadow-md mx-auto bg-white";
      pageHost.style.position = "relative";
      pageHost.style.marginBottom = "24px";
      pageHost.style.width = `${viewport.width}px`;
      pageHost.style.height = `${viewport.height}px`;
      pageHost.dataset.pageNumber = String(i);
      pageHost.dataset.pageIndex = String(i - 1);
      
      container.appendChild(pageHost);

      const pageView = new PDFPageView({
        container: pageHost,
        id: i,
        scale: currentScale,
        defaultViewport: viewport,
        eventBus,
        textLayerMode: 0,
        annotationMode: 0,
      });
      // @ts-ignore
      pageView.setPdfPage(page);
      await pageView.draw();
    }
    drawRedactions();
  };

  useEffect(() => {
    drawRedactions();
  }, [redactions, scale, currentRect, loading]);

  const drawRedactions = () => {
    if (!containerRef.current) return;
    const pages = containerRef.current.querySelectorAll(".pdf-page");
    pages.forEach((pageEl) => {
      const oldOverlay = pageEl.querySelector(".redact-layer");
      if (oldOverlay) oldOverlay.remove();

      const pageIndex = Number((pageEl as HTMLElement).dataset.pageIndex);
      const meta = pagesMeta.current[pageIndex];
      if (!meta) return;

      const overlay = document.createElement("div");
      overlay.className = "redact-layer";
      overlay.style.position = "absolute";
      overlay.style.inset = "0";
      overlay.style.pointerEvents = "none";
      overlay.style.zIndex = "100";
      pageEl.appendChild(overlay);

      const pageRedactions = redactions.find(r => r.page === pageIndex + 1);
      if (pageRedactions) {
        pageRedactions.rects.forEach(r => {
          const [x0, y0, x1, y1] = r;
          const x = x0 * scale;
          const w = (x1 - x0) * scale;
          const h = (y1 - y0) * scale;
          const y = (meta.height - y1) * scale;

          const box = document.createElement("div");
          box.style.position = "absolute";
          box.style.left = `${x}px`;
          box.style.top = `${y}px`;
          box.style.width = `${w}px`;
          box.style.height = `${h}px`;
          box.style.backgroundColor = "black";
          overlay.appendChild(box);
        });
      }

      if (isDrawing && startPoint && startPoint.pageIndex === pageIndex && currentRect) {
        const box = document.createElement("div");
        box.style.position = "absolute";
        box.style.left = `${currentRect.x}px`;
        box.style.top = `${currentRect.y}px`;
        box.style.width = `${currentRect.w}px`;
        box.style.height = `${currentRect.h}px`;
        box.style.backgroundColor = "rgba(0, 0, 0, 0.5)";
        box.style.border = "1px solid black";
        overlay.appendChild(box);
      }
    });
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    const target = e.target as HTMLElement;
    const pageEl = target.closest(".pdf-page") as HTMLElement;
    if (!pageEl) return;

    const rect = pageEl.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const pageIndex = Number(pageEl.dataset.pageIndex);

    setStartPoint({ x, y, pageIndex });
    setIsDrawing(true);
    setCurrentRect({ x, y, w: 0, h: 0 });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDrawing || !startPoint) return;
    
    const container = containerRef.current;
    if (!container) return;
    const pages = container.querySelectorAll(".pdf-page");
    const pageEl = pages[startPoint.pageIndex] as HTMLElement;
    if (!pageEl) return;

    const currentX = e.clientX - pageEl.getBoundingClientRect().left;
    const currentY = e.clientY - pageEl.getBoundingClientRect().top;

    const width = Math.abs(currentX - startPoint.x);
    const height = Math.abs(currentY - startPoint.y);
    const left = Math.min(currentX, startPoint.x);
    const top = Math.min(currentY, startPoint.y);

    setCurrentRect({ x: left, y: top, w: width, h: height });
  };

  const handleMouseUp = () => {
    if (!isDrawing || !startPoint || !currentRect) {
      setIsDrawing(false);
      setStartPoint(null);
      setCurrentRect(null);
      return;
    }

    const pageIndex = startPoint.pageIndex;
    const meta = pagesMeta.current[pageIndex];
    
    if (meta && currentRect.w > 5 && currentRect.h > 5) {
        const x0 = currentRect.x / scale;
        const x1 = (currentRect.x + currentRect.w) / scale;
        const y1_pdf = meta.height - (currentRect.y / scale);
        const y0_pdf = meta.height - ((currentRect.y + currentRect.h) / scale);

        const newRect: RedactionRect = [x0, y0_pdf, x1, y1_pdf];

        setRedactions(prev => {
            const pageRedactions = prev.find(p => p.page === pageIndex + 1);
            if (pageRedactions) {
                return prev.map(p => p.page === pageIndex + 1 ? { ...p, rects: [...p.rects, newRect] } : p);
            } else {
                return [...prev, { page: pageIndex + 1, rects: [newRect] }];
            }
        });
    }

    setIsDrawing(false);
    setStartPoint(null);
    setCurrentRect(null);
  };

  useEffect(() => {
    const handleResize = () => {
       if (resizeTimeoutRef.current) clearTimeout(resizeTimeoutRef.current);
       resizeTimeoutRef.current = setTimeout(() => {
         if (pdfDocRef.current) renderPages(pdfDocRef.current);
       }, 300);
    };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  useEffect(() => {
    if (pdfDocRef.current && !loading) renderPages(pdfDocRef.current);
  }, [scale]);

  const handleZoomIn = () => setScale(s => Math.min(s + 0.25, 3.0));
  const handleZoomOut = () => setScale(s => Math.max(s - 0.25, 0.5));
  const handleUndo = () => {
      setRedactions(prev => {
          const nonEmpty = prev.filter(p => p.rects.length > 0);
          if (nonEmpty.length === 0) return prev;
          const lastPage = nonEmpty[nonEmpty.length - 1];
          const newRects = lastPage.rects.slice(0, -1);
          return prev.map(p => p.page === lastPage.page ? { ...p, rects: newRects } : p).filter(p => p.rects.length > 0);
      });
  };

  return (
    <div 
        className="fixed inset-0 bg-background/95 backdrop-blur-sm z-50 flex flex-col h-screen w-screen select-none"
        onMouseUp={handleMouseUp}
    >
      <div className="flex items-center justify-between px-6 py-3 border-b bg-background shadow-sm shrink-0">
        <div className="font-medium text-lg">PDF manuell schwärzen</div>
        
        <div className="flex items-center gap-2 bg-secondary/50 rounded-lg p-1">
            <Button variant="ghost" size="icon" onClick={handleZoomOut} disabled={scale <= 0.5}>
                <ZoomOut className="h-4 w-4" />
            </Button>
            <span className="text-xs font-mono w-12 text-center">{Math.round(scale * 100)}%</span>
            <Button variant="ghost" size="icon" onClick={handleZoomIn} disabled={scale >= 3.0}>
                <ZoomIn className="h-4 w-4" />
            </Button>
        </div>

        <div className="flex items-center gap-2">
          <Button variant="outline" size="icon" onClick={handleUndo} title="Letzte Box entfernen">
              <RotateCcw className="h-4 w-4" />
          </Button>
          <div className="w-px h-6 bg-border mx-2"></div>
          <Button variant="ghost" onClick={onCancel}>Abbrechen</Button>
          <Button onClick={() => onApply(redactions)} disabled={loading || !!error}>
             {loading ? "Lädt..." : "Anwenden"}
          </Button>
        </div>
      </div>
      
      <div 
        className="flex-1 overflow-auto bg-accent/10 p-8 flex flex-col items-center cursor-crosshair"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
      >
        {loading && (
            <div className="flex flex-col items-center justify-center h-full gap-3 text-muted-foreground">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
                <p>Lade PDF...</p>
            </div>
        )}
        {error && (
            <div className="flex flex-col items-center justify-center h-full gap-2 text-destructive">
                <p className="font-medium">Fehler beim Laden</p>
                <p className="text-sm opacity-80">{error}</p>
            </div>
        )}
        
        <div ref={containerRef} className={`transition-opacity duration-200 ${loading ? 'opacity-0' : 'opacity-100'}`} />
      </div>
    </div>
  );
};

export default PdfEditor;
