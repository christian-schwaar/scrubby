import { useState, useCallback, useEffect } from "react";
import { useDropzone } from "react-dropzone";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { FileTile } from "@/components/FileTile";
import { AlertCircle, RotateCcw, Share2, FileText, Plus, X } from "lucide-react";
import { listen } from "@tauri-apps/api/event";
import { invoke, convertFileSrc } from "@tauri-apps/api/core";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { cn } from "@/lib/utils";

interface ProcessingState {
  isProcessing: boolean;
  progress: number;
  status: string;
  error?: string;
}

type SelectedFileLike = Pick<File, "name" | "type" | "size">;

interface NavItem {
  id: string;
  label: string;
}

interface SessionState {
  selectedFile: SelectedFileLike | null;
  selectedRealFile: File | null;
  selectedPath: string | null;
  outputFileName: string | null;
  outputPath: string | null;
  inputPreviewUrl: string | null;
  outputPreviewUrl: string | null;
  processingState: ProcessingState;
  accuracy: 0.6 | 0.85;
}

interface PersistedSessionState {
  selectedFile: SelectedFileLike | null;
  selectedPath: string | null;
  outputFileName: string | null;
  outputPath: string | null;
  accuracy?: 0.6 | 0.85;
}

const makeEmptySession = (): SessionState => ({
  selectedFile: null,
  selectedRealFile: null,
  selectedPath: null,
  outputFileName: null,
  outputPath: null,
  inputPreviewUrl: null,
  outputPreviewUrl: null,
  processingState: {
    isProcessing: false,
    progress: 0,
    status: "Bereit",
  },
  accuracy: 0.6,
});

function App() {
  const MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024; // 100 MB

  const [navItems, setNavItems] = useState<NavItem[]>([
    {
      id: "tab-1",
      label: "Neue Datei",
    },
  ]);
  const [activeNavId, setActiveNavId] = useState<string>("tab-1");

  const [sessions, setSessions] = useState<Record<string, SessionState>>({
    "tab-1": makeEmptySession(),
  });

  // Beim Start: Verlauf aus localStorage laden
  useEffect(() => {
    try {
      if (typeof window === "undefined" || !window.localStorage) return;
      const raw = window.localStorage.getItem("tauro-sanitizer-history-v1");
      if (!raw) return;
      const parsed = JSON.parse(raw) as {
        navItems?: NavItem[];
        activeNavId?: string;
        sessions?: Record<string, PersistedSessionState>;
      } | null;
      if (!parsed || !Array.isArray(parsed.navItems) || parsed.navItems.length === 0) return;

      const loadedNavItems = parsed.navItems;
      const persistedSessions = parsed.sessions ?? {};
      const nextSessions: Record<string, SessionState> = {};

      for (const item of loadedNavItems) {
        const base = makeEmptySession();
        const persisted = persistedSessions[item.id];
        if (persisted) {
          let outputPreviewUrl: string | null = null;
          if (persisted.outputPath) {
            try {
              outputPreviewUrl = convertFileSrc(persisted.outputPath);
            } catch {
              outputPreviewUrl = null;
            }
          }
          nextSessions[item.id] = {
            ...base,
            selectedFile: persisted.selectedFile,
            selectedPath: persisted.selectedPath,
            outputFileName: persisted.outputFileName,
            outputPath: persisted.outputPath,
            outputPreviewUrl,
            accuracy: persisted.accuracy ?? 0.6,
          };
        } else {
          nextSessions[item.id] = base;
        }
      }

      setNavItems(loadedNavItems);
      setSessions(nextSessions);
      const candidateActive = parsed.activeNavId;
      const validActive =
        candidateActive && loadedNavItems.some((i) => i.id === candidateActive)
          ? candidateActive
          : loadedNavItems[0].id;
      setActiveNavId(validActive);
    } catch {
      // Ignoriere defekten Speicher
    }
  }, []);

  const handleAddTab = () => {
    const nextIndex = navItems.length + 1;
    const newId = `tab-${Date.now()}-${nextIndex}`;
    const newItem: NavItem = {
      id: newId,
      label: "Neue Datei",
    };
    setNavItems((prev) => [...prev, newItem]);
    setSessions((prev) => ({
      ...prev,
      [newId]: makeEmptySession(),
    }));
    setActiveNavId(newId);
  };

  const handleRemoveTab = (id: string) => {
    if (navItems.length <= 1) {
      return; // Immer mindestens ein Tab behalten
    }

    const filtered = navItems.filter((item) => item.id !== id);
    let nextActiveId = activeNavId;

    if (id === activeNavId) {
      const fallback = filtered[filtered.length - 1];
      if (fallback) {
        nextActiveId = fallback.id;
      }
    }

    setNavItems(filtered);
    setActiveNavId(nextActiveId);
    setSessions((prev) => {
      const { [id]: _removed, ...rest } = prev;
      if (!rest[nextActiveId]) {
        rest[nextActiveId] = makeEmptySession();
      }
      return rest;
    });
  };

  // Verlauf im localStorage sichern, sobald sich Tabs oder Sessions ändern
  useEffect(() => {
    try {
      if (typeof window === "undefined" || !window.localStorage) return;
      const serializableSessions: Record<string, PersistedSessionState> = {};
      for (const [id, s] of Object.entries(sessions)) {
        serializableSessions[id] = {
          selectedFile: s.selectedFile,
          selectedPath: s.selectedPath,
          outputFileName: s.outputFileName,
          outputPath: s.outputPath,
          accuracy: s.accuracy,
        };
      }
      const payload = {
        navItems,
        activeNavId,
        sessions: serializableSessions,
      };
      window.localStorage.setItem(
        "tauro-sanitizer-history-v1",
        JSON.stringify(payload)
      );
    } catch {
      // Ignoriere fehlgeschlagenes Persistieren
    }
  }, [navItems, sessions, activeNavId]);

  // Filter settings for advanced options
  const [filterSettings] = useState({
    names: true,
    addresses: true,
    phoneNumbers: true,
    emails: true,
    iban: true,
  });

  const arrayBufferToBase64 = async (buffer: ArrayBuffer): Promise<string> => {
    // Convert safely in chunks to avoid call stack / memory spikes
    const bytes = new Uint8Array(buffer);
    const chunkSize = 0x8000; // 32k
    let binary = "";
    for (let i = 0; i < bytes.length; i += chunkSize) {
      const chunk = bytes.subarray(i, i + chunkSize);
      // Build substring for this chunk
      let sub = "";
      for (let j = 0; j < chunk.length; j++) {
        sub += String.fromCharCode(chunk[j]);
      }
      binary += sub;
    }
    return btoa(binary);
  };

  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      if (acceptedFiles.length > 0) {
        const f = acceptedFiles[0];
        if (f.size > MAX_FILE_SIZE_BYTES) {
          // Guard only in stub
          return;
        }
        setSessions((prev) => {
          const existing = prev[activeNavId] ?? makeEmptySession();
          let nextInputUrl: string | null = null;
          try {
            nextInputUrl = URL.createObjectURL(f);
          } catch {
            nextInputUrl = null;
          }
          return {
            ...prev,
            [activeNavId]: {
              ...existing,
              selectedFile: { name: f.name, type: f.type, size: f.size },
              selectedRealFile: f,
              selectedPath: null,
              inputPreviewUrl: nextInputUrl,
              outputFileName: null,
              outputPath: null,
              outputPreviewUrl: null,
              processingState: {
                isProcessing: false,
                progress: 0,
                status: "Datei bereit für Verarbeitung",
              },
            },
          };
        });
        // Label des aktiven Tabs anpassen
        setNavItems((prev) =>
          prev.map((item) =>
            item.id === activeNavId ? { ...item, label: f.name } : item
          )
        );
        // ignore additional files in stub
      }
    },
    [MAX_FILE_SIZE_BYTES, activeNavId]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'image/*': ['.png', '.jpg', '.jpeg'],
      'text/plain': ['.txt'],
      'text/markdown': ['.md'],
      'application/json': ['.json'],
    },
    multiple: false,
  });

  // Tauri OS-level file drop support → mirror to UI state
  useEffect(() => {
    let unlisten: (() => void) | undefined;
    (async () => {
      try {
        unlisten = await listen<any>("tauri://file-drop", (event) => {
          const payload: any = event.payload;
          const files: string[] | undefined = Array.isArray(payload)
            ? payload
            : (Array.isArray(payload?.paths) ? payload.paths : (Array.isArray(payload?.files) ? payload.files : undefined));
          if (!files || files.length === 0) return;
          const raw = String(files[0] ?? "");
          const filePath = raw.startsWith("file://") ? raw.replace(/^file:\/\//, "") : raw;
          const name = filePath.split("/").pop() || "file";
          const mock: SelectedFileLike = { name, type: "application/octet-stream", size: 0 };
          setSessions((prev) => {
            const existing = prev[activeNavId] ?? makeEmptySession();
            return {
              ...prev,
              [activeNavId]: {
                ...existing,
                selectedFile: mock,
                selectedRealFile: null,
                selectedPath: filePath,
                inputPreviewUrl: null,
                outputFileName: null,
                outputPath: null,
                outputPreviewUrl: null,
                processingState: {
                  isProcessing: false,
                  progress: 0,
                  status: "Datei bereit für Verarbeitung",
                },
              },
            };
          });
          setNavItems((prev) =>
            prev.map((item) =>
              item.id === activeNavId ? { ...item, label: name } : item
            )
          );
        });
      } catch (_) {
        // ignore if not running in tauri
      }
    })();
    return () => {
      if (unlisten) unlisten();
    };
  }, [activeNavId]);

  const handleOpenFinder = async () => {
    try {
      const session = sessions[activeNavId] ?? makeEmptySession();
      const outPath = session.outputPath;
      const target = outPath && outPath.includes('/')
        ? outPath.substring(0, outPath.lastIndexOf('/'))
        : 'data/output';
      await invoke("open_folder", { path: target });
    } catch (_) {
      // silently ignore in non-tauri context
    }
  };

  const handleStartProcessing = async () => {
    const session = sessions[activeNavId] ?? makeEmptySession();
    const { selectedFile, selectedRealFile, selectedPath, accuracy } = session;
    if (!selectedFile) return;
    setSessions((prev) => {
      const current = prev[activeNavId] ?? makeEmptySession();
      return {
        ...prev,
        [activeNavId]: {
          ...current,
          processingState: {
            isProcessing: true,
            progress: 0,
            status: "Verarbeitung gestartet...",
          },
        },
      };
    });

    // Prefer real engine when available (Tauri). Fallback to stub.
    try {
      const engineMode = "pseudo";
      const threshold = accuracy ?? 0.6;
      let pathToUse = selectedPath;
      if (!pathToUse && selectedRealFile) {
        const arrayBuf = await selectedRealFile.arrayBuffer();
        const base64 = await arrayBufferToBase64(arrayBuf);
        pathToUse = await invoke<string>('write_temp_file', { fileName: selectedRealFile.name, dataBase64: base64 });
        const finalPath = pathToUse;
        setSessions((prev) => {
          const current = prev[activeNavId] ?? makeEmptySession();
          return {
            ...prev,
            [activeNavId]: {
              ...current,
              selectedPath: finalPath,
            },
          };
        });
      }
      if (pathToUse) {
        const result = await invoke<string>('run_engine', {
          input: pathToUse,
          mode: engineMode,
          outputDir: 'data/output',
          language: 'de',
          ocr: true,
          threshold,
          filters: {
            names: filterSettings.names,
            addresses: filterSettings.addresses,
            phoneNumbers: filterSettings.phoneNumbers,
            emails: filterSettings.emails,
            iban: filterSettings.iban,
          }
        });
        let parsed: any;
        try {
          parsed = JSON.parse(result);
        } catch {
          throw new Error('Engine returned non-JSON output');
        }
        if (parsed.ok) {
          const outPath: string = parsed.output;
          let previewUrl: string | null = null;
          try {
            previewUrl = convertFileSrc(outPath);
          } catch {
            previewUrl = null;
          }
          const name = outPath.split('/').pop() || selectedFile.name;
          const finalPreviewUrl = previewUrl;
          setSessions((prev) => {
            const current = prev[activeNavId] ?? makeEmptySession();
            return {
              ...prev,
              [activeNavId]: {
                ...current,
                outputPath: outPath,
                outputPreviewUrl: finalPreviewUrl,
                outputFileName: name,
                processingState: {
                  isProcessing: false,
                  progress: 100,
                  status: "Verarbeitung abgeschlossen",
                },
              },
            };
          });
          return;
        }
        throw new Error(parsed.error || 'Engine error');
      }
    } catch (err: any) {
      const message = String(err?.message || err);
      setSessions((prev) => {
        const current = prev[activeNavId] ?? makeEmptySession();
        return {
          ...prev,
          [activeNavId]: {
            ...current,
            processingState: {
              isProcessing: false,
              progress: 0,
              status: "Fehler aufgetreten",
              error: message,
            },
          },
        };
      });
      return;
    }

    // If engine path missing
    setSessions((prev) => {
      const current = prev[activeNavId] ?? makeEmptySession();
      return {
        ...prev,
        [activeNavId]: {
          ...current,
          processingState: {
            isProcessing: false,
            progress: 0,
            status: "Fehler aufgetreten",
            error: "Kein Eingabepfad für Engine",
          },
        },
      };
    });
  };

  const handleReset = () => {
    const session = sessions[activeNavId];
    if (session) {
      try {
        if (session.inputPreviewUrl) URL.revokeObjectURL(session.inputPreviewUrl);
        if (session.outputPreviewUrl) URL.revokeObjectURL(session.outputPreviewUrl);
      } catch {}
    }
    setSessions((prev) => ({
      ...prev,
      [activeNavId]: makeEmptySession(),
    }));
  };

  // Icon/Preview handled by FileTile design
  const renderInputPreview = () => {
    const session = sessions[activeNavId] ?? makeEmptySession();
    const { selectedFile, inputPreviewUrl, selectedPath } = session;
    if (!selectedFile) return null;
    const nameLower = selectedFile.name.toLowerCase();
    const mime = selectedFile.type || "";

    const isPdf =
      mime === "application/pdf" || nameLower.endsWith(".pdf");
    const isImage =
      mime.startsWith("image/") ||
      /\.(png|jpe?g|gif|webp|bmp)$/i.test(nameLower);
    const isTextLike =
      mime.startsWith("text/") ||
      /\.(txt|md|json|log|csv)$/i.test(nameLower);

    let fileUrl: string | null = null;

    // Bevorzuge echten Dateipfad (z.B. bei OS-Level-Drops)
    if (selectedPath) {
      try {
        fileUrl = convertFileSrc(selectedPath);
      } catch {
        fileUrl = null;
      }
    }

    // Fallback: Blob-URL aus Drag & Drop
    if (!fileUrl && inputPreviewUrl) {
      fileUrl = inputPreviewUrl;
    }

    if (fileUrl && isPdf) {
      return (
        <iframe
          key={fileUrl}
          src={fileUrl}
          className="w-full h-full rounded-md border bg-white"
        />
      );
    }

    if (fileUrl && isImage) {
      return (
        <img
          key={fileUrl}
          src={fileUrl}
          alt={selectedFile.name}
          className="w-full h-full object-contain rounded-md border bg-white"
        />
      );
    }

    if (fileUrl && isTextLike) {
      return (
        <iframe
          key={fileUrl}
          src={fileUrl}
          className="w-full h-full rounded-md border bg-white"
        />
      );
    }

    return <FileTile fileName={selectedFile.name} showBadge={false} />;
  };

  const renderOutputPreview = () => {
    const session = sessions[activeNavId] ?? makeEmptySession();
    const { outputFileName, outputPath } = session;
    if (!outputFileName) return (
      <div className="border-2 border-dashed rounded-2xl p-12 text-center text-muted-foreground grid place-items-center h-full w-full">
        Keine Ausgabe
      </div>
    );
    const lower = outputFileName.toLowerCase();
    const isPdf = lower.endsWith(".pdf");
    const isImage = /\.(png|jpe?g|gif|webp|bmp)$/i.test(lower);
    const isTextLike = /\.(txt|md|json|log|csv)$/i.test(lower);

    if (outputPath) {
      let fileUrl: string | null = null;
      try {
        fileUrl = convertFileSrc(outputPath);
      } catch {
        fileUrl = null;
      }
      if (fileUrl) {
        if (isPdf) {
          return (
            <div className="w-full h-full rounded-md p-6 overflow-hidden">
              <iframe
                key={fileUrl}
                src={fileUrl}
                className="w-full h-full rounded-md border bg-white"
              />
            </div>
          );
        }
        if (isImage) {
          return (
            <div className="w-full h-full rounded-md p-6 overflow-hidden bg-white">
              <img
                key={fileUrl}
                src={fileUrl}
                alt={outputFileName}
                className="w-full h-full object-contain rounded-md border"
              />
            </div>
          );
        }
        if (isTextLike) {
          return (
            <div className="w-full h-full rounded-md p-6 overflow-hidden bg-white">
              <iframe
                key={fileUrl}
                src={fileUrl}
                className="w-full h-full rounded-md border bg-white"
              />
            </div>
          );
        }
      }
    }

    return <FileTile fileName={outputFileName} showBadge={true} />;
  };

  return (
    <>
      <div className="min-h-screen bg-background flex">
        {/* Apple-style Navigation als weiße, „floating“ Sidebar */}
        <aside className="shrink-0 flex p-0 w-[300px] min-w-[300px] max-w-[300px]">
          <div className="flex-1 px-4 py-4 mb-4">
            <div className="h-full p-3 rounded-3xl bg-white/95 border border-border/70 shadow-[0_18px_45px_rgba(15,23,42,0.16)] flex flex-col overflow-hidden">
              <div className="flex items-center justify-between px-2 pt-2 pb-3">
                <span className="text-[11px] font-medium text-muted-foreground/80 tracking-[0.08em] uppercase">
                  Verlauf
                </span>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-5 w-5 rounded-full hover:bg-muted/80"
                  type="button"
                  onClick={handleAddTab}
                >
                  <Plus className="h-3 w-3 text-muted-foreground" />
                </Button>
              </div>
              <nav className="flex-1 flex flex-col gap-[2px] mt-1 px-1 pb-2">
                {navItems.map((item) => (
                  <button
                    key={item.id}
                    type="button"
                    onClick={() => setActiveNavId(item.id)}
                    className={cn(
                      "group flex items-center gap-2.5 rounded-lg px-3 py-1.5 text-[13px] transition-colors text-left",
                      activeNavId === item.id
                        ? "bg-[rgba(0,113,227,0.10)] text-foreground"
                        : "text-muted-foreground hover:bg-muted/80 hover:text-foreground"
                    )}
                  >
                    <FileText
                      className={cn(
                        "h-4 w-4 shrink-0 opacity-70",
                        activeNavId === item.id
                          ? "text-sky-500 opacity-100"
                          : "group-hover:opacity-100"
                      )}
                    />
                    <span className="truncate font-medium max-w-[150px]">
                      {item.label.length > 22
                        ? `${item.label.slice(0, 22)}…`
                        : item.label}
                    </span>
                    {navItems.length > 1 && (
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleRemoveTab(item.id);
                        }}
                        className="ml-auto opacity-0 group-hover:opacity-100 hover:opacity-100 transition-opacity rounded-full hover:bg-muted/80 h-5 w-5 grid place-items-center"
                        aria-label="Tab schließen"
                        title="Tab schließen"
                      >
                        <X className="h-3 w-3 text-muted-foreground" />
                      </button>
                    )}
                  </button>
                ))}
              </nav>
            </div>
          </div>
        </aside>

        {/* Rechts: Header + Main-Content */}
        <div className="flex-1 flex flex-col mt-3">
          {/* Header nur über Content-Bereich */}
          <header className="">
            <div className="mx-auto w-full px-6 pl-4 py-3 flex items-center justify-between gap-4">
              <div className="flex items-center gap-3 min-w-0 ">
                <div className="flex flex-col min-w-0">
                  <span className="text-sm font-medium leading-none">
                    Scrubby
                  </span>
                  <span className="text-xs text-muted-foreground mt-0.5 truncate max-w-[260px]">
                    {(sessions[activeNavId]?.selectedFile?.name) ?? "Keine Datei ausgewählt"}
                  </span>
                </div>
              </div>

              {/* Rechte Finder-Style Toolbar: weiß, rund, „floating“ */}
              <div className="flex items-center justify-end">
                <div className="inline-flex items-center gap-3 rounded-full bg-white/95 border border-border/70 shadow-[0_18px_45px_rgba(15,23,42,0.18)] px-4 py-1.5 backdrop-blur">
                  <span className="text-[11px] font-medium uppercase tracking-[0.08em] text-muted-foreground">
                    Accuracy
                  </span>
                  <ToggleGroup
                    type="single"
                    value={(sessions[activeNavId]?.accuracy ?? 0.6).toString()}
                    onValueChange={(val) => {
                      if (!val) return;
                      const parsed = parseFloat(val);
                      const next = (parsed === 0.85 ? 0.85 : 0.6) as 0.6 | 0.85;
                      setSessions((prev) => {
                        const current = prev[activeNavId] ?? makeEmptySession();
                        return {
                          ...prev,
                          [activeNavId]: {
                            ...current,
                            accuracy: next,
                          },
                        };
                      });
                    }}
                    className="bg-muted/70 rounded-full px-0.5 py-0"
                  >
                    <ToggleGroupItem
                      value="0.6"
                      size="sm"
                      className="rounded-full px-3 text-[11px] font-medium"
                    >
                      0.60
                    </ToggleGroupItem>
                    <ToggleGroupItem
                      value="0.85"
                      size="sm"
                      className="rounded-full px-3 text-[11px] font-medium"
                    >
                      0.85
                    </ToggleGroupItem>
                  </ToggleGroup>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-7 w-7 rounded-full hover:bg-muted/80"
                    onClick={() => {
                      try {
                        if (navigator.share) {
                          navigator
                            .share({
                              title: "Scrubby",
                              text: sessions[activeNavId]?.selectedFile?.name
                                ? `Teile ${sessions[activeNavId]?.selectedFile?.name}`
                                : "Scrubby",
                            })
                            .catch(() => {});
                        }
                      } catch {
                        // ignore share errors in unsupported environments
                      }
                    }}
                  >
                    <Share2 className="h-3.5 w-3.5" />
                  </Button>
                </div>
              </div>
            </div>
          </header>

          {/* Main content */}
          <main className="flex-1">
            <div className="grid grid-cols-2 gap-4 px-6 pl-2 py-2 min-h-[calc(100vh-7rem)]">
              {/* Input Column */}
              <div className="space-y-3">
                <Card className="shadow-none h-full flex flex-col rounded-3xl">
                  {/*<CardHeader>
                <div className="flex items-center gap-2">
                  <div className="inline-flex rounded-md border bg-background p-0.5 text-xs">
                    <button className="px-3 py-1 text-muted-foreground" disabled>Text</button>
                    <button className="px-3 py-1 rounded-sm bg-accent text-accent-foreground">File</button>
                  </div>
                </div>
              </CardHeader>*/}
                  <CardContent className="flex-1 h-full pt-6 flex flex-col min-h-0 p-3">
                    <div
                      {...getRootProps()}
                      className={`border-2 h-full w-full flex-1 min-h-0 bg-accent/5 border-dashed rounded-2xl p-6 text-center cursor-pointer transition-colors grid ${
                        isDragActive
                          ? "border-accent bg-accent/5 "
                          : "border-accent/30 hover:border-accent/60 hover:bg-accent/5"
                      }`}
                    >
                      <input {...getInputProps()} />
                      {sessions[activeNavId]?.selectedFile ? (
                        <div className="w-full h-full grid place-items-center">
                          {renderInputPreview()}
                        </div>
                      ) : (
                        <div className="text-center self-center">
                          <p className="text-2xl font-semibold">Drag here</p>
                          <p className="text-sm text-muted-foreground mt-2">
                            *.pdf, *.png, *.jpg, *.jpeg
                          </p>
                        </div>
                      )}
                    </div>

                    {/*<div className="mt-4">
                  <Accordion type="single" collapsible>
                    <AccordionItem value="advanced">
                      <AccordionTrigger>Erweiterte Einstellungen</AccordionTrigger>
                      <AccordionContent>
                        <div className="grid gap-3">
                          <div className="flex items-center space-x-2">
                            <Checkbox
                              id="filter-all"
                              checked={allChecked ? true : (someChecked ? "indeterminate" : false)}
                              onCheckedChange={(checked) => {
                                if (checked === "indeterminate") return;
                                const next = checked === true;
                                setFilterSettings({
                                  names: next,
                                  addresses: next,
                                  phoneNumbers: next,
                                  emails: next,
                                  iban: next,
                                });
                              }}
                            />
                            <Label htmlFor="filter-all">Alles auswählen</Label>
                          </div>

                          <div className="flex items-center space-x-2">
                            <Checkbox
                              id="filter-names"
                              checked={filterSettings.names}
                              onCheckedChange={(v) => setFilterSettings((s) => ({ ...s, names: v === true }))}
                            />
                            <Label htmlFor="filter-names">Namen</Label>
                          </div>

                          <div className="flex items-center space-x-2">
                            <Checkbox
                              id="filter-addresses"
                              checked={filterSettings.addresses}
                              onCheckedChange={(v) => setFilterSettings((s) => ({ ...s, addresses: v === true }))}
                            />
                            <Label htmlFor="filter-addresses">Adressen</Label>
                          </div>

                          <div className="flex items-center space-x-2">
                            <Checkbox
                              id="filter-phone"
                              checked={filterSettings.phoneNumbers}
                              onCheckedChange={(v) => setFilterSettings((s) => ({ ...s, phoneNumbers: v === true }))}
                            />
                            <Label htmlFor="filter-phone">Telefonnummern</Label>
                          </div>

                          <div className="flex items-center space-x-2">
                            <Checkbox
                              id="filter-emails"
                              checked={filterSettings.emails}
                              onCheckedChange={(v) => setFilterSettings((s) => ({ ...s, emails: v === true }))}
                            />
                            <Label htmlFor="filter-emails">E-Mail-Adressen</Label>
                          </div>

                          <div className="flex items-center space-x-2">
                            <Checkbox
                              id="filter-iban"
                              checked={filterSettings.iban}
                              onCheckedChange={(v) => setFilterSettings((s) => ({ ...s, iban: v === true }))}
                            />
                            <Label htmlFor="filter-iban">IBAN</Label>
                          </div>
                        </div>
                      </AccordionContent>
                    </AccordionItem>
                  </Accordion>
                </div> */}

                    <div className="mt-4 flex justify-end gap-2 pb-1">
                      <Button
                        onClick={handleStartProcessing}
                        disabled={
                          !sessions[activeNavId]?.selectedFile ||
                          (sessions[activeNavId]?.processingState.isProcessing ?? false)
                        }
                        className="bg-accent rounded-full text-accent-foreground hover:bg-accent/90 px-6"
                      >
                        {sessions[activeNavId]?.processingState.isProcessing
                          ? "Verarbeitung…"
                          : "Start"}
                      </Button>
                      {sessions[activeNavId]?.selectedFile && (
                        <Button
                          type="button"
                          variant="outline"
                          className="aspect-square p-0 h-10 w-10 shrink-0 group rounded-full"
                          onClick={handleReset}
                          aria-label="Reset"
                          title="Zurücksetzen"
                        >
                          <RotateCcw className="h-4 w-4 transition-transform duration-300 group-hover:-rotate-180 text-gray-500" />
                        </Button>
                      )}
                    </div>
                  </CardContent>
                </Card>
              </div>

              {/* Output Column */}
              <div className="space-y-3">
                <Card className="shadow-none h-full flex flex-col rounded-3xl">
                  <CardContent className="w-full h-full flex-1 flex flex-col min-h-0 rounded-2xl p-3">
                    <div className="h-full w-full flex flex-col justify-between min-h-0">
                      <div className="w-full flex-1 min-h-0">
                        {renderOutputPreview()}
                      </div>
                      <div className="flex justify-end">
                        <Button
                          variant="outline"
                          className="mt-4 mb-2 rounded-full"
                          disabled={!sessions[activeNavId]?.outputFileName}
                          onClick={handleOpenFinder}
                        >
                          im Finder anzeigen
                        </Button>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </div>

            {/* Loading overlay */}
            {sessions[activeNavId]?.processingState.isProcessing && (
              <div className="fixed inset-0 bg-background/50 backdrop-blur-sm grid place-items-center z-50">
                <div className="flex flex-col items-center gap-3">
                  <div className="h-8 w-8 animate-spin rounded-full border-2 border-foreground/20 border-t-foreground" />
                  <div className="text-sm text-muted-foreground">
                    {sessions[activeNavId]?.processingState.status}
                  </div>
                </div>
              </div>
            )}

            {/* Error */}
            {sessions[activeNavId]?.processingState.error && (
              <Card className="border-destructive mt-4 mx-6">
                <CardContent className="pt-6">
                  <div className="flex items-start gap-2">
                    <AlertCircle className="h-5 w-5 text-destructive mt-0.5 shrink-0" />
                    <div>
                      <p className="font-medium text-destructive">
                        Fehler aufgetreten:
                      </p>
                      <p className="text-sm text-destructive/80 mt-1">
                        {sessions[activeNavId]?.processingState.error}
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}
          </main>
        </div>
      </div>

      {/* no toasts needed for stub */}
    </>
  );
}

export default App;
