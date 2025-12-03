import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Copy, ExternalLink, FolderOpen, Info } from "lucide-react";
import { invoke } from "@tauri-apps/api/core";

interface OutputActionsProps {
  outputPath: string;
  onError?: (error: string) => void;
}

export function OutputActions({ outputPath, onError }: OutputActionsProps) {
  const [copied, setCopied] = useState(false);

  const handleCopyPath = async () => {
    try {
      await navigator.clipboard.writeText(outputPath);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (error) {
      onError?.("Fehler beim Kopieren des Pfads");
    }
  };

  const handleOpenFolder = async () => {
    try {
      // Extrahiere den Ordner-Pfad aus dem Datei-Pfad
      const folderPath = outputPath.substring(0, outputPath.lastIndexOf('/'));
      
      // Verwende Tauri Shell Command für plattformspezifisches Öffnen
      await invoke('open_folder', { path: folderPath });
    } catch (error) {
      onError?.(`Fehler beim Öffnen des Ordners: ${error}`);
    }
  };

  const handleOpenFile = async () => {
    try {
      // Versuche die Datei direkt zu öffnen
      await invoke('open_file', { path: outputPath });
    } catch (error) {
      onError?.(`Fehler beim Öffnen der Datei: ${error}`);
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Ergebnis</CardTitle>
        <CardDescription>
          Die verarbeitete Datei steht bereit
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Output Path */}
        <div className="space-y-2">
          <label className="text-sm font-medium">Ausgabepfad:</label>
          <div className="flex items-center space-x-2">
            <Input
              value={outputPath}
              readOnly
              className="flex-1 font-mono text-sm"
            />
            <Button
              variant="outline"
              size="icon"
              onClick={handleCopyPath}
              title="Pfad kopieren"
            >
              <Copy className="h-4 w-4" />
            </Button>
            {copied && (
              <span className="text-sm text-green-600">Kopiert!</span>
            )}
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex flex-wrap gap-2">
          <Button
            variant="outline"
            onClick={handleOpenFolder}
            className="flex items-center gap-2"
          >
            <FolderOpen className="h-4 w-4" />
            Ordner öffnen
          </Button>
          
          <Button
            variant="outline"
            onClick={handleOpenFile}
            className="flex items-center gap-2"
          >
            <ExternalLink className="h-4 w-4" />
            Datei öffnen
          </Button>
        </div>

        {/* Drag-Out Fallback Info */}
        <div className="bg-blue-50 dark:bg-blue-950 p-4 rounded-lg border border-blue-200 dark:border-blue-800">
          <div className="flex items-start gap-2">
            <Info className="h-5 w-5 text-blue-600 dark:text-blue-400 mt-0.5 shrink-0" />
            <div className="text-sm text-blue-800 dark:text-blue-200">
              <p className="font-medium mb-1">Drag-Out Fallback</p>
              <p className="mb-2">
                Echte Drag-Out-Funktionalität kommt später. Bis dahin:
              </p>
              <ul className="list-disc list-inside space-y-1 text-xs">
                <li>Klicke "Ordner öffnen" um den Ausgabeordner zu öffnen</li>
                <li>Ziehe die Datei aus dem Finder/Explorer in andere Apps</li>
                <li>Oder kopiere den Pfad und füge ihn in ChatGPT ein</li>
              </ul>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}


