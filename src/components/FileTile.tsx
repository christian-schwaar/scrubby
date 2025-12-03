import { FileText } from "lucide-react";

interface FileTileProps {
  fileName: string;
  showBadge?: boolean;
}

export function FileTile({ fileName, showBadge = false }: FileTileProps) {
  return (
    <div className="flex items-center gap-3">
      <div className="relative">
        <div className="h-12 w-10 rounded-md bg-muted/40 grid place-items-center border border-muted">
          <FileText className="h-6 w-6" />
        </div>
        {showBadge && (
          <div className="absolute -right-2 -top-2 h-5 w-5 rounded-full bg-red-500 text-white text-[10px] grid place-items-center shadow-sm">A</div>
        )}
      </div>
      <div className="text-sm font-medium truncate max-w-48" title={fileName}>{fileName}</div>
    </div>
  );
}


