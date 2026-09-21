import { Sparkles } from "lucide-react";

export function Brand({ compact = false, light = false }: { compact?: boolean; light?: boolean }) {
  return (
    <div className={`brand ${light ? "brand--light" : ""}`} aria-label="TOPAS Studio">
      <span className="brand__mark" aria-hidden="true">
        <span className="brand__orbit" />
        <Sparkles size={11} strokeWidth={2.2} />
      </span>
      {!compact && (
        <span className="brand__words">
          <strong>TOPAS</strong>
          <small>Studio</small>
        </span>
      )}
    </div>
  );
}

