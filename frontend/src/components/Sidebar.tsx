import {
  Blocks,
  Bot,
  ChevronRight,
  DatabaseZap,
  FlaskConical,
  Home,
  LogOut,
  ScanSearch,
  SlidersHorizontal,
  Sparkles,
} from "lucide-react";
import type { PipelineStage, User } from "../types";
import { Brand } from "./Brand";

const stages = [
  { label: "Extraction", icon: ScanSearch, key: "extraction" as PipelineStage },
  { label: "Refinement", icon: SlidersHorizontal, key: "refinement" as PipelineStage },
  { label: "Annotation", icon: DatabaseZap, key: null },
  { label: "RL training", icon: FlaskConical, key: null },
  { label: "Deployment", icon: Bot, key: null },
];

interface Props {
  user: User;
  page: "dashboard" | "project";
  stage: PipelineStage;
  onHome: () => void;
  onStage: (stage: PipelineStage) => void;
  onLogout: () => void;
}

export function Sidebar({ user, page, stage, onHome, onStage, onLogout }: Props) {
  const initials = user.email.slice(0, 2).toUpperCase();
  return (
    <aside className="sidebar">
      <Brand light />
      <nav className="sidebar-nav" aria-label="Primary">
        <button className={page === "dashboard" ? "nav-item is-active" : "nav-item"} onClick={onHome}>
          <Home size={18} /> Overview
        </button>
        <div className="sidebar-label">Pipeline</div>
        {stages.map((item, index) => (
          <button className={`nav-item nav-stage ${item.key === stage && page === "project" ? "is-active" : ""}`} key={item.label} disabled={!item.key || page !== "project"} onClick={() => item.key && onStage(item.key)}>
            <span className="stage-num">{String(index + 1).padStart(2, "0")}</span>
            <item.icon size={17} />
            <span>{item.label}</span>
            {!item.key && <span className="soon-dot" title="Coming next" />}
          </button>
        ))}
      </nav>
      <div className="sidebar-callout">
        <span><Sparkles size={15} /> In focus</span>
        <strong>{stage === "refinement" ? "State-space refinement" : "Late-fusion extraction"}</strong>
        <p>{stage === "refinement" ? "Find missing state dimensions and preserve a rationale trail." : "Trace every source from summary to final component."}</p>
        <div className="mini-flow"><i /><ChevronRight size={12} /><i /><ChevronRight size={12} /><i /></div>
      </div>
      <div className="sidebar-user">
        <span className="avatar">{initials}</span>
        <span className="sidebar-user__text"><strong>{user.email.split("@")[0]}</strong><small>{user.email}</small></span>
        <button className="icon-button icon-button--dark" onClick={onLogout} title="Sign out"><LogOut size={17} /></button>
      </div>
    </aside>
  );
}
