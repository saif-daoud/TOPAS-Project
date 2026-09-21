import { ArrowRight, BookOpen, CalendarDays, Check, Clock3, FolderOpen, Layers3, Plus, Sparkles } from "lucide-react";
import type { Project, User } from "../types";

interface Props {
  user: User;
  projects: Project[];
  loading: boolean;
  onCreate: () => void;
  onOpen: (id: string) => void;
}

function relativeDate(value: string) {
  const date = new Date(value);
  const days = Math.max(0, Math.round((Date.now() - date.getTime()) / 86400000));
  if (days === 0) return "Today";
  if (days === 1) return "Yesterday";
  if (days < 7) return `${days} days ago`;
  return date.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

function statusLabel(project: Project) {
  if (project.is_demo) return "Showcase";
  if (project.status === "completed") return "Extracted";
  if (project.status === "running") return `${project.progress}% · Running`;
  if (project.status === "failed") return "Needs attention";
  return project.book_count ? "Ready to extract" : "Setup";
}

export function Dashboard({ user, projects, loading, onCreate, onOpen }: Props) {
  const firstName = user.email.split("@")[0].split(/[._-]/)[0];
  const completed = projects.filter((item) => item.status === "completed").length;
  const totalBooks = projects.reduce((sum, item) => sum + Number(item.book_count || 0), 0);
  return (
    <div className="dashboard page-enter">
      <header className="topbar">
        <div><span className="topbar-kicker">Workspace</span><h1>Overview</h1></div>
        <button className="button button--primary" onClick={onCreate}><Plus size={18} /> New project</button>
      </header>

      <section className="welcome-panel">
        <div className="welcome-panel__content">
          <span className="eyebrow"><Sparkles size={13} /> Knowledge to policy</span>
          <h2>Good to see you, <em>{firstName}.</em></h2>
          <p>Create, refine, train, and deploy domain-grounded agents in one workspace.</p>
          <button className="text-link" onClick={onCreate}>Start from your textbooks <ArrowRight size={16} /></button>
        </div>
        <div className="welcome-diagram" aria-hidden="true">
          <span className="diagram-book diagram-book--one" />
          <span className="diagram-book diagram-book--two" />
          <span className="diagram-line" />
          <span className="diagram-node diagram-node--one">S</span>
          <span className="diagram-node diagram-node--two">A</span>
          <span className="diagram-node diagram-node--three">R</span>
          <span className="diagram-core"><Layers3 size={25} /></span>
        </div>
      </section>

      <section className="stats-row" aria-label="Workspace totals">
        <div className="stat"><span className="stat__icon"><FolderOpen size={19} /></span><div><strong>{projects.length}</strong><small>Projects</small></div></div>
        <div className="stat"><span className="stat__icon"><BookOpen size={19} /></span><div><strong>{totalBooks}</strong><small>Textbooks</small></div></div>
        <div className="stat"><span className="stat__icon"><Check size={19} /></span><div><strong>{completed}</strong><small>Extractions</small></div></div>
        <div className="stat stat--quiet"><span className="stat__icon"><Clock3 size={19} /></span><div><strong>Late</strong><small>Fusion method</small></div></div>
      </section>

      <div className="section-heading"><div><h2>Your projects</h2><p>Everything is saved to this profile.</p></div></div>
      <section className="project-grid">
        <button className="project-create-card" onClick={onCreate}>
          <span><Plus size={24} /></span>
          <strong>Create a new domain</strong>
          <small>Configure roles, add sources, and extract</small>
        </button>
        {loading && [0, 1].map((item) => <div className="project-card skeleton" key={item} />)}
        {!loading && projects.map((project, index) => (
          <button className="project-card" key={project.id} onClick={() => onOpen(project.id)} style={{ animationDelay: `${index * 45}ms` }}>
            <div className="project-card__top">
              <span className={`status-badge status-badge--${project.status}`}>{project.status === "completed" && <Check size={12} />}{statusLabel(project)}</span>
              <span className="project-card__arrow"><ArrowRight size={17} /></span>
            </div>
            <div className="project-glyph"><span>{project.system_name.slice(0, 1)}</span><i /><span>{project.user_name.slice(0, 1)}</span></div>
            <span className="project-card__domain">{project.domain}</span>
            <h3>{project.name}</h3>
            <p>{project.system_name} ↔ {project.user_name} · {project.interaction_unit}</p>
            <div className="project-card__foot">
              <span><BookOpen size={14} /> {project.book_count || 0} sources</span>
              <span><CalendarDays size={14} /> {relativeDate(project.updated_at)}</span>
              <span className="model-chip">{project.model}</span>
            </div>
            {project.status === "running" && <span className="project-progress"><i style={{ width: `${project.progress}%` }} /></span>}
          </button>
        ))}
      </section>
    </div>
  );
}
