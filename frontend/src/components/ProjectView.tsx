import { useEffect, useRef, useState, type CSSProperties, type DragEvent } from "react";
import {
  Activity,
  AlertTriangle,
  ArrowLeft,
  ArrowRight,
  BookCheck,
  BookOpen,
  Bot,
  Braces,
  Check,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  CircleDot,
  Clock3,
  Download,
  FileJson,
  FileCheck2,
  FileText,
  GitBranch,
  Layers3,
  LoaderCircle,
  Network,
  Maximize2,
  Pencil,
  Play,
  RefreshCw,
  ScanSearch,
  Search,
  ShieldAlert,
  SlidersHorizontal,
  Sparkles,
  UploadCloud,
  UserRound,
  X,
} from "lucide-react";
import { api } from "../api";
import type { Book, ComponentPhase, Manifest, PipelineStage, Project, ProjectEvent, Role } from "../types";
import { ComponentEditor } from "./ComponentEditor";
import { KnowledgeGraphEditor } from "./KnowledgeGraphEditor";

const componentMeta: Record<string, { label: string; description: string; icon: typeof Layers3 }> = {
  macro_actions: { label: "Macro actions", description: "High-level strategies and their long-term objectives", icon: Layers3 },
  micro_actions: { label: "Micro actions", description: "Concrete responses available inside each strategy", icon: GitBranch },
  conversation_states: { label: "Conversation states", description: "Variables tracked as the interaction evolves", icon: Activity },
  knowledge_graph: { label: "Knowledge graph", description: "Memory schema for domain facts and relationships", icon: Network },
  cautions: { label: "Cautions", description: "Unsafe or counterproductive system behaviors", icon: ShieldAlert },
  user_profile: { label: "User profile", description: "Static and dynamic dimensions of the user simulator", icon: UserRound },
};

function humanBytes(bytes: number) {
  if (!bytes) return "Source file";
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function friendlyTime(value: string) {
  return new Date(value).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function polishText(value: unknown): unknown {
  if (typeof value === "string") {
    return value
      .replaceAll("�", "’")
      .replaceAll("â€™", "’")
      .replaceAll("â€“", "–")
      .replaceAll("â€”", "—");
  }
  if (Array.isArray(value)) return value.map(polishText);
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, polishText(item)]));
  }
  return value;
}

function countItems(component: string, data: unknown): number {
  if (Array.isArray(data)) return data.length;
  if (!data || typeof data !== "object") return 0;
  const object = data as Record<string, unknown>;
  if (component === "macro_actions" && Array.isArray(object.macro_actions)) return object.macro_actions.length;
  if (component === "knowledge_graph" && Array.isArray(object.nodes)) return object.nodes.length;
  if (component === "user_profile") {
    return [object.static_dimensions, object.dynamic_dimensions].reduce<number>((sum, value) => sum + (Array.isArray(value) ? value.length : 0), 0);
  }
  return Object.keys(object).length;
}

interface DropzoneProps {
  role: Role;
  title: string;
  subtitle: string;
  books: Book[];
  disabled?: boolean;
  busy: boolean;
  onUpload: (role: Role, files: File[]) => Promise<void>;
  onRemove: (bookId: string) => Promise<void>;
}

function UploadLane({ role, title, subtitle, books, disabled, busy, onUpload, onRemove }: DropzoneProps) {
  const input = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const accept = (files: FileList | null) => {
    const pdfs = Array.from(files || []).filter((file) => file.name.toLowerCase().endsWith(".pdf"));
    if (pdfs.length) void onUpload(role, pdfs);
  };
  const drop = (event: DragEvent) => {
    event.preventDefault();
    setDragging(false);
    if (!disabled) accept(event.dataTransfer.files);
  };
  return (
    <section className={`upload-lane ${dragging ? "is-dragging" : ""}`}>
      <div className="upload-lane__heading">
        <span className={`role-symbol role-symbol--${role}`}>{role === "system" ? <Bot size={19} /> : <UserRound size={19} />}</span>
        <div><strong>{title}</strong><p>{subtitle}</p></div>
        <span className="source-count">{books.length}</span>
      </div>
      <button
        type="button"
        className="dropzone"
        disabled={disabled || busy}
        onClick={() => input.current?.click()}
        onDragEnter={(event) => { event.preventDefault(); setDragging(true); }}
        onDragOver={(event) => event.preventDefault()}
        onDragLeave={() => setDragging(false)}
        onDrop={drop}
      >
        {busy ? <LoaderCircle className="spin" size={23} /> : <UploadCloud size={24} />}
        <span><strong>{busy ? "Uploading…" : "Drop PDF textbooks here"}</strong><small>or click to browse · up to 250 MB each</small></span>
      </button>
      <input ref={input} className="visually-hidden" type="file" accept="application/pdf,.pdf" multiple onChange={(event) => { accept(event.target.files); event.target.value = ""; }} />
      {books.length > 0 && <div className="book-list">
        {books.map((book, index) => <div className="book-row" key={book.id}>
          <span className="pdf-icon">PDF</span>
          <span className="book-row__copy"><strong>{book.original_name}</strong><small>{book.page_count ? `${book.page_count} pages · ` : ""}{humanBytes(book.size_bytes)}</small></span>
          {!disabled && !book.id.startsWith("demo") && <button className="icon-button" type="button" onClick={() => void onRemove(book.id)} title="Remove"><X size={16} /></button>}
          {disabled && <Check size={16} className="success-icon" />}
          <span className="book-index">{String(index + 1).padStart(2, "0")}</span>
        </div>)}
      </div>}
    </section>
  );
}

function PipelineHeader({ project, stage, onStage }: { project: Project; stage: PipelineStage; onStage: (stage: PipelineStage) => void }) {
  const steps = ["Extraction", "Refinement", "Annotation", "RL training", "Deployment"];
  return <div className="pipeline-header">
    {steps.map((step, index) => {
      const key = index === 0 ? "extraction" : index === 1 ? "refinement" : null;
      const complete = index === 0 ? project.status === "completed" : index === 1 && project.refinement_status === "completed";
      return <button type="button" className={`${key && stage === key ? "is-active" : ""} ${complete ? "is-complete" : ""} ${!key ? "is-locked" : ""}`} key={step} disabled={!key} onClick={() => key && onStage(key)}>
      <span>{complete ? <Check size={13} /> : index + 1}</span>
      <small>{step}</small>
      {index < steps.length - 1 && <i />}
    </button>;
    })}
  </div>;
}

function LiveActivity({ events, project }: { events: ProjectEvent[]; project: Project }) {
  const shown = events.slice(-8).reverse();
  const running = project.status === "running" || project.refinement_status === "running";
  return <aside className="activity-panel">
    <div className="panel-title"><div><Activity size={17} /><strong>Live activity</strong></div>{running && <span className="live-pill"><i /> Live</span>}</div>
    <div className="activity-list">
      {shown.length === 0 && <div className="activity-empty"><Clock3 size={19} /><span>Activity will appear here when extraction starts.</span></div>}
      {shown.map((event, index) => <div className={`activity-item activity-item--${event.kind}`} key={event.id}>
        <span className="activity-dot">{event.kind.includes("completed") || event.kind.includes("ready") ? <Check size={12} /> : event.kind === "failed" ? <AlertTriangle size={12} /> : <CircleDot size={11} />}</span>
        <div><strong>{event.message}</strong><small>{friendlyTime(event.created_at)}</small></div>
        {index === 0 && running && <span className="activity-pulse" />}
      </div>)}
    </div>
  </aside>;
}

function ExtractionSteps({ project, manifest }: { project: Project; manifest: Manifest }) {
  const summaryCount = manifest.roles.system.summaries.length + manifest.roles.user.summaries.length;
  const perBookCount = Object.values(manifest.roles.system.book_components).flat().length + Object.values(manifest.roles.user.book_components).flat().length;
  const mergedCount = manifest.roles.system.merged_components.length + manifest.roles.user.merged_components.length;
  const stages = [
    { label: "Read sources", sub: `${project.books?.length || 0} textbooks indexed`, complete: project.status !== "draft", icon: BookOpen },
    { label: "Chapter summaries", sub: summaryCount ? `${summaryCount} book summaries ready` : "Waiting for summaries", complete: summaryCount > 0, icon: FileText },
    { label: "Per-book extraction", sub: perBookCount ? `${perBookCount} source components ready` : "Starts after summaries", complete: perBookCount > 0, icon: ScanSearch },
    { label: "Late fusion", sub: mergedCount ? `${mergedCount} merged components ready` : "Deduplicate across sources", complete: mergedCount > 0, icon: Layers3 },
  ];
  return <div className="extraction-steps">
    {stages.map((stage, index) => <div className={`${stage.complete ? "is-complete" : ""} ${!stage.complete && project.status === "running" && (index === 0 || stages[index - 1].complete) ? "is-current" : ""}`} key={stage.label}>
      <span className="extraction-step__icon">{stage.complete ? <Check size={17} /> : <stage.icon size={18} />}</span>
      <span><strong>{stage.label}</strong><small>{stage.sub}</small></span>
      {index < stages.length - 1 && <i />}
    </div>)}
  </div>;
}

function SummaryBrowser({ project, manifest }: { project: Project; manifest: Manifest }) {
  const available = (["system", "user"] as Role[]).flatMap((role) => manifest.roles[role].summaries.map((item) => ({ ...item, role })));
  const [selected, setSelected] = useState(available[0] || null);
  const [data, setData] = useState<Record<string, unknown>[] | null>(null);
  const [query, setQuery] = useState("");
  useEffect(() => {
    if (!selected) return;
    setData(null);
    api.summary(project.id, selected.role, selected.book_index).then((response) => setData(polishText(response.data) as Record<string, unknown>[])).catch(() => setData([]));
  }, [project.id, selected?.role, selected?.book_index]);
  useEffect(() => {
    if (!selected && available.length) setSelected(available[0]);
  }, [available.length]);
  const booksByRole = project.books || [];
  const filtered = (data || []).filter((chapter) => JSON.stringify(chapter).toLowerCase().includes(query.toLowerCase()));
  const relevantCount = (chapter: Record<string, unknown>) => Object.entries(chapter).filter(([key, value]) => componentMeta[key] && JSON.stringify(value) !== '"None"' && JSON.stringify(value) !== '{"summary":"None"}').length;
  const firstRelevant = Math.max(0, filtered.findIndex((chapter) => relevantCount(chapter) > 0));
  if (!available.length) return <EmptyArtifact icon={FileText} title="Summaries are on the way" text="Each textbook summary will appear here as soon as TOPAS finishes reading it." />;
  return <div className="artifact-layout">
    <aside className="artifact-nav">
      <span className="artifact-nav__label">Available summaries</span>
      {(["system", "user"] as Role[]).map((role) => manifest.roles[role].summaries.length > 0 && <div className="summary-role" key={role}>
        <span>{role === "system" ? project.system_name : project.user_name}</span>
        {manifest.roles[role].summaries.map((item) => {
          const roleBooks = booksByRole.filter((book) => book.role === role);
          return <button className={selected?.role === role && selected.book_index === item.book_index ? "is-active" : ""} key={item.book_index} onClick={() => setSelected({ ...item, role })}>
            <BookCheck size={16} /><span><strong>{roleBooks[item.book_index]?.original_name || `Textbook ${item.book_index + 1}`}</strong><small>{item.chapter_count} chapters</small></span><ChevronRight size={15} />
          </button>;
        })}
      </div>)}
    </aside>
    <section className="artifact-content summary-content">
      <div className="artifact-content__head"><div><span className="artifact-kicker">Source synthesis</span><h3>{selected ? `${selected.role === "system" ? project.system_name : project.user_name} textbook ${selected.book_index + 1}` : "Summary"}</h3><p>Chapter-level evidence grouped by extracted component.</p></div><label className="search-box"><Search size={16} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search this summary" /></label></div>
      {!data ? <div className="artifact-loading"><span className="spinner spinner--dark" /> Loading summary…</div> : <div className="chapter-list">
        {filtered.map((chapter, index) => <details className="chapter-card" key={index} open={index === firstRelevant}>
          <summary><span className="chapter-number">{String(index + 1).padStart(2, "0")}</span><span><strong>Chapter {index + 1}</strong><small>{relevantCount(chapter)} relevant dimensions</small></span><ChevronDown size={17} /></summary>
          <div className="chapter-components">{Object.entries(chapter).map(([key, value]) => {
            if (!componentMeta[key]) return null;
            const summary = typeof value === "object" && value && "summary" in value ? (value as { summary: unknown }).summary : value;
            if (summary === "None" || summary == null) return null;
            return <div key={key}><span>{componentMeta[key]?.label || key.replaceAll("_", " ")}</span><FormattedText value={summary} /></div>;
          })}</div>
        </details>)}
        {!filtered.length && <div className="no-results">No summary content matches “{query}”.</div>}
      </div>}
    </section>
  </div>;
}

function FormattedText({ value }: { value: unknown }) {
  if (Array.isArray(value)) return <ul>{value.slice(0, 20).map((item, index) => <li key={index}>{typeof item === "string" ? item : JSON.stringify(item)}</li>)}</ul>;
  if (typeof value === "object" && value) return <pre>{JSON.stringify(value, null, 2)}</pre>;
  return <p>{String(value)}</p>;
}

function EmptyArtifact({ icon: Icon, title, text }: { icon: typeof FileText; title: string; text: string }) {
  return <div className="empty-artifact"><span><Icon size={25} /></span><h3>{title}</h3><p>{text}</p></div>;
}

function ComponentBrowser({ project, manifest, phase = "extraction", onSaved }: { project: Project; manifest: Manifest; phase?: ComponentPhase; onSaved?: () => Promise<void> }) {
  const componentsFor = (role: Role) => phase === "refinement" ? manifest.roles[role].refined_components : manifest.roles[role].merged_components;
  const available = (["system", "user"] as Role[]).flatMap((role) => componentsFor(role).map((component) => ({ role, component })));
  const [selected, setSelected] = useState(available[0] || null);
  const [data, setData] = useState<unknown>(null);
  const [raw, setRaw] = useState(false);
  const [editing, setEditing] = useState(false);
  const [saved, setSaved] = useState(false);
  useEffect(() => {
    if (!selected) return;
    setData(null);
    api.component(project.id, selected.role, selected.component, phase).then((response) => setData(polishText(response.data))).catch(() => setData({}));
  }, [project.id, selected?.role, selected?.component, phase]);
  useEffect(() => { if (!selected && available.length) setSelected(available[0]); }, [available.length]);
  if (!available.length) return <EmptyArtifact icon={Layers3} title={phase === "refinement" ? "Refined components are not ready" : "Components will arrive live"} text={phase === "refinement" ? "Run refinement to create an editable, refined component set." : "Final late-fused components appear here independently—you won’t need to wait for the whole run."} />;
  const meta = selected ? componentMeta[selected.component] : null;
  async function saveComponent(next: unknown) {
    if (!selected) return;
    const response = await api.updateComponent(project.id, selected.role, selected.component, phase, next);
    setData(polishText(response.data));
    setEditing(false);
    setSaved(true);
    window.setTimeout(() => setSaved(false), 2800);
    await onSaved?.();
  }
  return <><div className="artifact-layout">
    <aside className="artifact-nav component-nav">
      <span className="artifact-nav__label">{phase === "refinement" ? "Refined components" : "Extracted components"}</span>
      {(["system", "user"] as Role[]).map((role) => componentsFor(role).length > 0 && <div className="summary-role" key={role}>
        <span>{role === "system" ? project.system_name : project.user_name}</span>
        {componentsFor(role).map((component) => {
          const Icon = componentMeta[component]?.icon || Braces;
          return <button className={selected?.role === role && selected.component === component ? "is-active" : ""} key={component} onClick={() => { setSelected({ role, component }); setRaw(false); }}><Icon size={16} /><span><strong>{componentMeta[component]?.label || component}</strong><small>{componentMeta[component]?.description}</small></span><ChevronRight size={15} /></button>;
        })}
      </div>)}
    </aside>
    <section className="artifact-content component-content">
      <div className="artifact-content__head"><div><span className="artifact-kicker">{phase === "refinement" ? "Refined" : "Late-fused"} · {selected?.role}</span><h3>{meta?.label}</h3><p>{meta?.description}</p></div><div className="artifact-tools">{saved && <span className="saved-chip"><Check size={13} /> Saved</span>}<span className="item-count">{data === null || !selected ? "—" : countItems(selected.component, data)} items</span><button className="button button--tiny button--edit" disabled={data === null} onClick={() => setEditing(true)}>{selected?.component === "knowledge_graph" ? <Maximize2 size={15} /> : <Pencil size={15} />}{selected?.component === "knowledge_graph" ? "Open workspace" : "Edit"}</button><button className={`button button--tiny ${raw ? "is-active" : ""}`} onClick={() => setRaw(!raw)}><FileJson size={15} /> JSON</button><button className="icon-button" title="Download JSON" onClick={() => selected && downloadJson(`${selected.component}.json`, data)}><Download size={16} /></button></div></div>
      {data === null ? <div className="artifact-loading"><span className="spinner spinner--dark" /> Loading component…</div> : raw ? <pre className="raw-json">{JSON.stringify(data, null, 2)}</pre> : <ComponentRenderer component={selected!.component} data={data} onEditGraph={() => setEditing(true)} />}
    </section>
  </div>{editing && selected && data !== null && (selected.component === "knowledge_graph" ? <KnowledgeGraphEditor data={data as { nodes?: Record<string, unknown>[]; edges?: Record<string, unknown>[] }} onClose={() => setEditing(false)} onSave={saveComponent} /> : <ComponentEditor component={selected.component} label={meta?.label || selected.component} data={data} onClose={() => setEditing(false)} onSave={saveComponent} />)}</>;
}

function downloadJson(name: string, data: unknown) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }));
  const anchor = document.createElement("a"); anchor.href = url; anchor.download = name; anchor.click(); URL.revokeObjectURL(url);
}

function objectItems(data: unknown, key?: string): Record<string, unknown>[] {
  if (Array.isArray(data)) return data as Record<string, unknown>[];
  if (data && typeof data === "object" && key && Array.isArray((data as Record<string, unknown>)[key])) return (data as Record<string, unknown>)[key] as Record<string, unknown>[];
  return [];
}

function ComponentRenderer({ component, data, onEditGraph }: { component: string; data: unknown; onEditGraph: () => void }) {
  if (component === "knowledge_graph") return <KnowledgeGraph data={data as { nodes?: Record<string, unknown>[]; edges?: Record<string, unknown>[] }} onOpen={onEditGraph} />;
  if (component === "user_profile") return <UserProfile data={data as Record<string, unknown>} />;
  const items = objectItems(data, component === "macro_actions" ? "macro_actions" : undefined);
  if (component === "conversation_states") return <div className="state-grid">{items.map((item, index) => <article key={index}><div className="state-card__head"><span className="state-index">S{String(index + 1).padStart(2, "0")}</span><span>Tracked state</span></div><h4>{String(item["Variable Name"] || item.name || "State")}</h4><p>{String(item.Description || item.description || "")}</p><StateValues label="Categories" values={item["Categorical values"]} /><StateValues label="Numeric range" values={item["Numerical values"]} /></article>)}</div>;
  if (component === "cautions") return <div className="caution-list">{items.map((item, index) => <article key={index}><span><ShieldAlert size={18} /></span><div><span className="caution-label">Behavior to avoid</span><h4>{String(item.negative_action || item.name || `Caution ${index + 1}`)}</h4><p>{String(item.description || "")}</p>{item.risk != null && <small><strong>Potential risk</strong> · {String(item.risk)}</small>}</div><Confidence value={item.confidence_score} /></article>)}</div>;
  if (component === "micro_actions") return <div className="macro-list">{items.map((item, index) => <details key={index} open={index === 0}><summary><span className="macro-index">{String(index + 1).padStart(2, "0")}</span><span><h4>{String(item.name || "Action group")}</h4><p>{String(item.description || "")}</p></span><span className="micro-count">{Array.isArray(item.micro_actions) ? item.micro_actions.length : 0} actions</span><ChevronDown size={17} /></summary><div className="micro-list">{Array.isArray(item.micro_actions) && (item.micro_actions as Record<string, unknown>[]).map((micro, i) => <div key={i}><span>{i + 1}</span><div><strong>{String(micro.name || "Micro action")}</strong><p>{String(micro.description || "")}</p></div><Confidence value={micro.confidence_score} /></div>)}</div></details>)}</div>;
  return <div className="macro-list">{items.map((item, index) => <details key={index} open={index < 2}><summary><span className="macro-index">{String(index + 1).padStart(2, "0")}</span><span><h4>{String(item.name || item.title || `Item ${index + 1}`)}</h4><p>{String(item.description || "")}</p></span><Confidence value={item.confidence_score} /><ChevronDown size={17} /></summary><div className="macro-detail">{item.goal != null && <div><span>Goal</span><StructuredValue value={item.goal} /></div>}{Array.isArray(item.states) && <div><span>Applicable states</span><div className="tag-row">{(item.states as unknown[]).map((state, i) => <span key={i}>{String(state)}</span>)}</div></div>}</div></details>)}</div>;
}

function StateValues({ label, values }: { label: string; values: unknown }) {
  if (!Array.isArray(values) || !values.length) return null;
  return <div className="state-values"><span>{label}</span><div className="tag-row">{values.slice(0, 8).map((value, index) => <span key={index}>{String(value)}</span>)}</div></div>;
}

function StructuredValue({ value }: { value: unknown }) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return <FormattedText value={value} />;
  return <div className="structured-value">{Object.entries(value as Record<string, unknown>).map(([key, item]) => <div key={key}><strong>{key.replaceAll("_", " ")}</strong><FormattedText value={item} /></div>)}</div>;
}

function Confidence({ value }: { value: unknown }) {
  if (typeof value !== "number") return null;
  const percentage = Math.max(0, Math.min(100, Math.round(value * 100)));
  return <span className="confidence" title={`Extraction confidence: ${percentage}%`} aria-label={`Extraction confidence: ${percentage}%`}><i aria-hidden="true" style={{ width: `${percentage}%` }} /><span className="confidence__label">Extraction confidence</span><strong>{percentage}%</strong></span>;
}

function UserProfile({ data }: { data: Record<string, unknown> }) {
  return <div className="profile-dimensions">{(["static_dimensions", "dynamic_dimensions"] as const).map((key) => {
    const items = Array.isArray(data[key]) ? data[key] as Record<string, unknown>[] : [];
    return <section key={key}><div className="dimension-heading"><span>{key.startsWith("static") ? "Static" : "Dynamic"}</span><h4>{key.startsWith("static") ? "Persistent attributes" : "Evolving attributes"}</h4><small>{items.length} dimensions</small></div><div className="dimension-grid">{items.map((item, index) => <article key={index}><span className="state-index">{key.startsWith("static") ? "S" : "D"}{String(index + 1).padStart(2, "0")}</span><h5>{String(item.dimension_name || item.name || "Dimension")}</h5><p>{String(item.description || "")}</p>{Array.isArray(item.options) && <div className="tag-row">{(item.options as unknown[]).slice(0, 5).map((option, i) => <span key={i}>{String(option)}</span>)}{item.options.length > 5 && <span>+{item.options.length - 5}</span>}</div>}</article>)}</div></section>;
  })}</div>;
}

function KnowledgeGraph({ data, onOpen }: { data: { nodes?: Record<string, unknown>[]; edges?: Record<string, unknown>[] }; onOpen: () => void }) {
  const nodes = (data.nodes || []).slice(0, 9);
  const edges = (data.edges || []).slice(0, 12);
  const positions = nodes.map((_, index) => {
    if (index === 0) return { x: 50, y: 50 };
    const angle = ((index - 1) / Math.max(1, nodes.length - 1)) * Math.PI * 2 - Math.PI / 2;
    return { x: 50 + Math.cos(angle) * 37, y: 50 + Math.sin(angle) * 38 };
  });
  return <div className="kg-presentation"><div className="kg-summary-strip"><div><span><Network size={16} /></span><div><strong>Interactive memory schema</strong><small>{data.nodes?.length || 0} concepts connected by {data.edges?.length || 0} relationships</small></div></div><button className="button button--primary button--small" type="button" onClick={onOpen}><Maximize2 size={15} /> Open graph workspace</button></div><div className="kg-wrap">
    <div className="kg-canvas" onDoubleClick={onOpen} title="Double-click to open the graph workspace">
      <svg viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">{positions.slice(1).map((position, index) => <line key={index} x1="50" y1="50" x2={position.x} y2={position.y} />)}</svg>
      {nodes.map((node, index) => <button className={index === 0 ? "kg-node kg-node--root" : "kg-node"} style={{ left: `${positions[index].x}%`, top: `${positions[index].y}%` }} key={index} title={String(node.description || "")}><Network size={index === 0 ? 17 : 13} /><span>{String(node.type || node.name || `Node ${index + 1}`)}</span></button>)}
      {(data.nodes?.length || 0) > nodes.length && <span className="kg-more">+{(data.nodes?.length || 0) - nodes.length} nodes</span>}
    </div>
    <div className="edge-list"><div className="edge-list__head"><strong>Schema relationships</strong><span>{data.edges?.length || 0} total</span></div>{edges.map((edge, index) => <div key={index}><span>{String(edge.source || "Source")}</span><i>{String(edge.relation || "relates to")}</i><span>{String(edge.target || "Target")}</span></div>)}</div>
  </div></div>;
}

function RefinementReport({ project, manifest }: { project: Project; manifest: Manifest }) {
  const [data, setData] = useState<Record<string, unknown> | null>(null);
  useEffect(() => {
    if (!manifest.refinement.summary_available) return;
    api.refinementReport(project.id).then((response) => setData(polishText(response.data) as Record<string, unknown>)).catch(() => setData({}));
  }, [project.id, manifest.refinement.summary_available]);
  if (!manifest.refinement.summary_available) return <EmptyArtifact icon={FileCheck2} title="The refinement report will appear here" text="TOPAS records each new state dimension and the action-space gap that motivated it." />;
  if (!data) return <div className="artifact-loading"><span className="spinner spinner--dark" /> Loading refinement report…</div>;
  const dimensions = Array.isArray(data.new_dimensions) ? data.new_dimensions as Record<string, unknown>[] : [];
  const byMacro = Array.isArray(data.by_macro_action) ? data.by_macro_action as Record<string, unknown>[] : [];
  return <div className="refinement-report">
    <div className="report-hero"><span><FileCheck2 size={21} /></span><div><small>Refinement outcome</small><strong>{Number(data.num_added_dimensions || dimensions.length)} new state dimensions</strong><p>The extracted state model was audited against every macro and micro action.</p></div></div>
    <div className="report-layout"><section><div className="report-section-head"><div><span>New dimensions</span><h3>Added to conversation states</h3></div><small>{dimensions.length} additions</small></div><div className="report-dimension-grid">{dimensions.map((dimension, index) => <article key={index}><span className="state-index">R{String(index + 1).padStart(2, "0")}</span><h4>{String(dimension["Variable Name"] || dimension.name || `Dimension ${index + 1}`)}</h4><p>{String(dimension.Description || dimension.description || "")}</p><StateValues label="Categories" values={dimension["Categorical values"]} /></article>)}</div></section><aside><div className="report-section-head"><div><span>Coverage audit</span><h3>By action group</h3></div></div><div className="macro-audit-list">{byMacro.map((entry, index) => <div key={index}><span>{String(index + 1).padStart(2, "0")}</span><div><strong>{String(entry.macro_action || `Action group ${index + 1}`)}</strong><small>{Number(entry.num_added || 0)} dimensions added</small></div></div>)}</div></aside></div>
  </div>;
}

function RefinementWorkspace({ project, manifest, events, onRun, onRefresh }: { project: Project; manifest: Manifest; events: ProjectEvent[]; onRun: () => Promise<void>; onRefresh: () => Promise<void> }) {
  const [tab, setTab] = useState<"overview" | "components" | "report">("overview");
  const [busy, setBusy] = useState(false);
  const refinedCount = manifest.roles.system.refined_components.length + manifest.roles.user.refined_components.length;
  const status = project.refinement_status || "not_started";
  const complete = status === "completed";
  const stale = status === "stale";
  async function run() { setBusy(true); try { await onRun(); setTab("overview"); } finally { setBusy(false); } }
  return <main className="project-main refinement-main">
    <section className="project-hero refinement-hero"><div><span className="eyebrow"><SlidersHorizontal size={13} /> Refinement workspace</span><h1>Strengthen the agent blueprint</h1><p>Audit conversation states against the full action space and add only the dimensions that are still missing.</p><div className="project-meta"><span>{project.model}</span><i /><span>Prompt-preserving</span><i /><span>Versioned outputs</span></div></div><div className="progress-orb" style={{ "--progress": `${project.refinement_progress * 3.6}deg` } as CSSProperties}><div><strong>{project.refinement_progress}%</strong><small>{complete ? "Refined" : "Progress"}</small></div></div></section>
    <nav className="project-tabs"><button className={tab === "overview" ? "is-active" : ""} onClick={() => setTab("overview")}><Activity size={16} /> Overview</button><button className={tab === "components" ? "is-active" : ""} onClick={() => setTab("components")} disabled={!refinedCount}><Layers3 size={16} /> Refined components {refinedCount > 0 && <span>{refinedCount}</span>}</button><button className={tab === "report" ? "is-active" : ""} onClick={() => setTab("report")} disabled={!manifest.refinement.summary_available}><FileCheck2 size={16} /> Report</button></nav>
    {tab === "overview" && <div className="overview-layout page-enter"><div className="overview-main">
      <section className="card-panel refinement-purpose"><div className="section-heading section-heading--compact"><div><span className="artifact-kicker">State-space audit</span><h2>What TOPAS refines</h2><p>The original refinement prompt reviews each action group against the current conversation state schema.</p></div><span className="refinement-method"><SlidersHorizontal size={15} /> Coverage refinement</span></div><div className="refinement-benefits"><article><span>01</span><div><strong>Preserve the extracted blueprint</strong><p>Every extracted component is copied into a separate refined version.</p></div></article><article><span>02</span><div><strong>Find missing state dimensions</strong><p>Macro and micro actions are checked against what the policy can currently observe.</p></div></article><article><span>03</span><div><strong>Keep a rationale trail</strong><p>New dimensions and their gap reasons are recorded in the refinement report.</p></div></article></div></section>
      <section className="card-panel refinement-run-card"><div className="refinement-status-row"><span className={`refinement-status-icon refinement-status-icon--${status}`}>{complete ? <Check size={21} /> : status === "failed" ? <AlertTriangle size={21} /> : <SlidersHorizontal size={21} />}</span><div><span className="artifact-kicker">{stale ? "Update recommended" : status.replaceAll("_", " ")}</span><h2>{stale ? "Extraction changed since the last refinement" : project.refinement_step}</h2><p>{complete ? "The refined components and audit report are ready to review and edit." : stale ? "Run refinement again to bring the refined state space in sync." : status === "running" ? "Results and progress are saved continuously to this profile." : status === "failed" ? project.refinement_error : "Start when the extracted macro actions, micro actions, and conversation states are ready."}</p></div>{status !== "running" && project.status === "completed" && <button className="button button--primary" disabled={busy} onClick={() => void run()}>{busy ? <span className="spinner" /> : <><Play size={16} fill="currentColor" /> {complete || stale ? "Run again" : "Start refinement"}</>}</button>}</div><div className="refinement-progress-track"><i style={{ width: `${project.refinement_progress}%` }} /></div>{status === "failed" && <div className="run-error"><AlertTriangle size={18} /><div><strong>Refinement stopped</strong><p>{project.refinement_error}</p></div></div>}</section>
      {refinedCount > 0 && <section className="card-panel refined-preview"><div className="section-heading section-heading--compact"><div><span className="artifact-kicker">Editable result</span><h2>{refinedCount} refined components ready</h2><p>Review the structured views or make profile-specific edits.</p></div><button className="button button--quiet" onClick={() => setTab("components")}><Pencil size={15} /> Review components</button></div></section>}
    </div><LiveActivity events={events} project={project} /></div>}
    {tab === "components" && <div className="card-panel artifact-panel page-enter"><ComponentBrowser project={project} manifest={manifest} phase="refinement" onSaved={onRefresh} /></div>}
    {tab === "report" && <div className="card-panel artifact-panel report-panel page-enter"><RefinementReport project={project} manifest={manifest} /></div>}
  </main>;
}

interface Props {
  project: Project;
  manifest: Manifest;
  events: ProjectEvent[];
  stage: PipelineStage;
  refreshing: boolean;
  onBack: () => void;
  onRefresh: () => Promise<void>;
  onUpload: (role: Role, files: File[]) => Promise<void>;
  onRemove: (bookId: string) => Promise<void>;
  onRun: () => Promise<void>;
  onRunRefinement: () => Promise<void>;
  onStage: (stage: PipelineStage) => void;
}

export function ProjectView({ project, manifest, events, stage, refreshing, onBack, onRefresh, onUpload, onRemove, onRun, onRunRefinement, onStage }: Props) {
  const [tab, setTab] = useState<"overview" | "summaries" | "components">("overview");
  const [uploadingRole, setUploadingRole] = useState<Role | null>(null);
  const [runBusy, setRunBusy] = useState(false);
  const books = project.books || [];
  const systemBooks = books.filter((book) => book.role === "system");
  const userBooks = books.filter((book) => book.role === "user");
  const mergedCount = manifest.roles.system.merged_components.length + manifest.roles.user.merged_components.length;
  const summaryCount = manifest.roles.system.summaries.length + manifest.roles.user.summaries.length;
  const statusText = stage === "refinement" ? project.refinement_status === "running" ? "Refining live" : project.refinement_status === "completed" ? "Refinement complete" : project.refinement_status === "stale" ? "Refinement update available" : project.refinement_status === "failed" ? "Refinement needs attention" : "Ready for refinement" : project.is_demo ? "Completed showcase" : project.status === "running" ? "Extracting live" : project.status === "completed" ? "Extraction complete" : project.status === "failed" ? "Needs attention" : books.length ? "Ready to extract" : "Awaiting textbooks";
  const statusClass = stage === "refinement" ? project.refinement_status : project.status;
  async function upload(role: Role, files: File[]) { setUploadingRole(role); try { await onUpload(role, files); } finally { setUploadingRole(null); } }
  async function run() { setRunBusy(true); try { await onRun(); setTab("overview"); } finally { setRunBusy(false); } }
  useEffect(() => setTab("overview"), [stage]);
  return <div className="project-page page-enter">
    <header className="project-topbar">
      <button className="breadcrumb" onClick={onBack}><ArrowLeft size={17} /><span>Projects</span></button>
      <div className="project-topbar__title"><span className="project-topbar__glyph">{project.system_name.slice(0, 1)}<i />{project.user_name.slice(0, 1)}</span><div><strong>{project.name}</strong><small>{project.domain}</small></div></div>
      <div className="project-topbar__actions"><span className={`run-status run-status--${statusClass}`}><i />{statusText}</span><button className="icon-button" title="Refresh" onClick={() => void onRefresh()}><RefreshCw className={refreshing ? "spin" : ""} size={17} /></button></div>
    </header>
    <PipelineHeader project={project} stage={stage} onStage={onStage} />

    {stage === "refinement" ? <RefinementWorkspace project={project} manifest={manifest} events={events} onRun={onRunRefinement} onRefresh={onRefresh} /> : <main className="project-main">
      <section className="project-hero">
        <div><span className="eyebrow"><Sparkles size={13} /> Extraction workspace</span><h1>{project.domain}</h1><p>Turning <strong>{project.system_name}</strong> and <strong>{project.user_name}</strong> sources into an executable agent blueprint.</p><div className="project-meta"><span>{project.model}</span><i /><span>Late fusion</span><i /><span>{project.interaction_unit}</span></div></div>
        <div className="progress-orb" style={{ "--progress": `${project.progress * 3.6}deg` } as CSSProperties}><div><strong>{project.progress}%</strong><small>{project.status === "completed" ? "Complete" : "Extracted"}</small></div></div>
      </section>

      <nav className="project-tabs">
        <button className={tab === "overview" ? "is-active" : ""} onClick={() => setTab("overview")}><Activity size={16} /> Overview</button>
        <button className={tab === "summaries" ? "is-active" : ""} onClick={() => setTab("summaries")}><FileText size={16} /> Summaries {summaryCount > 0 && <span>{summaryCount}</span>}</button>
        <button className={tab === "components" ? "is-active" : ""} onClick={() => setTab("components")}><Layers3 size={16} /> Components {mergedCount > 0 && <span>{mergedCount}</span>}</button>
      </nav>

      {tab === "overview" && <div className="overview-layout page-enter">
        <div className="overview-main">
          <section className="source-section card-panel">
            <div className="section-heading section-heading--compact"><div><span className="artifact-kicker">Source library</span><h2>Textbooks by role</h2><p>Add either side now; the other can be added before a later run.</p></div>{books.length > 0 && <span className="book-total"><BookOpen size={15} /> {books.length} source{books.length === 1 ? "" : "s"}</span>}</div>
            <div className="upload-grid">
              <UploadLane role="system" title={`${project.system_name} textbooks`} subtitle="System strategies, states, memory & cautions" books={systemBooks} disabled={project.status === "running" || project.is_demo} busy={uploadingRole === "system"} onUpload={upload} onRemove={onRemove} />
              <UploadLane role="user" title={`${project.user_name} textbooks`} subtitle="User profile and simulator dimensions" books={userBooks} disabled={project.status === "running" || project.is_demo} busy={uploadingRole === "user"} onUpload={upload} onRemove={onRemove} />
            </div>
            {!project.is_demo && project.status !== "running" && project.status !== "completed" && <div className="run-bar"><div><span className="run-bar__spark"><Sparkles size={17} /></span><span><strong>{books.length ? "Your source library is ready" : "Add your first textbook"}</strong><small>{books.length ? `TOPAS will process ${books.length} source${books.length === 1 ? "" : "s"} with ${project.model}.` : "Upload one or both roles—you can add the rest later."}</small></span></div><button className="button button--primary" disabled={!books.length || runBusy} onClick={() => void run()}>{runBusy ? <span className="spinner" /> : <><Play size={17} fill="currentColor" /> Start extraction</>}</button></div>}
            {project.status === "failed" && <div className="run-error"><AlertTriangle size={18} /><div><strong>Extraction stopped</strong><p>{project.error || "Review the backend logs and try again."}</p></div></div>}
          </section>
          <section className="card-panel progress-panel">
            <div className="section-heading section-heading--compact"><div><span className="artifact-kicker">TOPAS late fusion</span><h2>Extraction progress</h2></div><span className={`status-badge status-badge--${project.status}`}>{project.current_step}</span></div>
            <ExtractionSteps project={project} manifest={manifest} />
          </section>
          <section className="card-panel output-plan">
            <div className="section-heading section-heading--compact"><div><span className="artifact-kicker">Output contract</span><h2>Components being built</h2><p>Final schemas appear here independently as they complete.</p></div></div>
            <div className="component-plan">{Object.entries(componentMeta).map(([key, meta]) => { const Icon = meta.icon; const ready = manifest.roles.system.merged_components.includes(key) || manifest.roles.user.merged_components.includes(key); return <button key={key} className={ready ? "is-ready" : ""} onClick={() => ready && setTab("components")}><span><Icon size={18} /></span><div><strong>{meta.label}</strong><small>{key === "user_profile" ? project.user_name : project.system_name}</small></div>{ready ? <CheckCircle2 size={16} /> : <Clock3 size={15} />}</button>; })}</div>
          </section>
          {project.status === "completed" && <section className="phase-handoff"><span><SlidersHorizontal size={20} /></span><div><small>Next pipeline step</small><strong>Refine the state space</strong><p>Audit extracted states against the action space while preserving this version.</p></div><button className="button button--primary" onClick={() => onStage("refinement")}>Open refinement <ArrowRight size={16} /></button></section>}
        </div>
        <LiveActivity events={events} project={project} />
      </div>}
      {tab === "summaries" && <div className="card-panel artifact-panel page-enter"><SummaryBrowser project={project} manifest={manifest} /></div>}
      {tab === "components" && <div className="card-panel artifact-panel page-enter"><ComponentBrowser project={project} manifest={manifest} onSaved={onRefresh} /></div>}
    </main>}
  </div>;
}
