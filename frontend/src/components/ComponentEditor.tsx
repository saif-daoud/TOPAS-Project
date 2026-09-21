import { useMemo, useState } from "react";
import { createPortal } from "react-dom";
import { Braces, Check, ChevronRight, Plus, Save, Trash2, X } from "lucide-react";

type JsonRecord = Record<string, unknown>;
type PathPart = string | number;

interface Props {
  component: string;
  label: string;
  data: unknown;
  onClose: () => void;
  onSave: (data: unknown) => Promise<void>;
}

interface Collection {
  key: string;
  label: string;
  path: PathPart[];
  items: JsonRecord[];
}

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

function setAt(root: unknown, path: PathPart[], value: unknown): unknown {
  if (!path.length) return value;
  const [head, ...tail] = path;
  const copy: any = Array.isArray(root) ? [...root] : { ...((root || {}) as object) };
  copy[head] = setAt(copy[head], tail, value);
  return copy;
}

function collectionsFor(component: string, data: unknown): Collection[] {
  if (component === "macro_actions") {
    const root = data as JsonRecord;
    return [{ key: "macro_actions", label: "Macro actions", path: ["macro_actions"], items: Array.isArray(root?.macro_actions) ? root.macro_actions as JsonRecord[] : [] }];
  }
  if (component === "user_profile") {
    const root = data as JsonRecord;
    return [
      { key: "static_dimensions", label: "Static dimensions", path: ["static_dimensions"], items: Array.isArray(root?.static_dimensions) ? root.static_dimensions as JsonRecord[] : [] },
      { key: "dynamic_dimensions", label: "Dynamic dimensions", path: ["dynamic_dimensions"], items: Array.isArray(root?.dynamic_dimensions) ? root.dynamic_dimensions as JsonRecord[] : [] },
    ];
  }
  return [{ key: component, label: component.replaceAll("_", " "), path: [], items: Array.isArray(data) ? data as JsonRecord[] : [] }];
}

function blankRecord(component: string, collection: string): JsonRecord {
  if (component === "conversation_states") return { "Variable Name": "New state", Description: "", "Categorical values": [], "Numerical values": [] };
  if (component === "cautions") return { negative_action: "New caution", description: "", risk: "", confidence_score: 0.8 };
  if (component === "user_profile") return { dimension_name: `New ${collection.startsWith("static") ? "static" : "dynamic"} dimension`, description: "", options: [] };
  if (component === "micro_actions") return { name: "New strategy", description: "", goal: "", states: [], confidence_score: 0.8, micro_actions: [] };
  return { name: "New action", description: "", goal: "", states: [], confidence_score: 0.8 };
}

function displayName(item: JsonRecord, index: number): string {
  return String(item.name || item.dimension_name || item["Variable Name"] || item.negative_action || `Item ${index + 1}`);
}

function humanLabel(value: string): string {
  return value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function ValueField({ label, value, path, onChange, depth = 0 }: { label: string; value: unknown; path: PathPart[]; onChange: (path: PathPart[], value: unknown) => void; depth?: number }) {
  if (label === "confidence_score" && typeof value === "number") {
    const percent = Math.round(value * 100);
    return <label className="editor-field editor-field--confidence"><span><span>Extraction confidence</span><strong>{percent}%</strong></span><input type="range" min="0" max="1" step="0.01" value={value} onChange={(event) => onChange(path, Number(event.target.value))} /></label>;
  }
  if (typeof value === "string") {
    const multiline = value.length > 70 || /description|goal|risk|rationale/i.test(label);
    return <label className={`editor-field ${multiline ? "editor-field--wide" : ""}`}><span>{humanLabel(label)}</span>{multiline ? <textarea value={value} rows={Math.min(8, Math.max(3, Math.ceil(value.length / 90)))} onChange={(event) => onChange(path, event.target.value)} /> : <input value={value} onChange={(event) => onChange(path, event.target.value)} />}</label>;
  }
  if (typeof value === "number") return <label className="editor-field"><span>{humanLabel(label)}</span><input type="number" value={value} onChange={(event) => onChange(path, Number(event.target.value))} /></label>;
  if (typeof value === "boolean") return <label className="editor-toggle"><input type="checkbox" checked={value} onChange={(event) => onChange(path, event.target.checked)} /><span>{humanLabel(label)}</span></label>;
  if (Array.isArray(value) && value.every((item) => typeof item !== "object" || item === null)) {
    return <label className="editor-field editor-field--wide"><span>{humanLabel(label)} <small>one value per line</small></span><textarea rows={Math.min(7, Math.max(3, value.length + 1))} value={value.join("\n")} onChange={(event) => {
      const values = event.target.value.split("\n").map((item) => item.trim()).filter(Boolean);
      const next = value.length && typeof value[0] === "number" ? values.map(Number).filter((item) => !Number.isNaN(item)) : values;
      onChange(path, next);
    }} /></label>;
  }
  if (Array.isArray(value)) {
    return <div className="nested-editor editor-field--wide"><div className="nested-editor__head"><span>{humanLabel(label)}</span><button type="button" onClick={() => onChange(path, [...value, label === "micro_actions" ? { name: "New micro action", description: "", states: [], confidence_score: 0.8 } : {}])}><Plus size={14} /> Add</button></div>{value.map((entry, index) => <div className="nested-editor__item" key={index}><div className="nested-editor__title"><strong>{typeof entry === "object" && entry ? displayName(entry as JsonRecord, index) : `Item ${index + 1}`}</strong><button type="button" aria-label="Remove item" onClick={() => onChange(path, value.filter((_, itemIndex) => itemIndex !== index))}><Trash2 size={14} /></button></div>{entry && typeof entry === "object" ? <div className="editor-form-grid">{Object.entries(entry as JsonRecord).map(([key, nested]) => <ValueField key={key} label={key} value={nested} path={[...path, index, key]} onChange={onChange} depth={depth + 1} />)}</div> : null}</div>)}</div>;
  }
  if (value && typeof value === "object") {
    return <fieldset className="object-editor editor-field--wide"><legend>{humanLabel(label)}</legend><div className="editor-form-grid">{Object.entries(value as JsonRecord).map(([key, nested]) => <ValueField key={key} label={key} value={nested} path={[...path, key]} onChange={onChange} depth={depth + 1} />)}</div></fieldset>;
  }
  return <label className="editor-field"><span>{humanLabel(label)}</span><input value={value == null ? "" : String(value)} onChange={(event) => onChange(path, event.target.value)} /></label>;
}

export function ComponentEditor({ component, label, data, onClose, onSave }: Props) {
  const [draft, setDraft] = useState<unknown>(() => clone(data));
  const [collectionKey, setCollectionKey] = useState(() => collectionsFor(component, data)[0]?.key || component);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [advanced, setAdvanced] = useState(false);
  const [jsonText, setJsonText] = useState(() => JSON.stringify(data, null, 2));
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const collections = useMemo(() => collectionsFor(component, draft), [component, draft]);
  const collection = collections.find((item) => item.key === collectionKey) || collections[0];
  const selected = collection?.items[selectedIndex];

  function update(path: PathPart[], value: unknown) {
    setDraft((current: unknown) => setAt(current, path, value));
  }

  function replaceCollection(items: JsonRecord[]) {
    if (!collection) return;
    setDraft((current: unknown) => setAt(current, collection.path, items));
  }

  function addItem() {
    if (!collection) return;
    replaceCollection([...collection.items, blankRecord(component, collection.key)]);
    setSelectedIndex(collection.items.length);
  }

  function removeItem() {
    if (!collection || !selected) return;
    replaceCollection(collection.items.filter((_, index) => index !== selectedIndex));
    setSelectedIndex(Math.max(0, selectedIndex - 1));
  }

  function toggleAdvanced() {
    setError("");
    if (advanced) {
      try {
        setDraft(JSON.parse(jsonText));
        setAdvanced(false);
      } catch {
        setError("Fix the JSON syntax before returning to the form editor.");
      }
    } else {
      setJsonText(JSON.stringify(draft, null, 2));
      setAdvanced(true);
    }
  }

  async function save() {
    setError("");
    let value = draft;
    if (advanced) {
      try {
        value = JSON.parse(jsonText);
      } catch {
        setError("The JSON is not valid yet.");
        return;
      }
    }
    setSaving(true);
    try {
      await onSave(value);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not save this component.");
      setSaving(false);
    }
  }

  return createPortal(<div className="workspace-overlay" role="dialog" aria-modal="true" aria-label={`Edit ${label}`}>
    <div className="component-editor-window">
      <header className="workspace-window__header"><div><span className="artifact-kicker">Structured component editor</span><h2>Edit {label}</h2><p>Changes are saved to this profile and the previous version is retained.</p></div><div><button className={`button button--quiet ${advanced ? "is-active" : ""}`} type="button" onClick={toggleAdvanced}><Braces size={16} /> {advanced ? "Form editor" : "Advanced JSON"}</button><button className="icon-button" type="button" aria-label="Close editor" onClick={onClose}><X size={19} /></button></div></header>
      {advanced ? <main className="json-editor-pane"><textarea spellCheck={false} value={jsonText} onChange={(event) => setJsonText(event.target.value)} /></main> : <div className="component-editor-body">
        <aside className="editor-index">
          {collections.length > 1 && <div className="editor-collections">{collections.map((item) => <button className={item.key === collection?.key ? "is-active" : ""} type="button" key={item.key} onClick={() => { setCollectionKey(item.key); setSelectedIndex(0); }}>{item.label}<span>{item.items.length}</span></button>)}</div>}
          <div className="editor-index__head"><span>{collection?.label}</span><button type="button" onClick={addItem}><Plus size={14} /> Add</button></div>
          <div className="editor-index__list">{collection?.items.map((item, index) => <button className={selectedIndex === index ? "is-active" : ""} type="button" key={index} onClick={() => setSelectedIndex(index)}><span>{String(index + 1).padStart(2, "0")}</span><strong>{displayName(item, index)}</strong><ChevronRight size={14} /></button>)}</div>
        </aside>
        <main className="record-editor">{selected ? <><div className="record-editor__head"><div><span>Item {selectedIndex + 1} of {collection.items.length}</span><h3>{displayName(selected, selectedIndex)}</h3></div><button className="danger-button" type="button" onClick={removeItem}><Trash2 size={15} /> Remove item</button></div><div className="editor-form-grid">{Object.entries(selected).map(([key, value]) => <ValueField key={key} label={key} value={value} path={[...collection.path, selectedIndex, key]} onChange={update} />)}</div></> : <div className="editor-empty"><Check size={24} /><h3>No items yet</h3><p>Add the first item to this component.</p><button className="button button--primary" type="button" onClick={addItem}><Plus size={15} /> Add item</button></div>}</main>
      </div>}
      <footer className="workspace-window__footer"><span>{error || "Edits are validated before they are written."}</span><div><button className="button button--quiet" type="button" onClick={onClose}>Cancel</button><button className="button button--primary" type="button" disabled={saving} onClick={() => void save()}>{saving ? <span className="spinner" /> : <Save size={16} />} Save changes</button></div></footer>
    </div>
  </div>, document.body);
}
