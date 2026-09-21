import { useMemo, useRef, useState, type PointerEvent } from "react";
import { createPortal } from "react-dom";
import { GitBranchPlus, Minus, Network, Plus, Save, Trash2, X, ZoomIn } from "lucide-react";

type JsonRecord = Record<string, unknown>;
type Selection = { kind: "node" | "edge"; index: number } | null;

interface GraphData extends JsonRecord {
  nodes?: JsonRecord[];
  edges?: JsonRecord[];
}

interface Props {
  data: GraphData;
  onClose: () => void;
  onSave: (data: unknown) => Promise<void>;
}

function positionsFor(count: number): { x: number; y: number }[] {
  const capacities = [6, 10, 14, 18];
  const radii = [16, 28, 39, 47];
  return Array.from({ length: count }, (_, index) => {
    if (index === 0) return { x: 50, y: 50 };
    let remaining = index - 1;
    let ring = 0;
    while (ring < capacities.length - 1 && remaining >= capacities[ring]) {
      remaining -= capacities[ring];
      ring += 1;
    }
    const populated = Math.min(capacities[ring], Math.max(1, count - 1 - capacities.slice(0, ring).reduce((sum, value) => sum + value, 0)));
    const angle = (remaining / populated) * Math.PI * 2 - Math.PI / 2;
    return { x: 50 + Math.cos(angle) * radii[ring], y: 50 + Math.sin(angle) * radii[ring] * 0.9 };
  });
}

export function KnowledgeGraphEditor({ data, onClose, onSave }: Props) {
  const [draft, setDraft] = useState<GraphData>(() => JSON.parse(JSON.stringify(data)));
  const nodes = draft.nodes || [];
  const edges = draft.edges || [];
  const [positions, setPositions] = useState(() => positionsFor(nodes.length));
  const [selection, setSelection] = useState<Selection>(nodes.length ? { kind: "node", index: 0 } : null);
  const [dragging, setDragging] = useState<number | null>(null);
  const [zoom, setZoom] = useState(1);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const innerRef = useRef<HTMLDivElement>(null);
  const nodeIndex = useMemo(() => new Map(nodes.map((node, index) => [String(node.type || node.name || ""), index])), [nodes]);

  function updateNode(index: number, field: string, value: string) {
    const nextNodes = nodes.map((node, itemIndex) => itemIndex === index ? { ...node, [field]: value } : node);
    let nextEdges = edges;
    if (field === "type") {
      const previous = String(nodes[index]?.type || nodes[index]?.name || "");
      nextEdges = edges.map((edge) => ({ ...edge, source: edge.source === previous ? value : edge.source, target: edge.target === previous ? value : edge.target }));
    }
    setDraft({ ...draft, nodes: nextNodes, edges: nextEdges });
  }

  function updateEdge(index: number, field: string, value: string) {
    setDraft({ ...draft, edges: edges.map((edge, itemIndex) => itemIndex === index ? { ...edge, [field]: value } : edge) });
  }

  function addNode() {
    const index = nodes.length;
    setDraft({ ...draft, nodes: [...nodes, { type: `New concept ${index + 1}`, description: "Describe this concept." }] });
    setPositions([...positions, { x: 50 + ((index * 9) % 28) - 14, y: 50 + ((index * 13) % 28) - 14 }]);
    setSelection({ kind: "node", index });
  }

  function deleteNode(index: number) {
    const identity = String(nodes[index]?.type || nodes[index]?.name || "");
    setDraft({ ...draft, nodes: nodes.filter((_, itemIndex) => itemIndex !== index), edges: edges.filter((edge) => edge.source !== identity && edge.target !== identity) });
    setPositions(positions.filter((_, itemIndex) => itemIndex !== index));
    setSelection(null);
  }

  function addEdge() {
    if (nodes.length < 2) return;
    const index = edges.length;
    setDraft({ ...draft, edges: [...edges, { source: String(nodes[0].type || nodes[0].name), target: String(nodes[1].type || nodes[1].name), relation: "relates to" }] });
    setSelection({ kind: "edge", index });
  }

  function pointerMove(event: PointerEvent<HTMLDivElement>) {
    if (dragging == null || !innerRef.current) return;
    const rect = innerRef.current.getBoundingClientRect();
    const x = Math.max(4, Math.min(96, ((event.clientX - rect.left) / rect.width) * 100));
    const y = Math.max(6, Math.min(94, ((event.clientY - rect.top) / rect.height) * 100));
    setPositions((current) => current.map((position, index) => index === dragging ? { x, y } : position));
  }

  async function save() {
    setSaving(true);
    setError("");
    try {
      await onSave(draft);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not save the graph.");
      setSaving(false);
    }
  }

  const selectedNode = selection?.kind === "node" ? nodes[selection.index] : null;
  const selectedEdge = selection?.kind === "edge" ? edges[selection.index] : null;

  return createPortal(<div className="workspace-overlay graph-overlay" role="dialog" aria-modal="true" aria-label="Knowledge graph workspace">
    <div className="graph-editor-window">
      <header className="workspace-window__header"><div><span className="artifact-kicker">Interactive schema workspace</span><h2>Knowledge graph</h2><p>Drag concepts, edit their meaning, and connect them without leaving TOPAS.</p></div><div><span className="graph-count"><Network size={15} /> {nodes.length} nodes · {edges.length} relationships</span><button className="icon-button" type="button" aria-label="Close graph workspace" onClick={onClose}><X size={19} /></button></div></header>
      <div className="graph-editor-body">
        <aside className="graph-list-panel"><div className="graph-list-head"><span>Schema objects</span><button type="button" onClick={addNode}><Plus size={14} /> Node</button></div><div className="graph-section-label">Concepts</div>{nodes.map((node, index) => <button className={selection?.kind === "node" && selection.index === index ? "is-active" : ""} type="button" key={index} onClick={() => setSelection({ kind: "node", index })}><Network size={14} /><span>{String(node.type || node.name || `Node ${index + 1}`)}</span></button>)}<div className="graph-section-label graph-section-label--relations">Relationships <button type="button" disabled={nodes.length < 2} onClick={addEdge}><GitBranchPlus size={14} /></button></div>{edges.map((edge, index) => <button className={selection?.kind === "edge" && selection.index === index ? "is-active" : ""} type="button" key={index} onClick={() => setSelection({ kind: "edge", index })}><GitBranchPlus size={14} /><span>{String(edge.relation || "relates to")}</span></button>)}</aside>
        <main className="graph-stage" onPointerMove={pointerMove} onPointerUp={() => setDragging(null)} onPointerLeave={() => setDragging(null)}>
          <div className="graph-stage__toolbar"><button type="button" aria-label="Zoom out" onClick={() => setZoom((value) => Math.max(0.7, value - 0.1))}><Minus size={15} /></button><button type="button" onClick={() => setZoom(1)}>{Math.round(zoom * 100)}%</button><button type="button" aria-label="Zoom in" onClick={() => setZoom((value) => Math.min(1.5, value + 0.1))}><ZoomIn size={15} /></button></div>
          <div className="graph-stage__inner" ref={innerRef} style={{ transform: `scale(${zoom})` }}>
            <svg viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">{edges.map((edge, index) => {
              const source = nodeIndex.get(String(edge.source)); const target = nodeIndex.get(String(edge.target));
              if (source == null || target == null || !positions[source] || !positions[target]) return null;
              return <line className={selection?.kind === "edge" && selection.index === index ? "is-selected" : ""} key={index} x1={positions[source].x} y1={positions[source].y} x2={positions[target].x} y2={positions[target].y} />;
            })}</svg>
            {nodes.map((node, index) => <button className={`graph-edit-node ${selection?.kind === "node" && selection.index === index ? "is-selected" : ""}`} type="button" style={{ left: `${positions[index]?.x || 50}%`, top: `${positions[index]?.y || 50}%` }} key={index} onPointerDown={(event) => { event.currentTarget.setPointerCapture(event.pointerId); setDragging(index); setSelection({ kind: "node", index }); }}><Network size={15} /><span>{String(node.type || node.name || `Node ${index + 1}`)}</span></button>)}
          </div>
        </main>
        <aside className="graph-inspector"><span className="graph-inspector__kicker">Inspector</span>{selectedNode && selection?.kind === "node" ? <><h3>Concept</h3><label><span>Name / type</span><input value={String(selectedNode.type || selectedNode.name || "")} onChange={(event) => updateNode(selection.index, "type", event.target.value)} /></label><label><span>Description</span><textarea rows={8} value={String(selectedNode.description || "")} onChange={(event) => updateNode(selection.index, "description", event.target.value)} /></label><button className="danger-button" type="button" onClick={() => deleteNode(selection.index)}><Trash2 size={15} /> Delete concept</button></> : selectedEdge && selection?.kind === "edge" ? <><h3>Relationship</h3><label><span>Source</span><select value={String(selectedEdge.source || "")} onChange={(event) => updateEdge(selection.index, "source", event.target.value)}>{nodes.map((node, index) => <option key={index}>{String(node.type || node.name)}</option>)}</select></label><label><span>Relation</span><input value={String(selectedEdge.relation || "")} onChange={(event) => updateEdge(selection.index, "relation", event.target.value)} /></label><label><span>Target</span><select value={String(selectedEdge.target || "")} onChange={(event) => updateEdge(selection.index, "target", event.target.value)}>{nodes.map((node, index) => <option key={index}>{String(node.type || node.name)}</option>)}</select></label><button className="danger-button" type="button" onClick={() => { setDraft({ ...draft, edges: edges.filter((_, index) => index !== selection.index) }); setSelection(null); }}><Trash2 size={15} /> Delete relationship</button></> : <div className="graph-inspector__empty"><Network size={25} /><p>Select a concept or relationship to edit it.</p></div>}</aside>
      </div>
      <footer className="workspace-window__footer"><span>{error || "Layout changes are visual; schema content is saved to the component JSON."}</span><div><button className="button button--quiet" type="button" onClick={onClose}>Cancel</button><button className="button button--primary" type="button" disabled={saving} onClick={() => void save()}>{saving ? <span className="spinner" /> : <Save size={16} />} Save graph</button></div></footer>
    </div>
  </div>, document.body);
}
