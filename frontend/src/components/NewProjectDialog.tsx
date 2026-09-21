import { useState, type FormEvent } from "react";
import { ArrowLeft, ArrowRight, Bot, Check, MessageCircleMore, Sparkles, UserRound, X } from "lucide-react";
import type { ProjectDraft } from "../types";

const defaults: ProjectDraft = {
  name: "",
  domain: "",
  system_name: "",
  user_name: "",
  interaction_unit: "session",
  model: "gpt-5.1",
};

interface Props {
  onClose: () => void;
  onCreate: (draft: ProjectDraft) => Promise<void>;
}

export function NewProjectDialog({ onClose, onCreate }: Props) {
  const [step, setStep] = useState(0);
  const [draft, setDraft] = useState(defaults);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const update = <K extends keyof ProjectDraft>(key: K, value: ProjectDraft[K]) => setDraft((current) => ({ ...current, [key]: value }));

  async function next(event: FormEvent) {
    event.preventDefault();
    setError("");
    if (step === 0 && (!draft.name.trim() || !draft.domain.trim())) return setError("Name the project and its domain to continue.");
    if (step === 1 && (!draft.system_name.trim() || !draft.user_name.trim())) return setError("Name both sides of the interaction.");
    if (step < 2) return setStep(step + 1);
    setBusy(true);
    try {
      await onCreate(draft);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not create the project.");
      setBusy(false);
    }
  }

  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <form className="project-dialog" onSubmit={next}>
        <button type="button" className="icon-button dialog-close" onClick={onClose} aria-label="Close"><X size={19} /></button>
        <div className="dialog-aside">
          <div><span className="eyebrow eyebrow--light"><Sparkles size={13} /> New blueprint</span><h2>Start with the shape<br />of the interaction.</h2><p>This context is inserted into TOPAS’s original extraction prompts.</p></div>
          <div className="dialog-steps">
            {["Domain", "Participants", "Model"].map((label, index) => (
              <div className={`${index === step ? "is-active" : ""} ${index < step ? "is-done" : ""}`} key={label}>
                <span>{index < step ? <Check size={14} /> : index + 1}</span><strong>{label}</strong>
              </div>
            ))}
          </div>
          <small>Late fusion · Source-grounded</small>
        </div>
        <div className="dialog-main">
          {step === 0 && <div className="dialog-step page-enter">
            <span className="step-count">Step 1 of 3</span>
            <h3>What are we building?</h3>
            <p>Give your workspace a recognizable name and define the knowledge domain.</p>
            <label className="field-label" htmlFor="project-name">Project name</label>
            <input className="text-input" id="project-name" value={draft.name} onChange={(e) => update("name", e.target.value)} placeholder="e.g. Therapy agent" autoFocus />
            <label className="field-label" htmlFor="domain-name">Domain name</label>
            <input className="text-input" id="domain-name" value={draft.domain} onChange={(e) => update("domain", e.target.value)} placeholder="e.g. Cognitive behavioral therapy" />
          </div>}
          {step === 1 && <div className="dialog-step page-enter">
            <span className="step-count">Step 2 of 3</span>
            <h3>Name the two parties</h3>
            <p>These labels keep the extracted planner grounded in the real interaction.</p>
            <div className="role-input-grid">
              <label className="role-input"><span><Bot size={18} /> System party</span><input value={draft.system_name} onChange={(e) => update("system_name", e.target.value)} placeholder="e.g. Counselor" autoFocus /></label>
              <span className="role-connector"><ArrowRight size={17} /></span>
              <label className="role-input"><span><UserRound size={18} /> User party</span><input value={draft.user_name} onChange={(e) => update("user_name", e.target.value)} placeholder="e.g. Client" /></label>
            </div>
            <label className="field-label">Interaction unit</label>
            <div className="segmented">
              {(["session", "conversation"] as const).map((unit) => <button type="button" className={draft.interaction_unit === unit ? "is-selected" : ""} onClick={() => update("interaction_unit", unit)} key={unit}><MessageCircleMore size={17} /><span><strong>{unit[0].toUpperCase() + unit.slice(1)}</strong><small>{unit === "session" ? "One meeting in a longer journey" : "One complete dialogue"}</small></span></button>)}
            </div>
          </div>}
          {step === 2 && <div className="dialog-step page-enter">
            <span className="step-count">Step 3 of 3</span>
            <h3>Choose the extraction model</h3>
            <p>Models are served through the configured Azure AI endpoint. You can add sources next.</p>
            <div className="model-options">
              {(["gpt-5.1", "gpt-4.1", "DeepSeek-V4-Pro"] as const).map((model, index) => (
                <button type="button" className={draft.model === model ? "is-selected" : ""} onClick={() => update("model", model)} key={model}>
                  <span className="model-radio">{draft.model === model && <i />}</span>
                  <span><strong>{model}</strong><small>{index === 0 ? "Highest reasoning quality · Recommended" : index === 1 ? "Fast, capable default" : "Alternative reasoning model"}</small></span>
                  {index === 0 && <em>Recommended</em>}
                </button>
              ))}
            </div>
            <div className="review-strip"><span>{draft.domain}</span><i /> <span>{draft.system_name} ↔ {draft.user_name}</span><i /> <span>{draft.interaction_unit}</span></div>
          </div>}
          {error && <div className="form-error" role="alert">{error}</div>}
          <div className="dialog-actions">
            {step > 0 ? <button type="button" className="button button--secondary" onClick={() => setStep(step - 1)}><ArrowLeft size={17} /> Back</button> : <span />}
            <button className="button button--primary" disabled={busy}>{busy ? <span className="spinner" /> : step === 2 ? <>Create & add books <ArrowRight size={17} /></> : <>Continue <ArrowRight size={17} /></>}</button>
          </div>
        </div>
      </form>
    </div>
  );
}
