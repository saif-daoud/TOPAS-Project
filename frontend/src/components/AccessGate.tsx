import { useState, type FormEvent } from "react";
import { ArrowRight, Eye, EyeOff, KeyRound, LockKeyhole, Mail, ShieldCheck, Sparkles } from "lucide-react";
import { Brand } from "./Brand";

interface Props {
  onLogin: (email: string, accessCode: string) => Promise<void>;
}

export function AccessGate({ onLogin }: Props) {
  const [email, setEmail] = useState("");
  const [accessCode, setAccessCode] = useState("");
  const [showCode, setShowCode] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function submit(event: FormEvent) {
    event.preventDefault();
    setError("");
    if (!email.trim() || !accessCode.trim()) {
      setError("Enter your email and access code to continue.");
      return;
    }
    setBusy(true);
    try {
      await onLogin(email, accessCode);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not sign in.");
      setBusy(false);
    }
  }

  return (
    <main className="gate-shell">
      <section className="gate-story">
        <div className="gate-story__top"><Brand light /></div>
        <div className="gate-visual" aria-hidden="true">
          <div className="gate-visual__glow" />
          <div className="orbit orbit--one"><i /><i /><i /></div>
          <div className="orbit orbit--two"><i /><i /></div>
          <div className="orbit orbit--three"><i /></div>
          <div className="gate-visual__core">
            <span>Domain<br />knowledge</span>
          </div>
          <span className="orbit-label orbit-label--a">Planner</span>
          <span className="orbit-label orbit-label--b">Memory</span>
          <span className="orbit-label orbit-label--c">Policy</span>
          <span className="orbit-label orbit-label--d">Simulator</span>
        </div>
        <div className="gate-story__copy">
          <span className="eyebrow eyebrow--light"><Sparkles size={13} /> From textbooks to proactive systems</span>
          <h1>Build agents that can<br /><em>plan ahead.</em></h1>
          <p>
            Turn domain textbooks into structured dialogue planners, memory, and policies—inside one traceable workspace.
          </p>
        </div>
        <div className="gate-story__foot">
          <span><ShieldCheck size={15} /> Research workspace</span>
          <span>QCRI · TOPAS</span>
        </div>
      </section>

      <section className="gate-form-wrap">
        <div className="gate-form-top"><Brand /></div>
        <form className="gate-card" onSubmit={submit}>
          <div className="gate-card__icon"><LockKeyhole size={23} /></div>
          <span className="eyebrow">Private workspace</span>
          <h2>Welcome to TOPAS</h2>
          <p className="gate-card__intro">Sign in to continue your projects and pick up exactly where you left off.</p>

          <label className="field-label" htmlFor="gate-email">Work email</label>
          <div className="field-control">
            <Mail size={18} />
            <input
              id="gate-email"
              type="email"
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              placeholder="name@organization.org"
              autoComplete="email"
              autoFocus
            />
          </div>

          <label className="field-label" htmlFor="gate-code">Access code</label>
          <div className="field-control">
            <KeyRound size={18} />
            <input
              id="gate-code"
              type={showCode ? "text" : "password"}
              value={accessCode}
              onChange={(event) => setAccessCode(event.target.value)}
              placeholder="Enter your access code"
              autoComplete="current-password"
            />
            <button className="icon-button icon-button--field" type="button" onClick={() => setShowCode(!showCode)} aria-label={showCode ? "Hide access code" : "Show access code"}>
              {showCode ? <EyeOff size={17} /> : <Eye size={17} />}
            </button>
          </div>

          {error && <div className="form-error" role="alert">{error}</div>}
          <button className="button button--primary button--wide" type="submit" disabled={busy}>
            {busy ? <span className="spinner" /> : <>Enter workspace <ArrowRight size={18} /></>}
          </button>
          <p className="privacy-note"><ShieldCheck size={14} /> Your projects are securely saved to your profile.</p>
        </form>
        <p className="gate-legal">Restricted research preview · Authorized users only</p>
      </section>
    </main>
  );
}
