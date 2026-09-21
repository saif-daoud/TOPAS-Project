import { useCallback, useEffect, useRef, useState } from "react";
import { AlertCircle, CheckCircle2, X } from "lucide-react";
import { api, authStore, emptyManifest } from "./api";
import { AccessGate } from "./components/AccessGate";
import { Dashboard } from "./components/Dashboard";
import { NewProjectDialog } from "./components/NewProjectDialog";
import { ProjectView } from "./components/ProjectView";
import { Sidebar } from "./components/Sidebar";
import type { Manifest, PipelineStage, Project, ProjectDraft, ProjectEvent, Role, User } from "./types";

interface Toast {
  id: number;
  message: string;
  tone: "success" | "error";
}

export default function App() {
  const [user, setUser] = useState<User | null>(null);
  const [booting, setBooting] = useState(Boolean(authStore.get()));
  const [projects, setProjects] = useState<Project[]>([]);
  const [loadingProjects, setLoadingProjects] = useState(false);
  const [activeProject, setActiveProject] = useState<Project | null>(null);
  const [activeStage, setActiveStage] = useState<PipelineStage>("extraction");
  const [manifest, setManifest] = useState<Manifest>(emptyManifest());
  const [events, setEvents] = useState<ProjectEvent[]>([]);
  const [showCreate, setShowCreate] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const lastEvent = useRef(0);

  const toast = useCallback((message: string, tone: Toast["tone"] = "success") => {
    const id = Date.now() + Math.random();
    setToasts((current) => [...current, { id, message, tone }]);
    window.setTimeout(() => setToasts((current) => current.filter((item) => item.id !== id)), 4200);
  }, []);

  const loadProjects = useCallback(async () => {
    setLoadingProjects(true);
    try {
      const response = await api.projects();
      setProjects(response.projects);
    } catch (reason) {
      toast(reason instanceof Error ? reason.message : "Could not load projects.", "error");
    } finally {
      setLoadingProjects(false);
    }
  }, [toast]);

  const openProject = useCallback(async (id: string, quiet = false) => {
    if (!quiet) setRefreshing(true);
    try {
      const response = await api.project(id);
      setActiveProject(response.project);
      setManifest(response.manifest);
      const eventResponse = await api.events(id, 0);
      setEvents(eventResponse.events);
      lastEvent.current = eventResponse.events.at(-1)?.id || 0;
    } catch (reason) {
      toast(reason instanceof Error ? reason.message : "Could not open the project.", "error");
    } finally {
      setRefreshing(false);
    }
  }, [toast]);

  useEffect(() => {
    if (!authStore.get()) {
      setBooting(false);
      return;
    }
    api.me().then((profile) => {
      setUser(profile);
      setBooting(false);
    }).catch(() => {
      authStore.clear();
      setBooting(false);
    });
  }, []);

  useEffect(() => {
    if (user) void loadProjects();
  }, [user, loadProjects]);

  useEffect(() => {
    if (!activeProject || (activeProject.status !== "running" && activeProject.refinement_status !== "running")) return;
    const interval = window.setInterval(async () => {
      try {
        const response = await api.events(activeProject.id, lastEvent.current);
        if (response.events.length) {
          setEvents((current) => [...current, ...response.events]);
          lastEvent.current = response.events.at(-1)?.id || lastEvent.current;
        }
        setActiveProject(response.project);
        const detail = await api.project(activeProject.id);
        setManifest(detail.manifest);
        setActiveProject(detail.project);
        const extractionFinished = activeProject.status === "running" && detail.project.status !== "running";
        const refinementFinished = activeProject.refinement_status === "running" && detail.project.refinement_status !== "running";
        if (extractionFinished || refinementFinished) {
          const phase = extractionFinished ? "Extraction" : "Refinement";
          const successful = extractionFinished ? detail.project.status === "completed" : detail.project.refinement_status === "completed";
          toast(successful ? `${phase} completed successfully.` : `${phase} stopped. Review the run details.`, successful ? "success" : "error");
          void loadProjects();
        }
      } catch {
        // A transient poll failure should not interrupt the active extraction.
      }
    }, 1500);
    return () => window.clearInterval(interval);
  }, [activeProject?.id, activeProject?.status, activeProject?.refinement_status, loadProjects, toast]);

  async function login(email: string, accessCode: string) {
    const response = await api.login(email, accessCode);
    authStore.set(response.token);
    setUser(response.user);
    toast(response.returning ? "Welcome back. Your workspace is restored." : "Your TOPAS workspace is ready.");
  }

  function logout() {
    authStore.clear();
    setUser(null);
    setProjects([]);
    setActiveProject(null);
    setActiveStage("extraction");
    setManifest(emptyManifest());
  }

  async function createProject(draft: ProjectDraft) {
    const response = await api.createProject(draft);
    setShowCreate(false);
    await loadProjects();
    setActiveStage("extraction");
    await openProject(response.project.id);
    toast("Project created. Add either set of textbooks to begin.");
  }

  async function refreshProject() {
    if (!activeProject) return;
    await openProject(activeProject.id);
  }

  async function upload(role: Role, files: File[]) {
    if (!activeProject) return;
    try {
      const response = await api.uploadBooks(activeProject.id, role, files);
      setActiveProject(response.project);
      await openProject(activeProject.id, true);
      toast(`${files.length} ${role} textbook${files.length === 1 ? "" : "s"} added.`);
    } catch (reason) {
      toast(reason instanceof Error ? reason.message : "Upload failed.", "error");
      throw reason;
    }
  }

  async function removeBook(bookId: string) {
    if (!activeProject) return;
    try {
      const response = await api.removeBook(activeProject.id, bookId);
      setActiveProject(response.project);
      toast("Textbook removed.");
    } catch (reason) {
      toast(reason instanceof Error ? reason.message : "Could not remove the textbook.", "error");
    }
  }

  async function runExtraction() {
    if (!activeProject) return;
    try {
      await api.run(activeProject.id);
      await new Promise((resolve) => window.setTimeout(resolve, 250));
      await openProject(activeProject.id, true);
      toast("Extraction started. Results will appear here live.");
    } catch (reason) {
      toast(reason instanceof Error ? reason.message : "Could not start extraction.", "error");
      throw reason;
    }
  }

  async function runRefinement() {
    if (!activeProject) return;
    try {
      await api.refine(activeProject.id);
      await new Promise((resolve) => window.setTimeout(resolve, 250));
      await openProject(activeProject.id, true);
      toast("Refinement started. The state-space audit will update here live.");
    } catch (reason) {
      toast(reason instanceof Error ? reason.message : "Could not start refinement.", "error");
      throw reason;
    }
  }

  if (booting) return <div className="app-boot"><span className="boot-mark"><i /></span><strong>TOPAS</strong><small>Restoring your workspace…</small></div>;
  if (!user) return <AccessGate onLogin={login} />;

  return <div className="app-shell">
    <Sidebar user={user} page={activeProject ? "project" : "dashboard"} stage={activeStage} onStage={setActiveStage} onHome={() => { setActiveProject(null); setActiveStage("extraction"); void loadProjects(); }} onLogout={logout} />
    <div className="app-content">
      {activeProject ? <ProjectView project={activeProject} manifest={manifest} events={events} stage={activeStage} refreshing={refreshing} onBack={() => { setActiveProject(null); setActiveStage("extraction"); void loadProjects(); }} onRefresh={refreshProject} onUpload={upload} onRemove={removeBook} onRun={runExtraction} onRunRefinement={runRefinement} onStage={setActiveStage} /> : <Dashboard user={user} projects={projects} loading={loadingProjects} onCreate={() => setShowCreate(true)} onOpen={(id) => { setActiveStage("extraction"); void openProject(id); }} />}
    </div>
    {showCreate && <NewProjectDialog onClose={() => setShowCreate(false)} onCreate={createProject} />}
    <div className="toast-stack" aria-live="polite">{toasts.map((item) => <div className={`toast toast--${item.tone}`} key={item.id}>{item.tone === "success" ? <CheckCircle2 size={18} /> : <AlertCircle size={18} />}<span>{item.message}</span><button onClick={() => setToasts((current) => current.filter((toastItem) => toastItem.id !== item.id))}><X size={15} /></button></div>)}</div>
  </div>;
}
