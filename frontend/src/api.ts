import type { ComponentPhase, Manifest, Project, ProjectDetail, ProjectDraft, ProjectEvent, Role, User } from "./types";

const API_BASE = String(import.meta.env.VITE_API_BASE || "").replace(/\/$/, "");
const TOKEN_KEY = "topas_studio_token";

export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

export const authStore = {
  get: () => localStorage.getItem(TOKEN_KEY) || "",
  set: (token: string) => localStorage.setItem(TOKEN_KEY, token),
  clear: () => localStorage.removeItem(TOKEN_KEY),
};

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const token = authStore.get();
  const headers = new Headers(options.headers);
  if (token) headers.set("Authorization", `Bearer ${token}`);
  if (options.body && !(options.body instanceof FormData)) headers.set("Content-Type", "application/json");
  const response = await fetch(`${API_BASE}${path}`, { ...options, headers });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    if (response.status === 401 && path !== "/api/auth/login") authStore.clear();
    throw new ApiError(String(data.detail || "Something went wrong."), response.status);
  }
  return data as T;
}

export const api = {
  login: (email: string, accessCode: string) =>
    request<{ token: string; user: User; returning: boolean }>("/api/auth/login", {
      method: "POST",
      body: JSON.stringify({ email, access_code: accessCode }),
    }),
  me: () => request<User>("/api/me"),
  projects: () => request<{ projects: Project[] }>("/api/projects"),
  createProject: (draft: ProjectDraft) =>
    request<{ project: Project }>("/api/projects", { method: "POST", body: JSON.stringify(draft) }),
  project: (id: string) => request<ProjectDetail>(`/api/projects/${id}`),
  uploadBooks: (id: string, role: Role, files: File[]) => {
    const body = new FormData();
    body.set("role", role);
    files.forEach((file) => body.append("files", file));
    return request<{ project: Project }>(`/api/projects/${id}/books`, { method: "POST", body });
  },
  removeBook: (projectId: string, bookId: string) =>
    request<{ project: Project }>(`/api/projects/${projectId}/books/${bookId}`, { method: "DELETE" }),
  run: (id: string) => request<{ accepted: boolean }>(`/api/projects/${id}/run`, { method: "POST" }),
  refine: (id: string) => request<{ accepted: boolean }>(`/api/projects/${id}/refine`, { method: "POST" }),
  events: (id: string, after = 0) =>
    request<{ events: ProjectEvent[]; project: Project }>(`/api/projects/${id}/events?after=${after}`),
  summary: (id: string, role: Role, bookIndex: number) =>
    request<{ data: unknown }>(`/api/projects/${id}/summaries/${role}/${bookIndex}`),
  component: (id: string, role: Role, component: string, phase: ComponentPhase = "extraction") =>
    request<{ data: unknown }>(`/api/projects/${id}/components/${role}/${component}?phase=${phase}`),
  updateComponent: (id: string, role: Role, component: string, phase: ComponentPhase, data: unknown) =>
    request<{ saved: boolean; data: unknown; project: Project }>(`/api/projects/${id}/components/${role}/${component}`, {
      method: "PUT",
      body: JSON.stringify({ phase, data }),
    }),
  refinementReport: (id: string, report: "summary" | "details" = "summary") =>
    request<{ data: unknown }>(`/api/projects/${id}/refinement/${report}`),
};

export const emptyManifest = (): Manifest => ({
  roles: {
    system: { summaries: [], book_components: {}, merged_components: [], refined_components: [] },
    user: { summaries: [], book_components: {}, merged_components: [], refined_components: [] },
  },
  refinement: { report_available: false, summary_available: false },
});
