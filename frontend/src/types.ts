export type Role = "system" | "user";
export type ProjectStatus = "draft" | "queued" | "running" | "completed" | "failed";
export type RefinementStatus = "not_started" | "running" | "completed" | "failed" | "stale";
export type PipelineStage = "extraction" | "refinement";
export type ComponentPhase = "extraction" | "refinement";

export interface User {
  id: string;
  email: string;
}

export interface Book {
  id: string;
  project_id: string;
  role: Role;
  original_name: string;
  size_bytes: number;
  page_count: number;
  created_at: string;
}

export interface Project {
  id: string;
  name: string;
  domain: string;
  system_name: string;
  user_name: string;
  interaction_unit: "session" | "conversation";
  model: "gpt-5.1" | "gpt-4.1" | "DeepSeek-V4-Pro";
  status: ProjectStatus;
  progress: number;
  current_step: string;
  error?: string | null;
  is_demo: boolean;
  book_count?: number;
  system_book_count?: number;
  user_book_count?: number;
  books?: Book[];
  created_at: string;
  updated_at: string;
  completed_at?: string | null;
  refinement_status: RefinementStatus;
  refinement_progress: number;
  refinement_step: string;
  refinement_error?: string | null;
  refined_at?: string | null;
}

export interface ProjectEvent {
  id: number;
  project_id: string;
  kind: string;
  message: string;
  payload: Record<string, unknown>;
  created_at: string;
}

export interface RoleManifest {
  summaries: { book_index: number; chapter_count: number }[];
  book_components: Record<string, string[]>;
  merged_components: string[];
  refined_components: string[];
}

export interface Manifest {
  roles: Record<Role, RoleManifest>;
  refinement: {
    report_available: boolean;
    summary_available: boolean;
  };
}

export interface ProjectDetail {
  project: Project;
  manifest: Manifest;
}

export interface ProjectDraft {
  name: string;
  domain: string;
  system_name: string;
  user_name: string;
  interaction_unit: "session" | "conversation";
  model: Project["model"];
}
