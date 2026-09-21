PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS users (
  id TEXT PRIMARY KEY,
  email TEXT NOT NULL UNIQUE,
  created_at TEXT NOT NULL,
  last_seen_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS projects (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  domain TEXT NOT NULL,
  system_name TEXT NOT NULL,
  user_name TEXT NOT NULL,
  interaction_unit TEXT NOT NULL,
  model TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'draft',
  progress INTEGER NOT NULL DEFAULT 0,
  current_step TEXT NOT NULL DEFAULT 'Ready for textbooks',
  error TEXT,
  is_demo INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  completed_at TEXT,
  refinement_status TEXT NOT NULL DEFAULT 'not_started',
  refinement_progress INTEGER NOT NULL DEFAULT 0,
  refinement_step TEXT NOT NULL DEFAULT 'Ready after extraction',
  refinement_error TEXT,
  refined_at TEXT
);

CREATE TABLE IF NOT EXISTS books (
  id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  role TEXT NOT NULL,
  original_name TEXT NOT NULL,
  object_key TEXT NOT NULL,
  size_bytes INTEGER NOT NULL,
  page_count INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  kind TEXT NOT NULL,
  message TEXT NOT NULL,
  payload_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS artifacts (
  id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  phase TEXT NOT NULL,
  role TEXT NOT NULL DEFAULT '',
  kind TEXT NOT NULL,
  book_index INTEGER NOT NULL DEFAULT -1,
  component TEXT NOT NULL DEFAULT '',
  report TEXT NOT NULL DEFAULT '',
  object_key TEXT NOT NULL,
  chapter_count INTEGER NOT NULL DEFAULT 0,
  updated_at TEXT NOT NULL,
  UNIQUE(project_id, phase, role, kind, book_index, component, report)
);

CREATE INDEX IF NOT EXISTS idx_projects_user_updated ON projects(user_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_books_project_role ON books(project_id, role);
CREATE INDEX IF NOT EXISTS idx_events_project_id ON events(project_id, id);
CREATE INDEX IF NOT EXISTS idx_artifacts_project ON artifacts(project_id, phase);
