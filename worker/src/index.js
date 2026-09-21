const SYSTEM_COMPONENTS = ["macro_actions", "micro_actions", "conversation_states", "knowledge_graph", "cautions"];
const USER_COMPONENTS = ["user_profile"];
const COMPONENTS = { system: SYSTEM_COMPONENTS, user: USER_COMPONENTS };
const MODELS = ["gpt-5.1", "gpt-4.1", "DeepSeek-V4-Pro"];
const encoder = new TextEncoder();

const now = () => new Date().toISOString();
const uuid = () => crypto.randomUUID().replaceAll("-", "");
const b64url = (bytes) => btoa(String.fromCharCode(...bytes)).replaceAll("+", "-").replaceAll("/", "_").replaceAll("=", "");
const unb64url = (text) => Uint8Array.from(atob(text.replaceAll("-", "+").replaceAll("_", "/")), (c) => c.charCodeAt(0));

function cors(request, env) {
  const origin = request.headers.get("Origin") || "";
  const allowed = String(env.ALLOWED_ORIGINS || "").split(",").map((item) => item.trim());
  return allowed.includes(origin) ? origin : allowed[0] || "https://saif-daoud.github.io";
}

function response(request, env, body, status = 200, extra = {}) {
  return new Response(body === null ? null : JSON.stringify(body), {
    status,
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      "Access-Control-Allow-Origin": cors(request, env),
      "Access-Control-Allow-Headers": "Authorization, Content-Type, X-TOPAS-JOB-TOKEN",
      "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
      "Vary": "Origin",
      ...extra,
    },
  });
}

const error = (request, env, detail, status = 400) => response(request, env, { detail }, status);

async function sha256Hex(value) {
  return [...new Uint8Array(await crypto.subtle.digest("SHA-256", encoder.encode(value)))].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

async function hmac(secret, value) {
  const key = await crypto.subtle.importKey("raw", encoder.encode(secret), { name: "HMAC", hash: "SHA-256" }, false, ["sign"]);
  return new Uint8Array(await crypto.subtle.sign("HMAC", key, encoder.encode(value)));
}

async function createToken(env, user) {
  const payload = b64url(encoder.encode(JSON.stringify({ uid: user.id, email: user.email, exp: Date.now() + 30 * 86400000 })));
  return `${payload}.${b64url(await hmac(env.TOKEN_SECRET, payload))}`;
}

async function currentUser(request, env) {
  const auth = request.headers.get("Authorization") || "";
  if (!auth.startsWith("Bearer ")) return null;
  const [payload, signature] = auth.slice(7).split(".");
  if (!payload || !signature) return null;
  const expected = b64url(await hmac(env.TOKEN_SECRET, payload));
  if (signature.length !== expected.length) return null;
  let mismatch = 0;
  for (let index = 0; index < expected.length; index += 1) mismatch |= signature.charCodeAt(index) ^ expected.charCodeAt(index);
  if (mismatch) return null;
  try {
    const parsed = JSON.parse(new TextDecoder().decode(unb64url(payload)));
    if (parsed.exp < Date.now()) return null;
    return await env.DB.prepare("SELECT * FROM users WHERE id = ?").bind(parsed.uid).first();
  } catch {
    return null;
  }
}

async function addEvent(env, projectId, kind, message, payload = {}) {
  await env.DB.prepare("INSERT INTO events (project_id, kind, message, payload_json, created_at) VALUES (?, ?, ?, ?, ?)")
    .bind(projectId, kind, message, JSON.stringify(payload), now()).run();
}

function publicProject(row, books = undefined) {
  if (!row) return null;
  const project = { ...row, is_demo: Boolean(row.is_demo) };
  delete project.user_id;
  if (books) project.books = books.map(({ object_key, ...book }) => book);
  return project;
}

async function projectFor(env, projectId, userId) {
  const project = await env.DB.prepare("SELECT * FROM projects WHERE id = ? AND user_id = ?").bind(projectId, userId).first();
  if (!project) return null;
  const { results: books } = await env.DB.prepare("SELECT * FROM books WHERE project_id = ? ORDER BY role, created_at").bind(projectId).all();
  return { raw: project, public: publicProject(project, books), books };
}

async function manifestFor(env, projectId) {
  const manifest = {
    roles: {
      system: { summaries: [], book_components: {}, merged_components: [], refined_components: [] },
      user: { summaries: [], book_components: {}, merged_components: [], refined_components: [] },
    },
    refinement: { report_available: false, summary_available: false },
  };
  const { results } = await env.DB.prepare("SELECT * FROM artifacts WHERE project_id = ? ORDER BY book_index, component").bind(projectId).all();
  for (const artifact of results) {
    if (artifact.kind === "summary") manifest.roles[artifact.role].summaries.push({ book_index: artifact.book_index, chapter_count: artifact.chapter_count });
    if (artifact.kind === "book_component") {
      const key = String(artifact.book_index);
      (manifest.roles[artifact.role].book_components[key] ||= []).push(artifact.component);
    }
    if (artifact.kind === "component" && artifact.phase === "extraction") manifest.roles[artifact.role].merged_components.push(artifact.component);
    if (artifact.kind === "component" && artifact.phase === "refinement") manifest.roles[artifact.role].refined_components.push(artifact.component);
    if (artifact.kind === "report" && artifact.report === "details") manifest.refinement.report_available = true;
    if (artifact.kind === "report" && artifact.report === "summary") manifest.refinement.summary_available = true;
  }
  return manifest;
}

async function addArtifact(env, values, data) {
  const key = values.object_key || `artifacts/${values.project_id}/${values.phase}/${values.role || "global"}/${values.kind}/${values.book_index ?? -1}/${values.component || values.report || uuid()}.json`;
  await env.FILES.put(key, typeof data === "string" ? data : JSON.stringify(data), { metadata: { project_id: values.project_id } });
  await env.DB.prepare(`INSERT INTO artifacts
    (id, project_id, phase, role, kind, book_index, component, report, object_key, chapter_count, updated_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(project_id, phase, role, kind, book_index, component, report)
    DO UPDATE SET object_key = excluded.object_key, chapter_count = excluded.chapter_count, updated_at = excluded.updated_at`)
    .bind(uuid(), values.project_id, values.phase, values.role || "", values.kind, Number(values.book_index ?? -1), values.component || "", values.report || "", key, Number(values.chapter_count || 0), now()).run();
  return key;
}

async function seedDemo(env, userId) {
  const existing = await env.DB.prepare("SELECT id FROM projects WHERE user_id = ? AND is_demo = 1").bind(userId).first();
  if (existing) return;
  const projectId = `demo${userId}`;
  const timestamp = now();
  await env.DB.prepare(`INSERT INTO projects
    (id,user_id,name,domain,system_name,user_name,interaction_unit,model,status,progress,current_step,is_demo,created_at,updated_at,completed_at,refinement_status,refinement_progress,refinement_step,refined_at)
    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)`)
    .bind(projectId, userId, "CBT agent blueprint", "Cognitive behavioral therapy", "Therapist", "Patient", "session", "gpt-4.1", "completed", 100, "Extraction complete", 1, timestamp, timestamp, timestamp, "completed", 100, "Refinement complete", timestamp).run();
  for (const [role, count] of [["system", 7], ["user", 6]]) {
    for (let index = 0; index < count; index += 1) {
      await env.DB.prepare("INSERT INTO books (id,project_id,role,original_name,object_key,size_bytes,page_count,created_at) VALUES (?,?,?,?,?,0,0,?)")
        .bind(uuid(), projectId, role, `${role[0].toUpperCase()}${role.slice(1)} textbook ${String(index + 1).padStart(2, "0")}.pdf`, "", timestamp).run();
      await env.DB.prepare("INSERT INTO artifacts (id,project_id,phase,role,kind,book_index,component,report,object_key,chapter_count,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)")
        .bind(uuid(), projectId, "extraction", role, "summary", index, "", "", `demo/extraction/${role}/summaries_book_${index}.json`, 1, timestamp).run();
    }
  }
  for (const [role, components] of Object.entries(COMPONENTS)) {
    for (const component of components) {
      await env.DB.prepare("INSERT INTO artifacts (id,project_id,phase,role,kind,book_index,component,report,object_key,chapter_count,updated_at) VALUES (?,?,?,?,?,-1,?,?,?,0,?)")
        .bind(uuid(), projectId, "extraction", role, "component", component, "", `demo/extraction/${role}/merged_components/${component}.json`, timestamp).run();
      await env.DB.prepare("INSERT INTO artifacts (id,project_id,phase,role,kind,book_index,component,report,object_key,chapter_count,updated_at) VALUES (?,?,?,?,?,-1,?,?,?,0,?)")
        .bind(uuid(), projectId, "refinement", role, "component", component, "", `demo/refinement/${component}.json`, timestamp).run();
    }
  }
  for (const report of ["summary", "details"]) {
    const filename = report === "summary" ? "refinement_new_dimensions_summary.json" : "refinement_report.json";
    await env.DB.prepare("INSERT INTO artifacts (id,project_id,phase,role,kind,book_index,component,report,object_key,chapter_count,updated_at) VALUES (?,?,?,'','report',-1,'',?,?,0,?)")
      .bind(uuid(), projectId, "refinement", report, `demo/refinement/${filename}`, timestamp).run();
  }
  await addEvent(env, projectId, "completed", "Late-fusion extraction completed", { demo: true });
}

async function dispatchJob(env, projectId, jobType, apiUrl) {
  const result = await fetch(`https://api.github.com/repos/${env.GITHUB_REPOSITORY}/actions/workflows/topas_job.yml/dispatches`, {
    method: "POST",
    headers: { Authorization: `Bearer ${env.GITHUB_TOKEN}`, Accept: "application/vnd.github+json", "User-Agent": "TOPAS-Studio" },
    body: JSON.stringify({ ref: "main", inputs: { project_id: projectId, job_type: jobType, api_url: apiUrl } }),
  });
  if (!result.ok) throw new Error(`GitHub job dispatch failed (${result.status}): ${await result.text()}`);
}

async function artifactJson(env, projectId, filters) {
  const clauses = ["project_id = ?"];
  const bindings = [projectId];
  for (const [key, value] of Object.entries(filters)) { clauses.push(`${key} = ?`); bindings.push(value); }
  const row = await env.DB.prepare(`SELECT object_key FROM artifacts WHERE ${clauses.join(" AND ")}`).bind(...bindings).first();
  if (!row) return null;
  const object = await env.FILES.get(row.object_key, "text");
  if (object === null) return null;
  return JSON.parse(object);
}

async function handlePublic(request, env, url) {
  const method = request.method;
  if (method === "GET" && url.pathname === "/api/health") return response(request, env, { ok: true, service: "TOPAS Studio", models: MODELS, azure_configured: env.AZURE_CONFIGURED === "true" });
  if (method === "POST" && url.pathname === "/api/auth/login") {
    const body = await request.json();
    const email = String(body.email || "").trim().toLowerCase();
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) return error(request, env, "Enter a valid email address.", 422);
    const digest = await sha256Hex(String(body.access_code || ""));
    if (!String(env.ACCESS_CODE_HASHES || "").split(",").map((x) => x.trim()).includes(digest)) return error(request, env, "That access code is not valid.", 401);
    let user = await env.DB.prepare("SELECT * FROM users WHERE email = ?").bind(email).first();
    const returning = Boolean(user);
    if (user) {
      await env.DB.prepare("UPDATE users SET last_seen_at = ? WHERE id = ?").bind(now(), user.id).run();
    } else {
      user = { id: uuid(), email, created_at: now(), last_seen_at: now() };
      await env.DB.prepare("INSERT INTO users (id,email,created_at,last_seen_at) VALUES (?,?,?,?)").bind(user.id, email, user.created_at, user.last_seen_at).run();
    }
    await seedDemo(env, user.id);
    return response(request, env, { token: await createToken(env, user), user: { id: user.id, email }, returning });
  }

  const user = await currentUser(request, env);
  if (!user) return error(request, env, "Authentication required.", 401);
  if (method === "GET" && url.pathname === "/api/me") return response(request, env, { id: user.id, email: user.email });

  if (method === "GET" && url.pathname === "/api/projects") {
    const { results } = await env.DB.prepare(`SELECT p.*,
      COUNT(b.id) book_count,
      SUM(CASE WHEN b.role='system' THEN 1 ELSE 0 END) system_book_count,
      SUM(CASE WHEN b.role='user' THEN 1 ELSE 0 END) user_book_count
      FROM projects p LEFT JOIN books b ON b.project_id=p.id WHERE p.user_id=? GROUP BY p.id ORDER BY p.is_demo ASC,p.updated_at DESC`).bind(user.id).all();
    return response(request, env, { projects: results.map((row) => publicProject(row)) });
  }

  if (method === "POST" && url.pathname === "/api/projects") {
    const body = await request.json();
    for (const key of ["name", "domain", "system_name", "user_name"]) {
      body[key] = String(body[key] || "").trim().replace(/\s+/g, " ");
      if (body[key].length < 2 || body[key].includes("..") || /[\\/\0]/.test(body[key])) return error(request, env, `Invalid ${key}.`, 422);
    }
    if (!["session", "conversation"].includes(body.interaction_unit) || !MODELS.includes(body.model)) return error(request, env, "Invalid project configuration.", 422);
    const id = uuid(), timestamp = now();
    await env.DB.prepare(`INSERT INTO projects (id,user_id,name,domain,system_name,user_name,interaction_unit,model,status,progress,current_step,is_demo,created_at,updated_at,refinement_status,refinement_progress,refinement_step)
      VALUES (?,?,?,?,?,?,?,?, 'draft',0,'Ready for textbooks',0,?,?, 'not_started',0,'Ready after extraction')`)
      .bind(id, user.id, body.name, body.domain, body.system_name, body.user_name, body.interaction_unit, body.model, timestamp, timestamp).run();
    await addEvent(env, id, "created", "Project workspace created");
    return response(request, env, { project: (await projectFor(env, id, user.id)).public }, 201);
  }

  let match = url.pathname.match(/^\/api\/projects\/([^/]+)$/);
  if (method === "GET" && match) {
    const owned = await projectFor(env, match[1], user.id);
    if (!owned) return error(request, env, "Project not found.", 404);
    return response(request, env, { project: owned.public, manifest: await manifestFor(env, match[1]) });
  }

  match = url.pathname.match(/^\/api\/projects\/([^/]+)\/events$/);
  if (method === "GET" && match) {
    const owned = await projectFor(env, match[1], user.id);
    if (!owned) return error(request, env, "Project not found.", 404);
    const after = Math.max(0, Number(url.searchParams.get("after") || 0));
    const { results } = await env.DB.prepare("SELECT * FROM events WHERE project_id=? AND id>? ORDER BY id LIMIT 250").bind(match[1], after).all();
    return response(request, env, { events: results.map(({ payload_json, ...row }) => ({ ...row, payload: JSON.parse(payload_json || "{}") })), project: owned.public });
  }

  match = url.pathname.match(/^\/api\/projects\/([^/]+)\/books$/);
  if (method === "POST" && match) {
    const owned = await projectFor(env, match[1], user.id);
    if (!owned) return error(request, env, "Project not found.", 404);
    if (owned.raw.is_demo || owned.raw.status === "running" || owned.raw.status === "queued") return error(request, env, "Books cannot be changed right now.", 409);
    const form = await request.formData();
    const role = String(form.get("role") || "");
    if (!COMPONENTS[role]) return error(request, env, "Choose a valid role.", 422);
    const files = form.getAll("files").filter((item) => item instanceof File);
    if (!files.length) return error(request, env, "Choose at least one PDF.", 422);
    for (const file of files) {
      if (!file.name.toLowerCase().endsWith(".pdf") || (file.type && file.type !== "application/pdf")) return error(request, env, `${file.name} is not a PDF.`, 415);
      if (file.size > 24 * 1024 * 1024) return error(request, env, `${file.name} exceeds the deployed 24 MB per-file limit.`, 413);
      const id = uuid(), key = `uploads/${user.id}/${match[1]}/${role}/${id}.pdf`, timestamp = now();
      await env.FILES.put(key, await file.arrayBuffer(), { metadata: { original_name: file.name, project_id: match[1] } });
      await env.DB.prepare("INSERT INTO books (id,project_id,role,original_name,object_key,size_bytes,page_count,created_at) VALUES (?,?,?,?,?,?,0,?)")
        .bind(id, match[1], role, file.name, key, file.size, timestamp).run();
      await addEvent(env, match[1], "book_uploaded", `Added ${file.name} to ${role} sources`, { role, name: file.name, pages: 0 });
    }
    await env.DB.prepare("UPDATE projects SET status='draft',progress=0,current_step='Ready to extract',refinement_status='not_started',refinement_progress=0,refinement_step='Ready after extraction',refinement_error=NULL,refined_at=NULL,updated_at=? WHERE id=?").bind(now(), match[1]).run();
    return response(request, env, { project: (await projectFor(env, match[1], user.id)).public }, 201);
  }

  match = url.pathname.match(/^\/api\/projects\/([^/]+)\/books\/([^/]+)$/);
  if (method === "DELETE" && match) {
    const owned = await projectFor(env, match[1], user.id);
    if (!owned) return error(request, env, "Project not found.", 404);
    if (owned.raw.is_demo || ["running", "queued"].includes(owned.raw.status)) return error(request, env, "This book cannot be removed right now.", 409);
    const book = await env.DB.prepare("SELECT * FROM books WHERE id=? AND project_id=?").bind(match[2], match[1]).first();
    if (!book) return error(request, env, "Book not found.", 404);
    if (book.object_key) await env.FILES.delete(book.object_key);
    await env.DB.prepare("DELETE FROM books WHERE id=?").bind(match[2]).run();
    await addEvent(env, match[1], "book_removed", `Removed ${book.original_name}`);
    return response(request, env, { project: (await projectFor(env, match[1], user.id)).public });
  }

  match = url.pathname.match(/^\/api\/projects\/([^/]+)\/(run|refine)$/);
  if (method === "POST" && match) {
    const owned = await projectFor(env, match[1], user.id);
    if (!owned) return error(request, env, "Project not found.", 404);
    const jobType = match[2] === "run" ? "extraction" : "refinement";
    if (owned.raw.is_demo) return error(request, env, "The showcase run is already complete.", 409);
    if (jobType === "extraction" && !owned.books.length) return error(request, env, "Upload at least one system or user textbook.", 422);
    if (jobType === "refinement" && owned.raw.status !== "completed") return error(request, env, "Complete extraction before starting refinement.", 409);
    const field = jobType === "extraction" ? "status" : "refinement_status";
    if (["running", "queued"].includes(owned.raw[field])) return error(request, env, `${jobType} is already running for this project.`, 409);
    try {
      await dispatchJob(env, match[1], jobType, url.origin);
      if (jobType === "extraction") await env.DB.prepare("UPDATE projects SET status='queued',progress=1,current_step='Waiting for a TOPAS runner',error=NULL,updated_at=? WHERE id=?").bind(now(), match[1]).run();
      else await env.DB.prepare("UPDATE projects SET refinement_status='running',refinement_progress=1,refinement_step='Waiting for a TOPAS runner',refinement_error=NULL,updated_at=? WHERE id=?").bind(now(), match[1]).run();
      await addEvent(env, match[1], `${jobType}_queued`, `${jobType[0].toUpperCase()}${jobType.slice(1)} queued`);
      return response(request, env, { accepted: true, project_id: match[1] }, 202);
    } catch (caught) { return error(request, env, String(caught.message || caught), 503); }
  }

  match = url.pathname.match(/^\/api\/projects\/([^/]+)\/summaries\/(system|user)\/(\d+)$/);
  if (method === "GET" && match) {
    if (!(await projectFor(env, match[1], user.id))) return error(request, env, "Project not found.", 404);
    const data = await artifactJson(env, match[1], { phase: "extraction", role: match[2], kind: "summary", book_index: Number(match[3]) });
    return data === null ? error(request, env, "That summary is not ready yet.", 404) : response(request, env, { data });
  }

  match = url.pathname.match(/^\/api\/projects\/([^/]+)\/components\/(system|user)\/([^/]+)$/);
  if ((method === "GET" || method === "PUT") && match) {
    const owned = await projectFor(env, match[1], user.id);
    if (!owned) return error(request, env, "Project not found.", 404);
    if (!COMPONENTS[match[2]].includes(match[3])) return error(request, env, "Component not found.", 404);
    const body = method === "PUT" ? await request.json() : null;
    const phase = method === "PUT" ? body.phase : (url.searchParams.get("phase") || "extraction");
    if (!["extraction", "refinement"].includes(phase)) return error(request, env, "Invalid component phase.", 422);
    if (method === "GET") {
      const data = await artifactJson(env, match[1], { phase, role: match[2], kind: "component", component: match[3] });
      return data === null ? error(request, env, "That component is not ready yet.", 404) : response(request, env, { data });
    }
    if (["running", "queued"].includes(owned.raw.status) || owned.raw.refinement_status === "running") return error(request, env, "Wait for the active pipeline step to finish before editing.", 409);
    const existing = await env.DB.prepare("SELECT object_key FROM artifacts WHERE project_id=? AND phase=? AND role=? AND kind='component' AND component=?").bind(match[1], phase, match[2], match[3]).first();
    if (!existing) return error(request, env, "That component is not ready yet.", 404);
    await env.FILES.put(existing.object_key, JSON.stringify(body.data));
    if (phase === "extraction" && owned.raw.refinement_status === "completed") await env.DB.prepare("UPDATE projects SET refinement_status='stale',refinement_step='Extracted components changed',updated_at=? WHERE id=?").bind(now(), match[1]).run();
    await addEvent(env, match[1], "component_edited", `Updated ${match[3].replaceAll("_", " ")}`, { role: match[2], component: match[3], phase });
    return response(request, env, { saved: true, data: body.data, project: (await projectFor(env, match[1], user.id)).public });
  }

  match = url.pathname.match(/^\/api\/projects\/([^/]+)\/refinement\/(summary|details)$/);
  if (method === "GET" && match) {
    if (!(await projectFor(env, match[1], user.id))) return error(request, env, "Project not found.", 404);
    const data = await artifactJson(env, match[1], { phase: "refinement", kind: "report", report: match[2] });
    return data === null ? error(request, env, "That refinement report is not ready yet.", 404) : response(request, env, { data });
  }
  return error(request, env, "Not found.", 404);
}

async function handleInternal(request, env, url) {
  if ((request.headers.get("X-TOPAS-JOB-TOKEN") || "") !== env.JOB_TOKEN) return error(request, env, "Invalid job credential.", 401);
  let match = url.pathname.match(/^\/internal\/projects\/([^/]+)\/job$/);
  if (request.method === "GET" && match) {
    const project = await env.DB.prepare("SELECT * FROM projects WHERE id=?").bind(match[1]).first();
    if (!project) return error(request, env, "Project not found.", 404);
    const { results: books } = await env.DB.prepare("SELECT * FROM books WHERE project_id=? ORDER BY role,created_at").bind(match[1]).all();
    const { results: artifacts } = await env.DB.prepare("SELECT * FROM artifacts WHERE project_id=?").bind(match[1]).all();
    return response(request, env, { project, books, artifacts });
  }
  match = url.pathname.match(/^\/internal\/books\/([^/]+)$/);
  if (request.method === "GET" && match) {
    const book = await env.DB.prepare("SELECT * FROM books WHERE id=?").bind(match[1]).first();
    if (!book || !book.object_key) return error(request, env, "Book not found.", 404);
    const object = await env.FILES.get(book.object_key, "arrayBuffer");
    if (object === null) return error(request, env, "Book object not found.", 404);
    return new Response(object, { headers: { "Content-Type": "application/pdf" } });
  }
  match = url.pathname.match(/^\/internal\/artifacts\/([^/]+)$/);
  if (request.method === "GET" && match) {
    const artifact = await env.DB.prepare("SELECT * FROM artifacts WHERE id=? AND project_id=?").bind(url.searchParams.get("artifact_id"), match[1]).first();
    if (!artifact) return error(request, env, "Artifact not found.", 404);
    const object = await env.FILES.get(artifact.object_key, "text");
    return object !== null ? new Response(object, { headers: { "Content-Type": "application/json" } }) : error(request, env, "Artifact object not found.", 404);
  }
  if (request.method === "PUT" && match) {
    const values = {
      project_id: match[1], phase: url.searchParams.get("phase") || "extraction", role: url.searchParams.get("role") || "",
      kind: url.searchParams.get("kind") || "component", book_index: Number(url.searchParams.get("book_index") || -1),
      component: url.searchParams.get("component") || "", report: url.searchParams.get("report") || "", chapter_count: Number(url.searchParams.get("chapter_count") || 0),
    };
    await addArtifact(env, values, await request.text());
    return response(request, env, { saved: true });
  }
  match = url.pathname.match(/^\/internal\/projects\/([^/]+)\/progress$/);
  if (request.method === "POST" && match) {
    const body = await request.json(), timestamp = now();
    if (body.job_type === "refinement") {
      await env.DB.prepare("UPDATE projects SET refinement_status=?,refinement_progress=?,refinement_step=?,refinement_error=?,updated_at=? WHERE id=?")
        .bind(body.status || "running", Number(body.progress || 0), body.step || "Refining", body.error || null, timestamp, match[1]).run();
    } else {
      await env.DB.prepare("UPDATE projects SET status=?,progress=?,current_step=?,error=?,updated_at=?,completed_at=? WHERE id=?")
        .bind(body.status || "running", Number(body.progress || 0), body.step || "Extracting", body.error || null, timestamp, body.status === "completed" ? timestamp : null, match[1]).run();
    }
    if (body.kind && body.message) await addEvent(env, match[1], body.kind, body.message, body.payload || {});
    return response(request, env, { saved: true });
  }
  return error(request, env, "Not found.", 404);
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (request.method === "OPTIONS") return response(request, env, null, 204);
    try {
      if (url.pathname.startsWith("/internal/")) return await handleInternal(request, env, url);
      return await handlePublic(request, env, url);
    } catch (caught) {
      console.error(caught);
      return error(request, env, "The service could not complete that request.", 500);
    }
  },
};
