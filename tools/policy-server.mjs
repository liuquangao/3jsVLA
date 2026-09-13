import { readdir, stat } from "node:fs/promises";
import { spawn } from "node:child_process";
import path from "node:path";

/**
 * Keeps one `tools/policy_server.py` child alive and pipes newline-delimited JSON to it.
 * Loading a checkpoint costs seconds; an action chunk costs a fraction of one, so the
 * process is reused across steps and only replaced when a different model is selected.
 */
class PolicyProcess {
  constructor(child, model) {
    this.child = child;
    this.model = model;
    this.pending = [];
    this.buffer = "";
    this.stderr = "";
    child.stdout.setEncoding("utf8");
    child.stdout.on("data", (chunk) => {
      this.buffer += chunk;
      let newline;
      while ((newline = this.buffer.indexOf("\n")) >= 0) {
        const line = this.buffer.slice(0, newline).trim();
        this.buffer = this.buffer.slice(newline + 1);
        if (!line) continue;
        const waiter = this.pending.shift();
        if (!waiter) continue;
        try {
          waiter.resolve(JSON.parse(line));
        } catch {
          waiter.reject(new Error(`Policy returned malformed JSON: ${line.slice(0, 200)}`));
        }
      }
    });
    child.stderr.setEncoding("utf8");
    // Python tracebacks and library warnings land here; keep only enough to explain a failure.
    child.stderr.on("data", (chunk) => { this.stderr = (this.stderr + chunk).slice(-4000); });
    const fail = (error) => {
      for (const waiter of this.pending.splice(0)) waiter.reject(error);
    };
    child.on("error", fail);
    child.on("close", (code) => {
      this.closed = true;
      fail(new Error((this.stderr || `policy exited with code ${code}`).trim()));
    });
  }

  request(payload) {
    if (this.closed) return Promise.reject(new Error("Policy process has exited"));
    return new Promise((resolve, reject) => {
      this.pending.push({ resolve, reject });
      this.child.stdin.write(`${JSON.stringify(payload)}\n`);
    });
  }

  /** The ready line is unsolicited, so claim it with an empty request slot. */
  handshake() {
    return new Promise((resolve, reject) => this.pending.push({ resolve, reject }));
  }

  stop() {
    this.closed = true;
    // Stopping is a deliberate act, so settle in-flight requests as stopped rather than
    // failing them; the collector would otherwise log a server error for a normal click.
    for (const waiter of this.pending.splice(0)) waiter.resolve({ stopped: true });
    this.child.kill();
  }
}

/** Local-only policy runner. Checkpoints are addressed by name inside `checkpoints/`. */
export function policyServer() {
  return {
    name: "local-policy-runner",
    configureServer(server) {
      const projectRoot = server.config.root;
      const checkpointRoot = path.join(projectRoot, "checkpoints");
      let policy = null;

      server.httpServer?.on("close", () => policy?.stop());

      server.middlewares.use(async (req, res, next) => {
        const route = req.url?.split("?")[0];
        if (!route?.startsWith("/__policy/")) return next();
        const reply = (status, payload) => {
          res.statusCode = status;
          res.setHeader("Content-Type", "application/json");
          res.end(JSON.stringify(payload));
        };
        if (req.method !== "POST") return reply(405, { error: "Expected a JSON POST" });

        try {
          const chunks = [];
          for await (const chunk of req) chunks.push(chunk);
          const body = chunks.length ? Buffer.concat(chunks).toString("utf8") : "{}";
          const payload = JSON.parse(body);

          if (route === "/__policy/list") {
            let entries = [];
            try {
              entries = (await readdir(checkpointRoot)).filter((name) => name.endsWith(".pt"));
            } catch {
              return reply(200, { models: [], loaded: policy?.model ?? null });
            }
            const models = await Promise.all(entries.map(async (name) => {
              const info = await stat(path.join(checkpointRoot, name));
              return { name, bytes: info.size, modified: info.mtime.toISOString() };
            }));
            models.sort((a, b) => b.modified.localeCompare(a.modified));
            return reply(200, { models, loaded: policy?.model ?? null });
          }

          if (route === "/__policy/load") {
            const model = path.basename(String(payload.model ?? ""));
            if (!model.endsWith(".pt")) return reply(400, { error: "Expected a .pt checkpoint name" });
            if (policy?.model === model && !policy.closed) {
              return reply(200, { ready: true, model, reused: true });
            }
            policy?.stop();
            policy = null;
            const venvPython = path.join(
              projectRoot,
              ".venv",
              process.platform === "win32" ? "Scripts/python.exe" : "bin/python",
            );
            const python = process.env.PYTHON || venvPython;
            const child = spawn(
              python,
              [path.join(projectRoot, "tools", "policy_server.py"), path.join(checkpointRoot, model)],
              { cwd: projectRoot, windowsHide: true },
            );
            const started = new PolicyProcess(child, model);
            const ready = await started.handshake();
            if (ready.error) {
              started.stop();
              return reply(500, { error: ready.error });
            }
            policy = started;
            return reply(200, { ...ready, model });
          }

          if (route === "/__policy/act") {
            if (!policy || policy.closed) return reply(400, { error: "No policy is loaded" });
            const result = await policy.request({
              images: payload.images,
              state: payload.state,
              task: payload.task,
            });
            if (result.stopped) return reply(200, { actions: [] });
            if (result.error) return reply(500, { error: result.error });
            return reply(200, result);
          }

          if (route === "/__policy/stop") {
            policy?.stop();
            policy = null;
            return reply(200, { stopped: true });
          }

          return reply(404, { error: "Unknown policy endpoint" });
        } catch (error) {
          const detail = error instanceof Error ? error.message : String(error);
          const reason = detail.includes("No module named 'torch'")
            ? "PyTorch is not installed in .venv; install tools/requirements.txt"
            : detail;
          reply(500, { error: reason });
        }
      });
    },
  };
}
