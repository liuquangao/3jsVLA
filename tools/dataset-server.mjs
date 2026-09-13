import { mkdir, mkdtemp, writeFile } from "node:fs/promises";
import path from "node:path";

/** Local-only dataset writer. No client-supplied filesystem paths are accepted. */
export function datasetServer() {
  return {
    name: "local-dataset-writer",
    configureServer(server) {
      const dataRoot = path.join(server.config.root, "data");
      const runs = new Map();

      server.middlewares.use(async (req, res, next) => {
        const route = req.url?.split("?")[0];
        if (!route?.startsWith("/__dataset/")) return next();
        const reply = (status, payload) => {
          res.statusCode = status;
          res.setHeader("Content-Type", "application/json");
          res.end(JSON.stringify(payload));
        };
        const remote = req.socket.remoteAddress;
        if (!["127.0.0.1", "::1", "::ffff:127.0.0.1"].includes(remote)) {
          return reply(403, { error: "Dataset writes require a local connection" });
        }
        if (req.method !== "POST" || req.headers["content-type"] !== "application/json") {
          return reply(405, { error: "Expected a JSON POST" });
        }
        if (!req.headers.origin || req.headers.origin !== `http://${req.headers.host}`) {
          return reply(403, { error: "Expected a same-origin request" });
        }

        try {
          const chunks = [];
          let size = 0;
          for await (const chunk of req) {
            size += chunk.length;
            if (size > 64 * 1024 * 1024) {
              return reply(413, { error: "Episode exceeds the 64 MB request limit" });
            }
            chunks.push(chunk);
          }
          const payload = JSON.parse(Buffer.concat(chunks).toString("utf8"));
          if (route === "/__dataset/start") {
            await mkdir(dataRoot, { recursive: true });
            const directory = await mkdtemp(path.join(dataRoot, `${new Date().toISOString().replace(/[:.]/g, "-")}-`));
            const run = path.basename(directory);
            await mkdir(path.join(directory, "episodes"));
            runs.set(run, directory);
            return reply(200, { run, name: `data/${run}` });
          }

          const directory = runs.get(payload.run);
          if (!directory) return reply(400, { error: "Unknown dataset run; start a new generation" });
          if (route === "/__dataset/meta") {
            await writeFile(path.join(directory, "meta.json"), JSON.stringify(payload.meta, null, 2));
            return reply(200, { saved: true });
          }
          if (route !== "/__dataset/episode") return reply(404, { error: "Unknown dataset endpoint" });
          const { index, episode } = payload;
          if (!Number.isSafeInteger(index) || index < 0 || index > 1999 || !Array.isArray(episode?.frames)) {
            return reply(400, { error: "Invalid episode" });
          }
          for (const frame of episode.frames) {
            if (typeof frame.observation?.image !== "string" || !frame.observation.image.startsWith("data:image/jpeg;base64,")) {
              return reply(400, { error: "Expected JPEG observation frames" });
            }
          }
          const episodeDir = path.join(directory, "episodes", `episode_${String(index).padStart(5, "0")}`);
          await mkdir(episodeDir);
          await mkdir(path.join(episodeDir, "frames"));
          const frames = [];
          for (const [frameIndex, frame] of episode.frames.entries()) {
            const { image, ...observation } = frame.observation;
            const file = `${String(frameIndex).padStart(6, "0")}.jpg`;
            await writeFile(path.join(episodeDir, "frames", file), Buffer.from(image.slice(image.indexOf(",") + 1), "base64"));
            frames.push({ ...frame, observation: { ...observation, image_path: `frames/${file}` } });
          }
          await writeFile(path.join(episodeDir, "episode.json"), JSON.stringify({ ...episode, frames }));
          reply(200, { saved: true });
        } catch (error) {
          reply(500, { error: error instanceof Error ? error.message : String(error) });
        }
      });
    },
  };
}
