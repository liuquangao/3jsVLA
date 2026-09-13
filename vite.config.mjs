import { defineConfig } from "vite";
import { datasetServer } from "./tools/dataset-server.mjs";

export default defineConfig({
  plugins: [datasetServer()],
});
