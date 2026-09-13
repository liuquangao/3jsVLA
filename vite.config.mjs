import { defineConfig } from "vite";
import { datasetServer } from "./tools/dataset-server.mjs";
import { policyServer } from "./tools/policy-server.mjs";

export default defineConfig({
  plugins: [datasetServer(), policyServer()],
});
