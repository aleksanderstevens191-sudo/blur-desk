import { defineConfig } from "vite";

export default defineConfig({
  build: {
    target: "es2022",
    rollupOptions: {
      input: { demo: "index.html", check: "check.html", bench: "bench.html" },
    },
  },
});
