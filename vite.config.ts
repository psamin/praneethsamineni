import { defineConfig } from "vite";
import preact from "@preact/preset-vite";

// Preact (via preact/compat) runs the React components unchanged at a fraction
// of React's size. `base` is "/" for a custom domain or <user>.github.io; for a
// project page at <user>.github.io/<repo>/, build with BASE=/<repo>/ npm run build.
export default defineConfig({
  base: process.env.BASE ?? "/",
  plugins: [preact()],
});
