import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// host:true expone el dev server en todas las interfaces para poder abrir la UI
// desde otra maquina de la red local.
export default defineConfig({
  plugins: [react()],
  server: { host: true, port: 5173, strictPort: false },
  preview: { host: true, port: 4173 },
});
