import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  // Relative asset paths: the static Space serves the build from its root.
  base: './',
  server: {
    proxy: {
      '/api': 'http://localhost:7860',
    },
  },
  build: {
    outDir: 'dist',
  },
})
