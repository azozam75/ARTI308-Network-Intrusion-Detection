import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Proxy API calls to the FastAPI backend during development so the
// frontend can use relative `/api/*` URLs and ship unchanged to prod.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})
