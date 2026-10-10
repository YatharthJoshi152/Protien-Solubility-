import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/meta': 'http://127.0.0.1:8000',
      '/features': 'http://127.0.0.1:8000',
      '/predict': 'http://127.0.0.1:8000',
      '/predict_batch': 'http://127.0.0.1:8000',
      '/health': 'http://127.0.0.1:8000'
    }
  }
})
