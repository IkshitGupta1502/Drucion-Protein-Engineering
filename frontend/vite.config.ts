import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      output: {
        // Plotly is by far the largest dependency; splitting it out lets the
        // app shell render while the charting bundle is still arriving.
        manualChunks: {
          plotly: ['plotly.js-dist-min'],
          react: ['react', 'react-dom'],
        },
      },
    },
    chunkSizeWarningLimit: 6000,
  },
});
