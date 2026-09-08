import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  root: path.resolve(__dirname, 'app/interface'),
  plugins: [
      react({
    include: [/\.[jt]sx?$/]
  })],
  resolve: {
    alias: { '@': path.resolve(__dirname, 'app/interface/src') },
  },
  server: {
    proxy: { '/api': 'http://localhost:8000' },
  },
  build: {
    outDir: path.resolve(__dirname, 'dist'),
    emptyOutDir: true,
  },
});
