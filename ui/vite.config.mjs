import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
export default defineConfig(({ mode }) => ({
  plugins: [react({ jsxRuntime: 'classic' })],
  esbuild: { loader: 'jsx', include: /src\/.*\.jsx?$/ },
  optimizeDeps: { esbuildOptions: { loader: { '.js': 'jsx' } } },
  define: { 'process.env': JSON.stringify({ ...loadEnv(mode, process.cwd(), 'REACT_APP_'), NODE_ENV: mode, PUBLIC_URL: '' }) },
  build: { outDir: 'build' },
  test: { environment: 'jsdom', globals: true, setupFiles: ['./src/setupTests.js'] },
}));
