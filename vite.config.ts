import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [sveltekit()],
	server: {
		watch: {
			// Prevent runaway file watching (and HMR reload storms) when Python deps are present.
			ignored: ['**/.venv/**', '**/.data/**', '**/node_modules/**']
		}
	}
});
