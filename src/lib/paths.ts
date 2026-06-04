import { base } from '$app/paths';

export function appPath(path: string): string {
  if (!path.startsWith('/')) return `${base}/${path}`;
  return `${base}${path}`;
}

export function assetPath(path: string): string {
  if (/^https?:\/\//.test(path)) return path;
  return appPath(path);
}
