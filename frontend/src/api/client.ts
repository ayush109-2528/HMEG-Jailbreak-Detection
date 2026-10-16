export const API = (import.meta.env.VITE_API_URL || '').replace(/\/$/, '');
export async function api<T = any>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(`${API}/api${path}`, options);
  const body = await response.json().catch(() => null);
  if (!response.ok) throw new Error(typeof body?.detail === 'string' ? body.detail : `Request failed (${response.status}). Check the API connection.`);
  return body;
}
export function download(name: string, data: unknown) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' }));
  const a = document.createElement('a'); a.href = url; a.download = name; a.click(); URL.revokeObjectURL(url);
}
