import { useEffect, useState } from 'react';
import { api } from '../api/client';
export function useResource<T = any>(path: string) {
  const [data, setData] = useState<T | null>(null), [error, setError] = useState(''), [loading, setLoading] = useState(true);
  const [version, setVersion] = useState(0);
  useEffect(() => {
    const controller = new AbortController(); setLoading(true); setError('');
    api<T>(path, { signal: controller.signal }).then(setData).catch(e => { if (!controller.signal.aborted) setError(e.message); }).finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [path, version]);
  return { data, error, loading, refresh: () => setVersion(x => x + 1) };
}
