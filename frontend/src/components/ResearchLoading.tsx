import { useEffect, useState } from 'react';
import { Database, Radio } from 'lucide-react';

/** Render only while an actual resource request is pending. */
export function ResearchLoading({ title = 'Loading research data…', compact = false }: { title?: string; compact?: boolean }) {
  const [slow, setSlow] = useState(false);
  useEffect(() => {
    const timer = window.setTimeout(() => setSlow(true), 8000);
    return () => window.clearTimeout(timer);
  }, []);
  return <div className={'research-loader' + (compact ? ' compact' : '')} role="status">
    <span className="signal-path" aria-hidden="true"><Database size={18}/><i/><Radio size={18}/></span>
    <div>{!compact && <small>RESEARCH OBSERVATORY</small>}<strong>{title}</strong>
      {slow && <span className="signal-note">Still waiting for a response. You can explore another page.</span>}
    </div>
  </div>;
}
