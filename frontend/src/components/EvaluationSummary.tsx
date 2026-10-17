import { useResource } from '../hooks/useResource';
import { Panel, State, number } from './UI';
export function EvaluationSummary() {
  const manifest = useResource<any>('/reproducibility');
  const results = useResource<any[]>('/experiments');
  const m = manifest.data;
  const rows = (results.data || []).filter(x => x.model_version === m?.model?.model_version && ['hmeg', 'expert'].includes(x.model_type));
  return <Panel title="Which test dataset produced these results?">
    <State resource={manifest}/><State resource={results}/>
    <p><strong>{m?.dataset?.filename || 'Dataset identity unavailable'}</strong> - held-out test partition, with {number(m?.training?.used_split_counts?.test)} prompts.</p>
    <p>Experts trained on {number(m?.training?.used_split_counts?.train)} rows. Learned fusion fitted on {number(m?.training?.used_split_counts?.validation)} separate validation rows. Test rows were excluded from both fitting stages. Shared parent groups and normalized prompt hashes are excluded across partitions.</p>
    <p>Active model: <code>{m?.model?.model_version || 'Unavailable'}</code></p>
    {rows.length ? <div className="table-wrap"><table><thead><tr><th>Model</th><th>Test prompts</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead><tbody>{rows.map(x => <tr key={x.experiment_id}><td>{x.name}</td><td>{number(x.classification_sample_count ?? x.training?.used_split_counts?.test)}</td>{['accuracy','precision','recall','f1'].map(key => <td key={key}>{typeof x.metrics?.[key] === 'number' ? (x.metrics[key] * 100).toFixed(2) + '%' : 'Unavailable'}</td>)}</tr>)}</tbody></table></div> : <p>No saved evaluation matches the active model version.</p>}
    <details><summary>Dataset sources and identity</summary><ul>{m?.training?.sources?.map((source:string) => <li key={source}>{source}</li>)}</ul><p className="break">SHA-256: {m?.training?.dataset_sha256 || 'Unavailable'}</p></details>
    <div className="notice">These are measured results on this imported research corpus, which is predominantly synthetic. They do not establish accuracy on unseen sources or equivalence to published guardrail benchmarks. See Experiments for confusion matrices and latency sample sizes.</div>
  </Panel>;
}
