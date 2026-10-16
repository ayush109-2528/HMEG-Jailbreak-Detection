import { useState } from 'react';
import { ReactFlow, Background, Controls, MarkerType } from '@xyflow/react';
import '@xyflow/react/dist/style.css';
export function Architecture({ result, active = '', mode = 'HMEG' }: { result?: any; active?: string; mode?: string }) {
  const [selected, setSelected] = useState('Select a stage to inspect its role.');
  const specs: [string, string, number, number][] = mode === 'Single' ? [['input', 'User prompt', 180, 0], ['word', 'Single guardrail', 180, 100], ['decision', 'Final classification', 180, 200]] : [
    ['input', 'User prompt', 240, 0], ['preprocessing', 'Preprocessing', 240, 75], ['word', '01 · Word Expert', 100, 160], ['format', '02 · Format Expert', 380, 160], ['router', mode === 'Flat' ? 'Run every expert' : 'Confidence router', 240, 250], ['meaning', '03 · Meaning', 30, 350], ['pattern', '04 · Pattern', 240, 350], ['instruction', '05 · Instruction', 450, 350], ['meta', 'Learned meta-classifier', 240, 445], ['decision', 'Benign / Jailbreak', 240, 540]];
  const links = mode === 'Single' ? [['input', 'word'], ['word', 'decision']] : [['input', 'preprocessing'], ['preprocessing', 'word'], ['preprocessing', 'format'], ['word', 'router'], ['format', 'router'], ...['meaning', 'pattern', 'instruction'].flatMap(key => [['router', key], [key, 'meta']]), ['meta', 'decision'], ...(mode === 'Flat' ? [] : [['router', 'decision']])];
  const nodes = specs.map(([id, name, x, y]) => {
    const expert = result?.experts?.[id], skipped = expert && !expert.executed;
    return { id, position: { x, y }, data: { label: <><b>{name}</b>{expert && <small>{expert.executed ? `${(expert.score * 100).toFixed(1)}% · ${expert.latency_ms.toFixed(2)} ms` : expert.status}</small>}</> }, style: { width: 180, background: active === id ? '#154a48' : '#182332', color: skipped ? '#81909f' : '#e5eef5', border: `1px solid ${active === id ? '#48d9c2' : '#354354'}`, borderRadius: 10, opacity: skipped ? .55 : 1 } };
  });
  const edges = links.map(([source, target]) => ({ id: source + target, source, target, animated: active === source, markerEnd: { type: MarkerType.ArrowClosed }, style: { stroke: active === source ? '#48d9c2' : '#506074' }, label: source === 'router' && target === 'decision' ? 'Confident exit' : undefined }));
  return <><div className="flow"><ReactFlow nodes={nodes} edges={edges} fitView nodesDraggable={false} nodesConnectable={false} onNodeClick={(_, node) => setSelected(`${node.id === 'router' ? 'Scores outside the uncertainty interval exit at Layer 1. Only uncertain inputs enter Layer 2.' : node.id === 'meta' ? 'A classifier fitted on validation-partition expert scores produces the final estimate.' : `${node.id}: ${result?.experts?.[node.id]?.status || 'pipeline stage'}.`} ${result?.experts?.[node.id]?.executed ? 'This expert actually executed for this prompt.' : ''}`)}><Background color="#2c3948" gap={24}/><Controls showInteractive={false}/></ReactFlow></div><p className="caption">{selected}</p></>;
}
