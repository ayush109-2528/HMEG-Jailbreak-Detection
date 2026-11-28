export type Paper = {
 id:string; title:string; shortTitle:string; authors:string[]; year:number|null; venue:string|null; type:string;
 doi:string|null; arxiv:string|null; url:string; pdfUrl:string|null; publicPdf:string|null; source:string; pages:number;
 category:string; tags:string[]; components:string[]; languages:string[]; abstract:string|null; abstractPermission:string;
 summary:string; objective:string; methodology:string; models:string[]; features:string[]; datasets:string[];
 results:string; metrics:string[]; limitations:string; opportunity:string; findings:string; scopeNote:string;
 direct:string; indirect:string; ensemble:string; calibration:string; routing:string;
 steps:string[]; evidence:Record<string,number[]>; references:{title:string;page:number;url?:string}[];
 license:string; licenseUrl:string|null; versions:{label:string;url:string}[];
 figures:{caption:string;page:number;image?:string}[];
};
export type Relationship={id:string;source:string;target:string;type:'similarity'|'citation'|'relevance';reason:string;pages?:number[]};
export type Library={papers:Paper[];relationships:Relationship[];manifest:{[key:string]:any}};
export const missing='Not specified in the source.';
export const components:Record<string,string>={word:'Word Expert',format:'Format Expert',meaning:'Meaning Expert',pattern:'Pattern Expert',instruction:'Instruction Expert',router:'Confidence Router',meta:'Meta-Classifier',evaluation:'Evaluation Framework'};
export const categories:Record<string,string>={classical:'Classical Machine Learning',semantic:'Semantic & Embedding Detection',ensemble:'Hybrid & Ensemble Guardrails',structural:'Structural & Formatting Analysis',multilingual:'Multilingual Methodology · outside scope',evaluation:'Benchmarks & Evaluation',llm:'LLM-Based Guardrails',attacks:'Attack & Vulnerability Research'};
export const colors:Record<string,string>={classical:'#45b5c8',semantic:'#9d8be5',ensemble:'#39b7a0',structural:'#d5a64f',multilingual:'#b78cc5',evaluation:'#6396df',llm:'#d882a8',attacks:'#d68169'};
export function safeUrl(url:string|null|undefined){return !!url && (/^https:\/\//.test(url)||/^\/research\/[a-zA-Z0-9/_\-.]+$/.test(url));}
export function bibtex(p:Paper){const clean=(s:string)=>s.replace(/[{}\\]/g,'');return `@${p.type.includes('Conference')?'inproceedings':'article'}{${p.id},\n  title = {${clean(p.title)}},\n  author = {${p.authors.map(a=>clean(a)).join(' and ')}},${p.year?`\n  year = {${p.year}},`:''}${p.venue?`\n  ${p.type.includes('Conference')?'booktitle':'journal'} = {${clean(p.venue)}},`:''}${p.doi?`\n  doi = {${p.doi}},`:''}\n  url = {${p.url}}\n}`;}
function authorName(a:string){const s=a.split(' ');const last=s.pop();return `${last}, ${s.map(x=>x[0]+'.').join(' ')}`;}
export function citation(p:Paper,style:string){if(style==='BibTeX')return bibtex(p);if(style==='APA')return `${p.authors.map(authorName).join(', ')} (${p.year||'n.d.'}). ${p.title}. ${p.venue||p.type}. ${p.doi?'https://doi.org/'+p.doi:p.url}`;return `${p.authors.join(', ')}, “${p.title},” ${p.venue||p.type}${p.year?', '+p.year:''}. ${p.doi?'doi: '+p.doi:p.url}`;}
export function exportBib(papers:Paper[]){const u=URL.createObjectURL(new Blob([papers.map(bibtex).join('\n\n')],{type:'application/x-bibtex'}));const a=document.createElement('a');a.href=u;a.download='hmeG-research-library.bib';a.click();URL.revokeObjectURL(u);}
