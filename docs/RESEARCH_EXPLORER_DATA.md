# Research Explorer data workflow

The Explorer is a static, reviewed literature knowledge base. It does not write MongoDB records or change model artifacts. HMEG research scope is English-language jailbreak and prompt-injection detection. Papers covering other languages are methodological background only.

## Source files and permissions

Place PDFs in `research_sources/papers/`. That directory and `research_sources/extracted/` are ignored by Git. Keep restricted PDFs out of `frontend/public`. The public reader can open a device-local copy without uploading it. Licensed public PDF copies require explicit CC BY, CC BY-SA, CC0 or public-domain status, attribution, and evidence. Open accessibility alone is not permission to redistribute.

The current source inventory is `research_sources/inventory.json`. Additional downloads can be recorded in `research_sources/additional_downloads.json` (or `additional_inventory.json`). A damaged original is retained as a source inventory item; a replacement must be identified as such. Files containing several papers must be manually partitioned: the extractor reports candidate abstract pages but never equates one file with one paper automatically.

## Offline ingestion

From `backend`:

```powershell
uv pip install --python .venv/Scripts/python.exe -r requirements-research.txt
.venv/Scripts/python.exe -m research.ingest_papers
```

The report records SHA-256 hashes, byte duplicates, failed PDFs, page text and candidate boundaries. PDF text is untrusted research data. No instructions in a document are executed. Extraction does not publish papers, infer complete authorship, or authorize redistribution.

## Curate and build

Review original pages and add records to `research_sources/curation/base.json` or `additional.json`. Use the TypeScript `Paper` schema in `frontend/src/components/research/types.ts`. Record actual PDF page indexes (starting at 1), not inferred printed page numbers. Missing fields remain empty/null and render as “Not specified in the source.” Separate author findings and limitations from HMEG opportunities. Never copy an inaccessible abstract or guess result values.

```powershell
.venv/Scripts/python.exe -m research.build_research_graph
.venv/Scripts/python.exe -m research.validate_research_data
.venv/Scripts/python.exe -m pytest tests/test_research_library.py
```

The build generates `backend/data/research/{papers,relationships,expert_connections,timeline,research_manifest}.json` and the frontend `src/data/research/library.json`. No API or new MongoDB collection is needed. Backend generated JSON is an audit mirror; the frontend reads its bundled artifact directly, so a frontend rebuild is required after editing curated records.

## Relationship calculation

- Canonical records match normalized title, DOI, or arXiv identifier without version suffix. Distinct versions are preserved as links under the first reviewed canonical record; their claims are not silently merged.
- A methodological edge requires at least two exact shared curated method tags. It is a qualitative connection, not a calibrated similarity score. Tags must be justified by the methodology evidence.
- A directed citation requires an explicit reference title matching another canonical record and a verified page in the citing source. Similarity never creates citation edges.
- An HMEG edge is a researcher interpretation supported by the recorded opportunity and component mapping, not evidence of full integration or superior performance.
- All paper nodes have equal size. No citation counts are invented. Initial force-layout positions are deterministic; graph distances are not scholarly similarity values.

The named excluded paper is filtered before publication and is not added to the public literature map. A new source cannot bypass license or evidence validation just by being added to the ingestion folder.

Unretrieved paper links are listed in `research_sources/source_gaps.json`. They are not paper nodes and their reported claims are not invented. A recovered damaged original counts as one logical source in the manifest, with the damaged-original count reported separately.
