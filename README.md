# ai-arch
Mock-up for AI Agents architecture — secure and cost-optimized.

## Layout

```
ai-arch/
  .env                          # secrets (gitignored)
  ai-architect-env/             # Python venv only (gitignored)
  invoice_agent.py              # Gemini PDF → JSON extraction
  invoice_extraction_agent.py   # tool-calling extraction agent
  rag/                          # Mansfield invoice RAG
  apps/                         # LangGraph + Microsoft Agent Framework apps
  demos/                        # small Gemini / Azure demos
  data/                         # local PDFs / outputs
  rag_store/                    # Chroma index (gitignored)
```

## Setup

```bash
cd ai-arch
source ai-architect-env/bin/activate
# Ensure GEMINI_API_KEY is set in .env at repo root
```

## Invoice extraction

```bash
# Put PDFs in data/input_pdfs/
python invoice_extraction_agent.py
python invoice_extraction_agent.py --direct
```

## RAG (Mansfield invoice JSON)

```bash
python -m rag.ingest
python -m rag.ask "What is the total on invoice 6893?"
python -m rag.ask --vendor "best oil" --show-sources "diesel"
python -m rag.agent "List Holston invoices and one sample total"
```

## Orchestration frameworks

### LangGraph
```bash
python -m apps.langgraph_invoice_app "How many Best Oil invoices are indexed?"
```

### Microsoft Agent Framework (SequentialBuilder)
```bash
python -m apps.maf_sequential_rag "Summarize Best Oil diesel invoices"
```

> Free-tier Gemini has low RPM/RPD; if you see `429`, wait and retry (or enable billing).

### Demos
```bash
python demos/Hi-AI.py
python demos/code_execution.py
python demos/workflow.py   # FastAPI + MAF (Azure)
```

## Security notes

- Never commit `.env` or API keys.
- Prefer env vars / managed identity in production (`demos/workflow.py` pattern).
