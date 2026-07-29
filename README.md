# ai-arch
Mock-up for AI Agents architecture — secure and cost-optimized.

## Stack highlights

| Layer | Tech |
|---|---|
| Models | Google Gemini (`google-genai`) |
| Invoice extraction agent | Gemini tools / AFC |
| RAG | Chroma + Gemini embeddings over Mansfield LTL JSON |
| Orchestration | **LangGraph**, **Microsoft Agent Framework** (SequentialBuilder) |

## Setup

```bash
cd ai-architect-env
source bin/activate
# GEMINI_API_KEY must be set in .env
```

## Invoice extraction

```bash
# Put PDFs in input_pdfs/
python invoice_extraction_agent.py
python invoice_extraction_agent.py --direct
```

## RAG (Mansfield invoice JSON)

Corpus default: `RAG_DATA_DIR` in `.env`  
(e.g. Mansfield `Base prompt/output_files`)

```bash
python -m rag.ingest
python -m rag.ask "What is the total on invoice 6893?"
python -m rag.ask --vendor "best oil" --show-sources "diesel"
python -m rag.agent "List Holston invoices and one sample total"
```

## Orchestration frameworks

### LangGraph (graph routing)
Classifies intent → calls the right RAG tool node → synthesizes an answer.

```bash
python -m apps.langgraph_invoice_app "How many Best Oil invoices are indexed?"
python -m apps.langgraph_invoice_app "Find diesel deliveries for JP Fuels"
```

### Microsoft Agent Framework (SequentialBuilder)
RetrieverAgent (tools) → AnalystAgent (final cited answer).  
Gemini is wired through the **OpenAI Chat Completions** compatible endpoint (`OpenAIChatCompletionClient`) — same multi-agent orchestration style as Semantic Kernel / AutoGen patterns.

```bash
python -m apps.maf_sequential_rag "Summarize Best Oil diesel invoices"
```

> Free-tier Gemini has low RPM/RPD; if you see `429 RESOURCE_EXHAUSTED`, wait a minute and retry (or enable billing).

Also see `workflow.py` for a FastAPI + MAF enterprise chat gateway (Azure OpenAI / CLI credential).

## Security notes

- Never commit `.env` or API keys.
- Prefer env vars / managed identity in production (`workflow.py` pattern).
