# VTT Innovation Disambiguation Pipeline

**🏆 Hackathon Solution: VTT Innovation De-duplication & Aggregation Challenge**

An AI-powered pipeline that transforms VTT's fragmented innovation data into a clean, canonical knowledge graph using FAISS semantic search, GPT-4.1-mini, and intelligent graph curation.

## The Problem

VTT's innovation data exists as fragmented mentions across multiple sources, making it impossible to get accurate portfolio overviews or track true collaboration patterns.

## Our Solution

**Three-stage automated pipeline:**

1. **FAISS Semantic Analysis** - Identify potential duplicates using context-aware similarity
2. **LLM Agent + MCP Server** - GPT-4.1-mini makes intelligent SAME/DIFFERENT decisions
3. **Canonical Graph** - Clean Memgraph database with complete provenance

## Quick Demo

```bash
cd pipeline
# Install dependencies
curl -LsSf https://astral.sh/uv/install.sh | sh
curl -fsSL https://bun.sh/install | bash
uv sync && bun install

# Start services
docker compose up -d
uv run fastmcp run mcp/innovation_entity_server.py --transport sse --port 9000

# Run pipeline
uv run python scripts/innovations_analysis.py analyze
bun run agents/innovation-curator-agent.ts results.json
```

## Key Innovation

**Context-aware AI curation** that preserves complete audit trails while creating canonical entities. Our "thick edge" graph architecture maintains every original mention with full provenance.

## Results

- Eliminates innovation mention duplicates while preserving all source context
- Enables semantic discovery and accurate collaboration analysis
- Production-ready automated pipeline replacing manual processes

https://github.com/user-attachments/assets/f7600076-04a1-4dc7-8930-84706030be01


## Documentation

📋 **[Complete Solution Details](docs/SOLUTION.md)** - Full technical architecture and hackathon submission

🚀 **[Pipeline Usage Guide](pipeline/README.md)** - Detailed setup and usage instructions

📋 **[Jupyter Notebook EDA](notebooks/candidates_eda.ipynb)** - Exploratory Data Analysis of the candidates

## Services

- **Memgraph Database**: `localhost:7687` (Web UI: `localhost:3000`)
- **MCP Server**: `localhost:9000`
- **Pipeline Scripts**: Python + TypeScript automation

## Technology Stack

- **AI**: Azure OpenAI (GPT-4.1, text-embedding-3-large)
- **Search**: FAISS vector similarity
- **Graph**: Memgraph with native vector search
- **Orchestration**: Model Context Protocol (MCP)
- **Languages**: Python (uv), TypeScript (Bun)

---

_Built for the AaltoAI Hackathon 2025_
