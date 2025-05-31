# VTT Innovation Entity MCP Server

A Model Context Protocol (MCP) server for managing innovation entities in a Memgraph knowledge graph with Azure OpenAI embeddings.

## Quick Start

### 1. Install Dependencies

Install and sync dependencies using uv:

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies
uv sync
```

### 2. Setup Environment Variables

FastMCP supports environment variables with the `FASTMCP_SERVER_` prefix. Set these variables:

```bash
# Memgraph Configuration
export MEMGRAPH_HOST="bolt://localhost:7687"
export MEMGRAPH_USER=""
export MEMGRAPH_PASSWORD=""

# Azure OpenAI Configuration
export AZURE_CONFIG_PATH="/path/to/your/azure_config.json"
export AZURE_MODEL_KEY="gpt-4o-mini"
export EMBEDDING_DIMENSION="2048"

# FastMCP Server Configuration
export FASTMCP_SERVER_HOST="127.0.0.1"
export FASTMCP_SERVER_PORT="9000"
export FASTMCP_SERVER_LOG_LEVEL="INFO"
```

Or create a `.env` file in your project root with the same variables (without `export`).

### 3. Start Memgraph Database

```bash
docker compose up -d
```

### 4. Run the MCP Server

```bash
# Using FastMCP CLI with environment variables
uv run fastmcp run mcp/innovation_entity_server.py --transport sse --port 9000

# Or run directly with Python via uv
uv run python mcp/innovation_entity_server.py
```

## Azure Configuration

Create `azure_config.json` with your Azure OpenAI credentials:

```json
{
  "gpt-4o-mini": {
    "api_key": "your-azure-openai-api-key",
    "api_version": "2024-02-01",
    "api_base": "https://your-instance.openai.azure.com/",
    "emb_deployment": "your-embedding-deployment-name"
  }
}
```

## Available Tools

- `resolve_or_create_canonical_innovation` - Smart innovation entity resolution
- `resolve_or_create_organization` - Organization management by VAT ID
- `add_mention_to_link` - Link organizations to innovations with provenance
- `search_similar_innovations` - Semantic search using embeddings
- `get_random_organization` - Retrieve random organizations for testing

## Services

- **Memgraph**: Graph database at `localhost:7687`
- **Memgraph Lab**: Web UI at `http://localhost:3000`
- **MCP Server**: Running on `localhost:9000` (SSE transport)
