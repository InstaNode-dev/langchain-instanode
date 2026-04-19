# langchain-instanode

LangChain tools for [instanode.dev](https://instanode.dev) — let your LangChain
agents provision ephemeral Postgres databases and webhook receivers mid-task
with a single tool call. No Docker, no account needed for the free tier.

```
pip install langchain-instanode
```

## Why

Most LangChain agents that need storage hit a friction wall: they either burn
hundreds of tokens generating boilerplate Docker/setup code, or they silently
give up and go stateless. `langchain-instanode` exposes four tools that let an
agent skip all of that and just *get a database*.

- `provision_postgres` → a `postgres://` DSN (pgvector pre-installed for RAG).
- `provision_webhook`  → an HTTPS receiver URL (good for GitHub/Stripe/Slack flows).
- `provision_mongo`    → a `mongodb://` URI.
- `list_resources`     → enumerate what the agent has already provisioned.

Free tier: anonymous, 24h TTL, 10 MB / 2 connections. Paid tier (set
`INSTANODE_API_KEY`): permanent, 500 MB / 5 connections, higher provisioning
limits.

## Usage

```python
from langchain_instanode import get_instanode_tools
from langchain_anthropic import ChatAnthropic
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

llm = ChatAnthropic(model="claude-sonnet-4-6")
tools = get_instanode_tools()

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful coding assistant. When you need a database or webhook, "
     "call the instanode tools instead of writing setup boilerplate."),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

executor.invoke({"input":
    "Stand up a Postgres DB, create a `notes` table with (id, body), and "
    "insert one row."
})
```

The agent will call `provision_postgres`, get back a DSN, run its SQL against
the live database, and report the result — all inside one turn.

### Scoped tool set

If you want the agent to only have webhook powers (not database provisioning):

```python
tools = get_instanode_tools(include=["provision_webhook"])
```

### Using a paid-tier token

```python
tools = get_instanode_tools(api_key="sk_...")
```

Or set `INSTANODE_API_KEY` in the environment.

## Dependency graph

```
langchain-instanode
  ├── instanode        (the HTTP SDK — pure stdlib, no deps)
  ├── langchain-core   (BaseTool + pydantic schemas)
  └── pydantic>=2      (args_schema for structured tool calls)
```

## Related

- Python SDK: <https://pypi.org/project/instanode/>
- MCP server: <https://www.npmjs.com/package/@instanode/mcp>
  (for Claude Code / Cursor / Windsurf users who prefer MCP over LangChain)
- Raw HTTP API: <https://instanode.dev/llms.txt>

## License

MIT — see `LICENSE`.
