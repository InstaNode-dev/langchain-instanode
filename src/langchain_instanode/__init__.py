"""
langchain-instanode — LangChain tools for instanode.dev.

Expose instanode.dev's zero-setup provisioning as LangChain tools that an
agent can call mid-conversation. A single `curl` + no account required for
the free tier.

Quick start
-----------
    from langchain_instanode import get_instanode_tools
    from langchain.agents import create_tool_calling_agent
    from langchain_anthropic import ChatAnthropic

    llm = ChatAnthropic(model="claude-sonnet-4-6")
    tools = get_instanode_tools()
    agent = create_tool_calling_agent(llm, tools, prompt)
    # Now the agent can provision a DB mid-task:
    #   "I need a Postgres DB to store these embeddings."
    #   → calls provision_postgres → gets a DSN → continues work.

Authentication
--------------
Free tier works without any credentials (anonymous resources, 24h TTL).
For paid-tier limits and resource ownership, set INSTANODE_API_KEY in
your environment or pass api_key= to `get_instanode_tools()`.

- Homepage:      https://instanode.dev
- Docs:          https://instanode.dev/docs
- Underlying:    https://pypi.org/project/instanode/
"""

from langchain_instanode.tools import (
    get_instanode_tools,
    ProvisionPostgresTool,
    ProvisionWebhookTool,
    ProvisionMongoTool,
    ListResourcesTool,
)

__version__ = "0.1.0"
__all__ = [
    "get_instanode_tools",
    "ProvisionPostgresTool",
    "ProvisionWebhookTool",
    "ProvisionMongoTool",
    "ListResourcesTool",
]
