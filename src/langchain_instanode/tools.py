"""
tools.py — LangChain BaseTool subclasses wrapping instanode.dev provisioning.

Each tool is a thin adapter around a method on ``instanode.Client``. The tool
schema is inferred from a Pydantic model so the LLM sees structured args.

Design notes
------------
- One shared ``instanode.Client`` per ``get_instanode_tools()`` call. The
  client reuses urllib's connection pool; reusing it across tool invocations
  is faster than constructing a fresh client per call.
- Every tool returns a plain string (typically a connection URL or a short
  human-readable report). Tool returns are fed back to the LLM verbatim,
  so terse-but-informative is the right shape.
- Errors are caught and returned as strings starting with ``ERROR:`` so the
  LLM can recover (retry with different args, fall back, ask the user)
  rather than having an exception unwind the agent loop.
"""

from __future__ import annotations

from typing import Any, List, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

import instanode


# ---------------------------------------------------------------------------
# Input schemas — what the LLM sees in each tool's argument signature
# ---------------------------------------------------------------------------


class _ProvisionDBArgs(BaseModel):
    name: str = Field(
        description=(
            "Short kebab-case label for the database (e.g. 'embeddings-store'). "
            "Required. 1-120 characters, shown in the instanode dashboard. "
            "Has no effect on the connection URL."
        ),
    )


class _ProvisionWebhookArgs(BaseModel):
    name: str = Field(
        description=(
            "Short kebab-case label for the webhook receiver (e.g. "
            "'stripe-events'). Required. 1-120 characters."
        ),
    )


class _ListResourcesArgs(BaseModel):
    """No arguments — returns every resource owned by the configured api_key."""


# ---------------------------------------------------------------------------
# Tool classes
# ---------------------------------------------------------------------------


class _InstanodeTool(BaseTool):
    """Shared base carrying the Client that every subclass uses."""

    client: Any = Field(exclude=True)
    model_config = {"arbitrary_types_allowed": True}


class ProvisionPostgresTool(_InstanodeTool):
    name: str = "provision_postgres"
    description: str = (
        "Provision a new Postgres database from instanode.dev and return its "
        "connection URL. Use this when the user (or the task) needs a database "
        "to store structured data, run SQL queries, or hold embeddings "
        "(pgvector is pre-installed). Free-tier DBs last 24 hours; paid-tier "
        "are permanent. The returned URL is a standard postgres:// DSN usable "
        "with psycopg, asyncpg, SQLAlchemy, Prisma, and anything else that "
        "speaks Postgres."
    )
    args_schema: Type[BaseModel] = _ProvisionDBArgs

    def _run(self, name: str) -> str:
        try:
            res = self.client.provision_database(name=name)
        except instanode.InstanodeError as exc:
            return f"ERROR: {exc}"
        return (
            f"Postgres database provisioned. "
            f"DSN: {res.connection_url} "
            f"(tier={res.tier}, storage_mb={res.limits.storage_mb}, "
            f"expires_in={res.limits.expires_in or 'never'})"
        )


class ProvisionWebhookTool(_InstanodeTool):
    name: str = "provision_webhook"
    description: str = (
        "Provision a webhook receiver URL from instanode.dev. Use this when "
        "you need a live HTTP endpoint to receive callbacks from a third-party "
        "service (GitHub webhooks, Stripe events, Slack slash-commands, etc.). "
        "The URL accepts any POST, stores the body, and lets the user inspect "
        "recent requests from their dashboard."
    )
    args_schema: Type[BaseModel] = _ProvisionWebhookArgs

    def _run(self, name: str) -> str:
        try:
            res = self.client.provision_webhook(name=name)
        except instanode.InstanodeError as exc:
            return f"ERROR: {exc}"
        return (
            f"Webhook receiver provisioned. URL: {res.connection_url} "
            f"(tier={res.tier}, expires_in={res.limits.expires_in or 'never'})"
        )


class ListResourcesTool(_InstanodeTool):
    name: str = "list_resources"
    description: str = (
        "List every instanode.dev resource owned by the current account. "
        "Requires INSTANODE_API_KEY (paid tier). Returns resource type, tier, "
        "token, and creation time per resource. Use this before provisioning a "
        "new resource to see if a suitable one already exists."
    )
    args_schema: Type[BaseModel] = _ListResourcesArgs

    def _run(self) -> str:
        try:
            resources = self.client.list_resources()
        except instanode.InstanodeError as exc:
            return f"ERROR: {exc}"
        if not resources:
            return "No resources."
        lines = [
            f"- {r.resource_type} ({r.tier}) token={r.token} created={r.created_at}"
            for r in resources
        ]
        return "Resources:\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def get_instanode_tools(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    include: Optional[List[str]] = None,
) -> List[BaseTool]:
    """
    Return a list of LangChain tools bound to an instanode.Client.

    Parameters
    ----------
    api_key:
        Optional bearer JWT. Falls back to ``INSTANODE_API_KEY``. Without
        one, the client operates anonymously (free tier, 24h TTL resources).
    base_url:
        Override the API base URL. Defaults to ``https://api.instanode.dev``.
    include:
        Subset of tool names to expose. Defaults to all. Useful when you want
        to restrict what the agent can provision (e.g. only webhooks).

    Returns
    -------
    list[BaseTool]
        Ready to pass to ``create_tool_calling_agent`` / ``AgentExecutor``.
    """
    client = instanode.Client(api_key=api_key, base_url=base_url)
    all_tools: List[BaseTool] = [
        ProvisionPostgresTool(client=client),
        ProvisionWebhookTool(client=client),
        ListResourcesTool(client=client),
    ]
    if include is None:
        return all_tools
    return [t for t in all_tools if t.name in include]
