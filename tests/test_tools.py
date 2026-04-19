"""
test_tools.py — unit tests for the LangChain tool wrappers.

Each test stubs an ``instanode.Client`` so no network calls happen.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import instanode
from langchain_instanode import (
    ListResourcesTool,
    ProvisionPostgresTool,
    ProvisionWebhookTool,
    get_instanode_tools,
)


def _fake_provision_result(url: str = "postgres://u:p@h/db") -> SimpleNamespace:
    return SimpleNamespace(
        connection_url=url,
        tier="anonymous",
        limits=SimpleNamespace(storage_mb=10, connections=2, expires_in="24h"),
    )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def test_factory_returns_three_tools_by_default():
    tools = get_instanode_tools()
    names = {t.name for t in tools}
    assert names == {"provision_postgres", "provision_webhook", "list_resources"}


def test_factory_respects_include_filter():
    tools = get_instanode_tools(include=["provision_postgres"])
    assert len(tools) == 1
    assert tools[0].name == "provision_postgres"


def test_factory_returns_empty_when_include_matches_nothing():
    assert get_instanode_tools(include=["nonexistent_tool"]) == []


# ---------------------------------------------------------------------------
# Happy-path tool behaviour
# ---------------------------------------------------------------------------


def test_postgres_tool_returns_dsn():
    client = MagicMock()
    client.provision_database.return_value = _fake_provision_result("postgres://ok")
    tool = ProvisionPostgresTool(client=client)
    out = tool.invoke({"name": "my-db"})
    assert "postgres://ok" in out
    assert "DSN:" in out
    client.provision_database.assert_called_once_with(name="my-db")


def test_webhook_tool_returns_url():
    client = MagicMock()
    client.provision_webhook.return_value = _fake_provision_result(
        "https://api.instanode.dev/webhook/receive/abc"
    )
    tool = ProvisionWebhookTool(client=client)
    out = tool.invoke({"name": "stripe-hook"})
    assert "/webhook/receive/abc" in out
    client.provision_webhook.assert_called_once_with(name="stripe-hook")


# ---------------------------------------------------------------------------
# Error path — InstanodeError becomes a returned string, not a raise
# ---------------------------------------------------------------------------


def test_postgres_tool_reports_error_as_string():
    client = MagicMock()
    client.provision_database.side_effect = instanode.InstanodeError(
        429, "rate_limited", "quota exceeded"
    )
    tool = ProvisionPostgresTool(client=client)
    out = tool.invoke({"name": "my-db"})
    assert out.startswith("ERROR:")
    assert "quota exceeded" in out


# ---------------------------------------------------------------------------
# list_resources
# ---------------------------------------------------------------------------


def test_list_resources_empty():
    client = MagicMock()
    client.list_resources.return_value = []
    tool = ListResourcesTool(client=client)
    assert tool.invoke({}) == "No resources."


def test_list_resources_formatted():
    client = MagicMock()
    client.list_resources.return_value = [
        SimpleNamespace(
            resource_type="postgres",
            tier="paid",
            token="tok-1",
            created_at="2026-04-19T10:00:00Z",
        ),
        SimpleNamespace(
            resource_type="webhook",
            tier="paid",
            token="tok-2",
            created_at="2026-04-19T11:00:00Z",
        ),
    ]
    tool = ListResourcesTool(client=client)
    out = tool.invoke({})
    assert "postgres" in out
    assert "webhook" in out
    assert "tok-1" in out
    assert "tok-2" in out


def test_list_resources_reports_auth_error():
    client = MagicMock()
    client.list_resources.side_effect = instanode.InstanodeError(
        401, "unauthorized", "missing or invalid api key"
    )
    tool = ListResourcesTool(client=client)
    out = tool.invoke({})
    assert out.startswith("ERROR:")
