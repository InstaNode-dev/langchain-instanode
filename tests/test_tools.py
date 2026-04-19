"""
test_tools.py — unit tests for the LangChain tool wrappers.

Each test stubs an instanode.Client so no network calls happen.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from langchain_instanode import (
    ProvisionMongoTool,
    ProvisionPostgresTool,
    ProvisionWebhookTool,
    ListResourcesTool,
    get_instanode_tools,
)
import instanode


def _fake_provision_result(connection_url: str = "postgres://u:p@h/db") -> SimpleNamespace:
    return SimpleNamespace(
        connection_url=connection_url,
        tier="anonymous",
        limits=SimpleNamespace(storage_mb=10, connections=2, expires_in="24h"),
    )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def test_factory_returns_four_tools_by_default():
    tools = get_instanode_tools()
    names = {t.name for t in tools}
    assert names == {
        "provision_postgres",
        "provision_webhook",
        "provision_mongo",
        "list_resources",
    }


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
    out = tool.invoke({})
    assert "postgres://ok" in out
    assert "DSN:" in out
    client.provision_database.assert_called_once_with(name=None)


def test_postgres_tool_forwards_name():
    client = MagicMock()
    client.provision_database.return_value = _fake_provision_result()
    tool = ProvisionPostgresTool(client=client)
    tool.invoke({"name": "my-db"})
    client.provision_database.assert_called_once_with(name="my-db")


def test_webhook_tool_returns_url():
    client = MagicMock()
    client.provision_webhook.return_value = _fake_provision_result("https://hooks/abc")
    tool = ProvisionWebhookTool(client=client)
    out = tool.invoke({})
    assert "https://hooks/abc" in out


def test_mongo_tool_returns_uri():
    client = MagicMock()
    client.provision_mongodb.return_value = _fake_provision_result("mongodb://ok/db")
    tool = ProvisionMongoTool(client=client)
    out = tool.invoke({})
    assert "mongodb://ok/db" in out


# ---------------------------------------------------------------------------
# Error path — InstanodeError must become a returned string, not a raise
# ---------------------------------------------------------------------------


def test_postgres_tool_reports_error_as_string():
    client = MagicMock()
    client.provision_database.side_effect = instanode.InstanodeError(
        429, "rate_limited", "quota exceeded"
    )
    tool = ProvisionPostgresTool(client=client)
    out = tool.invoke({})
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
            service="postgres",
            tier="hobby",
            connection_url="postgres://host/db_abcdef1234567890",
            created_at="2026-04-18T10:00:00Z",
        )
    ]
    tool = ListResourcesTool(client=client)
    out = tool.invoke({})
    assert "postgres" in out
    assert "hobby" in out


def test_list_resources_reports_auth_error():
    client = MagicMock()
    client.list_resources.side_effect = instanode.InstanodeError(
        401, "unauthorized", "missing or invalid api key"
    )
    tool = ListResourcesTool(client=client)
    out = tool.invoke({})
    assert out.startswith("ERROR:")
