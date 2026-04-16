"""
MCP Server Schema Compliance Tests.

Verifies that the MCP server correctly registers tools with proper schemas.
"""

import pytest

from infrastructure.config.dependency_injection import Container
from infrastructure.mcp_servers.crucible_server import TOOLS, create_crucible_mcp_server


class TestCrucibleMCPServer:
    @pytest.fixture
    async def container_and_server(self):
        container = Container()
        await container.init()
        server = create_crucible_mcp_server(container)
        yield container, server
        await container.shutdown()

    @pytest.mark.asyncio
    async def test_server_created(self, container_and_server):
        """MCP server should be created without errors."""
        _, server = container_and_server
        assert server is not None
        assert server.name == "crucible-service"

    def test_tool_schemas_defined(self):
        """All CRUCIBLE tools should have valid schemas."""
        assert len(TOOLS) == 5
        tool_names = {t.name for t in TOOLS}
        assert tool_names == {
            "register_agent", "capture_trace", "create_rubric",
            "evaluate_run", "generate_report",
        }

    def test_tool_schemas_have_required_fields(self):
        """Each tool schema should define required properties."""
        for tool in TOOLS:
            assert "properties" in tool.inputSchema
            assert "required" in tool.inputSchema
            assert tool.description  # Non-empty description

    def test_register_agent_schema(self):
        """register_agent tool should require name, description, agent_type."""
        tool = next(t for t in TOOLS if t.name == "register_agent")
        assert set(tool.inputSchema["required"]) == {"name", "description", "agent_type"}

    def test_evaluate_run_schema(self):
        """evaluate_run tool should require trace_id and rubric_id."""
        tool = next(t for t in TOOLS if t.name == "evaluate_run")
        assert "trace_id" in tool.inputSchema["required"]
        assert "rubric_id" in tool.inputSchema["required"]
