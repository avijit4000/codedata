"""
This module creates a unified FastAPI application that hosts multiple MCP (Model Context Protocol)
servers under different routes. Each MCP server runs with stateless HTTP transport and is managed
through a shared application lifespan.

Purpose
-------
This file acts as a gateway/orchestrator that:
1. Starts and manages multiple MCP server session managers
2. Mounts each MCP server as a sub-application
3. Exposes all servers through a single FastAPI service

Architecture Overview
---------------------
FastAPI App
│
├── /math        → Math MCP Server
├── /weather     → Weather MCP Server
├── /latency     → Latency Testing MCP Server
├── /dynatrace   → Dynatrace Monitoring MCP Plugin
├── /azure       → Azure Monitoring MCP Plugin
└── /servicenow  → ServiceNow Monitoring MCP Plugin

All MCP servers must:
- Use FastMCP
- Be configured with stateless_http=True
- Provide `.session_manager.run()` for lifecycle handling
- Provide `.streamable_http_app()` for mounting
"""

import contextlib
import os
import sys
from pathlib import Path
from fastapi import FastAPI

# -------------------------------------------------------------------
# Dynamic Path Setup
# -------------------------------------------------------------------
"""
Adds the local 'servers' directory to Python’s module search path.

This allows importing MCP server modules located in:
    ./servers/mathserver.py
    ./servers/weather.py
    ./servers/latency_server.py
"""
SERVERS_DIR = Path(__file__).parent / "servers"
sys.path.insert(0, str(SERVERS_DIR))


# -------------------------------------------------------------------
# MCP Server Imports
# -------------------------------------------------------------------
"""
Each import exposes an MCP instance named `mcp`.

These MCP objects provide:
- session_manager.run() → async context manager for server lifecycle
- streamable_http_app() → ASGI app used for HTTP mounting
"""

# Core example servers
from mathserver import mcp as math_mcp
from weather import mcp as weather_mcp
from latency_server import mcp as latency_mcp

# Monitoring / Enterprise plugins (MCP-compatible)
from src.LLM.plugins.dynatrace_monitoring.plugin_config_http import dynatrace_plugin
dynatrace_mcp = dynatrace_plugin.mcp  # Extract MCP instance from plugin wrapper

from src.LLM.plugins.azure_monitoring.plugin_config_azure import azure_plugin
azure_mcp = azure_plugin.mcp

from src.LLM.plugins.servicenow_monitoring.plugin_config_servicenow_http import servicenow_plugin
servicenow_mcp = servicenow_plugin.mcp


# -------------------------------------------------------------------
# Application Lifespan Manager
# -------------------------------------------------------------------
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context that starts and stops all MCP session managers.

    Why this is required:
    ---------------------
    Each MCP server maintains internal async resources such as:
    - Client sessions
    - Connection pools
    - Background workers
    - Streaming pipelines

    The session_manager.run() context ensures:
    ✔ Proper startup of MCP server resources
    ✔ Graceful shutdown on app termination
    ✔ No orphan background tasks

    The AsyncExitStack allows all MCP servers to be managed together.
    """
    async with contextlib.AsyncExitStack() as stack:
        await stack.enter_async_context(math_mcp.session_manager.run())
        await stack.enter_async_context(weather_mcp.session_manager.run())
        await stack.enter_async_context(latency_mcp.session_manager.run())
        await stack.enter_async_context(dynatrace_mcp.session_manager.run())
        await stack.enter_async_context(azure_mcp.session_manager.run())
        await stack.enter_async_context(servicenow_mcp.session_manager.run())
        yield


# -------------------------------------------------------------------
# FastAPI Application Initialization
# -------------------------------------------------------------------
"""
Creates the main FastAPI application and attaches the lifespan manager.
"""
app = FastAPI(lifespan=lifespan)


# -------------------------------------------------------------------
# MCP Server Route Mounting
# -------------------------------------------------------------------
"""
Each MCP server is mounted as a sub-application.

This allows:
- Independent tool namespaces per server
- Separation of domain responsibilities
- Scalable multi-agent tool routing
"""
app.mount("/math", math_mcp.streamable_http_app())
app.mount("/weather", weather_mcp.streamable_http_app())
app.mount("/latency", latency_mcp.streamable_http_app())
app.mount("/dynatrace", dynatrace_mcp.streamable_http_app())     # Requests sent to /latency → handled by latency MCP server
app.mount("/azure", azure_mcp.streamable_http_app())             # Requests sent to /dynatrace → handled by dynatrace MCP server
app.mount("/servicenow", servicenow_mcp.streamable_http_app())   # Requests sent to /servicenow → handled by servicenow MCP server


# -------------------------------------------------------------------
# Server Startup Configuration
# -------------------------------------------------------------------
"""
The service runs on a configurable port using the PORT environment variable.
Default port: 8012
"""
PORT = int(os.environ.get("PORT", 8012))


if __name__ == "__main__":
    """
    Entry point for local development.
    Accessible at:
        http://localhost:8012
        http://localhost:8012/latency/mcp
        http://localhost:8012/dynatrace/mcp
        http://localhost:8012/servicenow/mcp

    Example:
        PORT=8012 python multi_server_http_example.py
    """
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
