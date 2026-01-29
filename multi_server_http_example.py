import contextlib
import os
import sys
from pathlib import Path
from fastapi import FastAPI

"""
Example: multi_server_http_example.py with FastAPI and Multiple MCP Servers

This file demonstrates how to create a combined FastAPI application that manages
multiple MCP servers with a unified lifespan context manager.

IMPORTANT: Before using this configuration:
1. Ensure all required MCP servers are available
2. All servers should use FastMCP with stateless_http=True
3. Run this file as the main application server
"""
# Add the servers directory to the path for imports
SERVERS_DIR = Path(__file__).parent / "servers"
sys.path.insert(0, str(SERVERS_DIR))

# Import mcp instances from server files
from mathserver import mcp as math_mcp
from weather import mcp as weather_mcp
from latency_server import mcp as latency_mcp
# Import the Dynatrace plugin instead of the server
from src.LLM.plugins.dynatrace_monitoring.plugin_config_http import dynatrace_plugin
dynatrace_mcp = dynatrace_plugin.mcp  # Get the MCP instance from the plugin

from src.LLM.plugins.azure_monitoring.plugin_config_azure import azure_plugin
azure_mcp = azure_plugin.mcp

from src.LLM.plugins.servicenow_monitoring.plugin_config_servicenow_http import servicenow_plugin
servicenow_mcp = servicenow_plugin.mcp

# Create a combined lifespan to manage all session managers
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    async with contextlib.AsyncExitStack() as stack:
        await stack.enter_async_context(math_mcp.session_manager.run())
        await stack.enter_async_context(weather_mcp.session_manager.run())
        await stack.enter_async_context(latency_mcp.session_manager.run())
        await stack.enter_async_context(dynatrace_mcp.session_manager.run())
        await stack.enter_async_context(azure_mcp.session_manager.run())
        await stack.enter_async_context(servicenow_mcp.session_manager.run())
        yield


app = FastAPI(lifespan=lifespan)

# Mount all MCP servers as sub-applications

app.mount("/math", math_mcp.streamable_http_app())
app.mount("/weather", weather_mcp.streamable_http_app())
app.mount("/latency", latency_mcp.streamable_http_app())
app.mount("/dynatrace", dynatrace_mcp.streamable_http_app())
app.mount("/azure", azure_mcp.streamable_http_app())
app.mount("/servicenow", servicenow_mcp.streamable_http_app())

PORT = int(os.environ.get("PORT", 8012))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
