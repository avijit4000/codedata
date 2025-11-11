from fastmcp import FastMCP, tool
from pull_metric_api_2 import fetch_metric_data
from typing import Optional, List, Dict, Any
import json

# Initialize MCP server
mcp = FastMCP("matrix_server_stdio")

@tool
def get_metric_data() -> List[Dict[str, Any]]:
    """
    Fetch metric data from pull_metric_api_2.py.
    Returns the raw list of metric dictionaries.
    """
    try:
        data = fetch_metric_data()
        return data
    except Exception as e:
        return [{"error": str(e)}]

@tool
def get_metric_data_pretty() -> str:
    """
    Fetch metric data and return formatted JSON for readability.
    """
    try:
        data = fetch_metric_data()
        return json.dumps(data, indent=2)
    except Exception as e:
        return f"Error fetching metric data: {e}"

if __name__ == "__main__":
    print("🚀 Starting MCP server (matrix_server_stdio)...")
    mcp.run("stdio")
