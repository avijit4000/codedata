# from fastmcp import FastMCP, tool
# from typing import Optional, Iterable, Dict, Any
# from datetime import datetime
# import requests
#
# API_URL = "https://hcccloud-uhgdlm-dtlapi-dev.uhc.com/gpd-backend-python/Latency-Report-7day"
#
# # ---------------- Core Function ---------------- #
# def fetch_latency_report(
#     day: Optional[str] = None,
#     days: Optional[Iterable[str]] = None,
#     start_date: Optional[str] = None,
#     end_date: Optional[str] = None,
#     app: Optional[str] = None
# ) -> Dict[str, Any]:
#     """
#     Fetch latency report JSON and optionally filter by date(s).
#     """
#     resp = requests.get(API_URL, verify="standard_trusts.pem")
#     resp.raise_for_status()
#     data = resp.json()
#     if not isinstance(data, dict):
#         raise ValueError("Unexpected response shape (expected dict)")
#
#     single = {day} if day else None
#     multi = set(days) if days else None
#     range_start = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
#     range_end = datetime.strptime(end_date, "%Y-%m-%d") if end_date else range_start
#
#     def include(d: str) -> bool:
#         if single:
#             return d in single
#         if multi:
#             return d in multi
#         if range_start:
#             try:
#                 dt = datetime.strptime(d, "%Y-%m-%d")
#             except ValueError:
#                 return False
#             return range_start <= dt <= range_end
#         return True
#
#     filtered: Dict[str, Any] = {}
#     if app:
#         app = app.strip().upper()
#         if app not in data:
#             raise ValueError(f"Unknown app '{app}'. Available keys: {', '.join(sorted(data.keys()))}")
#         rows = data.get(app, [])
#         filtered[app] = [r for r in rows if isinstance(r, dict) and include(r.get("date", ""))]
#         return filtered
#
#     for product, rows in data.items():
#         if not isinstance(rows, list):
#             continue
#         keep = [r for r in rows if isinstance(r, dict) and include(r.get("date", ""))]
#         filtered[product] = keep
#
#     return filtered
#
#
# # ---------------- MCP Integration ---------------- #
# mcp = FastMCP("latency-metrics")
#
# @tool
# def get_latency_report(
#     day: Optional[str] = None,
#     days: Optional[list[str]] = None,
#     start_date: Optional[str] = None,
#     end_date: Optional[str] = None,
#     app: Optional[str] = None
# ) -> Dict[str, Any]:
#     """
#     Retrieve latency report with optional filters.
#     """
#     return fetch_latency_report(day, days, start_date, end_date, app)
#
#
# if __name__ == "__main__":
#     # Run as a streamable-http MCP server on port 8020
#     mcp.run("streamable-http", host="127.0.0.1", port=8020)


from fastmcp import FastMCP, tool
from pull_metric_api import fetch_latency_report
from typing import Optional, List, Dict, Any
import json

# Initialize MCP server
mcp = FastMCP("matrix_server")

@tool
def get_latency_data(
    day: Optional[str] = None,
    days: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    app: Optional[str] = None
) -> Dict[str, Any]:
    """
    Fetch dynamic latency report data from the pull_metric_api module.
    You can filter by:
        - day: single date (YYYY-MM-DD)
        - days: list of specific dates
        - start_date, end_date: date range
        - app: filter by application (e.g., UMS, AMP)
    """
    try:
        data = fetch_latency_report(day, days, start_date, end_date, app)
        return data
    except Exception as e:
        return {"error": str(e)}

@tool
def get_latency_data_pretty(
    app: Optional[str] = None,
    day: Optional[str] = None
) -> str:
    """
    Fetch latency report and return formatted as readable JSON text.
    """
    try:
        data = fetch_latency_report(day=day, app=app)
        return json.dumps(data, indent=2)
    except Exception as e:
        return f"Error fetching latency data: {e}"

if __name__ == "__main__":
    print("🚀 Starting MCP server (matrix_server)...")
    mcp.run("streamable-http", host="127.0.0.1", port=8010)
