import requests
from typing import Optional, Iterable, Dict, Any
from datetime import datetime

API_URL = "https://hcccloud-uhgdlm-dtlapi-dev.uhc.com/gpd-backend-python/Latency-Report-7day"

def fetch_latency_report(
    day: Optional[str] = None,
    days: Optional[Iterable[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
    , app: Optional[str] = None
) -> Dict[str, Any]:
    """
    Fetch latency report JSON and optionally filter by date(s).

    Date format: YYYY-MM-DD
    Filtering precedence:
      1. If day provided -> return only that day.
      2. Else if days iterable provided -> return those days.
      3. Else if start_date (and optional end_date) provided -> range filter.
      4. Else return all data.

    Returns dict with same top-level keys (e.g. UMS, AMP) each containing filtered list.
    """
    resp = requests.get(API_URL, verify="standard_trusts.pem")
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict):
        raise ValueError("Unexpected response shape (expected dict)")

    # Normalize filter sets
    single = {day} if day else None
    multi = set(days) if days else None
    range_start = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
    range_end = datetime.strptime(end_date, "%Y-%m-%d") if end_date else range_start

    def include(d: str) -> bool:
        if single:
            return d in single
        if multi:
            return d in multi
        if range_start:
            try:
                dt = datetime.strptime(d, "%Y-%m-%d")
            except ValueError:
                return False
            return range_start <= dt <= range_end
        return True  # no filters

    filtered: Dict[str, Any] = {}
    # validate app if provided
    if app:
        app = app.strip().upper()
        if app not in data:
            raise ValueError(f"Unknown app '{app}'. Available keys: {', '.join(sorted(data.keys()))}")
        rows = data.get(app, [])
        if not isinstance(rows, list):
            raise ValueError(f"Unexpected data for app {app}")
        filtered[app] = [r for r in rows if isinstance(r, dict) and include(r.get("date", ""))]
        return filtered

    for product, rows in data.items():
        if not isinstance(rows, list):
            continue
        keep = [r for r in rows if isinstance(r, dict) and include(r.get("date", ""))]
        filtered[product] = keep

    return filtered

# Example (remove or comment out in production):
if __name__ == "__main__":
    result = fetch_latency_report()
    print(result)