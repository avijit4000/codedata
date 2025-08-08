import requests
from fastapi import HTTPException
from pydantic import BaseModel
from src.components.logger import logger
import traceback

class SummaryRequest(BaseModel):
    api_url: str

class CallSummary:
    def fetch_call_summary(self, api_url: str) -> str:
        try:
            response = requests.get(api_url, timeout=10)
            response.raise_for_status()
            data = response.json()
            last_value = str(list(list(data[0].items())[-1])[-1])
            logger.info("Call summary fetched successfully.")
            return last_value
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {str(e)}\n{traceback.format_exc()}")
            raise HTTPException(status_code=400, detail=f"Failed to fetch call summary: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail="Unexpected error while processing the call summary.")

getsummary = SummaryRequest
summary_data = CallSummary()

print("ok data int")