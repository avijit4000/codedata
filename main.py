from fastapi import FastAPI, HTTPException
import uvicorn
import traceback
from src.components.logger import logger
from src.components import data_ingestion
from src.llm import llm_template
from src.llm import llm_model

app = FastAPI()


getsummary = data_ingestion.getsummary
summary_data=data_ingestion.summary_data

promp=llm_template.classtem
summary_generate=llm_model.summary_generator

@app.post("/summary/", summary="Generate a user-friendly summary")
async def get_summary(request: getsummary):
    logger.info(f"Received request for summary generation: {request.api_url}")
    try:
        conversation_data = summary_data.fetch_call_summary(request.api_url)
        promptem = promp.summarytemplate(conversation_data)
        summary = summary_generate.generate_summary(promptem)
        return {"consumer_friendly_summary": summary}
    except HTTPException as http_ex:
        logger.warning(f"Handled HTTPException: {http_ex.detail}")
        raise http_ex
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8080, reload=True)
