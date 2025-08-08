# from fastapi import FastAPI, HTTPException
# import uvicorn
# import traceback
# from src.components.logger import logger
# from src.components import data_ingestion
# from src.llm import llm_template
# from src.llm import llm_model
#
# app = FastAPI()
#
#
# getsummary = data_ingestion.getsummary
# summary_data=data_ingestion.summary_data
#
# promp=llm_template.classtem
# summary_generate=llm_model.summary_generator
#
# @app.post("/summary/", summary="Generate a user-friendly summary")
# async def get_summary(request: getsummary):
#     logger.info(f"Received request for summary generation: {request.api_url}")
#     try:
#         conversation_data = summary_data.fetch_call_summary(request.api_url)
#         promptem = promp.summarytemplate(conversation_data)
#         summary = summary_generate.generate_summary(promptem)
#         return {"consumer_friendly_summary": summary}
#     except HTTPException as http_ex:
#         logger.warning(f"Handled HTTPException: {http_ex.detail}")
#         raise http_ex
#     except Exception as e:
#         logger.error(f"Unhandled exception: {str(e)}\n{traceback.format_exc()}")
#         raise HTTPException(status_code=500, detail="Internal server error")
#
# if __name__ == "__main__":
#     uvicorn.run("src.main:app", host="0.0.0.0", port=8080, reload=True)


from flask import Flask, request, jsonify
from werkzeug.exceptions import HTTPException
import traceback
from src.components.logger import logger
from src.components import data_ingestion
from src.llm import llm_template
from src.llm import llm_model

app = Flask(__name__)

# Import required functions/classes
getsummary = data_ingestion.getsummary
summary_data = data_ingestion.summary_data
promp = llm_template.classtem
summary_generate = llm_model.summary_generator

@app.route("/summary/", methods=["POST"])
def get_summary():
    try:
        req_data = request.get_json()

        if not req_data or "api_url" not in req_data:
            return jsonify({"error": "Missing 'api_url' in request body"}), 400

        logger.info(f"Received request for summary generation: {req_data['api_url']}")

        conversation_data = summary_data.fetch_call_summary(req_data["api_url"])
        promptem = promp.summarytemplate(conversation_data)
        summary = summary_generate.generate_summary(promptem)

        return jsonify({"consumer_friendly_summary": summary})

    except HTTPException as http_ex:
        logger.warning(f"Handled HTTPException: {str(http_ex)}")
        return jsonify({"error": str(http_ex)}), http_ex.code
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
