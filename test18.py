from flask import Flask, request, jsonify
from langchain_openai import AzureChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import os
import logging
from dotenv import load_dotenv

# ------------------- SETUP LOGGING -------------------
logging.basicConfig(
    filename="chatbot.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()

load_dotenv(dotenv_path="D:/Pyn/Test_work/LLMs/.env")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")
azure_model = os.getenv("AZURE_OAI_MODEL")


app = Flask(__name__)

class FormChatBot:
    def __init__(self):
        self.questions = {
            "first_name": "What is your first name?",
            "last_name": "What is your last name?",
            "address": "Please provide your address.",
            "gender": "What is your gender? (Male/Female/Other)",
            "terms_and_conditions": "Do you agree to the terms and conditions? (Yes/No)"
        }
        self.responses = {}
        self.asked_keys = list(self.questions.keys())
        self.index = 0
        self.return_index = None

        # Initialize Azure Model
        self.chat_model = AzureChatOpenAI(
            openai_api_key=api_key,
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_model,
            api_version=api_version,
            temperature=0.3
        )

        # Validation Prompt Setup
        validation_prompt = PromptTemplate(
            input_variables=["asking_question", "user_input", "questions_list"],
            template="""
            You are a validation AI. A form chatbot asked the user a question: "{asking_question}".

            User replied: "{user_input}"

            List of possible fields: {questions_list}

            Your task:
            - If this is a valid answer to the question, return: ANSWER
            - If this is a correction like "I need to change first name" or "wrong last name", return: CORRECT <field_name>

            Only reply exactly "ANSWER" or "CORRECT <field_name>".
            """
        )
        self.validator = LLMChain(llm=self.chat_model, prompt=validation_prompt, verbose=False)

    def get_next_question(self):
        if self.index < len(self.asked_keys):
            return self.questions[self.asked_keys[self.index]]
        return None

    def process_input(self, user_input):
        key = self.asked_keys[self.index]
        question = self.questions[key]

        logger.info(f"User input received for {key}: {user_input}")

        validation_result = self.validator.run({
            "asking_question": question,
            "user_input": user_input,
            "questions_list": ", ".join(self.questions.keys())
        }).strip()

        logger.info(f"Validation result: {validation_result}")

        # Handle correction
        if validation_result.startswith("CORRECT"):
            correction_field = validation_result.replace("CORRECT", "").strip()
            logger.info(f"Correction requested for: {correction_field}")
            if correction_field in self.questions:
                if self.return_index is None:
                    self.return_index = self.index
                self.index = self.asked_keys.index(correction_field)
                return f"Okay, updating {correction_field}."

        # Store valid response
        self.responses[key] = user_input
        if self.return_index is not None and self.index == self.return_index:
            self.return_index = None
        else:
            self.index += 1

        return "Recorded"


chatbot = FormChatBot()

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message", "").strip()

    if user_input.lower() in ["exit", "quit"]:
        logger.info("Chatbot exited by user.")
        return jsonify({"message": "Chatbot: Exiting. Goodbye!", "end": True})

    bot_response = chatbot.process_input(user_input)
    next_question = chatbot.get_next_question()

    if not next_question:
        return jsonify({"message": "Form Completed", "responses": chatbot.responses, "end": True})

    return jsonify({"message": bot_response, "next_question": next_question})

if __name__ == "__main__":
    app.run(debug=True)
