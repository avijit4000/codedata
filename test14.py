# import os
# from flask import Flask, jsonify, request
# from dotenv import load_dotenv
# from langchain_openai import AzureChatOpenAI
# from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
# from langchain.chains import LLMChain
# from langchain.memory import ChatMessageHistory
# from langchain_core.runnables.history import RunnableWithMessageHistory
# load_dotenv(dotenv_path="D:/Pyn/Test_work/LLMs/.env")
# azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
# api_key = os.getenv("AZURE_OPENAI_API_KEY")
# api_version = os.getenv("AZURE_OPENAI_API_VERSION")
# azure_model = os.getenv("AZURE_OAI_MODEL")
# app = Flask(__name__)
# store = {}
# def get_session_history(session_id: str):
#     if session_id not in store:
#         store[session_id] = ChatMessageHistory()
#     return store[session_id]
# llm = AzureChatOpenAI(
#     openai_api_key=api_key,
#     azure_endpoint=azure_endpoint,
#     azure_deployment=azure_model,
#     api_version=api_version,
#     temperature=0.0
# )
# class prompTemplate:
#     def question_ask(self, user_input: str) -> str:
#         question_template = f"""
#                 You are a helpful and precise assistant that converts a form field name into a clear, natural question.
#
#                 Instructions:
#                 - Use simple, human-like question phrasing.
#                 - Do not add any extra words or explanation.
#                 - Only return the question text.
#                 - Examples are very important — follow them closely.
#
#                 Examples:
#                 - "First Name" → "What is your first name?"
#                 - "Last Name" → "What is your last name?"
#                 - "Date of Birth" → "What is your date of birth?"
#                 - "Sex" → "What is your sex? (Male/Female/Other)"
#                 - "Phone Type" → "Please select your phone type (e.g., Mobile, Home, Work)."
#
#                 Input: "{user_input}"
#
#                 Output:
#                 """
#         return question_template
#     def question_answer(self, asking_question: str, user_ans: str) -> str:
#         template = f"""
#             You are a helpful assistant. A question was asked: "{asking_question}".
#
#             Based on the user's input below, provide **only the exact answer** relevant to the question.
#
#             - If the question is about "date of birth" or any date field, detect the date in any format (e.g., "April 19, 1999", "19/04/1999") and return it in DD-MM-YYYY format.
#             - For other fields, return the relevant answer as-is, without any extra text, punctuation, or explanation.
#
#             User input: "{user_ans}"
#
#             Output:
#             """
#         return template
# template_builder = prompTemplate()
# @app.route('/ask_question', methods=['POST'])
# def ask_question():
#     data = request.get_json()
#     user_input = data.get('field_name', '').strip()
#     session_id = data.get('session_id', 'user1')
#     history = get_session_history(session_id)
#     for msg in history.messages:
#         if user_input.lower() in msg.content.lower():
#             return jsonify({"extracted_answer": msg.content})
#     question_template = template_builder.question_ask(user_input)
#     prompt = PromptTemplate(input_variables=["user_input"], template=question_template)
#     chain = LLMChain(llm=llm, prompt=prompt, verbose=False)
#     result = chain.run(user_input=user_input).strip()
#     return jsonify({"question": result})
# @app.route('/extract_answer', methods=['POST'])
# def extract_answer():
#     data = request.get_json()
#     asking_question = data.get('asking_question', '')
#     user_ans = data.get('user_answer', '')
#     session_id = data.get('session_id', 'user1')
#     history = get_session_history(session_id)
#     answer_template = template_builder.question_answer(asking_question, user_ans)
#     prompt = ChatPromptTemplate.from_messages([
#         ("system", answer_template),
#         MessagesPlaceholder(variable_name="history"),
#         ("human", "{input}")
#     ])
#     chain = prompt | llm
#     with_memory = RunnableWithMessageHistory(
#         chain,
#         get_session_history,
#         input_messages_key="input",
#         history_messages_key="history"
#     )
#     config = {"configurable": {"session_id": session_id}}
#     if user_ans:
#         with_memory.invoke({"input": user_ans}, config=config)
#     response = with_memory.invoke({"input": asking_question}, config=config)
#     return jsonify({"extracted_answer": response.content})
#
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)



import os
from flask import Flask, jsonify, request
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain.memory import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv(dotenv_path="D:/Pyn/Test_work/LLMs/.env")

azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")
azure_model = os.getenv("AZURE_OAI_MODEL")

app = Flask(__name__)

# Store for sessions
store = {}  # { session_id: { "history": ChatMessageHistory(), "form_data": {} } }

def get_session(session_id: str):
    if session_id not in store:
        store[session_id] = {
            "history": ChatMessageHistory(),
            "form_data": {}
        }
    return store[session_id]

llm = AzureChatOpenAI(
    openai_api_key=api_key,
    azure_endpoint=azure_endpoint,
    azure_deployment=azure_model,
    api_version=api_version,
    temperature=0.0
)

class prompTemplate:
    def question_ask(self, user_input: str) -> str:
        return f"""
        You are a helpful and precise assistant that converts a form field name into a clear, natural question.

        Examples:
        - "First Name" → "What is your first name?"
        - "Last Name" → "What is your last name?"
        - "Date of Birth" → "What is your date of birth?"
        - "Sex" → "What is your sex? (Male/Female/Other)"

        Input: "{user_input}"
        Output:
        """

    def question_answer(self, asking_question: str, user_ans: str) -> str:
        return f"""
        You are a helpful assistant. A question was asked: "{asking_question}".
        Based on the user's input below, provide only the exact answer relevant to the question.

        - If it's a date (like date of birth), return in DD-MM-YYYY format.
        - For other fields, return the relevant answer only.

        User input: "{user_ans}"
        Output:
        """

template_builder = prompTemplate()

@app.route('/ask_question', methods=['POST'])
def ask_question():
    data = request.get_json()
    user_input = data.get('field_name', '').strip()
    session_id = data.get('session_id', 'user1')

    session = get_session(session_id)
    history = session["history"]

    question_template = template_builder.question_ask(user_input)
    prompt = PromptTemplate(input_variables=["user_input"], template=question_template)
    chain = LLMChain(llm=llm, prompt=prompt, verbose=False)
    question = chain.run(user_input=user_input).strip()

    return jsonify({"question": question})


@app.route('/extract_answer', methods=['POST'])
def extract_answer():
    data = request.get_json()
    asking_question = data.get('asking_question', '')
    user_ans = data.get('user_answer', '')
    field_name = data.get('field_name', '')  # NEW
    session_id = data.get('session_id', 'user1')

    session = get_session(session_id)
    history = session["history"]
    form_data = session["form_data"]

    # Create LLM prompt for extraction
    answer_template = template_builder.question_answer(asking_question, user_ans)
    prompt = ChatPromptTemplate.from_messages([
        ("system", answer_template),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    chain = prompt | llm
    with_memory = RunnableWithMessageHistory(
        chain,
        lambda _: history,
        input_messages_key="input",
        history_messages_key="history"
    )
    config = {"configurable": {"session_id": session_id}}

    response = with_memory.invoke({"input": user_ans}, config=config)
    extracted = response.content.strip()

    # Save the extracted answer
    if field_name:
        form_data[field_name] = extracted

    return jsonify({
        "extracted_answer": extracted,
        "form_data": form_data
    })


# 🆕 Allow user to update/change a previously entered field
@app.route('/update_answer', methods=['POST'])
def update_answer():
    data = request.get_json()
    field_name = data.get('field_name', '').strip()
    new_answer = data.get('new_answer', '').strip()
    session_id = data.get('session_id', 'user1')

    session = get_session(session_id)
    form_data = session["form_data"]

    if field_name not in form_data:
        return jsonify({"error": f"No previous answer found for '{field_name}'"}), 400

    # Update the stored answer
    form_data[field_name] = new_answer

    return jsonify({
        "message": f"Updated '{field_name}' successfully.",
        "form_data": form_data
    })


@app.route('/get_form_data', methods=['GET'])
def get_form_data():
    session_id = request.args.get('session_id', 'user1')
    session = get_session(session_id)
    return jsonify(session["form_data"])


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
