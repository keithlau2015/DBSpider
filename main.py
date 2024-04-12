import os
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents.agent_types import AgentType
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
import time
import gradio as gr

#Init
load_dotenv()
database_uri = os.getenv("DATABASE_URI")
llm_model = os.getenv("LLM_MODEL")
llm = Ollama(model=llm_model)
db = SQLDatabase.from_uri(database_uri)
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)
template = """
You are a sql specialist expert. Given an input question, answer the question with below provided data.
If necessary you can query information from the database
Only use following tables:
{schema}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
answer = prompt | agent_executor | StrOutputParser()
def getSchema(_):
    return db.get_table_info()

chain = (
    RunnablePassthrough
    .assign(schema=getSchema) | answer
)

print(db.dialect)
print(db.get_table_info())

with gr.Blocks() as ui:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")
    
    def user(user_msg, history):
        return "", history + [[user_msg, None]]

    def bot(history):
        bot_msg = chain.invoke({"question": history[-1][0]})
        history[-1][1] = ""
        for character in bot_msg:
            history[-1][1] += character
            time.sleep(0.05)
            yield history

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )

    clear.click(lambda: None, None, chatbot, queue=False)


ui.queue()
ui.launch()
#db.run("SELECT * FROM Artist LIMIT 10;")
