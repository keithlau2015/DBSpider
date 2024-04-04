import os
from operator import itemgetter
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain_community.utilities.sql_database import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
import gradio as gr

#Init
load_dotenv()
database_uri = os.getenv("DATABASE_URI")
llm_model = os.getenv("LLM_MODEL")
llm = Ollama(model=llm_model)
db = SQLDatabase.from_uri(database_uri)
exec_query = QuerySQLDataBaseTool(db=db)
write_query = create_sql_query_chain(llm, db)

template = """
You are a database specialist expert. Given an input question, first create a syntactically correct sql query to run, then look at the results of the query and return the answer to the input question.
Unless the user specifies in the question a specific number of examples to obtain, query for at most 5 results using the LIMIT clause as per sql. You can order the results to return the most informative data in the database.
Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in double quotes (") to denote them as delimited identifiers.
Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
Pay attention to use date('now') function to get the current date, if the question involves "today".

Only use following tables:
{schema}

Question: {question}
SQL Query: {query}
SQL Response: {response}
Answer: 
"""
prompt = PromptTemplate.from_template(template)

answer = prompt | llm | StrOutputParser

chain = (
    RunnablePassthrough
    .assign(query=write_query)
    .assign(response=itemgetter("query") | exec_query)
    | answer
)

def chatSQL(question):
    response = chain.invoke({"question": question, "schema":db.get_table_info()})    
    print(response)
    return response

ui = gr.Interface(
    fn=chatSQL,
    inputs=["text"],
    outputs=["text"],
)

ui.launch()
#db.run("SELECT * FROM Artist LIMIT 10;")