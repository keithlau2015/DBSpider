import getpass
import os

from langchain_community.llms import Ollama
from langchain_community.utilities import SQLDatabase

llm = Ollama(model="mistral")
llm.invoke("The first man on the moon was ...")
db = SQLDatabase.from_uri("sqlite:///Chinook.db")
print(db.dialect)
print(db.get_usable_table_names())
db.run("SELECT * FROM Artist LIMIT 10;")