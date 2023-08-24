
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
import os
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
# conversation = ConversationChain(
#     llm=chat,
#     memory=ConversationBufferMemory()
# )
load_dotenv()
API_KEY = os.environ.get("OPENAI_API_KEY")

loader = TextLoader("/home/edmond/Test/doc.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
embedding_function = OpenAIEmbeddings()

db = Chroma.from_documents(docs, embedding_function)
model_name = "gpt-3.5-turbo"
llm = ChatOpenAI(model_name=model_name)

retrieval_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=db.as_retriever())


while True:
    try:
        query = input("Ask your quetion (or 'exit'): ")
        if query.lower() == 'exit':
            break
        response = retrieval_chain.run(query)
        print(response)
    except Exception as e:
        print(f"Произошла ошибка: {e}")










  
