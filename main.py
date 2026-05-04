# ═══════════════════════════════════════════════════════
# Project  : AI RAG WebURL Chatbot
# Author   : Ponraj P
# GitHub   : https://github.com/Ponraj1405
# LinkedIn : https://www.linkedin.com/in/ponraj-p-804368202/
# Tech     : LangChain | FAISS | Groq API | Llama 3.1 | Streamlit
# Description: A RAG chatbot that answers questions from
#              any web URL using LangChain, FAISS,
#              Llama 3.1 and Groq API
# ═══════════════════════════════════════════════════════
import os
import streamlit as st
import pickle
import time

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader


from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate

# UI

st.title("AI RAG Chatbot")
st.sidebar.title("URLs")

urls=[]
for i in range(3):
    url=st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked=st.sidebar.button("Process URLs")
file_path="faiss_store_hf.pkl"
main_placeholder=st.empty()

#llm - Groq llama 3.1

@st.cache_resource
def load_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.1,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
llm=load_llm()


#Custom Prompt for better answer

prompt_template="""Read the context below and answer the questions directly and specifically.
context:{summaries}
Question:{question}
Answer(be specific and use facts from the context):"""

PROMPT=PromptTemplate(
    template=prompt_template,
    input_variables=["summaries","question"]
)

#Process URLs
if process_url_clicked:
    if os.path.exists(file_path):
        os.remove(file_path)
    clean_urls=[u for u in urls if u.strip()]
    if not clean_urls:
        st.sidebar.warning("Please enter at least one URL.")
    else:
        try:
            loader=WebBaseLoader(web_paths=clean_urls)   
            loader.requests_kwargs={"timeout":15}
            main_placeholder.text("Loading data from URLs...")
            data=loader.load()

            if not data:
                st.error("No data loaded. Please check your URLs.")
            else:
                text_splitter=RecursiveCharacterTextSplitter(
                    separators=["\n\n","\n",".",","],
                    chunk_size=500,
                    chunk_overlap=150,
                )
                main_placeholder.text("Splitting text...")
                docs=text_splitter.split_documents(data)

                main_placeholder.text(f"Creating embeddings for {len(docs)} chunks...")
                embeddings=HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                vectorstore=FAISS.from_documents(docs,embeddings)
                main_placeholder.text("Embeddings ready...")
                time.sleep(1)

                with open(file_path,"wb") as f:
                    pickle.dump(vectorstore,f)

                main_placeholder.success(f"Done! Processed {len(docs)} chunks. Ask your question below ")

        except Exception as e:
            st.error(f"Error processing URLs:{e}")

#Query(user input)

query=st.text_input("Ask Question:")

if query:
    if not os.path.exists(file_path):
        st.warning("Please process some URLs first.")
    else:
        with open(file_path,"rb") as f:
            vectorstore=pickle.load(f)

        chain=RetrievalQAWithSourcesChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(
                search_kwargs={"k":6}
            ),
            combine_prompt=PROMPT
        )
        
        with st.spinner("Thinking..."):
            result=chain({"question":query},return_only_outputs=True)

        st.header("Answer")
        st.write(result["answer"])

        sources=result.get("sources","")
        if sources:
            st.subheader("Sources:")
            for source in sources.split("\n"):
                if source.strip():
                    st.write(source)
