## **AI\_RAG\_WEBURL\_Chatbot 🤖**

A RAG (Retrieval Augmented Generation) chatbot that loads any public web URL, understands the content, and answers your questions accurately using **Llama 3.1** via **Groq API**.

---

### **🏗️ Full Flow**

Step 1 — You enter a web URL (e.g. Wikipedia page)  
                ↓  
Step 2 — WebBaseLoader downloads and cleans the page  
                ↓  
Step 3 — RecursiveCharacterTextSplitter chunks text  
         into 500 character pieces with 150 overlap  
                ↓  
Step 4 — HuggingFace sentence-transformers converts  
         each chunk into 384 numbers (vectors)  
                ↓  
Step 5 — FAISS stores all vectors locally on your PC  
         saved as faiss\_store\_hf.pkl  
                ↓  
Step 6 — You type a question  
                ↓  
Step 7 — Your question also becomes a vector  
         FAISS finds top 6 most similar chunks  
                ↓  
Step 8 — Chunks \+ Question sent to Groq API  
         Llama 3.1 reads context and generates answer  
                ↓  
Step 9 — Answer displayed on Streamlit UI with sources  
---

### **🛠️ Tech Stack**

| Tool | Purpose |
| :---- | :---- |
| Streamlit | Web UI |
| LangChain | RAG pipeline framework |
| WebBaseLoader | Load and clean web URLs |
| RecursiveCharacterTextSplitter | Chunk large text |
| HuggingFace sentence-transformers | Convert text to vectors |
| FAISS | Vector database for similarity search |
| Groq API | Fast LLM inference |
| Llama 3.1 8B | Language model for answering questions |
| Python dotenv | Manage API keys securely |

---

### **✨ Features**

* Load up to 3 public web URLs simultaneously  
* Automatically chunks and embeds content  
* Fast answers using Groq LPU (Language Processing Unit)  
* Shows source URLs with every answer  
* Custom prompt engineering for accurate responses  
* Saves FAISS index to disk (no reprocessing needed)

---

### **🤔 Why Groq instead of OpenAI or HuggingFace?**

| Feature | Groq (Llama 3.1) | OpenAI (GPT-4) | HuggingFace (flan-t5) |
| :---- | :---- | ----- | ----- |
| Cost | ✅ Free | ❌ Paid ($$$) | ✅ Free |
| Speed | ✅ Very fast (LPU) | ✅ Fast | ❌ Slow |
| Accuracy | ✅ Very good | ✅✅ Best | ❌ Weak |
| Download needed | ✅ No | ✅ No | ❌ Yes (GBs) |
| RAM usage | ✅ Zero | ✅ Zero | ❌ 8-20GB |
| API Key | ✅ Free | ❌ Paid | ✅ Free (limited) |

**Groq wins because:** Free, zero RAM usage, no downloads, accurate answers, responds in 2 seconds using custom LPU chip.

---

### **🔄 Alternative LLM Options (instead of Groq)**

#### **Option 1 — OpenAI GPT (Paid)**

python  
from langchain\_openai import ChatOpenAI  
llm \= ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)

#### **Option 2 — Google Gemini (Free tier)**

python  
from langchain\_google\_genai import ChatGoogleGenerativeAI  
llm \= ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.1)

#### **Option 3 — Ollama (Fully Local, Free)**

python  
from langchain\_community.llms import Ollama  
llm \= Ollama(model="llama3")

#### **Option 4 — HuggingFace Local (Free, needs RAM)**

python  
from langchain\_community.llms import HuggingFacePipeline  
from transformers import pipeline  
pipe \= pipeline("text2text-generation", model="google/flan-t5-large")  
llm \= HuggingFacePipeline(pipeline=pipe)

#### **Option 5 — Anthropic Claude (Paid)**

python  
from langchain\_anthropic import ChatAnthropic  
llm \= ChatAnthropic(model="claude-3-haiku-20240307", temperature=0.1)  
---

### **🔑 How to Create Groq API Key (Free)**

1\. Go to         → https://console.groq.com  
2\. Click           "Sign Up" (free, no credit card)  
3\. After login   → click "API Keys" in left sidebar  
4\. Click           "Create API Key"  
5\. Name it         "rag-chatbot"  
6\. Click           "Submit"  
7\. COPY the key  → starts with gsk\_xxxxxxxxxx  
   ⚠️ Copy immediately — you won't see it again\!  
8\. Add to your   → .env file (see setup below)  
---

### **⚙️ Setup and Installation**

#### **Step 1 — Clone the repository**

bash  
git clone https://github.com/Ponraj1405/AI\_RAG\_WEBURL\_Chatbot.git  
cd AI\_RAG\_WEBURL\_Chatbot

#### **Step 2 — Create virtual environment**

bash  
python3.10 \-m venv rag\_env

#### **Step 3 — Activate virtual environment**

bash  
\# Linux / Mac:  
source rag\_env/bin/activate

\# Windows:  
rag\_env\\Scripts\\activate

#### **Step 4 — Install all dependencies**

bash  
pip install \-r requirements.txt

#### **Step 5 — Create .env file**

bash  
touch .env

Add this inside .env file:

GROQ\_API\_KEY=gsk\_xxxxxxxxxxxxxxxxxx

#### **Step 6 — Run the application**

bash  
streamlit run main.py

#### **Step 7 — Open in browser**

Local URL: http://localhost:8501  
---

### **📖 How to Use**

1\. Enter up to 3 public web URLs in the sidebar  
   Example: https://en.wikipedia.org/wiki/Elon\_Musk

2\. Click "Process URLs" button  
   Wait for "Done\! Processed X chunks" message

3\. Type your question in the text box  
   Example: "Where was Elon Musk born?"

4\. Get accurate answers with sources instantly\!  
---

### **✅ Supported URLs**

✅ Wikipedia pages  
✅ News articles (BBC, Reuters, TechCrunch)  
✅ Blog posts (Medium, Substack)  
✅ Documentation pages  
✅ Any simple public webpage

❌ PDF files  
❌ Login required pages (LinkedIn, Twitter)  
❌ Paywalled articles (NYT, WSJ)  
❌ YouTube videos  
❌ Google Docs  
---

### **📁 Project Structure**

AI\_RAG\_WEBURL\_Chatbot/  
├── main.py              ← main application code  
├── requirements.txt     ← all dependencies  
├── README.md             
├── .env                 ← your API key   
├── .gitignore           ← files to ignore in git  
└── faiss\_store\_hf.pkl   ← saved vector store (auto created)  
---

### **🧠 Concepts Used**

| Concept | What it means in this project |
| :---- | :---- |
| RAG | Fetch real URL content → answer from that content |
| Embeddings | Convert text chunks to 384 numbers for search |
| Vector Search | Find chunks most similar to your question |
| Chunking | Split large webpage into 500 char pieces |
| Prompt Engineering | Custom instructions for accurate answers |
| LPU | Groq's custom chip — faster than GPU for LLMs |

---

### **🚀 Built With**

* Python 3.10  
* LangChain 0.2.16  
* Groq API (Free)  
* Llama 3.1 8B Instant  
* FAISS  
* Streamlit  
* HuggingFace sentence-transformers

---

### **👨‍💻 Author**

Ponraj P

* GitHub: \[@Ponraj1405\](https://github.com/Ponraj1405)  
* LinkedIn: \[Ponraj P\](https://www.linkedin.com/in/ponraj-p-804368202/)


---
