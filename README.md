# 🧠 Document Intelligence System using RAG (LangChain + Gemini)

## 📌 Overview
This project is an AI-powered document question-answering system that enables users to interact with PDF documents using natural language. It leverages **Retrieval-Augmented Generation (RAG)** to provide accurate and context-aware answers by combining semantic search with Large Language Models.

---

## 🚀 Features
- 📄 Upload and process PDF documents  
- 🔍 Semantic search using vector embeddings (Chroma DB)  
- 🤖 Context-aware question answering using Gemini LLM  
- 🧠 Reduced hallucination via retrieval-based grounding  
- ⚡ Fast and interactive UI using Streamlit  

---

## 🏗️ Architecture
User Query → Embedding → Chroma Vector DB → Relevant Chunks → Gemini LLM → Response


---

## ⚙️ Tech Stack
- **Language:** Python  
- **Framework:** LangChain  
- **LLM:** Gemini (Google Generative AI)  
- **Vector Database:** Chroma DB  
- **Embeddings:** Gemini Embeddings  
- **Frontend:** Streamlit  

---

## 🔄 Workflow

### 1. Document Ingestion
- Upload PDF files  
- Extract text and split into smaller chunks  

### 2. Embedding Generation
- Convert text chunks into vector embeddings  

### 3. Vector Storage
- Store embeddings in Chroma DB for fast retrieval  

### 4. Query Processing
- Convert user query into embedding  
- Retrieve most relevant chunks  

### 5. Answer Generation
- Pass retrieved context + query to Gemini  
- Generate accurate, grounded response  

---

## 🧪 Example

**User Query:**
> Summarize the key points from this document

**System Output:**
> Context-aware summary generated using retrieved document sections

---

## 🧠 Prompt Engineering

Example prompt used to reduce hallucination:
Answer the question based only on the provided context.
If the answer is not present, respond with "I don't know".

Context:
{retrieved_chunks}

Question:
{user_query}


---

## 📊 Key Highlights
- Improves answer accuracy using retrieval-based grounding  
- Minimizes hallucinations compared to standalone LLMs  
- Efficient semantic search with Chroma DB  
- Scalable architecture for multi-document support  

---

---

## ▶️ How to Run

```bash
git clone https://github.com/Rohit-Shere/RAG.git
cd RAG

pip install -r requirements.txt

streamlit run app.py
```

## 👨‍💻 Author

Rohit Shere
