import streamlit as st
import os
import io
import time
from dotenv import load_dotenv

# LangChain/LLM Components
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from operator import itemgetter
from langchain.schema import format_document
from typing import List

# --- CONFIGURATION & CONSTANTS ---
load_dotenv()

DB_DIR = "chroma_db"
CONTEXT_TURNS = 10 

# Ensure API keys are set before proceeding
if not os.getenv("OPENAI_API_KEY"):
    st.error("❌ OpenAI API Key not found. Please set the OPENAI_API_KEY environment variable in a .env file.")
    st.stop()
if not os.getenv("TAVILY_API_KEY"):
    st.warning("⚠️ Tavily API Key not found. Web search will be disabled. Set TAVILY_API_KEY for real-time answers.")

# LLM Prompt for INTERNAL RAG 
INTERNAL_RAG_PROMPT = """
You are OnboardAI, a highly accurate and trustworthy AI Knowledge Assistant for a modern organization.
Your goal is to answer onboarding and workflow-related questions STRICTLY based on the provided context (Internal Documents).
Every answer MUST be synthesized naturally and conversationally.
Every key factual statement MUST be followed by a clear, in-line citation to its source document and page number.

Use the exact citation format: [Source: Document_Name, p. Page_Number]. E.g., [Source: Employee Handbook.pdf, p. 4].
If the page number is 'N/A' (e.g., a Markdown file), use: [Source: Document_Name].

If the answer is NOT in the documents, state clearly and professionally: "I apologize, I cannot find that specific information in the current company knowledge base."

Context:
{context}
"""

# System Prompt for the Web-Augmented Generation
WEB_SYSTEM_PROMPT = """
You are OnboardAI, a highly accurate and trustworthy AI Assistant.
Your goal is to answer the user's question using the best available source.

- **Primary Source:** The documents provided in the 'Context' section (Internal Documents).
- **Secondary Source:** The 'Web Search Results' provided below for recent events.

- **If the answer is in the 'Context' (Internal Docs):** Answer based on the context and use the citation format: [Source: Document_Name, p. Page_Number].
- **If the answer is NOT in 'Context' but is in 'Web Search Results':** Use the web results to answer and cite with [Source: Web Search].
- **If the answer is in NEITHER:** State clearly and professionally: "I apologize, I cannot find that specific information in the current knowledge base or the web search results."

Context (Internal Documents):
{context}

Web Search Results (For real-time events):
{web_search}
"""


# --- INITIALIZATION AND STATE MANAGEMENT ---

@st.cache_resource(show_spinner="Initializing Models and Database...")
def initialize_rag_components():
    """Initializes the LLM, Embeddings, Vector Store, and Search Tool."""
    
    # 1. Embeddings Model
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # 2. Vector Store (ChromaDB with persistence)
    vectorstore = Chroma(
        collection_name="onboarding_knowledge",
        embedding_function=embedding_model,
        persist_directory=DB_DIR
    )
    
    # 3. LLM for Generation
    llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0, streaming=True)
    
    # 4. Web Search Tool (Tavily)
    if os.getenv("TAVILY_API_KEY"):
        web_search_tool = TavilySearchResults(k=3, max_results=5)
    else:
        # Dummy runnable for missing key
        web_search_tool = RunnableLambda(lambda x: "Web search is disabled due to missing API key.")

    return llm, embedding_model, vectorstore, web_search_tool

# Function to clear the DB and Streamlit session state (UNCHANGED)
def clear_db():
    """Wipes the ChromaDB directory and resets the chat."""
    try:
        if os.path.exists(DB_DIR):
            import shutil
            shutil.rmtree(DB_DIR)
        
        st.session_state["messages"] = [{"role": "assistant", "content": "Knowledge base cleared. Please upload new documents."}]
        
        st.cache_resource.clear()
        initialize_rag_components() 
        st.success("✅ Knowledge base reset successfully!")
    except Exception as e:
        st.error(f"Error clearing DB: {e}")

# --- DOCUMENT INGESTION LOGIC (UNCHANGED) ---

def get_loader(file_path, file_type):
    """Returns the appropriate LangChain document loader."""
    if file_type == "pdf":
        return PyPDFLoader(file_path)
    elif file_type == "docx":
        return Docx2txtLoader(file_path)
    elif file_type in ["md", "txt"]:
        return TextLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

def ingest_documents(uploaded_files, vectorstore):
    """Loads, chunks, and indexes documents into the vector store."""
    if not uploaded_files:
        return 0

    documents = []
    
    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        file_type = file_name.split('.')[-1].lower()
        
        temp_file_path = f"./temp_{file_name}"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            loader = get_loader(temp_file_path, file_type)
            loaded_docs = loader.load()
            
            for doc in loaded_docs:
                doc.metadata['source'] = file_name
                page = doc.metadata.get('page', 'N/A')
                doc.metadata['page'] = str(page + 1) if isinstance(page, int) else str(page)
            documents.extend(loaded_docs)
            
        except Exception as e:
            st.error(f"Error loading {file_name}: {e}")
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    if not documents:
        return 0

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    
    vectorstore.add_documents(chunks)
    vectorstore.persist()
    
    return len(chunks)

# --- RAG CHAIN LOGIC ---

def create_rag_chain(llm, vectorstore, web_search_tool):
    """Builds the Conversational RAG chain with conditional web search and citation extraction."""
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4}) 
    parser = StrOutputParser() 

    # 2. Document Formatting for Context
    DOCUMENT_PROMPT_TEMPLATE = ChatPromptTemplate.from_template(
        "---Source: {source} (p. {page})---\n{page_content}"
    )
    def _combine_documents(docs, document_prompt=DOCUMENT_PROMPT_TEMPLATE, separator="\n\n"):
        doc_strings = [format_document(doc, document_prompt) for doc in docs]
        return separator.join(doc_strings)

    # 3. Contextualization (Rewriting follow-up questions)
    contextualize_q_system_prompt = (
        "Given the following conversation and a follow-up question, "
        "rephrase the follow-up question to be a standalone question for document retrieval. "
        "If the question is already standalone, do not change it."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )
    
    contextualize_q_chain = contextualize_q_prompt | llm | parser

    conditional_contextualizer = RunnableLambda(
        lambda x: contextualize_q_chain if x.get("chat_history") else x["question"]
    )

    # 4. Parallel Chains for Context Retrieval
    
    # a) Internal RAG Chain
    rag_context_chain = (
        RunnablePassthrough.assign(
            standalone_question=conditional_contextualizer,
        )
        | itemgetter("standalone_question")
        | retriever
        | _combine_documents
    ).with_config(run_name="Internal_RAG_Context_Retrieval")

    # b) Web Search Chain
    # 💥 FIX: Added a custom RunnableLambda to format the list of search results into a single string
    def format_web_results(results: List[dict]) -> str:
        """Formats the list of Tavily search results into a single string."""
        formatted = []
        for i, res in enumerate(results):
            # Format: 'Source 1: Title, URL, Content'
            formatted.append(f"Source {i+1}: {res.get('title', 'N/A')}\nURL: {res.get('url', 'N/A')}\nContent: {res.get('content', 'N/A')}")
        return "\n\n---\n\n".join(formatted)

    web_search_chain = (
        RunnablePassthrough.assign(
            standalone_question=conditional_contextualizer,
        )
        | itemgetter("standalone_question")
        | web_search_tool # TavilySearchResults tool returns List[dict]
        | RunnableLambda(format_web_results) # <-- CRITICAL FIX: Convert list to string
    ).with_config(run_name="Tavily_Web_Search")

    # c) Combine RAG and Web Search outputs in parallel
    combined_context_chain = RunnableParallel(
        context=rag_context_chain,
        web_search=web_search_chain,
        question=itemgetter("question"),
        chat_history=itemgetter("chat_history")
    )

    # 5. Final Generation Chain (RAG + Web)
    qa_prompt_web = ChatPromptTemplate.from_messages(
        [
            ("system", WEB_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    rag_chain = (
        combined_context_chain
        | qa_prompt_web
        | llm
    ).with_config(run_name="Final_LLM_Generation")
    
    return rag_chain

# --- STREAMLIT UI LAYOUT & EXECUTION (UNCHANGED) ---

def get_history_for_context(messages):
    """Formats the last N turns of the chat history for the RAG chain."""
    history = []
    history_subset = messages[-min(len(messages), CONTEXT_TURNS * 2 + 1):-1]
    
    for msg in history_subset:
        if msg["role"] == "user":
            history.append(("human", msg["content"]))
        elif msg["role"] == "assistant":
            history.append(("ai", msg["content"]))
    return history

def main():
    """Main function for the Streamlit application."""
    st.set_page_config(page_title="OnboardAI 🚀", layout="wide")
    st.title("🚀 OnboardAI - RAG Knowledge Assistant")
    
    llm, _, vectorstore, web_search_tool = initialize_rag_components()
    rag_chain = create_rag_chain(llm, vectorstore, web_search_tool)

    # Sidebar for Admin & Knowledge Base Management (UNCHANGED)
    with st.sidebar:
        st.header("Admin & Knowledge Base")
        
        with st.expander("📂 Document Ingestion", expanded=True):
            uploaded_files = st.file_uploader(
                "Upload PDFs, DOCX, or MD/TXT files", 
                type=["pdf", "docx", "md", "txt"],
                accept_multiple_files=True
            )
            if st.button("Ingest Documents & Update KB"):
                if uploaded_files:
                    with st.spinner(f"Ingesting {len(uploaded_files)} documents..."):
                        chunk_count = ingest_documents(uploaded_files, vectorstore)
                    if chunk_count > 0:
                        st.success(f"Successfully ingested {len(uploaded_files)} documents, creating {chunk_count} chunks.")
                    else:
                        st.warning("No new documents were processed.")
                else:
                    st.info("Please select documents to upload first.")

            st.markdown("---")
            if st.button("🚨 Clear All Data"):
                clear_db()

        st.markdown("---")
        st.subheader("Current KB Status")
        try:
            count = vectorstore._collection.count()
            st.info(f"Indexed Chunks: **{count}**")
        except Exception:
            st.warning("DB not yet initialized.")
            
        st.markdown("---")
        st.markdown("Powered by LLMs, Semantic Search, & Tavily.")

    # Main Chat Interface
    
    if "messages" not in st.session_state:
        initial_msg = "Hello! I'm OnboardAI, your company knowledge assistant. Upload your documents in the sidebar to start asking questions."
        st.session_state.messages = [{"role": "assistant", "content": initial_msg}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about policies, IT support, or project workflows..."):
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        history = get_history_for_context(st.session_state.messages)
        rag_input = {
            "question": prompt, 
            "chat_history": history
        }

        with st.chat_message("assistant"):
            response_container = st.empty() 
            try:
                stream = rag_chain.stream(rag_input)
                full_response = ""
                
                for chunk in stream:
                    content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                    full_response += content
                    response_container.markdown(full_response)
                
                if not full_response:
                    full_response = "I received an empty response from the model. Please check the model status."
                    response_container.markdown(full_response)

            except Exception as e:
                error_msg = f"An API or model error occurred: {e}"
                st.error(error_msg)
                full_response = f"I encountered an error. Please check the API keys (OpenAI and Tavily) and database status. Error: ({e})"
                response_container.markdown(full_response)


        st.session_state.messages.append({"role": "assistant", "content": full_response})
        
if __name__ == "__main__":
    if not os.path.exists(DB_DIR):
        os.makedirs(DB_DIR) 
    main()