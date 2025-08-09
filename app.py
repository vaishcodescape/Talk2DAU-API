#Talk2DAU GPT-API

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import time
from dotenv import load_dotenv
from typing import Dict, List, Optional

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load environment variables
load_dotenv()

# Set LangSmith environment variables
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"] = "DAu Chatbot"

# Retrieve Groq API key
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize FastAPI app
app = FastAPI(title="Talk2DAU API", description="AI-powered chatbot for DA-IICT queries")

# Configure CORS middleware for Next.js integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://talk2-dau.vercel.app/",  # Replace with your production domain
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Pydantic models for API requests and responses
class QuestionRequest(BaseModel):
    question: str

class QuestionResponse(BaseModel):
    answer: str
    response_time: float
    source_chunks: Optional[List[Dict]] = None

class StatusResponse(BaseModel):
    status: str
    message: str
    documents_loaded: Optional[int] = None
    chunks_created: Optional[int] = None

# Global variables for vector storage
vectors = None
embeddings = None
docs = None
final_documents = None

# Initialize LLM
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama3-70b-8192"
)

# Prompt template
prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the provided context.
<context>
{context}
</context>

Question: {input}
""")

# Function to handle embedding and vector storage
def initialize_vector_database():
    global vectors, embeddings, docs, final_documents
    
    try:
        if vectors is None:
            # Initialize embeddings
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            
            # Load PDF documents
            loader = PyPDFDirectoryLoader("files")
            docs = loader.load()
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            final_documents = text_splitter.split_documents(docs)
            
            # Create vector store
            vectors = FAISS.from_documents(final_documents, embeddings)
            
            return {
                "status": "success",
                "documents_loaded": len(docs),
                "chunks_created": len(final_documents)
            }
        else:
            return {
                "status": "already_initialized",
                "documents_loaded": len(docs) if docs else 0,
                "chunks_created": len(final_documents) if final_documents else 0
            }
    except Exception as e:
        return {"status": "error", "message": str(e)}

# FastAPI Endpoints

@app.get("/", response_model=Dict)
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Talk2DAU API - AI-powered chatbot for DAU queries",
        "note": "This Bot is not affiliated with Dhirubhai Ambani University.",
        "website": "https://www.daiict.ac.in",
        "warning": "Misinformation can be generated! For more information visit the official website."
    }

@app.post("/initialize", response_model=StatusResponse)
async def initialize_bot():
    """Initialize the vector database with PDF documents"""
    result = initialize_vector_database()
    
    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["message"])
    
    return StatusResponse(
        status=result["status"],
        message="Bot initialized successfully" if result["status"] == "success" else "Bot already initialized",
        documents_loaded=result.get("documents_loaded"),
        chunks_created=result.get("chunks_created")
    )

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question to the chatbot"""
    global vectors
    
    if vectors is None:
        raise HTTPException(
            status_code=400, 
            detail="Bot not initialized. Please call /initialize endpoint first."
        )
    
    try:
        # Create retrieval chain
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = vectors.as_retriever(search_kwargs={"k": 10})
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        # Process question
        start = time.process_time()
        response = retrieval_chain.invoke({'input': request.question})
        end = time.process_time()
        
        # Get source documents for transparency
        relevant_docs = retriever.get_relevant_documents(request.question)
        source_chunks = [
            {
                "chunk_id": i + 1,
                "source": doc.metadata.get('source', 'unknown'),
                "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
            }
            for i, doc in enumerate(relevant_docs[:5])  # Limit to first 5 chunks
        ]
        
        return QuestionResponse(
            answer=response['answer'],
            response_time=round(end - start, 2),
            source_chunks=source_chunks
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get the current status of the bot"""
    global vectors, docs, final_documents
    
    if vectors is None:
        return StatusResponse(
            status="not_initialized",
            message="Bot is not initialized. Call /initialize endpoint first."
        )
    else:
        return StatusResponse(
            status="initialized",
            message="Bot is ready to answer questions",
            documents_loaded=len(docs) if docs else 0,
            chunks_created=len(final_documents) if final_documents else 0
        )

# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)