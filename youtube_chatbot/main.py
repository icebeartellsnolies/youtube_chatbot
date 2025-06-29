from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

# ---- Input schema ----
class QueryInput(BaseModel):
    video_url: str
    language: str = "en"
    question: str

# ---- LLM Setup ----
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = ChatGroq(model='llama3-8b-8192', temperature=0.3)

# ---- Utility to extract video ID ----
def extract_video_id(url: str) -> str:
    query = urlparse(url).query
    params = parse_qs(query)
    return params.get("v", [None])[0]

# ---- Prompt Template ----
prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables=['context', 'question']
)

# ---- Main Endpoint ----
@app.post("/ask-question")
def ask_question(input_data: QueryInput):
    video_id = extract_video_id(input_data.video_url)
    if not video_id:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")

    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=[input_data.language])
        transcript = ''.join(chunk['text'] for chunk in transcript_list)
    except TranscriptsDisabled:
        raise HTTPException(status_code=404, detail="No captions available for this video.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Chunking the transcript
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.create_documents([transcript])

    # Vector store setup
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 4})

    def format_docs(retrieved_docs):
        return '\n\n'.join(chunk.page_content for chunk in retrieved_docs)

    parallel_chain = RunnableParallel({
        'question': RunnablePassthrough(),
        'context': retriever | RunnableLambda(format_docs)
    })

    parser = StrOutputParser()
    main_chain = parallel_chain | prompt | llm | parser

    answer = main_chain.invoke(input_data.question)
    return {"answer": answer}
