from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

persistent_directory = "db/chroma_db"

# --------------------------------------------------
# Load embedding model
# --------------------------------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# --------------------------------------------------
# Load ChromaDB
# --------------------------------------------------

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

# --------------------------------------------------
# Create retriever
# --------------------------------------------------

retriever = db.as_retriever(
    search_kwargs={"k": 5}
)

# --------------------------------------------------
# Initialize Groq LLM
# --------------------------------------------------

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

# --------------------------------------------------
# Terminal Question-Answer Loop
# --------------------------------------------------

print("\n========================================")
print("       RAG Document Q&A System")
print("========================================")
print("Ask questions about your documents.")
print("Type 'exit' or 'quit' to stop.\n")

while True:

    # Get question from terminal
    query = input("You: ")

    # Exit condition
    if query.lower() in ["exit", "quit"]:
        print("\nGoodbye!")
        break

    # Handle empty input
    if not query.strip():
        print("Please enter a question.\n")
        continue

    # --------------------------------------------------
    # Retrieve relevant documents
    # --------------------------------------------------

    relevant_docs = retriever.invoke(query)

    # --------------------------------------------------
    # Display retrieved context
    # --------------------------------------------------

    print("\n--- Retrieved Context ---")

    for i, doc in enumerate(relevant_docs, 1):
        print(f"\nDocument {i}:")
        print(doc.page_content)

    # --------------------------------------------------
    # Create prompt
    # --------------------------------------------------

    combined_input = f"""
Based on the following documents, answer the user's question.

Question:
{query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

Instructions:
- Answer using only the information provided in the documents.
- Give a clear and helpful answer.
- If the answer cannot be found in the documents, say:
  "I don't have enough information to answer that question
   based on the provided documents."
"""

    # --------------------------------------------------
    # Create messages
    # --------------------------------------------------

    messages = [
        SystemMessage(
            content="You are a helpful assistant that answers questions using provided documents."
        ),
        HumanMessage(
            content=combined_input
        )
    ]

    # --------------------------------------------------
    # Generate answer
    # --------------------------------------------------

    result = model.invoke(messages)

    # --------------------------------------------------
    # Display answer
    # --------------------------------------------------

    print("\n--- Answer ---")
    print(result.content)
    print("\n" + "=" * 50 + "\n")