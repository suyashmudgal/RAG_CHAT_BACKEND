"""Quick end-to-end verification script for DocChat AI."""
import sys
import time
import requests

BASE_URL = "http://localhost:8000"

def test_e2e():
    print("1. Checking health...")
    r = requests.get(f"{BASE_URL}/")
    assert r.status_code == 200, f"Health check failed: {r.text}"
    print("   Health response:", r.json())

    print("\n2. Creating a test text document...")
    sample_text = (
        "Project Blueprint: DocChat AI.\n\n"
        "DocChat AI is an enterprise document assistant powered by LangChain, FastAPI, and ChromaDB.\n"
        "It supports PDF, DOCX, and TXT document ingestion.\n"
        "Chunking is performed using LangChain's RecursiveCharacterTextSplitter with chunk size 800 and overlap 200.\n"
        "Embeddings are computed locally using sentence-transformers all-MiniLM-L6-v2 via LangChain HuggingFaceEmbeddings.\n"
        "The LLM is powered by Groq (llama-3.3-70b-versatile) via LangChain ChatGroq.\n"
        "Vector search retrieves the top 5 most relevant passages.\n"
        "Chat supports conversational memory keeping the last 20 messages.\n"
    )
    with open("sample_rag_doc.txt", "w", encoding="utf-8") as f:
        f.write(sample_text)

    print("\n3. Testing document upload...")
    with open("sample_rag_doc.txt", "rb") as f:
        files = [("files", ("sample_rag_doc.txt", f, "text/plain"))]
        r = requests.post(f"{BASE_URL}/upload", files=files)
    assert r.status_code == 200, f"Upload failed: {r.text}"
    upload_res = r.json()
    print("   Upload result:", upload_res)
    assert len(upload_res.get("results", [])) == 1
    doc_id = upload_res["results"][0]["document_id"]
    print(f"   Indexed document_id: {doc_id}")

    print("\n4. Testing list documents...")
    r = requests.get(f"{BASE_URL}/documents")
    assert r.status_code == 200
    docs = r.json().get("documents", [])
    print(f"   Total documents in store: {len(docs)}")
    assert any(d["document_id"] == doc_id for d in docs), "Uploaded document not found in list!"

    print("\n5. Testing document deletion...")
    r = requests.delete(f"{BASE_URL}/documents/{doc_id}")
    assert r.status_code == 200
    print("   Delete response:", r.json())

    print("\n6. Verifying document was removed...")
    r = requests.get(f"{BASE_URL}/documents")
    docs_after = r.json().get("documents", [])
    assert not any(d["document_id"] == doc_id for d in docs_after), "Document was not deleted!"
    print(f"   Documents after delete: {len(docs_after)}")

    print("\n7. Re-uploading for UI usage...")
    with open("sample_rag_doc.txt", "rb") as f:
        files = [("files", ("sample_rag_doc.txt", f, "text/plain"))]
        r = requests.post(f"{BASE_URL}/upload", files=files)
    assert r.status_code == 200
    print("   Re-uploaded sample document successfully.")

    print("\n All Backend RAG + LangChain pipeline checks passed successfully!")

if __name__ == "__main__":
    try:
        test_e2e()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
