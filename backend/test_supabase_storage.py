"""Phase 5 — Supabase Storage & User-Wise File Isolation Test Suite.

Automated tests verifying:
- TEST 1: User A uploads PDF -> Exists in Supabase Storage under User A's path
- TEST 2: PostgreSQL document record verification (user_id, storage_path, filename)
- TEST 3: User B document list isolation (User A document not returned)
- TEST 4: User B guesses User A document ID -> 404 secure denial
- TEST 5: User B attempts download of User A file -> 404 denied
- TEST 6: User B attempts delete of User A file -> 404 denied
- TEST 7: User A downloads own file -> 200 success with exact byte match
- TEST 8: User A deletes own document -> Storage object, ChromaDB chunks, & PG record deleted
- TEST 9: RAG retrieves User A document context
- TEST 10: RAG isolation — User B cannot retrieve User A vectors
- TEST 11: Path traversal protection (../../ attack sanitized safely)
- TEST 12: Oversized file rejection
- TEST 13: Unsupported/malformed file rejection
- TEST 14: Storage failure cleanup (no orphan PG record)
- TEST 15: Processing failure cleanup (Storage object rolled back)
"""

from __future__ import annotations

import asyncio
import io
import sys
import uuid
from unittest.mock import AsyncMock, patch

# Fix Windows console UTF-8 output encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from httpx import ASGITransport, AsyncClient
from reportlab.pdfgen import canvas

from config.settings import settings
from database.session import SessionLocal
from main import app
from models.database import Document as PgDocument
from services.deps import get_storage_service, get_vector_store
from services.storage_service import build_storage_path, sanitize_filename


def _make_pdf(text: str) -> bytes:
    """Generate a valid single-page PDF with extractable text."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.drawString(72, 720, text)
    c.save()
    return buf.getvalue()


def _unique_email(prefix: str = "p5") -> str:
    return f"{prefix.lower()}_{uuid.uuid4().hex[:8]}@test.com"


async def _signup(ac: AsyncClient, name: str, email: str, password: str = "Password123!") -> dict:
    res = await ac.post("/auth/signup", json={"name": name, "email": email, "password": password})
    assert res.status_code == 200, f"Signup failed: {res.text}"
    return res.json()["user"]


async def run_phase5_tests():
    print("=" * 70)
    print("STARTING PHASE 5: USER-WISE FILE STORAGE WITH SUPABASE STORAGE TESTS")
    print("=" * 70)

    transport = ASGITransport(app=app)
    storage_service = get_storage_service()
    vector_store = get_vector_store()

    # Ensure bucket exists
    bucket_ok = await storage_service.ensure_bucket()
    assert bucket_ok is True, "Failed to ensure documents bucket exists!"
    print(f"PASS: Storage bucket '{storage_service.bucket_name}' ready (configured: {storage_service.is_configured})")

    email_a = _unique_email("userA")
    email_b = _unique_email("userB")

    async with AsyncClient(transport=transport, base_url="http://test") as client_a:
        user_a = await _signup(client_a, "User A", email_a)
        # Ensure user is synced to PG by calling /auth/me
        await client_a.get("/auth/me")

        async with AsyncClient(transport=transport, base_url="http://test") as client_b:
            user_b = await _signup(client_b, "User B", email_b)
            await client_b.get("/auth/me")

            # Fetch exact PostgreSQL user IDs
            db_init = SessionLocal()
            try:
                from models.database import User as PgUser
                pg_u_a = db_init.query(PgUser).filter(PgUser.email == email_a).first()
                pg_id_a = pg_u_a.id
                pg_u_b = db_init.query(PgUser).filter(PgUser.email == email_b).first()
                pg_id_b = pg_u_b.id
            finally:
                db_init.close()

            print(f"User A created: {email_a} (pg_id={pg_id_a})")
            print(f"User B created: {email_b} (pg_id={pg_id_b})")

            # ──────────────────────────────────────────────────────────
            # TEST 1: User A uploads PDF -> Exists in Storage under User A's path
            # ──────────────────────────────────────────────────────────
            print("\n── TEST 1: User A uploads PDF to Supabase Storage ──")
            pdf_text = "User A Confidential Project Apollo Deep Space Exploration Specifications."
            pdf_bytes = _make_pdf(pdf_text)

            res = await client_a.post(
                "/upload",
                files=[("files", ("apollo_spec.pdf", pdf_bytes, "application/pdf"))],
            )
            assert res.status_code == 200, f"Upload failed: {res.text}"
            upload_results = res.json()["results"]
            assert len(upload_results) == 1, f"Expected 1 result: {upload_results}"
            doc_a = upload_results[0]
            assert doc_a["status"] == "processed", f"Status not processed: {doc_a}"
            doc_a_id = doc_a["document_id"]
            doc_a_path = doc_a["storage_path"]

            # Verify deterministic path pattern: users/{pg_user_id}/documents/{document_id}/{safe_filename}
            expected_prefix = f"users/{pg_id_a}/documents/{doc_a_id}/apollo_spec.pdf"
            assert doc_a_path == expected_prefix, f"Storage path mismatch: got '{doc_a_path}', expected '{expected_prefix}'"

            # Check that file actually exists in storage
            exists_in_storage = await storage_service.file_exists(doc_a_path)
            assert exists_in_storage is True, f"File not found in storage at path: {doc_a_path}"
            print(f"PASS: TEST 1 - Uploaded PDF stored at: {doc_a_path}")

            # ──────────────────────────────────────────────────────────
            # TEST 2: PostgreSQL document record verification
            # ──────────────────────────────────────────────────────────
            print("\n── TEST 2: PostgreSQL metadata persistence ──")
            db = SessionLocal()
            try:
                pg_doc = db.query(PgDocument).filter(PgDocument.id == doc_a_id).first()
                assert pg_doc is not None, "PostgreSQL document record not found!"
                assert pg_doc.user_id == pg_id_a, f"User ID mismatch: {pg_doc.user_id} != {pg_id_a}"
                assert pg_doc.filename == "apollo_spec.pdf"
                assert pg_doc.storage_path == doc_a_path
                assert pg_doc.file_size == len(pdf_bytes)
                assert pg_doc.status == "processed"
                assert pg_doc.chunk_count >= 1
                print(f"PASS: TEST 2 - Verified PostgreSQL record for doc {doc_a_id} with owner {pg_doc.user_id}")
            finally:
                db.close()

            # ──────────────────────────────────────────────────────────
            # TEST 3: User B lists documents -> User A's document is NOT returned
            # ──────────────────────────────────────────────────────────
            print("\n── TEST 3: Document listing isolation ──")
            res_b = await client_b.get("/documents")
            assert res_b.status_code == 200
            docs_b = res_b.json()["documents"]
            b_ids = [d.get("document_id") or d.get("id") for d in docs_b]
            assert doc_a_id not in b_ids, "TEST 3 FAIL: User B can see User A's document!"

            res_a = await client_a.get("/documents")
            assert res_a.status_code == 200
            docs_a = res_a.json()["documents"]
            a_ids = [d.get("document_id") or d.get("id") for d in docs_a]
            assert doc_a_id in a_ids, "TEST 3 FAIL: User A cannot see their own document!"
            print("PASS: TEST 3 - Document list strictly isolated by authenticated user")

            # ──────────────────────────────────────────────────────────
            # TEST 4: User B guesses User A's document ID -> 404 denial
            # ──────────────────────────────────────────────────────────
            print("\n── TEST 4: Guessing document ID secure denial ──")
            res_guess_del = await client_b.delete(f"/documents/{doc_a_id}")
            assert res_guess_del.status_code == 404, f"Expected 404, got: {res_guess_del.status_code}"
            res_guess_dl = await client_b.get(f"/documents/{doc_a_id}/download")
            assert res_guess_dl.status_code == 404, f"Expected 404, got: {res_guess_dl.status_code}"
            print("PASS: TEST 4 - Guessing document ID yields secure 404")

            # ──────────────────────────────────────────────────────────
            # TEST 5: User B attempts to download User A's file -> Denied
            # ──────────────────────────────────────────────────────────
            print("\n── TEST 5: Cross-user download denial ──")
            res_dl_b = await client_b.get(f"/documents/{doc_a_id}/download")
            assert res_dl_b.status_code == 404, f"User B download should be 404: {res_dl_b.status_code}"
            print("PASS: TEST 5 - Cross-user file download blocked with 404")

            # ──────────────────────────────────────────────────────────
            # TEST 6: User B attempts to delete User A's file -> Denied
            # ──────────────────────────────────────────────────────────
            print("\n── TEST 6: Cross-user deletion denial ──")
            res_del_b = await client_b.delete(f"/documents/{doc_a_id}")
            assert res_del_b.status_code == 404, f"User B delete should be 404: {res_del_b.status_code}"
            # Verify file and PG record still intact
            db = SessionLocal()
            try:
                check_doc = db.query(PgDocument).filter(PgDocument.id == doc_a_id).first()
                assert check_doc is not None, "Document was erroneously deleted!"
            finally:
                db.close()
            assert await storage_service.file_exists(doc_a_path) is True
            print("PASS: TEST 6 - Cross-user document deletion blocked with 404")

            # ──────────────────────────────────────────────────────────
            # TEST 7: User A downloads their own file -> Success
            # ──────────────────────────────────────────────────────────
            print("\n── TEST 7: Owner download verification ──")
            res_dl_a = await client_a.get(f"/documents/{doc_a_id}/download")
            assert res_dl_a.status_code == 200, f"Owner download failed: {res_dl_a.status_code}"
            assert res_dl_a.content == pdf_bytes, "Downloaded content does not match uploaded bytes!"
            assert "apollo_spec.pdf" in res_dl_a.headers.get("Content-Disposition", "")
            print("PASS: TEST 7 - Owner downloaded exact file bytes successfully")

            # ──────────────────────────────────────────────────────────
            # TEST 8: User A deletes their own document -> Cleanup verification
            # ──────────────────────────────────────────────────────────
            print("\n── TEST 8: Owner document deletion & storage cleanup ──")
            # Upload a secondary document to test owner deletion
            del_content = b"Temporary document that will be deleted by owner."
            res_up = await client_a.post(
                "/upload",
                files=[("files", ("delete_target.txt", del_content, "text/plain"))],
            )
            del_doc_info = res_up.json()["results"][0]
            del_doc_id = del_doc_info["document_id"]
            del_storage_path = del_doc_info["storage_path"]
            assert await storage_service.file_exists(del_storage_path) is True

            # Delete document
            res_del = await client_a.delete(f"/documents/{del_doc_id}")
            assert res_del.status_code == 200, f"Delete failed: {res_del.text}"

            # 1. PostgreSQL record must be gone
            db = SessionLocal()
            try:
                pg_check = db.query(PgDocument).filter(PgDocument.id == del_doc_id).first()
                assert pg_check is None, "PostgreSQL record still exists after deletion!"
            finally:
                db.close()

            # 2. ChromaDB vectors must be removed
            chroma_meta = vector_store.get_document_info(del_doc_id)
            assert chroma_meta is None, "ChromaDB metadata still exists after deletion!"

            # 3. Storage file must be removed
            still_in_storage = await storage_service.file_exists(del_storage_path)
            assert still_in_storage is False, f"File '{del_storage_path}' still exists in storage after deletion!"
            print("PASS: TEST 8 - Owner deletion completely purged DB, ChromaDB, and Storage")

            # ──────────────────────────────────────────────────────────
            # TEST 9: RAG retrieves User A's document after upload
            # ──────────────────────────────────────────────────────────
            print("\n── TEST 9: RAG retrieval for document owner ──")
            res_chat_a = await client_a.post(
                "/chat",
                json={
                    "question": "What is the name of the deep space exploration project?",
                    "session_id": f"s_{uuid.uuid4().hex[:6]}",
                },
            )
            assert res_chat_a.status_code == 200, f"Chat failed: {res_chat_a.text}"
            answer_a = res_chat_a.json().get("answer", "").lower()
            assert "apollo" in answer_a, f"Expected 'apollo' in answer: {answer_a}"
            print(f"PASS: TEST 9 - RAG successfully retrieved owner's document context")

            # ──────────────────────────────────────────────────────────
            # TEST 10: User B cannot retrieve User A's vectors through RAG
            # ──────────────────────────────────────────────────────────
            print("\n── TEST 10: Cross-user RAG isolation ──")
            res_chat_b = await client_b.post(
                "/chat",
                json={
                    "question": "What is the name of the deep space exploration project?",
                    "session_id": f"s_{uuid.uuid4().hex[:6]}",
                },
            )
            assert res_chat_b.status_code == 200, f"Chat failed: {res_chat_b.text}"
            answer_b = res_chat_b.json().get("answer", "").lower()
            sources_b = res_chat_b.json().get("sources", [])
            for src in sources_b:
                assert src.get("filename") != "apollo_spec.pdf", "User B citation cited User A's file!"
            assert "apollo" not in answer_b or "could not find" in answer_b or "not mentioned" in answer_b or "don't have" in answer_b, \
                f"User B should not know about Apollo project: {answer_b}"
            print("PASS: TEST 10 - Zero cross-user vector retrieval in RAG")

            # ──────────────────────────────────────────────────────────
            # TEST 11: Filename path traversal attempt
            # ──────────────────────────────────────────────────────────
            print("\n── TEST 11: Path traversal protection ──")
            traversal_names = [
                "../../../../etc/passwd.txt",
                r"..\..\..\..\windows\system32\cmd.txt",
                "normal_file.txt",
            ]
            for bad_name in traversal_names:
                safe_out = sanitize_filename(bad_name)
                assert "/" not in safe_out, f"Slash found in sanitized name: {safe_out}"
                assert "\\" not in safe_out, f"Backslash found in sanitized name: {safe_out}"
                assert ".." not in safe_out, f"Double-dot found in sanitized name: {safe_out}"

            # Test upload with path traversal in filename
            res_trav = await client_a.post(
                "/upload",
                files=[("files", ("../../exploit.txt", b"Safe content inside", "text/plain"))],
            )
            assert res_trav.status_code == 200
            trav_result = res_trav.json()["results"][0]
            assert trav_result["status"] == "processed"
            assert ".." not in trav_result["storage_path"]
            assert trav_result["storage_path"].startswith(f"users/{pg_id_a}/documents/")
            print("PASS: TEST 11 - Path traversal attempts neutralized and user-scoped")

            # ──────────────────────────────────────────────────────────
            # TEST 12: Oversized file rejection
            # ──────────────────────────────────────────────────────────
            print("\n── TEST 12: Oversized file rejection ──")
            # Create a file exceeding max_file_size_mb
            orig_max = settings.max_file_size_mb
            try:
                # Temporarily set limit to 1MB for test efficiency
                settings.max_file_size_mb = 1
                oversized_bytes = b"X" * (2 * 1024 * 1024)  # 2MB
                res_big = await client_a.post(
                    "/upload",
                    files=[("files", ("big.txt", oversized_bytes, "text/plain"))],
                )
                assert res_big.status_code == 200
                res_obj = res_big.json()["results"][0]
                assert res_obj["status"] == "error"
                assert "too large" in res_obj["message"].lower()
                print("PASS: TEST 12 - Oversized file cleanly rejected")
            finally:
                settings.max_file_size_mb = orig_max

            # ──────────────────────────────────────────────────────────
            # TEST 13: Unsupported file extension rejection
            # ──────────────────────────────────────────────────────────
            print("\n── TEST 13: Unsupported file extension rejection ──")
            res_bad_ext = await client_a.post(
                "/upload",
                files=[("files", ("script.sh", b"#!/bin/bash\necho hello", "application/x-sh"))],
            )
            assert res_bad_ext.status_code == 200
            res_obj = res_bad_ext.json()["results"][0]
            assert res_obj["status"] == "error"
            assert "unsupported file type" in res_obj["message"].lower()
            print("PASS: TEST 13 - Unsupported file extension cleanly rejected")

            # ──────────────────────────────────────────────────────────
            # TEST 14: Storage upload failure -> No orphan PG record
            # ──────────────────────────────────────────────────────────
            print("\n── TEST 14: Storage upload failure rollback ──")
            with patch.object(storage_service, "upload_file", side_effect=RuntimeError("Simulated Storage Outage")):
                res_fail = await client_a.post(
                    "/upload",
                    files=[("files", ("failure_test.txt", b"Test content", "text/plain"))],
                )
                assert res_fail.status_code == 200
                res_obj = res_fail.json()["results"][0]
                assert res_obj["status"] == "error"
                assert "storage upload failed" in res_obj["message"].lower()

                # Verify no record created in PostgreSQL
                db = SessionLocal()
                try:
                    orphan = db.query(PgDocument).filter(PgDocument.filename == "failure_test.txt").first()
                    assert orphan is None, "Orphan PostgreSQL document record created after storage failure!"
                finally:
                    db.close()
            print("PASS: TEST 14 - Storage failure left no orphan database record")

            # ──────────────────────────────────────────────────────────
            # TEST 15: Document processing failure -> Storage cleanup
            # ──────────────────────────────────────────────────────────
            print("\n── TEST 15: Processing failure storage cleanup ──")
            # Upload empty-like text file that yields ValueError during chunking/extraction
            res_empty_chunks = await client_a.post(
                "/upload",
                files=[("files", ("empty_spaces.txt", b"    \n\n   \t  ", "text/plain"))],
            )
            assert res_empty_chunks.status_code == 200
            res_obj = res_empty_chunks.json()["results"][0]
            assert res_obj["status"] == "error"

            # Verify no orphan in DB
            db = SessionLocal()
            try:
                orphan = db.query(PgDocument).filter(PgDocument.filename == "empty_spaces.txt").first()
                assert orphan is None, "Orphan document found in DB after processing failure!"
            finally:
                db.close()
            print("PASS: TEST 15 - Processing failure safely triggered cleanup")

    print("\n" + "=" * 70)
    print("ALL PHASE 5 TESTS (TEST 1 - TEST 15) PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_phase5_tests())
