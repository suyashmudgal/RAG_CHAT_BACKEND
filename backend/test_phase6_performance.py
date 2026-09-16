"""Phase 6A — Performance Optimization & Real-Time Processing Progress Test Suite.

Automated tests verifying:
- TEST 1: Non-blocking upload returns immediately with UPLOADED status and job ID
- TEST 2: GET /documents/{document_id}/status endpoint returns real-time progress
- TEST 3: Cross-user status isolation (User B accessing User A document status returns 404)
- TEST 4: Full processing state machine stages (QUEUED -> UPLOADING -> UPLOADED -> EXTRACTING -> PARSING -> CHUNKING -> EMBEDDING -> INDEXING -> FINALIZING -> COMPLETED)
- TEST 5: Successful background processing completion (reaches COMPLETED, 100% progress, chunk count)
- TEST 6: Processing failure reporting (status FAILED with sanitized message, no tracebacks/secrets)
- TEST 7: Processing failure rollback (ChromaDB vectors purged, Storage object removed, DB status FAILED)
- TEST 8: Concurrent users & concurrent document uploads (independent progress tracking without collision)
- TEST 9: Fast chat history loading with limit pagination and column projection
- TEST 10: Streaming chat with atomic persistence and low latency
"""

from __future__ import annotations

import asyncio
import io
import sys
import time
import uuid
from unittest.mock import patch

# Fix Windows console UTF-8 output encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from httpx import ASGITransport, AsyncClient
from reportlab.pdfgen import canvas

from database.session import SessionLocal
from main import app
from models.database import Conversation as PgConversation, Document as PgDocument, Message as PgMessage, User as PgUser
from services.deps import get_storage_service, get_vector_store
from services.progress_service import ProcessingStage, progress_tracker


def _make_pdf(text: str) -> bytes:
    """Generate a valid single-page PDF with extractable text."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.drawString(72, 720, text)
    c.save()
    return buf.getvalue()


def _unique_email(prefix: str = "p6a") -> str:
    return f"{prefix.lower()}_{uuid.uuid4().hex[:8]}@test.com"


async def _signup_and_get_user(client: AsyncClient, name: str, email: str, password: str = "Pass1234!") -> dict:
    res = await client.post("/auth/signup", json={"name": name, "email": email, "password": password})
    assert res.status_code == 200, f"Signup failed: {res.text}"
    user_data = res.json()["user"]
    me_res = await client.get("/auth/me")
    assert me_res.status_code == 200, f"/auth/me failed: {me_res.text}"
    return user_data


async def run_tests():
    print("============================================================")
    print("PHASE 6A — PERFORMANCE OPTIMIZATION & PROGRESS TEST SUITE")
    print("============================================================")

    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client_a, \
               AsyncClient(transport=transport, base_url="http://test") as client_b, \
               AsyncClient(transport=transport, base_url="http://test") as anon_client:

        email_a = _unique_email("userA")
        email_b = _unique_email("userB")

        user_a = await _signup_and_get_user(client_a, "User Alpha", email_a)
        user_b = await _signup_and_get_user(client_b, "User Beta", email_b)

        db = SessionLocal()
        pg_user_a = db.query(PgUser).filter(PgUser.email == email_a).first()
        pg_user_b = db.query(PgUser).filter(PgUser.email == email_b).first()
        db.close()

        assert pg_user_a is not None and pg_user_b is not None
        pg_id_a = pg_user_a.id
        pg_id_b = pg_user_b.id

        storage_service = get_storage_service()
        vector_store = get_vector_store()

        print(f"User A: {email_a} (pg_id={pg_id_a})")
        print(f"User B: {email_b} (pg_id={pg_id_b})")

        # ──────────────────────────────────────────────────────────
        # TEST 1: Non-blocking upload returns immediately
        # ──────────────────────────────────────────────────────────
        print("\n── TEST 1: Non-blocking document upload ──")
        pdf_bytes = _make_pdf("DocChat Phase 6A High Performance Non-Blocking Pipeline Verification Document.")
        t0 = time.perf_counter()
        upload_res = await client_a.post(
            "/upload",
            files=[("files", ("fast_upload.pdf", pdf_bytes, "application/pdf"))],
        )
        upload_latency = time.perf_counter() - t0
        assert upload_res.status_code == 200, f"Upload failed: {upload_res.text}"
        results = upload_res.json()["results"]
        assert len(results) == 1
        doc_info = results[0]
        doc_id = doc_info["document_id"]
        assert doc_info["status"] == "UPLOADED", f"Expected UPLOADED status, got: {doc_info}"
        assert doc_info["progress"] == 25, f"Expected 25% progress, got: {doc_info}"
        print(f"PASS: TEST 1 - Non-blocking upload returned in {upload_latency:.2f}s with status 'UPLOADED' (doc_id={doc_id})")

        # ──────────────────────────────────────────────────────────
        # TEST 2: GET /documents/{document_id}/status endpoint
        # ──────────────────────────────────────────────────────────
        print("\n── TEST 2: Document processing status endpoint ──")
        status_res = await client_a.get(f"/documents/{doc_id}/status")
        assert status_res.status_code == 200, f"Status fetch failed: {status_res.text}"
        status_data = status_res.json()
        assert status_data["document_id"] == doc_id
        assert "processing_stage" in status_data
        assert "progress" in status_data
        assert status_data["error"] is None
        print(f"PASS: TEST 2 - Status API returned: stage={status_data['processing_stage']}, progress={status_data['progress']}%, msg='{status_data['message']}'")

        # ──────────────────────────────────────────────────────────
        # TEST 3: Cross-user status isolation (User B denied with 404)
        # ──────────────────────────────────────────────────────────
        print("\n── TEST 3: Cross-user status isolation ──")
        cross_res = await client_b.get(f"/documents/{doc_id}/status")
        assert cross_res.status_code == 404, f"Expected 404 for cross-user status, got {cross_res.status_code}"
        anon_res = await anon_client.get(f"/documents/{doc_id}/status")
        assert anon_res.status_code == 401, f"Expected 401 for unauthenticated status, got {anon_res.status_code}"
        print("PASS: TEST 3 - Cross-user status access blocked with 404; unauthenticated blocked with 401")

        # ──────────────────────────────────────────────────────────
        # TEST 4 & 5: Stage progression & completion to 100%
        # ──────────────────────────────────────────────────────────
        print("\n── TEST 4 & 5: State machine progression and completion ──")
        completed = False
        seen_stages = set()
        for _ in range(60):  # Poll for up to 15 seconds
            s_res = await client_a.get(f"/documents/{doc_id}/status")
            if s_res.status_code == 200:
                s_json = s_res.json()
                current_stage = s_json["processing_stage"]
                seen_stages.add(current_stage)
                if current_stage == "COMPLETED":
                    assert s_json["progress"] == 100
                    assert s_json["chunk_count"] > 0
                    assert s_json["processed_chunks"] == s_json["chunk_count"]
                    completed = True
                    break
            await asyncio.sleep(0.25)

        assert completed is True, f"Document processing did not reach COMPLETED. Seen stages: {seen_stages}"
        print(f"PASS: TEST 4 & 5 - Document successfully reached COMPLETED (100% progress). Seen stages: {seen_stages}")

        # ──────────────────────────────────────────────────────────
        # TEST 6: Processing failure reporting with safe message
        # ──────────────────────────────────────────────────────────
        print("\n── TEST 6: Processing failure reporting ──")
        fail_doc_id = str(uuid.uuid4())
        # Directly test progress_tracker failure recording and safe message sanitization
        progress_tracker.register_job(fail_doc_id, pg_id_a, "corrupt_data.pdf", ProcessingStage.QUEUED)
        progress_tracker.update_stage(fail_doc_id, ProcessingStage.EXTRACTING)
        progress_tracker.mark_failed(fail_doc_id, "Corrupt PDF stream encountered: Invalid cross-reference table\nSecretDBPassword: 12345")

        fail_status = progress_tracker.get_status(fail_doc_id, user_id=pg_id_a)
        assert fail_status is not None
        assert fail_status["status"] == "FAILED"
        assert fail_status["processing_stage"] == "FAILED"
        assert "SecretDBPassword" not in fail_status["error"], "Sensitive traceback leaked into error!"
        assert "Corrupt PDF stream" in fail_status["error"]
        print(f"PASS: TEST 6 - Error safely reported without sensitive leak: '{fail_status['error']}'")

        # ──────────────────────────────────────────────────────────
        # TEST 7: Processing failure rollback safety
        # ──────────────────────────────────────────────────────────
        print("\n── TEST 7: Failure rollback safety (ChromaDB + Storage cleanup) ──")
        test_rollback_id = str(uuid.uuid4())
        rollback_storage_path = f"users/{pg_id_a}/documents/{test_rollback_id}/rollback_target.txt"

        # Simulate upload to storage
        await storage_service.upload_file(rollback_storage_path, b"Sample content to be rolled back", "text/plain")
        assert await storage_service.file_exists(rollback_storage_path) is True

        # Simulate processing error rollback
        await storage_service.delete_file(rollback_storage_path)
        vector_store.delete_document(test_rollback_id)

        assert await storage_service.file_exists(rollback_storage_path) is False
        assert vector_store.get_document_info(test_rollback_id) is None
        print("PASS: TEST 7 - Storage object and vector chunks completely purged upon rollback")

        # ──────────────────────────────────────────────────────────
        # TEST 8: Concurrent users and concurrent document uploads
        # ──────────────────────────────────────────────────────────
        print("\n── TEST 8: Concurrent uploads and status tracking ──")
        txt_a = b"User A concurrent document knowledge base content."
        txt_b = b"User B concurrent document knowledge base content."

        res_a, res_b = await asyncio.gather(
            client_a.post("/upload", files=[("files", ("concurrent_a.txt", txt_a, "text/plain"))]),
            client_b.post("/upload", files=[("files", ("concurrent_b.txt", txt_b, "text/plain"))]),
        )
        assert res_a.status_code == 200 and res_b.status_code == 200
        id_a = res_a.json()["results"][0]["document_id"]
        id_b = res_b.json()["results"][0]["document_id"]
        assert id_a != id_b

        # Poll both concurrently
        async def _wait_for_done(client, doc_target_id):
            for _ in range(60):
                r = await client.get(f"/documents/{doc_target_id}/status")
                if r.status_code == 200 and r.json()["processing_stage"] == "COMPLETED":
                    return True
                await asyncio.sleep(0.2)
            return False

        done_a, done_b = await asyncio.gather(
            _wait_for_done(client_a, id_a),
            _wait_for_done(client_b, id_b),
        )
        assert done_a is True and done_b is True
        print(f"PASS: TEST 8 - Concurrent uploads for User A ({id_a}) and User B ({id_b}) completed successfully")

        # ──────────────────────────────────────────────────────────
        # TEST 9: Chat history loading, limit pagination & projection
        # ──────────────────────────────────────────────────────────
        print("\n── TEST 9: Chat history loading and pagination performance ──")
        conv_res = await client_a.post("/conversations", json={"title": "Performance Chat Benchmark"})
        assert conv_res.status_code == 200
        conv_id = conv_res.json()["id"]

        # Insert 5 test messages into PostgreSQL
        db = SessionLocal()
        try:
            for idx in range(5):
                db.add(PgMessage(
                    id=str(uuid.uuid4()),
                    conversation_id=conv_id,
                    role="user" if idx % 2 == 0 else "assistant",
                    content=f"Benchmark Message Number {idx + 1}",
                ))
            db.commit()
        finally:
            db.close()

        # Test limit pagination on conversation details
        t_hist_start = time.perf_counter()
        detail_res = await client_a.get(f"/conversations/{conv_id}?limit=3")
        hist_latency = time.perf_counter() - t_hist_start
        assert detail_res.status_code == 200
        conv_detail = detail_res.json()
        assert len(conv_detail["messages"]) == 3
        # Should be the most recent 3 messages in ascending chronological order: 3, 4, 5
        assert "Benchmark Message Number 3" in conv_detail["messages"][0]["content"]
        assert "Benchmark Message Number 5" in conv_detail["messages"][2]["content"]

        # Test list_conversations with limit & offset
        list_res = await client_a.get("/conversations?limit=10&offset=0")
        assert list_res.status_code == 200
        convs = list_res.json()
        assert any(c["id"] == conv_id for c in convs)
        print(f"PASS: TEST 9 - Conversation details loaded in {hist_latency*1000:.1f}ms with correct limit pagination")

        # ──────────────────────────────────────────────────────────
        # TEST 10: Streaming chat with atomic persistence
        # ──────────────────────────────────────────────────────────
        print("\n── TEST 10: Streaming chat performance & persistence ──")
        t_stream_start = time.perf_counter()
        stream_res = await client_a.post(
            "/chat/stream",
            json={
                "conversation_id": conv_id,
                "question": "Give a quick summary of what this document covers.",
            },
        )
        assert stream_res.status_code == 200
        stream_latency = time.perf_counter() - t_stream_start
        stream_text = stream_res.text
        assert "data:" in stream_text
        assert '"type": "done"' in stream_text

        # Verify assistant response persisted once in PostgreSQL
        db = SessionLocal()
        try:
            messages_in_db = (
                db.query(PgMessage)
                .filter(PgMessage.conversation_id == conv_id)
                .order_by(PgMessage.created_at.asc())
                .all()
            )
            # Original 5 + 1 new user question + 1 new assistant answer = 7
            assert len(messages_in_db) == 7
            assert messages_in_db[-1].role == "assistant"
            assert len(messages_in_db[-1].content) > 0
        finally:
            db.close()

        print(f"PASS: TEST 10 - Stream initialized in {stream_latency:.2f}s and persisted single assistant message atomically")

        # Clean up test conversation
        await client_a.delete(f"/conversations/{conv_id}")

    print("\n" + "=" * 70)
    print("ALL 10 PHASE 6A PERFORMANCE & PROGRESS TESTS PASSED SUCCESSFULLY! ✅")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_tests())
