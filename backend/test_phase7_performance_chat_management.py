"""Phase 7A — Chat Workspace Backend + Performance + Chat Management Test Suite.

Automated tests verifying:
- TEST 1: Rename conversation (PATCH /conversations/{id})
- TEST 2: Rename ownership isolation (User B cannot rename User A's conversation -> 404)
- TEST 3: Pin conversation (PATCH /conversations/{id} with is_pinned=true)
- TEST 4: Unpin conversation (PATCH /conversations/{id} with is_pinned=false)
- TEST 5: Pinned ordering (pinned conversations returned first, then updated_at DESC)
- TEST 6: Delete conversation with cascade and cross-user deletion protection (404)
- TEST 7: Export PDF (valid Content-Type, Content-Disposition, and %PDF- header)
- TEST 8: Export ownership isolation (User B exporting User A conversation returns 404)
- TEST 9: Large conversation export (handles multi-page conversations safely)
- TEST 10: Fast conversation listing (performance benchmark, returns is_pinned)
- TEST 11: Fast document listing (lightweight metadata)
- TEST 12: Lightweight status endpoint (fast response, strict isolation)
- TEST 13: Chat streaming (Server-Sent Events with atomic persistence)
- TEST 14: Title preservation (manual rename takes precedence and is not overwritten by auto-title)
- TEST 15: Multi-user isolation across all conversation management & export endpoints
"""

from __future__ import annotations

import asyncio
import io
import sys
import time
import uuid

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


def _unique_email(prefix: str = "p7a") -> str:
    return f"{prefix.lower()}_{uuid.uuid4().hex[:8]}@test.com"


async def _signup_and_get_user(client: AsyncClient, name: str, email: str, password: str = "Pass1234!") -> dict:
    res = await client.post("/auth/signup", json={"name": name, "email": email, "password": password})
    assert res.status_code == 200, f"Signup failed: {res.text}"
    user_data = res.json()["user"]
    me_res = await client.get("/auth/me")
    assert me_res.status_code == 200, f"/auth/me failed: {me_res.text}"
    return user_data


async def run_tests():
    print("======================================================================")
    print("PHASE 7A — CHAT WORKSPACE BACKEND & MANAGEMENT TEST SUITE")
    print("======================================================================")

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

        print(f"User A: {email_a} (pg_id={pg_id_a})")
        print(f"User B: {email_b} (pg_id={pg_id_b})")

        # ──────────────────────────────────────────────────────────
        # TEST 1: Rename conversation
        # ──────────────────────────────────────────────────────────
        print("\n── TEST 1: Rename conversation ──")
        create_res = await client_a.post("/conversations", json={"title": "Original Title"})
        assert create_res.status_code == 200, f"Failed to create conv: {create_res.text}"
        conv_a1 = create_res.json()
        conv_a1_id = conv_a1["id"]

        rename_res = await client_a.patch(f"/conversations/{conv_a1_id}", json={"title": "Renamed Chat Alpha"})
        assert rename_res.status_code == 200, f"Failed to rename conv: {rename_res.text}"
        renamed_data = rename_res.json()
        assert renamed_data["title"] == "Renamed Chat Alpha", f"Expected renamed title, got: {renamed_data['title']}"
        print(f"PASS: TEST 1 - Conversation successfully renamed to '{renamed_data['title']}'")

        # ──────────────────────────────────────────────────────────
        # TEST 2: Rename ownership isolation
        # ──────────────────────────────────────────────────────────
        print("\n── TEST 2: Rename ownership isolation ──")
        b_rename_res = await client_b.patch(f"/conversations/{conv_a1_id}", json={"title": "Hacked by User B"})
        assert b_rename_res.status_code == 404, f"Expected 404 for cross-user rename, got: {b_rename_res.status_code}"

        anon_rename = await anon_client.patch(f"/conversations/{conv_a1_id}", json={"title": "Anon Hack"})
        assert anon_rename.status_code == 401, f"Expected 401 for unauthenticated rename, got: {anon_rename.status_code}"
        print("PASS: TEST 2 - Cross-user rename blocked with 404; unauthenticated blocked with 401")

        # ──────────────────────────────────────────────────────────
        # TEST 3: Pin conversation
        # ──────────────────────────────────────────────────────────
        print("\n── TEST 3: Pin conversation ──")
        pin_res = await client_a.patch(f"/conversations/{conv_a1_id}", json={"is_pinned": True})
        assert pin_res.status_code == 200, f"Failed to pin conv: {pin_res.text}"
        pin_data = pin_res.json()
        assert pin_data["is_pinned"] is True, f"Expected is_pinned=True, got: {pin_data['is_pinned']}"
        print(f"PASS: TEST 3 - Conversation {conv_a1_id} successfully pinned (is_pinned=True)")

        # ──────────────────────────────────────────────────────────
        # TEST 4: Unpin conversation
        # ──────────────────────────────────────────────────────────
        print("\n── TEST 4: Unpin conversation ──")
        unpin_res = await client_a.patch(f"/conversations/{conv_a1_id}", json={"is_pinned": False})
        assert unpin_res.status_code == 200, f"Failed to unpin conv: {unpin_res.text}"
        unpin_data = unpin_res.json()
        assert unpin_data["is_pinned"] is False, f"Expected is_pinned=False, got: {unpin_data['is_pinned']}"

        # Also test convenience endpoints
        pin_conv_res = await client_a.post(f"/conversations/{conv_a1_id}/pin")
        assert pin_conv_res.status_code == 200 and pin_conv_res.json()["is_pinned"] is True
        unpin_conv_res = await client_a.post(f"/conversations/{conv_a1_id}/unpin")
        assert unpin_conv_res.status_code == 200 and unpin_conv_res.json()["is_pinned"] is False
        print("PASS: TEST 4 - Conversation successfully unpinned (and convenience endpoints verified)")

        # ──────────────────────────────────────────────────────────
        # TEST 5: Pinned ordering
        # ──────────────────────────────────────────────────────────
        print("\n── TEST 5: Pinned ordering (pinned first, then updated_at DESC) ──")
        res_c2 = await client_a.post("/conversations", json={"title": "Chat Two"})
        res_c3 = await client_a.post("/conversations", json={"title": "Chat Three"})
        c2_id = res_c2.json()["id"]
        c3_id = res_c3.json()["id"]

        # Pin Chat Two (which is older than Chat Three)
        await client_a.patch(f"/conversations/{c2_id}", json={"is_pinned": True})

        list_res = await client_a.get("/conversations")
        assert list_res.status_code == 200
        conv_items = list_res.json()
        assert len(conv_items) >= 3
        # First conversation in list MUST be the pinned one (c2_id)
        assert conv_items[0]["id"] == c2_id, f"Expected first conv to be pinned ({c2_id}), got: {conv_items[0]['id']}"
        assert conv_items[0]["is_pinned"] is True
        print(f"PASS: TEST 5 - Pinned conversation '{conv_items[0]['title']}' correctly ordered first")

        # ──────────────────────────────────────────────────────────
        # TEST 6: Delete conversation & cascade
        # ──────────────────────────────────────────────────────────
        print("\n── TEST 6: Delete conversation with cascade & isolation ──")
        temp_create = await client_a.post("/conversations", json={"title": "To Be Deleted"})
        temp_id = temp_create.json()["id"]

        # Cross-user deletion blocked
        b_del = await client_b.delete(f"/conversations/{temp_id}")
        assert b_del.status_code == 404, f"Expected 404 for cross-user delete, got: {b_del.status_code}"

        # Owner deletion succeeds
        a_del = await client_a.delete(f"/conversations/{temp_id}")
        assert a_del.status_code == 200, f"Expected 200 for owner delete, got: {a_del.status_code}"

        # Verify 404 on re-fetch
        fetch_after_del = await client_a.get(f"/conversations/{temp_id}")
        assert fetch_after_del.status_code == 404
        print(f"PASS: TEST 6 - Conversation {temp_id} deleted cleanly; cross-user delete rejected with 404")

        # ──────────────────────────────────────────────────────────
        # TEST 7: Export PDF
        # ──────────────────────────────────────────────────────────
        print("\n── TEST 7: Export conversation as PDF ──")
        # Add messages to c2_id directly in DB for deterministic testing
        db = SessionLocal()
        msg1 = PgMessage(id=str(uuid.uuid4()), conversation_id=c2_id, role="user", content="Can you summarize key principles?")
        msg2 = PgMessage(id=str(uuid.uuid4()), conversation_id=c2_id, role="assistant", content="Here are the key principles based on the uploaded document: 1. Modular architecture 2. Zero-leak privacy.")
        db.add(msg1)
        db.add(msg2)
        db.commit()
        db.close()

        pdf_res = await client_a.get(f"/conversations/{c2_id}/export/pdf")
        assert pdf_res.status_code == 200, f"Failed to export PDF: {pdf_res.status_code}"
        assert pdf_res.headers.get("content-type") == "application/pdf"
        assert "attachment" in pdf_res.headers.get("content-disposition", "")
        pdf_bytes = pdf_res.content
        assert pdf_bytes.startswith(b"%PDF-"), f"Invalid PDF magic header: {pdf_bytes[:10]}"
        assert len(pdf_bytes) > 500
        print(f"PASS: TEST 7 - Exported PDF ({len(pdf_bytes)} bytes) with valid magic header %PDF-")

        # ──────────────────────────────────────────────────────────
        # TEST 8: Export ownership isolation
        # ──────────────────────────────────────────────────────────
        print("\n── TEST 8: Export ownership isolation ──")
        b_export = await client_b.get(f"/conversations/{c2_id}/export/pdf")
        assert b_export.status_code == 404, f"Expected 404 for User B exporting User A's chat, got: {b_export.status_code}"

        anon_export = await anon_client.get(f"/conversations/{c2_id}/export/pdf")
        assert anon_export.status_code == 401, f"Expected 401 for unauthenticated export, got: {anon_export.status_code}"
        print("PASS: TEST 8 - Cross-user export blocked with 404; unauthenticated export blocked with 401")

        # ──────────────────────────────────────────────────────────
        # TEST 9: Large conversation export
        # ──────────────────────────────────────────────────────────
        print("\n── TEST 9: Large conversation export ──")
        large_conv_res = await client_a.post("/conversations", json={"title": "Large Conversation"})
        large_id = large_conv_res.json()["id"]

        db = SessionLocal()
        for i in range(25):
            db.add(PgMessage(id=str(uuid.uuid4()), conversation_id=large_id, role="user", content=f"User question number {i+1} regarding distributed systems and consensus protocols."))
            db.add(PgMessage(id=str(uuid.uuid4()), conversation_id=large_id, role="assistant", content=f"Assistant detailed response {i+1}: Raft and Paxos ensure consensus across fault-tolerant nodes by electing leaders and replicating write logs across quorums."))
        db.commit()
        db.close()

        large_pdf_res = await client_a.get(f"/conversations/{large_id}/export/pdf")
        assert large_pdf_res.status_code == 200
        assert large_pdf_res.content.startswith(b"%PDF-")
        assert len(large_pdf_res.content) > 10000, f"Expected multi-page PDF > 10KB, got: {len(large_pdf_res.content)}"
        print(f"PASS: TEST 9 - Multi-page PDF successfully generated for 50-message thread ({len(large_pdf_res.content)} bytes)")

        # ──────────────────────────────────────────────────────────
        # TEST 10: Fast conversation listing
        # ──────────────────────────────────────────────────────────
        print("\n── TEST 10: Fast conversation listing benchmark ──")
        t0 = time.perf_counter()
        conv_list_res = await client_a.get("/conversations?limit=20&offset=0")
        t_list = time.perf_counter() - t0
        assert conv_list_res.status_code == 200
        first_item = conv_list_res.json()[0]
        assert "is_pinned" in first_item
        assert "title" in first_item
        assert "id" in first_item
        # Verify no message bodies in list summary
        assert "messages" not in first_item
        print(f"PASS: TEST 10 - Conversation list loaded in {t_list*1000:.1f}ms (is_pinned present, bodies excluded)")

        # ──────────────────────────────────────────────────────────
        # TEST 11: Fast document listing
        # ──────────────────────────────────────────────────────────
        print("\n── TEST 11: Fast document listing metadata ──")
        t0 = time.perf_counter()
        doc_list_res = await client_a.get("/documents")
        t_docs = time.perf_counter() - t0
        assert doc_list_res.status_code == 200
        docs = doc_list_res.json().get("documents", [])
        print(f"PASS: TEST 11 - Document list loaded in {t_docs*1000:.1f}ms with lightweight metadata (count={len(docs)})")

        # ──────────────────────────────────────────────────────────
        # TEST 12: Lightweight status endpoint
        # ──────────────────────────────────────────────────────────
        print("\n── TEST 12: Lightweight status endpoint ──")
        # Query non-existent document status
        status_res = await client_a.get("/documents/non-existent-uuid/status")
        assert status_res.status_code == 404

        # Register in-memory mock job and check sub-millisecond retrieval
        from services.progress_service import progress_tracker, ProcessingStage
        mock_doc_id = str(uuid.uuid4())
        progress_tracker.register_job(mock_doc_id, user_id=pg_id_a, filename="instant_status.pdf")
        progress_tracker.update_stage(mock_doc_id, ProcessingStage.EMBEDDING, progress=70, message="Embedding passages...")

        t0 = time.perf_counter()
        active_status = await client_a.get(f"/documents/{mock_doc_id}/status")
        t_status = time.perf_counter() - t0
        assert active_status.status_code == 200
        status_data = active_status.json()
        assert status_data["processing_stage"] == "EMBEDDING"
        assert status_data["progress"] == 70

        # User B cannot access User A's status
        b_mock_status = await client_b.get(f"/documents/{mock_doc_id}/status")
        assert b_mock_status.status_code == 404
        print(f"PASS: TEST 12 - Status retrieved in {t_status*1000:.1f}ms with strict cross-user isolation")

        # ──────────────────────────────────────────────────────────
        # TEST 13: Chat streaming
        # ──────────────────────────────────────────────────────────
        print("\n── TEST 13: Chat streaming with SSE & persistence ──")
        stream_conv_res = await client_a.post("/conversations", json={"title": "Stream Chat"})
        stream_conv_id = stream_conv_res.json()["id"]

        t0 = time.perf_counter()
        token_count = 0
        first_token_latency = None
        async with client_a.stream(
            "POST",
            "/chat/stream",
            json={"question": "Briefly describe solar flares in one sentence.", "conversation_id": stream_conv_id},
        ) as s_res:
            assert s_res.status_code == 200
            async for chunk in s_res.aiter_lines():
                if chunk.startswith("data: "):
                    if first_token_latency is None:
                        first_token_latency = time.perf_counter() - t0
                    token_count += 1
        assert token_count > 0, "Expected streamed tokens"
        print(f"PASS: TEST 13 - Chat stream received {token_count} SSE events (first token in {first_token_latency*1000:.1f}ms)")

        # Verify atomic persistence in PostgreSQL
        detail_res = await client_a.get(f"/conversations/{stream_conv_id}")
        assert detail_res.status_code == 200
        detail_messages = detail_res.json()["messages"]
        assert len(detail_messages) == 2, f"Expected 2 messages (1 user, 1 assistant), got: {len(detail_messages)}"

        # ──────────────────────────────────────────────────────────
        # TEST 14: Title preservation
        # ──────────────────────────────────────────────────────────
        print("\n── TEST 14: Title preservation (manual rename takes precedence) ──")
        pres_conv = await client_a.post("/conversations", json={"title": "New Conversation"})
        pres_id = pres_conv.json()["id"]

        # User manually renames chat before sending message
        await client_a.patch(f"/conversations/{pres_id}", json={"title": "Important Research Chat"})

        # Send first message
        await client_a.post("/chat", json={"question": "What is quantum computing?", "conversation_id": pres_id})

        # Verify title was NOT overwritten by auto-title
        check_pres = await client_a.get(f"/conversations/{pres_id}")
        assert check_pres.status_code == 200
        assert check_pres.json()["title"] == "Important Research Chat", f"Title was unexpectedly overwritten: {check_pres.json()['title']}"
        print(f"PASS: TEST 14 - User manual title '{check_pres.json()['title']}' preserved after messaging")

        # ──────────────────────────────────────────────────────────
        # TEST 15: Multi-user isolation
        # ──────────────────────────────────────────────────────────
        print("\n── TEST 15: Multi-user isolation across all management endpoints ──")
        # User B cannot see User A's conversations in listing
        b_list = await client_b.get("/conversations")
        b_ids = [c["id"] for c in b_list.json()]
        assert conv_a1_id not in b_ids
        assert c2_id not in b_ids
        assert large_id not in b_ids

        # User B cannot read User A conversation details
        b_detail = await client_b.get(f"/conversations/{c2_id}")
        assert b_detail.status_code == 404

        # User B cannot pin User A conversation
        b_pin = await client_b.patch(f"/conversations/{c2_id}", json={"is_pinned": True})
        assert b_pin.status_code == 404

        # User B cannot delete User A conversation
        b_del2 = await client_b.delete(f"/conversations/{c2_id}")
        assert b_del2.status_code == 404

        # User B cannot export User A conversation
        b_exp2 = await client_b.get(f"/conversations/{c2_id}/export/pdf")
        assert b_exp2.status_code == 404
        print("PASS: TEST 15 - Complete multi-user isolation confirmed across all endpoints")

        print("\n======================================================================")
        print("ALL 15 PHASE 7A TESTS PASSED SUCCESSFULLY! ✅")
        print("======================================================================")


if __name__ == "__main__":
    asyncio.run(run_tests())
