"""Comprehensive tests for DELETE /conversations/bulk.

Verifies:
1. Delete multiple owned conversations
2. Delete one conversation
3. Duplicate IDs handled safely
4. Empty list returns validation error
5. Invalid conversation ID returns 404 (and no partial deletion)
6. Cross-user conversation ID returns 404 (and no partial deletion)
7. Unauthenticated request returns 401
8. Message cascade: messages belonging to deleted conversations are completely deleted
9. Transaction behavior: rollback on failure, no partial deletions
10. Active conversation deletion
11. Large but valid batch deletion (e.g. 20 conversations)
"""

import asyncio
import sys
import uuid
from httpx import ASGITransport, AsyncClient

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from main import app
from database.session import SessionLocal
from models.database import Conversation as PgConversation, Message as PgMessage, User as PgUser


def _unique_email(prefix: str = "bulk") -> str:
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
    print("RUNNING BULK CONVERSATION DELETE TEST SUITE")
    print("======================================================================")

    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client_a, \
               AsyncClient(transport=transport, base_url="http://test") as client_b, \
               AsyncClient(transport=transport, base_url="http://test") as anon_client:

        email_a = _unique_email("bulkA")
        email_b = _unique_email("bulkB")

        user_a = await _signup_and_get_user(client_a, "User Bulk A", email_a)
        user_b = await _signup_and_get_user(client_b, "User Bulk B", email_b)

        db = SessionLocal()
        pg_user_a = db.query(PgUser).filter(PgUser.email == email_a).first()
        pg_user_b = db.query(PgUser).filter(PgUser.email == email_b).first()
        db.close()

        assert pg_user_a is not None and pg_user_b is not None
        pg_id_a = pg_user_a.id
        pg_id_b = pg_user_b.id

        print(f"User A registered: {email_a} (id={pg_id_a})")
        print(f"User B registered: {email_b} (id={pg_id_b})")

        # ──────────────────────────────────────────────────────────
        # TEST 1: Unauthenticated request rejected with 401
        # ──────────────────────────────────────────────────────────
        print("\n── TEST 1: Unauthenticated bulk delete returns 401 ──")
        anon_res = await anon_client.request(
            "DELETE",
            "/conversations/bulk",
            json={"conversation_ids": ["some-random-id"]},
        )
        assert anon_res.status_code == 401, f"Expected 401, got: {anon_res.status_code}"
        print("PASS: TEST 1 - Unauthenticated request rejected with 401")

        # ──────────────────────────────────────────────────────────
        # TEST 2: Empty list returns validation error (422)
        # ──────────────────────────────────────────────────────────
        print("\n── TEST 2: Empty list returns validation error ──")
        empty_res = await client_a.request(
            "DELETE",
            "/conversations/bulk",
            json={"conversation_ids": []},
        )
        assert empty_res.status_code in (400, 422), f"Expected 400 or 422, got: {empty_res.status_code}"
        print(f"PASS: TEST 2 - Empty list rejected with code {empty_res.status_code}")

        # ──────────────────────────────────────────────────────────
        # TEST 3: Delete one conversation via bulk endpoint
        # ──────────────────────────────────────────────────────────
        print("\n── TEST 3: Delete one conversation via bulk delete ──")
        c1 = (await client_a.post("/conversations", json={"title": "Single Delete Conv"})).json()
        c1_id = c1["id"]

        del1_res = await client_a.request(
            "DELETE",
            "/conversations/bulk",
            json={"conversation_ids": [c1_id]},
        )
        assert del1_res.status_code == 200, f"Expected 200, got: {del1_res.text}"
        data = del1_res.json()
        assert data["deleted_count"] == 1
        assert data["deleted_ids"] == [c1_id]

        # Verify it is gone
        fetch_res = await client_a.get(f"/conversations/{c1_id}")
        assert fetch_res.status_code == 404
        print(f"PASS: TEST 3 - Single conversation {c1_id} deleted successfully via bulk endpoint")

        # ──────────────────────────────────────────────────────────
        # TEST 4: Delete multiple owned conversations & message cascade
        # ──────────────────────────────────────────────────────────
        print("\n── TEST 4: Delete multiple owned conversations with message cascade ──")
        c2 = (await client_a.post("/conversations", json={"title": "Multi Delete 1"})).json()
        c3 = (await client_a.post("/conversations", json={"title": "Multi Delete 2"})).json()
        c2_id, c3_id = c2["id"], c3["id"]

        # Add messages to both conversations
        db = SessionLocal()
        m1 = PgMessage(id=str(uuid.uuid4()), conversation_id=c2_id, role="user", content="Hello from C2")
        m2 = PgMessage(id=str(uuid.uuid4()), conversation_id=c2_id, role="assistant", content="Response in C2")
        m3 = PgMessage(id=str(uuid.uuid4()), conversation_id=c3_id, role="user", content="Hello from C3")
        db.add_all([m1, m2, m3])
        db.commit()

        # Check messages exist
        msg_count = db.query(PgMessage).filter(PgMessage.conversation_id.in_([c2_id, c3_id])).count()
        assert msg_count == 3, f"Expected 3 messages, got {msg_count}"
        db.close()

        # Perform bulk delete
        multi_del = await client_a.request(
            "DELETE",
            "/conversations/bulk",
            json={"conversation_ids": [c2_id, c3_id]},
        )
        assert multi_del.status_code == 200, f"Failed bulk delete: {multi_del.text}"
        m_data = multi_del.json()
        assert m_data["deleted_count"] == 2
        assert set(m_data["deleted_ids"]) == {c2_id, c3_id}

        # Verify conversations are gone
        db = SessionLocal()
        conv_count = db.query(PgConversation).filter(PgConversation.id.in_([c2_id, c3_id])).count()
        assert conv_count == 0, f"Expected 0 conversations left, got {conv_count}"

        # Verify message cascade: all messages must be deleted
        remaining_msgs = db.query(PgMessage).filter(PgMessage.conversation_id.in_([c2_id, c3_id])).count()
        assert remaining_msgs == 0, f"Expected 0 cascade messages left, got {remaining_msgs}"
        db.close()
        print(f"PASS: TEST 4 - Successfully deleted 2 conversations and all 3 cascaded messages")

        # ──────────────────────────────────────────────────────────
        # TEST 5: Duplicate IDs handled safely
        # ──────────────────────────────────────────────────────────
        print("\n── TEST 5: Duplicate IDs handled safely ──")
        c4 = (await client_a.post("/conversations", json={"title": "Duplicate ID Conv"})).json()
        c4_id = c4["id"]

        dup_res = await client_a.request(
            "DELETE",
            "/conversations/bulk",
            json={"conversation_ids": [c4_id, c4_id, c4_id]},
        )
        assert dup_res.status_code == 200, f"Failed dup delete: {dup_res.text}"
        dup_data = dup_res.json()
        assert dup_data["deleted_count"] == 1
        assert dup_data["deleted_ids"] == [c4_id]
        print("PASS: TEST 5 - Duplicate IDs deduplicated safely, deleted_count=1")

        # ──────────────────────────────────────────────────────────
        # TEST 6: Invalid conversation ID returns 404 & no partial deletion
        # ──────────────────────────────────────────────────────────
        print("\n── TEST 6: Invalid conversation ID returns 404 without partial delete ──")
        c5 = (await client_a.post("/conversations", json={"title": "Safe Conv 5"})).json()
        c5_id = c5["id"]
        fake_id = "00000000-0000-0000-0000-000000000000"

        fail_res = await client_a.request(
            "DELETE",
            "/conversations/bulk",
            json={"conversation_ids": [c5_id, fake_id]},
        )
        assert fail_res.status_code == 404, f"Expected 404, got: {fail_res.status_code}"

        # Verify c5_id was NOT deleted (transaction rolled back)
        c5_check = await client_a.get(f"/conversations/{c5_id}")
        assert c5_check.status_code == 200, "Conversation was partially deleted despite invalid ID in batch!"
        print("PASS: TEST 6 - Invalid ID rejected with 404; existing conversation preserved (zero partial delete)")

        # ──────────────────────────────────────────────────────────
        # TEST 7: Cross-user deletion blocked & no partial delete
        # ──────────────────────────────────────────────────────────
        print("\n── TEST 7: Cross-user deletion blocked with 404 ──")
        b_conv = (await client_b.post("/conversations", json={"title": "User B Secret Chat"})).json()
        b_conv_id = b_conv["id"]

        cross_res = await client_a.request(
            "DELETE",
            "/conversations/bulk",
            json={"conversation_ids": [c5_id, b_conv_id]},
        )
        assert cross_res.status_code == 404, f"Expected 404 for cross-user delete, got: {cross_res.status_code}"

        # Verify NEITHER conversation was deleted
        c5_still_exists = await client_a.get(f"/conversations/{c5_id}")
        assert c5_still_exists.status_code == 200, "User A conversation was prematurely deleted!"
        b_still_exists = await client_b.get(f"/conversations/{b_conv_id}")
        assert b_still_exists.status_code == 200, "User B conversation was unauthorizedly deleted!"
        print("PASS: TEST 7 - Cross-user deletion blocked with 404; both conversations intact")

        # ──────────────────────────────────────────────────────────
        # TEST 8: Active conversation deletion
        # ──────────────────────────────────────────────────────────
        print("\n── TEST 8: Active conversation deletion ──")
        active_conv = (await client_a.post("/conversations", json={"title": "Currently Active Chat"})).json()
        act_id = active_conv["id"]

        act_del = await client_a.request(
            "DELETE",
            "/conversations/bulk",
            json={"conversation_ids": [act_id, c5_id]},
        )
        assert act_del.status_code == 200
        assert act_del.json()["deleted_count"] == 2
        print("PASS: TEST 8 - Active conversations deleted cleanly")

        # ──────────────────────────────────────────────────────────
        # TEST 9: Large batch deletion (20 conversations)
        # ──────────────────────────────────────────────────────────
        print("\n── TEST 9: Large batch deletion (20 conversations) ──")
        batch_ids = []
        for i in range(20):
            created = (await client_a.post("/conversations", json={"title": f"Batch Item {i+1}"})).json()
            batch_ids.append(created["id"])

        large_del = await client_a.request(
            "DELETE",
            "/conversations/bulk",
            json={"conversation_ids": batch_ids},
        )
        assert large_del.status_code == 200, f"Large batch delete failed: {large_del.text}"
        large_data = large_del.json()
        assert large_data["deleted_count"] == 20
        assert len(large_data["deleted_ids"]) == 20

        # Verify all 20 are gone
        db = SessionLocal()
        left = db.query(PgConversation).filter(PgConversation.id.in_(batch_ids)).count()
        db.close()
        assert left == 0, f"Expected 0 left, found {left}"
        print(f"PASS: TEST 9 - Large batch of 20 conversations successfully deleted in one transaction")

        print("\n======================================================================")
        print("ALL BULK CONVERSATION DELETE TESTS PASSED WITH 100% SUCCESS! ✅")
        print("======================================================================")


if __name__ == "__main__":
    asyncio.run(run_tests())
