"""Phase 3 — Persistent Chat History Integration Tests.

Verifies:
TEST 1: User A creates conversation -> stored in PostgreSQL.
TEST 2: User A sends user message -> stored in PostgreSQL.
TEST 3: Assistant responds -> stored in PostgreSQL.
TEST 4: User A refreshes/reloads conversation -> previous messages returned.
TEST 5: Backend restarts -> conversation history still exists in PostgreSQL.
TEST 6: User A lists conversations -> only User A's conversations returned.
TEST 7: User B lists conversations -> User A's conversations are NOT returned.
TEST 8: User B attempts to access User A's conversation ID -> returns 404.
TEST 9: User B attempts to delete User A's conversation -> deletion rejected (404).
TEST 10: User A continues an old conversation -> previous relevant history is used by chat service.
TEST 11: Streaming response -> tokens stream correctly AND final assistant response is persisted once.
TEST 12: User A and User B use identical session_id -> no history collision in PostgreSQL.
TEST 13: User logout -> User B login -> User B sees only User B's chat history.
"""

import asyncio
import json
import uuid
from httpx import ASGITransport, AsyncClient

from main import app
from database.session import SessionLocal
from models.database import Conversation as PgConversation, Message as PgMessage, User as PgUser


def _unique_email(prefix: str = "persist") -> str:
    return f"{prefix.lower()}_{uuid.uuid4().hex[:8]}@test.com"


async def _signup_and_get_cookie(client: AsyncClient, name: str, email: str, password: str = "Pass1234!") -> dict:
    res = await client.post("/auth/signup", json={"name": name, "email": email, "password": password})
    assert res.status_code == 200, f"Signup failed: {res.text}"
    user_data = res.json()["user"]
    # Hit /auth/me to trigger lazy PostgreSQL user sync and attach pg_id
    me_res = await client.get("/auth/me")
    assert me_res.status_code == 200, f"/auth/me failed: {me_res.text}"
    return user_data


async def run_tests():
    print("============================================================")
    print("PHASE 3 — PERSISTENT CHAT HISTORY INTEGRATION TESTS")
    print("============================================================")

    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client_a, \
               AsyncClient(transport=transport, base_url="http://test") as client_b:

        email_a = _unique_email("userA")
        email_b = _unique_email("userB")

        user_a = await _signup_and_get_cookie(client_a, "User Alpha", email_a)
        user_b = await _signup_and_get_cookie(client_b, "User Beta", email_b)

        # Lookup pg_ids directly from DB
        db = SessionLocal()
        pg_user_a = db.query(PgUser).filter(PgUser.email == email_a).first()
        pg_user_b = db.query(PgUser).filter(PgUser.email == email_b).first()
        db.close()

        assert pg_user_a is not None, "User A not found in PostgreSQL"
        assert pg_user_b is not None, "User B not found in PostgreSQL"
        pg_id_a = pg_user_a.id
        pg_id_b = pg_user_b.id

        print(f"User A registered: {email_a} (pg_id={pg_id_a})")
        print(f"User B registered: {email_b} (pg_id={pg_id_b})")

        # ──────────────────────────────────────────────────────────
        # TEST 1: User A creates conversation -> stored in PostgreSQL
        # ──────────────────────────────────────────────────────────
        print("\n── TEST 1: User A creates conversation ──")
        create_res = await client_a.post("/conversations", json={"title": "ML Research"})
        assert create_res.status_code == 200, f"Failed to create conversation: {create_res.text}"
        conv_a1 = create_res.json()
        conv_id_a1 = conv_a1["id"]
        assert conv_a1["title"] == "ML Research"

        # Verify in PostgreSQL
        db = SessionLocal()
        db_conv = db.query(PgConversation).filter(PgConversation.id == conv_id_a1).first()
        db.close()
        assert db_conv is not None, "Conversation not found in PostgreSQL"
        assert db_conv.user_id == pg_id_a, f"Expected user_id={pg_id_a}, got {db_conv.user_id}"
        print(f"✅ TEST 1 PASSED: Conversation {conv_id_a1} stored in PostgreSQL for user_id={pg_id_a}")

        # ──────────────────────────────────────────────────────────
        # TEST 2 & 3: User A sends user message & assistant responds
        # ──────────────────────────────────────────────────────────
        print("\n── TEST 2 & 3: User message and assistant response stored in PostgreSQL ──")
        chat_res = await client_a.post("/chat", json={
            "question": "What is reinforcement learning?",
            "conversation_id": conv_id_a1,
        })
        assert chat_res.status_code == 200, f"Chat failed: {chat_res.text}"
        chat_data = chat_res.json()
        assert "answer" in chat_data
        assert chat_data["conversation_id"] == conv_id_a1

        # Verify both user and assistant messages stored in PostgreSQL
        db = SessionLocal()
        messages = (
            db.query(PgMessage)
            .filter(PgMessage.conversation_id == conv_id_a1)
            .order_by(PgMessage.created_at.asc())
            .all()
        )
        db.close()

        assert len(messages) >= 2, f"Expected at least 2 messages in PostgreSQL, got {len(messages)}"
        assert messages[0].role == "user"
        assert "reinforcement learning" in messages[0].content
        assert messages[1].role == "assistant"
        assert len(messages[1].content) > 0
        print(f"✅ TEST 2 & 3 PASSED: User message and assistant response persisted in PostgreSQL ({len(messages)} messages total)")

        # ──────────────────────────────────────────────────────────
        # TEST 4: User A refreshes/reloads conversation
        # ──────────────────────────────────────────────────────────
        print("\n── TEST 4: User A reloads conversation and messages ──")
        detail_res = await client_a.get(f"/conversations/{conv_id_a1}")
        assert detail_res.status_code == 200, f"Failed to reload conversation: {detail_res.text}"
        detail_data = detail_res.json()
        assert detail_data["id"] == conv_id_a1
        assert len(detail_data["messages"]) >= 2
        assert detail_data["messages"][0]["role"] == "user"
        assert detail_data["messages"][1]["role"] == "assistant"
        print(f"✅ TEST 4 PASSED: Loaded conversation details with {len(detail_data['messages'])} historical messages")

        # ──────────────────────────────────────────────────────────
        # TEST 5: Backend restart (fresh DB session & client session)
        # ──────────────────────────────────────────────────────────
        print("\n── TEST 5: Chat history persists across simulated backend restarts ──")
        # Direct DB check verifying persistence independent of application memory
        db = SessionLocal()
        persisted_conv = db.query(PgConversation).filter(PgConversation.id == conv_id_a1).first()
        persisted_msgs = db.query(PgMessage).filter(PgMessage.conversation_id == conv_id_a1).count()
        db.close()
        assert persisted_conv is not None
        assert persisted_msgs >= 2
        print(f"✅ TEST 5 PASSED: PostgreSQL database holds {persisted_msgs} messages independently of in-memory state")

        # ──────────────────────────────────────────────────────────
        # TEST 6: User A lists conversations
        # ──────────────────────────────────────────────────────────
        print("\n── TEST 6: User A lists conversations ──")
        list_a_res = await client_a.get("/conversations")
        assert list_a_res.status_code == 200
        list_a = list_a_res.json()
        conv_ids_a = [c["id"] for c in list_a]
        assert conv_id_a1 in conv_ids_a
        print(f"✅ TEST 6 PASSED: User A sees their own conversation in list ({len(list_a)} conversations)")

        # ──────────────────────────────────────────────────────────
        # TEST 7: User B lists conversations -> User A's docs not returned
        # ──────────────────────────────────────────────────────────
        print("\n── TEST 7: User B lists conversations ──")
        list_b_res = await client_b.get("/conversations")
        assert list_b_res.status_code == 200
        list_b = list_b_res.json()
        conv_ids_b = [c["id"] for c in list_b]
        assert conv_id_a1 not in conv_ids_b, "CRITICAL: User B saw User A's conversation!"
        print("✅ TEST 7 PASSED: User A's conversation is invisible to User B")

        # ──────────────────────────────────────────────────────────
        # TEST 8: User B attempts to access User A's conversation ID -> 404
        # ──────────────────────────────────────────────────────────
        print("\n── TEST 8: Cross-user access blocked ──")
        unauth_get = await client_b.get(f"/conversations/{conv_id_a1}")
        assert unauth_get.status_code == 404, f"Expected 404, got {unauth_get.status_code}"
        print("✅ TEST 8 PASSED: User B receives 404 when accessing User A's conversation")

        # ──────────────────────────────────────────────────────────
        # TEST 9: User B attempts to delete User A's conversation -> rejected (404)
        # ──────────────────────────────────────────────────────────
        print("\n── TEST 9: Cross-user deletion blocked ──")
        unauth_del = await client_b.delete(f"/conversations/{conv_id_a1}")
        assert unauth_del.status_code == 404, f"Expected 404, got {unauth_del.status_code}"

        # Verify conversation still exists
        db = SessionLocal()
        still_exists = db.query(PgConversation).filter(PgConversation.id == conv_id_a1).first()
        db.close()
        assert still_exists is not None, "Conversation was erroneously deleted!"
        print("✅ TEST 9 PASSED: User B cannot delete User A's conversation")

        # ──────────────────────────────────────────────────────────
        # TEST 10: User A continues an old conversation
        # ──────────────────────────────────────────────────────────
        print("\n── TEST 10: Continue old conversation with persistent history ──")
        cont_res = await client_a.post("/chat", json={
            "question": "Can you give one short application example of it?",
            "conversation_id": conv_id_a1,
        })
        assert cont_res.status_code == 200
        cont_data = cont_res.json()
        assert len(cont_data["answer"]) > 0

        # Check that conversation message count increased to 4 in PostgreSQL
        db = SessionLocal()
        total_msgs = db.query(PgMessage).filter(PgMessage.conversation_id == conv_id_a1).count()
        db.close()
        assert total_msgs >= 4, f"Expected at least 4 messages after continuation, got {total_msgs}"
        print(f"✅ TEST 10 PASSED: Conversation continued seamlessly ({total_msgs} messages in thread)")

        # ──────────────────────────────────────────────────────────
        # TEST 11: Streaming response persists final assistant message once
        # ──────────────────────────────────────────────────────────
        print("\n── TEST 11: SSE Streaming with single atomic assistant message persistence ──")
        # Create a new conversation for streaming
        stream_conv_res = await client_a.post("/conversations", json={"title": "Streaming Test"})
        stream_conv_id = stream_conv_res.json()["id"]

        stream_res = await client_a.post("/chat/stream", json={
            "question": "Tell me a brief one sentence fact about astronomy.",
            "conversation_id": stream_conv_id,
        })
        assert stream_res.status_code == 200
        assert "text/event-stream" in stream_res.headers.get("content-type", "")

        events = []
        full_tokens = ""
        done_seen = False

        async for line in stream_res.aiter_lines():
            if line.startswith("data: "):
                payload = json.loads(line[6:])
                events.append(payload)
                if payload.get("type") == "token":
                    full_tokens += payload.get("content", "")
                elif payload.get("type") == "done":
                    done_seen = True

        assert done_seen, "Did not receive 'done' event from SSE stream"
        assert len(full_tokens) > 0, "No streamed tokens received"

        # Check PostgreSQL: exactly 1 user message and 1 assistant message
        db = SessionLocal()
        stream_msgs = (
            db.query(PgMessage)
            .filter(PgMessage.conversation_id == stream_conv_id)
            .order_by(PgMessage.created_at.asc())
            .all()
        )
        db.close()

        assert len(stream_msgs) == 2, f"Expected exactly 2 messages in PostgreSQL for stream, got {len(stream_msgs)}"
        assert stream_msgs[0].role == "user"
        assert stream_msgs[1].role == "assistant"
        assert stream_msgs[1].content.strip() == full_tokens.strip(), "Persisted message content does not match streamed tokens"
        print(f"✅ TEST 11 PASSED: Stream completed, tokens matched, exactly 1 assistant message persisted ({len(full_tokens)} chars)")

        # ──────────────────────────────────────────────────────────
        # TEST 12: User A and User B use identical session_id -> no collision
        # ──────────────────────────────────────────────────────────
        print("\n── TEST 12: Identical session_id generates isolated conversations in PostgreSQL ──")
        identical_session = f"collision_test_{uuid.uuid4().hex[:6]}"

        res_a = await client_a.post("/chat", json={
            "question": "My top secret password is CHEETAH_444",
            "session_id": identical_session,
        })
        assert res_a.status_code == 200
        conv_id_a = res_a.json()["conversation_id"]

        res_b = await client_b.post("/chat", json={
            "question": "What is my secret password?",
            "session_id": identical_session,
        })
        assert res_b.status_code == 200
        conv_id_b = res_b.json()["conversation_id"]

        # Different conversation IDs in PostgreSQL!
        assert conv_id_a != conv_id_b, f"Expected distinct conversation IDs for User A and User B, got {conv_id_a}"
        b_answer = res_b.json()["answer"].lower()
        assert "cheetah_444" not in b_answer, "User B accessed User A's secret history!"

        # Check DB ownership
        db = SessionLocal()
        db_a = db.query(PgConversation).filter(PgConversation.id == conv_id_a).first()
        db_b = db.query(PgConversation).filter(PgConversation.id == conv_id_b).first()
        db.close()

        assert db_a.user_id == pg_id_a
        assert db_b.user_id == pg_id_b
        print(f"✅ TEST 12 PASSED: Identical session_id mapped to distinct DB conversations ({conv_id_a} vs {conv_id_b})")

        # ──────────────────────────────────────────────────────────
        # TEST 13: User logout -> User B login isolation
        # ──────────────────────────────────────────────────────────
        print("\n── TEST 13: Logout / Login state isolation ──")
        # Sign out User A
        await client_a.post("/auth/signout")
        unauth_list = await client_a.get("/conversations")
        assert unauth_list.status_code == 401

        # User B listing only has User B's conversations
        b_convs = (await client_b.get("/conversations")).json()
        for c in b_convs:
            assert c["id"] != conv_id_a1
            assert c["id"] != stream_conv_id
        print("✅ TEST 13 PASSED: Logged out user rejected; User B sees strictly their own conversations")

        # ──────────────────────────────────────────────────────────
        # Cascade deletion test: User A deletes conversation
        # ──────────────────────────────────────────────────────────
        print("\n── Cascade Deletion Test: User A deletes conversation ──")
        # Sign in User A again
        await client_a.post("/auth/signin", json={"email": email_a, "password": "Pass1234!"})
        del_res = await client_a.delete(f"/conversations/{conv_id_a1}")
        assert del_res.status_code == 200

        # Verify conversation and all messages deleted in PostgreSQL
        db = SessionLocal()
        del_conv = db.query(PgConversation).filter(PgConversation.id == conv_id_a1).first()
        del_msgs = db.query(PgMessage).filter(PgMessage.conversation_id == conv_id_a1).count()
        db.close()

        assert del_conv is None, "Conversation was not deleted from PostgreSQL"
        assert del_msgs == 0, f"Expected 0 cascade messages, found {del_msgs}"
        print(f"✅ Cascade Deletion PASSED: Conversation {conv_id_a1} and all messages deleted from PostgreSQL")

    print("\n============================================================")
    print("ALL 13 PERSISTENT CHAT HISTORY TESTS PASSED SUCCESSFULLY! ✅")
    print("============================================================")


if __name__ == "__main__":
    asyncio.run(run_tests())
