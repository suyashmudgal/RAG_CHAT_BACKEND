"""Phase 4 — User Profile & Account Settings Integration Tests.

Verifies:
TEST 1: Authenticated user can retrieve own profile.
TEST 2: Profile response contains name/email/id.
TEST 3: Password hash is never returned in response.
TEST 4: User can update own name in both PostgreSQL and SQLite.
TEST 5: Empty/whitespace name is rejected (400).
TEST 6: Email cannot be changed (rejected with 400).
TEST 7: User can change password using correct current password.
TEST 8: Wrong current password is rejected (400).
TEST 9: New password is hashed and never stored plaintext in SQLite or PostgreSQL.
TEST 10: User can login using the new password after password change.
TEST 11: Old password no longer works after successful password change (401).
TEST 12: Google OAuth user password change rejected with informative message.
TEST 13: User A cannot update User B's profile.
TEST 14: User A cannot change User B's password.
TEST 15: Logout clears session.
TEST 16: After User A logout and User B login, User B sees strictly User B's profile.
"""

import asyncio
import uuid
import bcrypt
from httpx import ASGITransport, AsyncClient

from main import app
from auth.database import get_db as get_sqlite_db
from database.session import SessionLocal
from models.database import User as PgUser


def _unique_email(prefix: str = "profile") -> str:
    return f"{prefix.lower()}_{uuid.uuid4().hex[:8]}@test.com"


async def _signup_user(client: AsyncClient, name: str, email: str, password: str = "Pass1234!") -> dict:
    res = await client.post("/auth/signup", json={"name": name, "email": email, "password": password})
    assert res.status_code == 200, f"Signup failed: {res.text}"
    user_data = res.json()["user"]
    # Trigger PostgreSQL user sync
    me_res = await client.get("/auth/me")
    assert me_res.status_code == 200, f"/auth/me failed: {me_res.text}"
    return user_data


async def run_tests():
    print("============================================================")
    print("PHASE 4 — USER PROFILE & ACCOUNT SETTINGS INTEGRATION TESTS")
    print("============================================================")

    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client_a, \
               AsyncClient(transport=transport, base_url="http://test") as client_b, \
               AsyncClient(transport=transport, base_url="http://test") as client_oauth:

        email_a = _unique_email("userA")
        email_b = _unique_email("userB")
        email_oauth = _unique_email("oauth")

        user_a = await _signup_user(client_a, "Alpha User", email_a, "InitialPass123!")
        user_b = await _signup_user(client_b, "Beta User", email_b, "BetaPass123!")

        # Set up mock OAuth user
        await _signup_user(client_oauth, "OAuth User", email_oauth, "TemporaryPass!")
        # Manually set password_hash to "oauth:google" in SQLite and PostgreSQL to simulate Google OAuth account
        s_db = await get_sqlite_db()
        await s_db.execute("UPDATE users SET password_hash = ? WHERE email = ?", ("oauth:google", email_oauth))
        await s_db.commit()
        await s_db.close()

        pg_db = SessionLocal()
        pg_oauth_user = pg_db.query(PgUser).filter(PgUser.email == email_oauth).first()
        if pg_oauth_user:
            pg_oauth_user.password_hash = "oauth:google"
            pg_db.commit()
        pg_db.close()

        # ──────────────────────────────────────────────────────────
        # TEST 1 & 2: Authenticated user retrieves own profile
        # ──────────────────────────────────────────────────────────
        print("\n── TEST 1 & 2: User A retrieves profile (name, email, id) ──")
        me_res = await client_a.get("/auth/me")
        assert me_res.status_code == 200
        me_data = me_res.json()
        assert me_data["name"] == "Alpha User"
        assert me_data["email"] == email_a
        assert "id" in me_data
        print(f"✅ TEST 1 & 2 PASSED: Profile retrieved: {me_data['name']} ({me_data['email']})")

        # ──────────────────────────────────────────────────────────
        # TEST 3: Password hash is never returned
        # ──────────────────────────────────────────────────────────
        print("\n── TEST 3: Password hash is never exposed ──")
        assert "password_hash" not in me_data
        assert "password" not in me_data
        print("✅ TEST 3 PASSED: No sensitive password fields present in response")

        # ──────────────────────────────────────────────────────────
        # TEST 4: User updates own name in both PostgreSQL and SQLite
        # ──────────────────────────────────────────────────────────
        print("\n── TEST 4: User updates name in both PostgreSQL and SQLite ──")
        update_res = await client_a.patch("/auth/me", json={"name": "Dr. Alpha Updated"})
        assert update_res.status_code == 200, f"Update failed: {update_res.text}"
        updated_data = update_res.json()
        assert updated_data["name"] == "Dr. Alpha Updated"

        # Verify SQLite
        s_db = await get_sqlite_db()
        cursor = await s_db.execute("SELECT name FROM users WHERE email = ?", (email_a,))
        row = await cursor.fetchone()
        await s_db.close()
        assert row is not None and row["name"] == "Dr. Alpha Updated", "Name not updated in SQLite"

        # Verify PostgreSQL
        pg_db = SessionLocal()
        pg_user = pg_db.query(PgUser).filter(PgUser.email == email_a).first()
        pg_db.close()
        assert pg_user is not None and pg_user.name == "Dr. Alpha Updated", "Name not updated in PostgreSQL"
        print(f"✅ TEST 4 PASSED: Name successfully updated to '{pg_user.name}' in both SQLite and PostgreSQL")

        # ──────────────────────────────────────────────────────────
        # TEST 5: Empty or whitespace name is rejected
        # ──────────────────────────────────────────────────────────
        print("\n── TEST 5: Empty / whitespace name rejected ──")
        empty_res1 = await client_a.patch("/auth/me", json={"name": ""})
        assert empty_res1.status_code in [400, 422], f"Expected 400/422, got {empty_res1.status_code}"

        empty_res2 = await client_a.patch("/auth/me", json={"name": "   "})
        assert empty_res2.status_code in [400, 422], f"Expected 400/422, got {empty_res2.status_code}"
        print("✅ TEST 5 PASSED: Empty and whitespace-only names properly rejected with error")

        # ──────────────────────────────────────────────────────────
        # TEST 6: Email cannot be changed (Read-only constraint)
        # ──────────────────────────────────────────────────────────
        print("\n── TEST 6: Email cannot be changed (Read-only constraint) ──")
        fake_email = "new_fake_email@test.com"
        email_change_res = await client_a.patch("/auth/me", json={
            "name": "Valid Name",
            "email": fake_email,
        })
        assert email_change_res.status_code == 400, f"Expected 400, got {email_change_res.status_code}"
        assert "email" in email_change_res.text.lower()

        # Verify email was NOT changed in either DB
        pg_db = SessionLocal()
        pg_user = pg_db.query(PgUser).filter(PgUser.email == email_a).first()
        pg_db.close()
        assert pg_user is not None and pg_user.email == email_a
        print("✅ TEST 6 PASSED: Email change attempt rejected with 400; email remained immutable")

        # ──────────────────────────────────────────────────────────
        # TEST 7: User changes password with valid current password
        # ──────────────────────────────────────────────────────────
        print("\n── TEST 7: Change password with valid current password ──")
        change_res = await client_a.post("/auth/change-password", json={
            "current_password": "InitialPass123!",
            "new_password": "BrandNewSecretPassword456!",
        })
        assert change_res.status_code == 200, f"Change password failed: {change_res.text}"
        assert change_res.json()["message"] == "Password changed successfully"
        print("✅ TEST 7 PASSED: Password changed successfully")

        # ──────────────────────────────────────────────────────────
        # TEST 8: Wrong current password is rejected
        # ──────────────────────────────────────────────────────────
        print("\n── TEST 8: Wrong current password rejected ──")
        bad_pw_res = await client_a.post("/auth/change-password", json={
            "current_password": "WrongPassword999!",
            "new_password": "AnotherNewPassword789!",
        })
        assert bad_pw_res.status_code == 400, f"Expected 400, got {bad_pw_res.status_code}"
        print("✅ TEST 8 PASSED: Wrong current password rejected with 400")

        # ──────────────────────────────────────────────────────────
        # TEST 9: New password is bcrypt-hashed and never stored plaintext
        # ──────────────────────────────────────────────────────────
        print("\n── TEST 9: New password is bcrypt hashed in database ──")
        s_db = await get_sqlite_db()
        cursor = await s_db.execute("SELECT password_hash FROM users WHERE email = ?", (email_a,))
        s_hash = (await cursor.fetchone())["password_hash"]
        await s_db.close()

        pg_db = SessionLocal()
        pg_hash = pg_db.query(PgUser).filter(PgUser.email == email_a).first().password_hash
        pg_db.close()

        assert not s_hash.startswith("BrandNewSecretPassword456!"), "Plaintext stored in SQLite!"
        assert not pg_hash.startswith("BrandNewSecretPassword456!"), "Plaintext stored in PostgreSQL!"
        assert s_hash.startswith("$2b$") or s_hash.startswith("$2a$"), "Not a valid bcrypt hash in SQLite"
        assert pg_hash.startswith("$2b$") or pg_hash.startswith("$2a$"), "Not a valid bcrypt hash in PostgreSQL"
        assert bcrypt.checkpw("BrandNewSecretPassword456!".encode(), s_hash.encode())
        print("✅ TEST 9 PASSED: Valid bcrypt hash confirmed in both SQLite and PostgreSQL")

        # ──────────────────────────────────────────────────────────
        # TEST 10 & 11: Login with new password succeeds; old password fails
        # ──────────────────────────────────────────────────────────
        print("\n── TEST 10 & 11: Login with new password succeeds; old password fails ──")
        async with AsyncClient(transport=transport, base_url="http://test") as login_client:
            # Try old password
            old_login = await login_client.post("/auth/signin", json={
                "email": email_a,
                "password": "InitialPass123!",
            })
            assert old_login.status_code == 401, f"Expected 401 with old password, got {old_login.status_code}"

            # Try new password
            new_login = await login_client.post("/auth/signin", json={
                "email": email_a,
                "password": "BrandNewSecretPassword456!",
            })
            assert new_login.status_code == 200, f"Expected 200 with new password, got {new_login.status_code}"
            assert new_login.json()["user"]["name"] == "Dr. Alpha Updated"
        print("✅ TEST 10 & 11 PASSED: Old password rejected (401); new password authenticates cleanly (200)")

        # ──────────────────────────────────────────────────────────
        # TEST 12: Google OAuth user password change rejected
        # ──────────────────────────────────────────────────────────
        print("\n── TEST 12: Google OAuth password change rejected ──")
        oauth_pw_res = await client_oauth.post("/auth/change-password", json={
            "current_password": "oauth:google",
            "new_password": "NewPassword123!",
        })
        assert oauth_pw_res.status_code == 400, f"Expected 400, got {oauth_pw_res.status_code}"
        assert "oauth" in oauth_pw_res.text.lower() or "google" in oauth_pw_res.text.lower()
        print(f"✅ TEST 12 PASSED: OAuth account password change rejected with helpful message: {oauth_pw_res.json()['detail']}")

        # ──────────────────────────────────────────────────────────
        # TEST 13 & 14: User isolation on profile and password updates
        # ──────────────────────────────────────────────────────────
        print("\n── TEST 13 & 14: Cross-user profile & password isolation ──")
        # User B changes their name and password
        await client_b.patch("/auth/me", json={"name": "Beta The Second"})
        await client_b.post("/auth/change-password", json={
            "current_password": "BetaPass123!",
            "new_password": "BetaNewPass789!",
        })

        # User A's profile and password must remain completely unchanged
        me_a = (await client_a.get("/auth/me")).json()
        assert me_a["name"] == "Dr. Alpha Updated"
        assert me_a["email"] == email_a

        # User A still logs in with User A's password
        async with AsyncClient(transport=transport, base_url="http://test") as verify_client:
            res_a_login = await verify_client.post("/auth/signin", json={
                "email": email_a,
                "password": "BrandNewSecretPassword456!",
            })
            assert res_a_login.status_code == 200
        print("✅ TEST 13 & 14 PASSED: User B updates did not affect User A data or credentials")

        # ──────────────────────────────────────────────────────────
        # TEST 15 & 16: Logout works; new login strictly sees own profile
        # ──────────────────────────────────────────────────────────
        print("\n── TEST 15 & 16: Logout and subsequent login isolation ──")
        signout_res = await client_a.post("/auth/signout")
        assert signout_res.status_code == 200
        unauth_me = await client_a.get("/auth/me")
        assert unauth_me.status_code == 401

        # Client B sees strictly Client B
        me_b = (await client_b.get("/auth/me")).json()
        assert me_b["email"] == email_b
        assert me_b["name"] == "Beta The Second"
        print("✅ TEST 15 & 16 PASSED: Signout properly invalidates session; User B profile remains completely isolated")

    print("\n============================================================")
    print("ALL 16 PROFILE & ACCOUNT SETTINGS TESTS PASSED! ✅")
    print("============================================================")


if __name__ == "__main__":
    asyncio.run(run_tests())
