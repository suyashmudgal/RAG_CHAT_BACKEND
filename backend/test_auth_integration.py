import uuid
import asyncio
from httpx import AsyncClient, ASGITransport
from main import app

async def run_tests():
    test_email = f"alex_{uuid.uuid4().hex[:8]}@test.com"
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # 1. Health check
        res = await ac.get("/")
        print("Health Check:", res.status_code, res.json())
        assert res.status_code == 200

        # 2. Check /auth/me without token -> 401
        res = await ac.get("/auth/me")
        print("Unauthenticated /auth/me:", res.status_code)
        assert res.status_code == 401

        # 3. Signup a test user
        signup_payload = {
            "name": "Alex Developer",
            "email": test_email,
            "password": "Password123!"
        }
        res = await ac.post("/auth/signup", json=signup_payload)
        print("Signup response:", res.status_code, res.json())
        assert res.status_code == 200
        assert "token" in res.cookies

        # 4. Check /auth/me with the cookie from signup
        res = await ac.get("/auth/me")
        print("Authenticated /auth/me:", res.status_code, res.json())
        assert res.status_code == 200
        assert res.json()["email"] == test_email
        assert res.json()["name"] == "Alex Developer"

        # 5. Sign out
        res = await ac.post("/auth/signout")
        print("Signout response:", res.status_code, res.json())
        assert res.status_code == 200

        # 6. Check /auth/me after signout -> 401
        res = await ac.get("/auth/me")
        print("After signout /auth/me:", res.status_code)
        assert res.status_code == 401

        # 7. Sign in with valid credentials
        signin_payload = {
            "email": test_email,
            "password": "Password123!"
        }
        res = await ac.post("/auth/signin", json=signin_payload)
        print("Signin response:", res.status_code, res.json())
        assert res.status_code == 200
        assert "token" in res.cookies

        # 8. Sign in with invalid password -> 401
        bad_signin = {
            "email": test_email,
            "password": "WrongPassword!"
        }
        res = await ac.post("/auth/signin", json=bad_signin)
        print("Bad signin response:", res.status_code)
        assert res.status_code == 401

        # 9. Duplicate signup -> 409
        res = await ac.post("/auth/signup", json=signup_payload)
        print("Duplicate signup response:", res.status_code)
        assert res.status_code == 409

    print("ALL AUTH INTEGRATION TESTS PASSED SUCCESSFULLY! [OK]")

if __name__ == "__main__":
    asyncio.run(run_tests())
