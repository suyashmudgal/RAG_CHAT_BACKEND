"""Integration test for Google Authentication endpoints (GIS & OAuth Redirect Flow)."""

import asyncio
from unittest.mock import AsyncMock, patch
from urllib.parse import parse_qs, urlparse
from httpx import ASGITransport, AsyncClient, Response
from main import app
from config.settings import settings


async def run_google_auth_tests():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # 1. Config endpoint
        res = await ac.get("/auth/config")
        assert res.status_code == 200, f"Config failed: {res.text}"
        data = res.json()
        assert data["google_client_id"] == settings.google_client_id, f"Client ID mismatch: {data}"
        assert data["google_enabled"] is True
        print("PASS: /auth/config returned active client ID:", data["google_client_id"])

        # 2. Empty credential to /auth/google -> 400 or 422
        res = await ac.post("/auth/google", json={"credential": ""})
        assert res.status_code in (400, 422)
        print("PASS: /auth/google handles empty credential with status:", res.status_code)

        # 3. Valid Google token flow (GIS Tokeninfo Flow)
        mock_google_profile = {
            "iss": "https://accounts.google.com",
            "sub": "google-user-9999",
            "aud": settings.google_client_id,
            "email": "google.testuser@example.com",
            "email_verified": "true",
            "name": "Google Test User",
            "picture": "https://example.com/photo.jpg",
        }

        with patch("auth.router.httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = Response(
                status_code=200,
                json=mock_google_profile,
            )
            res = await ac.post("/auth/google", json={"credential": "mock_id_token_12345"})

        assert res.status_code == 200, f"Mock google sign-in failed: {res.text}"
        res_data = res.json()
        assert res_data["user"]["email"] == "google.testuser@example.com"
        assert res_data["user"]["name"] == "Google Test User"
        assert "token" in res.cookies
        print("PASS: /auth/google authenticated user successfully:", res_data["user"])

        # 4. Verify authenticated session via /auth/me
        me_res = await ac.get("/auth/me")
        assert me_res.status_code == 200
        assert me_res.json()["email"] == "google.testuser@example.com"
        print("PASS: /auth/me recognized Google user session:", me_res.json())

        # 5. Sign out
        signout_res = await ac.post("/auth/signout")
        assert signout_res.status_code == 200
        print("PASS: /auth/signout cleared Google user session")

        # 6. Verify session revoked
        after_me = await ac.get("/auth/me")
        assert after_me.status_code == 401
        print("PASS: /auth/me rejected unauthenticated request after signout (401)")

        # 7. OAuth Redirect Flow: /auth/google and /auth/google/login
        for endpoint in ["/auth/google", "/auth/google/login"]:
            login_res = await ac.get(endpoint, follow_redirects=False)
            assert login_res.status_code in (302, 307), f"Expected redirect on {endpoint}, got: {login_res.status_code}"
            redirect_url = login_res.headers["location"]
            parsed_url = urlparse(redirect_url)
            assert parsed_url.hostname == "accounts.google.com"
            assert parsed_url.path == "/o/oauth2/v2/auth"
            query_params = parse_qs(parsed_url.query)
            assert query_params["client_id"][0] == settings.google_client_id
            assert query_params["redirect_uri"][0] == settings.google_redirect_uri
            assert query_params["redirect_uri"][0] == "http://localhost:8001/auth/google/callback"
            print(f"PASS: {endpoint} generates correct Google OAuth URL with callback:", settings.google_redirect_uri)

        # 8. OAuth Redirect Callback Flow: /auth/google/callback (First Login: New User)
        mock_tokens = {
            "access_token": "mock_google_access_token_xyz",
            "token_type": "Bearer",
            "expires_in": 3600,
            "scope": "openid email profile",
        }
        mock_userinfo = {
            "id": "123456789",
            "email": "oauth.callback.user@example.com",
            "verified_email": True,
            "name": "Callback User",
            "picture": "https://example.com/avatar.jpg",
        }

        mock_client = AsyncMock()
        mock_client.post.return_value = Response(200, json=mock_tokens)
        mock_client.get.return_value = Response(200, json=mock_userinfo)
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        with patch("auth.router.httpx.AsyncClient", return_value=mock_client):
            cb_res = await ac.get(
                "/auth/google/callback",
                params={"code": "mock_auth_code_9876"},
                follow_redirects=False,
            )
            assert cb_res.status_code in (302, 307), f"Expected redirect, got: {cb_res.status_code}"
            assert cb_res.headers["location"] == f"{settings.frontend_url}/app"
            assert "token" in cb_res.cookies
            print("PASS: /auth/google/callback (first login) exchanged code and set auth cookie, redirected to:", cb_res.headers["location"])

        # 9. Verify callback user session via /auth/me
        cb_me_res = await ac.get("/auth/me")
        assert cb_me_res.status_code == 200
        first_user = cb_me_res.json()
        assert first_user["email"] == "oauth.callback.user@example.com"
        assert first_user["is_oauth"] is True
        print("PASS: /auth/me recognized OAuth callback user session:", first_user)

        # 9b. Second Login with SAME Google Account (Idempotency: No duplicate user)
        with patch("auth.router.httpx.AsyncClient", return_value=mock_client):
            cb_res2 = await ac.get(
                "/auth/google/callback",
                params={"code": "mock_auth_code_repeat"},
                follow_redirects=False,
            )
            assert cb_res2.status_code in (302, 307)
            print("PASS: /auth/google/callback (second login) succeeded")

        cb_me_res2 = await ac.get("/auth/me")
        assert cb_me_res2.status_code == 200
        second_user = cb_me_res2.json()
        assert second_user["id"] == first_user["id"]
        assert second_user["email"] == first_user["email"]
        print("PASS: Second Google login reused existing user (id=%s) without duplicates" % second_user["id"])

        # 10. Callback error handling
        err_cb_res = await ac.get(
            "/auth/google/callback",
            params={"error": "access_denied"},
            follow_redirects=False,
        )
        assert err_cb_res.status_code in (302, 307)
        assert "access_denied" in err_cb_res.headers["location"]
        print("PASS: /auth/google/callback handled error parameter gracefully")

    print("\nALL GOOGLE AUTH & OAUTH REDIRECT TESTS PASSED SUCCESSFULLY! [OK]")


if __name__ == "__main__":
    asyncio.run(run_google_auth_tests())

