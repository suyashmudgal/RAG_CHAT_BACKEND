"""Verify the 5 citation test cases on the live running FastAPI backend."""
import json
import sys
import requests

sys.stdout.reconfigure(encoding='utf-8')

BASE_URL = "http://localhost:8001"

TEST_CASES = [
    {
        "id": 1,
        "question": "How many attention heads does the base Transformer use?",
        "expected_answer_keywords": ["8"],
        "expected_pages": [5],
        "forbidden_pages": [2, 8, 3],
    },
    {
        "id": 2,
        "question": "What are dk and dv in the base Transformer?",
        "expected_answer_keywords": ["64"],
        "expected_pages": [5, 9],  # either page 5 or page 9
        "forbidden_pages": [2, 8, 3, 4],
    },
    {
        "id": 3,
        "question": "What is the exact dollar cost of training the Transformer?",
        "expected_negative": True,
        "expected_sources_count": 0,
    },
    {
        "id": 4,
        "question": "How many encoder and decoder layers does the Transformer have?",
        "expected_answer_keywords": ["6", "six"],
        "expected_pages": [2, 3],
        "forbidden_pages": [5, 6, 8],
    },
    {
        "id": 5,
        "question": "Explain the difference between encoder self-attention and decoder masked self-attention.",
        "expected_answer_keywords": ["mask", "attend"],
        "min_sources": 1,
        "max_sources": 3,
    },
]

def run_tests():
    print("=" * 70)
    print("RUNNING CITATION QUALITY VERIFICATION SUITE")
    print("=" * 70)

    all_passed = True

    for tc in TEST_CASES:
        t_id = tc["id"]
        q = tc["question"]
        print(f"\n--- TEST {t_id}: {q} ---")

        # Test streaming endpoint (SSE)
        r = requests.post(
            f"{BASE_URL}/chat/stream",
            json={"question": q, "session_id": f"verify_suite_{t_id}"},
            stream=True,
            timeout=30,
        )
        assert r.status_code == 200, f"HTTP Error {r.status_code}"

        tokens = []
        sources = []
        for line in r.iter_lines():
            line_str = line.decode('utf-8', errors='ignore')
            if line_str.startswith("data: "):
                data = json.loads(line_str[6:])
                if data.get("type") == "token":
                    tokens.append(data.get("content", ""))
                elif data.get("type") == "sources":
                    sources = data.get("sources", [])

        full_answer = "".join(tokens)
        print(f"ANSWER: {full_answer[:120]}...")
        print(f"SOURCES ({len(sources)}):")
        pages = []
        for s in sources:
            p = s.get("page_number")
            score = s.get("relevance_score")
            pages.append(p)
            print(f"  -> {s['filename']} | Page: {p} | relevance_score: {score}")

        # Check expectations
        if tc.get("expected_negative"):
            if len(sources) != 0:
                print(f"FAILED: Expected 0 sources for not-found answer, got {len(sources)}")
                all_passed = False
            else:
                print(f"PASSED: 0 sources returned for not-found question.")
        else:
            if "expected_pages" in tc:
                # Must contain at least one expected page
                matched = any(p in tc["expected_pages"] for p in pages)
                if not matched:
                    print(f"FAILED: Expected pages from {tc['expected_pages']}, got {pages}")
                    all_passed = False
                else:
                    print(f"PASSED: Supporting page matched ({[p for p in pages if p in tc['expected_pages']]}).")

            if "forbidden_pages" in tc:
                # Check for noise pages
                noise = [p for p in pages if p in tc["forbidden_pages"]]
                if noise:
                    print(f"FAILED: Found unrelated noise pages: {noise}")
                    all_passed = False
                else:
                    print(f"PASSED: No unrelated noise pages found.")

        print(f"TEST {t_id} RESULT: {'SUCCESS' if all_passed else 'NEEDS ATTENTION'}")

    print("\n" + "=" * 70)
    if all_passed:
        print("ALL CITATION TESTS COMPLETED SUCCESSFULLY!")
    else:
        print("SOME TESTS FAILED.")
    print("=" * 70)
    return all_passed

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
