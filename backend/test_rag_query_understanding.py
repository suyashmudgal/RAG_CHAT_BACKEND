"""Comprehensive test suite for RAG Query Understanding, Multi-Document Retrieval,
ATS-Readiness Safety, Follow-up Contextualization, and User Isolation.
"""

import asyncio
import re
import sys
import uuid

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from services.deps import get_chat_service, get_vector_store


async def run_query_tests():
    chat_service = get_chat_service()
    store = get_vector_store()
    user_id = 112

    # Verify user 112 has the uploaded resumes
    docs = store.list_documents_for_user(user_id)
    filenames = [d["filename"] for d in docs]
    print("=" * 70, flush=True)
    print(f"VERIFYING RAG / QUERY UNDERSTANDING FOR USER {user_id}", flush=True)
    print(f"Available Documents ({len(docs)}): {filenames}", flush=True)
    print("=" * 70, flush=True)

    assert len(docs) >= 3, f"Expected at least 3 resumes for user {user_id}, got {len(docs)}"

    all_passed = True

    # ── PART 1: Core Comparison, ATS, Summary, and Synthesis Queries ──
    conv_main = f"test-main-{uuid.uuid4().hex[:8]}"

    core_queries = [
        # Original problem query
        "tell me which resume is best which have high ats score",
        # 1. AI/ML role suitability
        "Which resume is strongest for an AI/ML role?",
        # 2. Compare all three resumes
        "Compare all three resumes.",
        # 3. Project comparison
        "Which resume has better projects?",
        # 4. Technical skills
        "Which one has stronger technical skills?",
        # 5. ATS-friendliness
        "Which one appears more ATS-friendly and why?",
        # 6. Common skills
        "What common skills do these resumes have?",
        # 7. Missing skills
        "What skills are missing from these resumes for an AI/ML fresher?",
        # 8. Data Analyst role
        "Which resume is more suitable for a Data Analyst role?",
        # 9. Single doc summary
        "Summarize Suyash's resume.",
    ]

    for idx, q in enumerate(core_queries, 1):
        print("\n" + "-" * 70)
        print(f"[CORE {idx}/{len(core_queries)}] QUERY: '{q}'")

        await asyncio.sleep(1.5)  # Pace requests to respect free-tier TPM
        res = await chat_service.get_answer(
            question=q,
            conversation_id=conv_main,
            user_id=user_id,
        )

        answer = res["answer"]
        sources = res.get("sources", [])

        print(f"ANSWER PREVIEW ({len(answer)} chars):", flush=True)
        print(answer[:220] + ("..." if len(answer) > 220 else ""), flush=True)
        print(f"SOURCES COUNT: {len(sources)}", flush=True)
        for s in sources:
            print(f"  * {s.get('filename')} (Page {s.get('page_number')}, score: {s.get('relevance_score')})", flush=True)

        # 1. Check: Must NOT falsely reject with blunt refusal
        rejection_phrases = [
            "couldn't find this information in the uploaded documents",
            "could not find this information in the uploaded documents",
            "no relevant document context was found",
        ]
        is_rejected = any(rp in answer.lower() for rp in rejection_phrases) and len(answer.strip()) < 200
        if is_rejected:
            print(f"❌ FAIL: Query was falsely rejected with blunt refusal!", flush=True)
            all_passed = False
        else:
            print(f"✅ PASS: Not rejected; informative response generated.", flush=True)

        # 2. Check ATS safety: Must not fabricate an exact numeric score
        if "ats" in q.lower():
            fabricated_score_match = re.search(r"\b(?:ats\s*score\s*(?:of|is|:)?\s*\d{1,3}%?|\d{1,3}\s*/\s*100)\b", answer.lower())
            if fabricated_score_match:
                print(f"❌ FAIL: Fabricated ATS score detected: {fabricated_score_match.group(0)}", flush=True)
                all_passed = False
            else:
                print(f"✅ PASS: ATS safety preserved (evidence-based analysis, no fabricated score).", flush=True)

        # 3. Check document grounding / citations for comparison queries
        if len(sources) > 0:
            print(f"✅ PASS: Document-grounded citations provided ({len(sources)} sources).", flush=True)

    # ── PART 2: Follow-up Conversation Context Chain ──
    print("\n" + "=" * 70)
    print("VERIFYING FOLLOW-UP QUESTIONS WITH CONVERSATION CONTEXT")
    print("=" * 70)

    conv_followup = f"test-followup-{uuid.uuid4().hex[:8]}"

    followup_chain = [
        # Seed conversation with resume context
        ("Compare these resumes.", "Initial comparison establishing context"),
        # 10. Follow-up: certifications
        ("What about certifications?", "Follow-up 1: Certifications"),
        # 11. Follow-up: projects
        ("What about projects?", "Follow-up 2: Projects"),
        # 12. Follow-up: experience
        ("Which one has better experience?", "Follow-up 3: Experience"),
    ]

    for q, label in followup_chain:
        print("\n" + "-" * 70)
        print(f"[{label}] QUERY: '{q}'")

        await asyncio.sleep(1.5)  # Pace requests
        res = await chat_service.get_answer(
            question=q,
            conversation_id=conv_followup,
            user_id=user_id,
        )

        answer = res["answer"]
        sources = res.get("sources", [])

        print(f"ANSWER PREVIEW ({len(answer)} chars):")
        print(answer[:220] + ("..." if len(answer) > 220 else ""))
        print(f"SOURCES COUNT: {len(sources)}")
        for s in sources:
            print(f"  * {s.get('filename')} (Page {s.get('page_number')}, score: {s.get('relevance_score')})")

        rejection_phrases = [
            "couldn't find this information in the uploaded documents",
            "could not find this information in the uploaded documents",
            "no relevant document context was found",
        ]
        is_rejected = any(rp in answer.lower() for rp in rejection_phrases) and len(answer.strip()) < 200
        if is_rejected:
            print(f"❌ FAIL: Follow-up query was falsely rejected with blunt refusal!")
            all_passed = False
        else:
            print(f"✅ PASS: Follow-up recognized resume context successfully.")

    # ── PART 3: User Isolation Verification ──
    print("\n" + "=" * 70)
    print("VERIFYING USER ISOLATION (CROSS-USER RETRIEVAL BLOCKED)")
    print("=" * 70)

    unauthorized_user_id = 999999
    # Test retrieval pipeline isolation
    context, unauth_raw, analysis = await asyncio.to_thread(
        chat_service._retrieve_context, "Which resume is strongest for an AI/ML role?", unauthorized_user_id, []
    )
    unauth_candidates = chat_service.vector_store.similarity_search_for_user(
        "machine learning python", user_id=unauthorized_user_id
    )
    unauth_multi = chat_service.vector_store.similarity_search_multi_doc_for_user(
        ["machine learning", "python"], user_id=unauthorized_user_id
    )

    print(f"Unauthorized User {unauthorized_user_id} RAG Chunks: {len(unauth_raw)}")
    print(f"Unauthorized User {unauthorized_user_id} Single Search Chunks: {len(unauth_candidates)}")
    print(f"Unauthorized User {unauthorized_user_id} Multi Search Chunks: {len(unauth_multi)}")

    assert len(unauth_raw) == 0, f"SECURITY LEAK: Unauthorized user retrieved {len(unauth_raw)} chunks in RAG!"
    assert len(unauth_candidates) == 0, f"SECURITY LEAK: Unauthorized user retrieved {len(unauth_candidates)} chunks in vector store!"
    assert len(unauth_multi) == 0, f"SECURITY LEAK: Unauthorized user retrieved {len(unauth_multi)} chunks in multi-search!"
    assert context == "", f"SECURITY LEAK: Unauthorized user received context: {context}"
    print(f"✅ PASS: Cross-user retrieval strictly blocked! User {unauthorized_user_id} cannot access User {user_id}'s documents.")

    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 ALL 12 QUERIES AND ISOLATION TESTS PASSED SUCCESSFULLY!")
    else:
        print("❌ SOME CHECKS FAILED.")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_query_tests())
