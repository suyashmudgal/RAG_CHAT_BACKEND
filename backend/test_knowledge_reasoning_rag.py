"""Automated test suite for True Knowledge-Grounded + Reasoning RAG System.

Validates the 6 required scenarios:
1. Math conceptual derivation ("What is 3 + 4?" -> 7, no false refusal)
2. Practical code implementation for code-less paper (Transformer PyTorch code + disclaimer)
3. Concept application ("If gradient is positive, how does the parameter change?" -> decreases)
4. Multi-document reasoning & synthesis (Python basics + Transformer architecture paper)
5. Conversational follow-up resolution ("Explain Transformer" -> "Now give me code for it")
6. Truly missing document-specific fact (clear statement of insufficient evidence)
7. Cross-user isolation (unauthorized user 999999 gets 0 chunks)
"""

import asyncio
import os
import re
import sys
import uuid
from pathlib import Path

# Ensure UTF-8 output encoding for Windows PowerShell
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from langchain_core.documents import Document
from services.deps import get_chat_service, get_vector_store
from services.query_understanding import AnswerType


async def seed_test_documents(vector_store, user_id: int):
    """Seed test documents for user_id to ensure a reproducible knowledge base."""
    docs_to_seed = [
        (
            "math_addition_concept.txt",
            (
                "Addition is a fundamental mathematical operation that represents combining two or more quantities into a single sum. "
                "For example: 2 + 1 = 3. Addition satisfies the commutative property (a + b = b + a) and associative property. "
                "Subtraction is the inverse operation: 5 - 2 = 3. "
                "When you add two numbers together, you calculate the combined total quantity of those numbers."
            ),
        ),
        (
            "gradient_descent_concept.txt",
            (
                "Gradient descent is an iterative optimization algorithm used to minimize objective loss functions in machine learning and deep learning. "
                "The core update rule modifies model parameters opposite to the direction of the gradient: "
                "parameter_new = parameter_old - learning_rate * gradient. "
                "When the gradient of the loss with respect to a parameter is positive, subtracting the positive quantity causes the parameter value to decrease. "
                "Conversely, when the gradient is negative, the parameter value increases to move toward the minimum of the loss landscape."
            ),
        ),
        (
            "python_programming_fundamentals.txt",
            (
                "Python is a high-level, object-oriented programming language designed for readability and modularity. "
                "In deep learning applications, Python integrates with frameworks like PyTorch and NumPy. "
                "PyTorch neural network modules are implemented by creating a class that inherits from torch.nn.Module "
                "and defining the __init__ constructor and forward method to specify the computational graph."
            ),
        ),
        (
            "transformer_architecture_paper.txt",
            (
                "The Transformer model architecture relies entirely on self-attention mechanisms without recurrence or convolutions. "
                "The architecture consists of an encoder stack and a decoder stack, each composed of 6 identical layers. "
                "Each layer contains two sub-layers: a Multi-Head Attention mechanism and a position-wise fully connected feed-forward network. "
                "Multi-Head Attention computes Scaled Dot-Product Attention in parallel across h=8 heads with key/value dimensions d_k=d_v=64 and model dimension d_model=512. "
                "Positional encodings using sinusoidal functions are added to the input embeddings to convey sequential order. "
                "Note: This theoretical paper describes the architectural mathematics and specifications, but does not provide source code or software implementation scripts."
            ),
        ),
    ]

    existing_docs = {d.get("filename"): d for d in vector_store.list_documents_for_user(user_id)}

    for filename, content in docs_to_seed:
        if filename in existing_docs:
            continue
        doc_id = str(uuid.uuid4())
        chunk = Document(
            page_content=content,
            metadata={
                "document_id": doc_id,
                "filename": filename,
                "page_number": 1,
                "user_id": str(user_id),
            },
        )
        doc_info = {
            "document_id": doc_id,
            "filename": filename,
            "status": "processed",
            "chunk_count": 1,
            "user_id": user_id,
        }
        vector_store.add_documents(
            document_id=doc_id,
            chunks=[chunk],
            doc_info=doc_info,
            user_id=user_id,
        )
        print(f"  + Seeded test document: {filename} (id={doc_id})")


async def run_knowledge_reasoning_tests():
    user_id = 112
    chat_service = get_chat_service()
    vector_store = get_vector_store()

    print("=" * 75)
    print(f"VERIFYING KNOWLEDGE-GROUNDED + REASONING RAG SYSTEM (User ID: {user_id})")
    print("=" * 75)

    print("\nEnsuring required conceptual documents exist...")
    await seed_test_documents(vector_store, user_id)

    user_docs = vector_store.list_documents_for_user(user_id)
    doc_names = [d.get("filename") for d in user_docs]
    print(f"Active documents for user {user_id}: {doc_names}\n")

    all_passed = True
    conv_id = f"test-reasoning-{uuid.uuid4().hex[:8]}"

    # =========================================================================
    # TEST 1: Mathematics Conceptual Derivation
    # =========================================================================
    print("-" * 75)
    print("[TEST 1: Mathematics Derivation]")
    print("Question: 'What is 3 + 4?'")
    print("Expected: Calculates 7; grounded in addition concept; NO false refusal.")

    await asyncio.sleep(1.0)
    res1 = await chat_service.get_answer("What is 3 + 4?", conversation_id=conv_id, user_id=user_id)
    ans1 = res1["answer"]
    sources1 = res1.get("sources", [])
    ans_type1 = res1.get("answer_type")

    print(f"Answer Preview: {ans1[:220]}...")
    print(f"Answer Type: {ans_type1}, Sources: {len(sources1)}")
    for s in sources1:
        print(f"  * {s.get('filename')} (Page {s.get('page_number')})")

    has_7 = "7" in ans1 or "seven" in ans1.lower()
    has_refusal = "couldn't find" in ans1.lower() or "could not find" in ans1.lower()
    if has_7 and not has_refusal:
        print("✅ PASS: Correctly derived 3 + 4 = 7 using addition concept without refusal.")
    else:
        print("❌ FAIL: Failed to derive 7 or falsely refused.")
        all_passed = False

    # =========================================================================
    # TEST 2: Code Implementation for Code-less Research Paper
    # =========================================================================
    print("\n" + "-" * 75)
    print("[TEST 2: Code Implementation for Code-less Paper]")
    print("Question: 'Give me Python/PyTorch code to implement a Transformer.'")
    print("Expected: Practical PyTorch code + clear statement that the paper does not contain code.")

    await asyncio.sleep(1.0)
    res2 = await chat_service.get_answer(
        "Give me Python/PyTorch code to implement a Transformer.", conversation_id=conv_id, user_id=user_id
    )
    ans2 = res2["answer"]
    sources2 = res2.get("sources", [])
    ans_type2 = res2.get("answer_type")

    print(f"Answer Preview: {ans2[:240]}...")
    print(f"Answer Type: {ans_type2}, Sources: {len(sources2)}")
    for s in sources2:
        print(f"  * {s.get('filename')} (Page {s.get('page_number')})")

    ans2_clean = re.sub(r'[*_`]', '', ans2.lower())
    has_code = "```python" in ans2 or "import torch" in ans2 or "nn.Module" in ans2
    notes_paper_no_code = (
        "does not provide" in ans2_clean
        or "does not contain" in ans2_clean
        or "do not contain" in ans2_clean
        or "not contain" in ans2_clean
        or "neither contains" in ans2_clean
        or "no source code" in ans2_clean
        or "not present" in ans2_clean
        or "practical" in ans2_clean
        or "based on the architecture" in ans2_clean
    )

    if has_code and notes_paper_no_code:
        print("✅ PASS: Provided practical PyTorch code and acknowledged paper's conceptual nature.")
    else:
        print(f"❌ FAIL: has_code={has_code}, notes_paper_no_code={notes_paper_no_code}")
        all_passed = False

    # =========================================================================
    # TEST 3: Concept Application (Gradient Descent)
    # =========================================================================
    print("\n" + "-" * 75)
    print("[TEST 3: Concept Application]")
    print("Question: 'If gradient is positive, how does the parameter change?'")
    print("Expected: Derives that parameter decreases / moves in negative direction.")

    await asyncio.sleep(1.0)
    res3 = await chat_service.get_answer(
        "If gradient is positive, how does the parameter change in gradient descent?",
        conversation_id=conv_id,
        user_id=user_id,
    )
    ans3 = res3["answer"]
    sources3 = res3.get("sources", [])
    ans_type3 = res3.get("answer_type")

    print(f"Answer Preview: {ans3[:220]}...")
    print(f"Answer Type: {ans_type3}, Sources: {len(sources3)}")
    for s in sources3:
        print(f"  * {s.get('filename')} (Page {s.get('page_number')})")

    has_decrease = "decrease" in ans3.lower() or "reduced" in ans3.lower() or "smaller" in ans3.lower() or "opposite" in ans3.lower()
    if has_decrease and len(sources3) > 0:
        print("✅ PASS: Correctly derived that parameter decreases when gradient is positive.")
    else:
        print("❌ FAIL: Did not correctly derive parameter decrease.")
        all_passed = False

    # =========================================================================
    # TEST 4: Multi-Document Reasoning & Synthesis
    # =========================================================================
    print("\n" + "-" * 75)
    print("[TEST 4: Multi-Document Reasoning & Synthesis]")
    print("Question: 'Use Python to implement the Transformer architecture described in the paper.'")
    print("Expected: Synthesizes Python fundamentals + Transformer architecture concepts.")

    await asyncio.sleep(1.0)
    res4 = await chat_service.get_answer(
        "Use Python to implement the Transformer architecture described in the paper.",
        conversation_id=conv_id,
        user_id=user_id,
    )
    ans4 = res4["answer"]
    sources4 = res4.get("sources", [])
    ans_type4 = res4.get("answer_type")

    print(f"Answer Preview: {ans4[:240]}...")
    print(f"Answer Type: {ans_type4}, Sources: {len(sources4)}")
    for s in sources4:
        print(f"  * {s.get('filename')} (Page {s.get('page_number')})")

    has_synthesis = ("python" in ans4.lower() or "torch" in ans4.lower()) and "transformer" in ans4.lower()
    if has_synthesis and ("```" in ans4 or "def " in ans4 or "class " in ans4):
        print("✅ PASS: Successfully synthesized concepts across Python and Transformer documents.")
    else:
        print("❌ FAIL: Multi-document synthesis failed.")
        all_passed = False

    # =========================================================================
    # TEST 5: Conversational Follow-up Resolution
    # =========================================================================
    print("\n" + "-" * 75)
    print("[TEST 5: Conversational Follow-up Resolution]")
    followup_conv = f"test-followup-{uuid.uuid4().hex[:8]}"

    print("Step 1: 'Explain the Transformer architecture.'")
    await chat_service.get_answer(
        "Explain the Transformer architecture.", conversation_id=followup_conv, user_id=user_id
    )

    print("Step 2: 'Now give me code for it.'")
    await asyncio.sleep(1.0)
    res5 = await chat_service.get_answer(
        "Now give me code for it.", conversation_id=followup_conv, user_id=user_id
    )
    ans5 = res5["answer"]
    sources5 = res5.get("sources", [])
    ans_type5 = res5.get("answer_type")

    print(f"Answer Preview: {ans5[:240]}...")
    print(f"Answer Type: {ans_type5}, Sources: {len(sources5)}")

    resolves_it = (
        "transformer" in ans5.lower()
        or "attention" in ans5.lower()
        or "torch" in ans5.lower()
    ) and ("```" in ans5 or "class " in ans5 or "def " in ans5)

    if resolves_it:
        print("✅ PASS: Successfully resolved 'it' to Transformer and provided implementation code.")
    else:
        print("❌ FAIL: Failed to resolve pronoun 'it' to conversational subject.")
        all_passed = False

    # =========================================================================
    # TEST 6: Truly Insufficient Information
    # =========================================================================
    print("\n" + "-" * 75)
    print("[TEST 6: Truly Insufficient Information Handling]")
    print("Question: 'What was the exact dollar cost in US Dollars to train the Transformer in the paper?'")
    print("Expected: Correctly states this document-specific fact is not available.")

    await asyncio.sleep(1.0)
    res6 = await chat_service.get_answer(
        "What was the exact dollar cost in US Dollars to train the Transformer in the paper?",
        conversation_id=conv_id,
        user_id=user_id,
    )
    ans6 = res6["answer"]
    sources6 = res6.get("sources", [])
    ans_type6 = res6.get("answer_type")

    print(f"Answer Preview: {ans6[:220]}...")
    print(f"Answer Type: {ans_type6}, Sources: {len(sources6)}")

    ans6_clean = re.sub(r'[*_`]', '', ans6.lower())
    states_unavailable = (
        "couldn't find" in ans6_clean
        or "could not find" in ans6_clean
        or "does not mention" in ans6_clean
        or "do not contain" in ans6_clean
        or "does not contain" in ans6_clean
        or "not provide" in ans6_clean
        or "not mentioned" in ans6_clean
        or "not present" in ans6_clean
        or "not stated" in ans6_clean
        or "does not specify" in ans6_clean
    )

    if states_unavailable:
        print("✅ PASS: Correctly recognized truly missing document-specific fact without hallucinating.")
    else:
        print("❌ FAIL: Did not properly state insufficient information.")
        all_passed = False

    # =========================================================================
    # PART 7: User Isolation Check
    # =========================================================================
    print("\n" + "=" * 75)
    print("[PART 7: User Isolation Verification]")
    unauthorized_user_id = 999999
    context, unauth_raw, analysis = await asyncio.to_thread(
        chat_service._retrieve_context, "What is 3 + 4 in the math document?", unauthorized_user_id, []
    )
    unauth_multi = chat_service.vector_store.similarity_search_multi_doc_for_user(
        ["addition", "Transformer"], user_id=unauthorized_user_id
    )
    print(f"Unauthorized User {unauthorized_user_id} RAG Chunks: {len(unauth_raw)}")
    print(f"Unauthorized User {unauthorized_user_id} Multi Search Chunks: {len(unauth_multi)}")

    assert len(unauth_raw) == 0, "SECURITY LEAK: Unauthorized user retrieved chunks!"
    assert len(unauth_multi) == 0, "SECURITY LEAK: Unauthorized user retrieved multi-search chunks!"
    assert context == "", "SECURITY LEAK: Non-empty context returned for unauthorized user!"
    print("✅ PASS: Cross-user isolation strictly maintained (0 chunks returned).")

    print("\n" + "=" * 75)
    if all_passed:
        print("🎉 ALL 6 KNOWLEDGE-GROUNDED REASONING TESTS AND ISOLATION PASSED!")
    else:
        print("❌ ONE OR MORE TESTS FAILED.")
    print("=" * 75)


if __name__ == "__main__":
    asyncio.run(run_knowledge_reasoning_tests())
