"""3-stage LLM Council orchestration (local via Ollama)."""

from typing import List, Dict, Any, Tuple, Optional
import asyncio
import random

from .ollama_client import ollama_chat
from .config import (
    COUNCIL_MODELS,
    CHAIRMAN_MODEL,
    COUNCIL_BASE_URL,
    CHAIRMAN_BASE_URL,
    MAX_PARALLEL_REQUESTS,
    RETRY_COUNT,
    RETRY_SLEEP_SECONDS,
)


async def _call_with_retries(base_url: str, model: str, messages: List[Dict[str, str]], timeout: float) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Returns (model, text, error)
    """
    last_err = None
    for attempt in range(RETRY_COUNT + 1):
        try:
            text = await ollama_chat(
                base_url=base_url,
                model=model,
                messages=messages,
                timeout=timeout,
            )
            return model, text, None
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            # petit backoff pour éviter de spam Ollama
            if attempt < RETRY_COUNT:
                await asyncio.sleep(RETRY_SLEEP_SECONDS + random.random() * 0.3)
    return model, None, last_err


async def stage1_collect_responses(user_query: str) -> List[Dict[str, Any]]:
    """
    Stage 1: Collect individual responses from all council models.
    Returns: [{"model": "...", "response": "...", "error": None|"..."}]
    """
    messages = [
    {
        "role": "system",
        "content": (
            "Answer in the SAME language as the user's question. "
            "Do not switch languages. Be clear and concise."
        ),
    },
    {"role": "user", "content": user_query},
]
    sem = asyncio.Semaphore(MAX_PARALLEL_REQUESTS)

    async def run_one(model: str):
        async with sem:
            model, text, err = await _call_with_retries(
                base_url=COUNCIL_BASE_URL,
                model=model,
                messages=messages,
                timeout=300.0,
            )
            return {"model": model, "response": text, "error": err}

    tasks = [run_one(m) for m in COUNCIL_MODELS]
    results = await asyncio.gather(*tasks)

    # IMPORTANT: on garde aussi les erreurs dans les résultats (débug)
    # mais pour la suite, on utilisera seulement ceux qui ont une réponse.
    return results


async def stage2_collect_rankings(
    user_query: str,
    stage1_results: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Stage 2: Each model ranks anonymized responses.
    Returns: (stage2_results, label_to_model)
    """

    # On ne garde que les réponses valides
    valid_stage1 = [r for r in stage1_results if r.get("response")]

    if len(valid_stage1) < 2:
        # Pas assez de réponses pour faire un ranking
        return [], {}

    labels = [chr(65 + i) for i in range(len(valid_stage1))]  # A, B, C...
    label_to_model = {
        f"Response {label}": result["model"]
        for label, result in zip(labels, valid_stage1)
    }

    responses_text = "\n\n".join(
        [
            f"Response {label}:\n{result['response']}"
            for label, result in zip(labels, valid_stage1)
        ]
    )

    # Prompt dynamique: le modèle doit classer EXACTEMENT le nombre de réponses qu’il a
    expected_lines = "\n".join([f"{i}. Response {label}" for i, label in enumerate(labels, start=1)])

    ranking_prompt = f"""You are evaluating different responses to the following question:

Question: {user_query}

Here are the responses from different models (anonymized):

{responses_text}

Your task:
1. Evaluate each response briefly: strengths + weaknesses.
2. At the very end, provide a final ranking of ALL responses.

IMPORTANT: Your final ranking MUST be formatted EXACTLY like this (same labels, same count):

FINAL RANKING:
{expected_lines}

Now provide your evaluation and ranking:"""

    messages = [
    {
        "role": "system",
        "content": (
            "Answer in the SAME language as the original user question. "
            "Follow the required output format STRICTLY."
        ),
    },
    {"role": "user", "content": ranking_prompt},
]

    sem = asyncio.Semaphore(MAX_PARALLEL_REQUESTS)

    # Option conseillé : ne faire reviewer que les modèles qui ont déjà tenu Stage1
    reviewer_models = [r["model"] for r in valid_stage1]

    async def run_one(model: str):
        async with sem:
            model, text, err = await _call_with_retries(
                base_url=COUNCIL_BASE_URL,
                model=model,
                messages=messages,
                timeout=420.0,
            )
            return {"model": model, "ranking": text, "error": err}

    tasks = [run_one(m) for m in reviewer_models]
    results = await asyncio.gather(*tasks)

    stage2_results: List[Dict[str, Any]] = []
    for r in results:
        if r.get("ranking"):
            stage2_results.append(
                {
                    "model": r["model"],
                    "ranking": r["ranking"],
                    "parsed_ranking": parse_ranking_from_text(r["ranking"]),
                    "error": None,
                }
            )
        else:
            stage2_results.append(
                {
                    "model": r["model"],
                    "ranking": None,
                    "parsed_ranking": [],
                    "error": r.get("error"),
                }
            )

    return stage2_results, label_to_model


async def stage3_synthesize_final(
    user_query: str,
    stage1_results: List[Dict[str, Any]],
    stage2_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Stage 3: Chairman synthesizes final response.
    Returns: {"model": "...", "response": "..."}
    """

    valid_stage1 = [r for r in stage1_results if r.get("response")]
    valid_stage2 = [r for r in stage2_results if r.get("ranking")]

    stage1_text = "\n\n".join(
        [f"Model: {r['model']}\nResponse: {r['response']}" for r in valid_stage1]
    )

    stage2_text = "\n\n".join(
        [f"Model: {r['model']}\nRanking: {r['ranking']}" for r in valid_stage2]
    )

    chairman_prompt = f"""You are the Chairman of an LLM Council.

Original Question:
{user_query}

STAGE 1 - Individual Responses:
{stage1_text if stage1_text else "(No valid responses received.)"}

STAGE 2 - Peer Rankings:
{stage2_text if stage2_text else "(No valid rankings received.)"}

Task:
Write ONE final answer to the original question.
Be accurate, clear, and structured.
If there are disagreements, resolve them or mention uncertainty briefly.
"""

    try:
        final_text = await ollama_chat(
            base_url=CHAIRMAN_BASE_URL,
            model=CHAIRMAN_MODEL,
            messages=[
    {
        "role": "system",
        "content": (
            "Write the final answer in the SAME language as the user's question. "
            "Do not switch languages."
        ),
    },
    {"role": "user", "content": chairman_prompt},
],

            timeout=420.0,
        )
    except Exception as e:
        return {"model": CHAIRMAN_MODEL, "response": f"Error: Chairman failed: {type(e).__name__}: {e}"}

    if not final_text:
        return {"model": CHAIRMAN_MODEL, "response": "Error: Chairman returned empty response."}

    return {"model": CHAIRMAN_MODEL, "response": final_text}


def parse_ranking_from_text(ranking_text: str) -> List[str]:
    import re
    if not ranking_text:
        return []

    # Accepte FINAL RANKING ou Final Ranking, et isole tout ce qui suit
    m = re.search(r"(FINAL RANKING:|Final Ranking:)\s*(.*)", ranking_text, re.DOTALL)
    if not m:
        return []

    ranking_section = m.group(2)

    # Capture uniquement les lignes de classement numérotées (ex: "1. Response A") tout en tolérant du texte explicatif après le label (ex: "1. Response A - best answer").
    lines = re.findall(r"^\s*\d+\.\s*(Response [A-Z])\b.*$", ranking_section, re.MULTILINE)
    

    if not lines:
        return []

    # Déduplique en gardant l'ordre
    out = []
    for x in lines:
        if x not in out:
            out.append(x)

    return out



def calculate_aggregate_rankings(
    stage2_results: List[Dict[str, Any]],
    label_to_model: Dict[str, str],
) -> List[Dict[str, Any]]:
    from collections import defaultdict

    if not label_to_model:
        return []

    model_positions = defaultdict(list)

    for ranking in stage2_results:
        parsed_ranking = ranking.get("parsed_ranking") or parse_ranking_from_text(ranking.get("ranking") or "")
        
        parsed_ranking = parsed_ranking[:len(label_to_model)]
        
        for position, label in enumerate(parsed_ranking, start=1):
            if label in label_to_model:
                model_name = label_to_model[label]
                model_positions[model_name].append(position)

    aggregate = []
    for model, positions in model_positions.items():
        if positions:
            avg_rank = sum(positions) / len(positions)
            aggregate.append(
                {
                    "model": model,
                    "average_rank": round(avg_rank, 2),
                    "rankings_count": len(positions),
                }
            )

    aggregate.sort(key=lambda x: x["average_rank"])
    return aggregate


async def generate_conversation_title(user_query: str) -> str:
    title_prompt = f"""Generate a very short title (3-5 words max) summarizing this question.
No quotes, no punctuation.

Question: {user_query}

Title:"""

    try:
        title = await ollama_chat(
            base_url=CHAIRMAN_BASE_URL,
            model=CHAIRMAN_MODEL,
            messages=[{"role": "user", "content": title_prompt}],
            timeout=60.0,
        )
    except Exception:
        title = None

    if not title:
        return "New Conversation"

    title = title.strip().strip('"\'')
    return title[:50]


async def run_full_council(user_query: str) -> Tuple[List, List, Dict, Dict]:
    stage1_results = await stage1_collect_responses(user_query)

    valid_stage1 = [r for r in stage1_results if r.get("response")]
    if not valid_stage1:
        return stage1_results, [], {"model": "error", "response": "All models failed to respond."}, {}

    stage2_results, label_to_model = await stage2_collect_rankings(user_query, stage1_results)
    aggregate_rankings = calculate_aggregate_rankings(stage2_results, label_to_model)

    stage3_result = await stage3_synthesize_final(user_query, stage1_results, stage2_results)

    metadata = {"label_to_model": label_to_model, "aggregate_rankings": aggregate_rankings}
    return stage1_results, stage2_results, stage3_result, metadata
