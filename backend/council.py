"""3-stage LLM Council orchestration (local via Ollama).
Pipeline en 3 étapes :
1) Stage 1 : chaque modèle du Council répond à la question utilisateur.
2) Stage 2 : chaque modèle (reviewer) évalue et classe les réponses anonymisées.
3) Stage 3 : le Chairman synthétise une réponse finale à partir des réponses + rankings.

Objectif : améliorer la robustesse et la qualité de la réponse finale en combinant
plusieurs modèles (diversité) + une phase de peer-review + une synthèse finale."""

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
# Appelle un modèle via Ollama avec mécanisme de retry.
# Arguments : base_url: URL du serveur Ollama (Council ou Chairman), model: nom du modèle Ollama à appeler (ex: "llama3"), messages: conversation au format chat (system/user/assistant), timeout: timeout max pour la requête.
# Returns : model: nom du modèle appelé, text: réponse texte si succès, sinon None, error: message d'erreur si échec, sinon None 
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
    Stage 1: collecter une réponse de chaque modèle du Council.
    On envoie la même question à tous les modèles (en parallèle selon MAX_PARALLEL_REQUESTS).
    Args:
        user_query: question / prompt utilisateur
    Returns: Liste de dicts :
        [
          {"model": "...", "response": "...", "error": None|"..."},
          ...
        ]
    """

    # Messages de conversation envoyés aux modèles :
    # - system: impose des contraintes globales (langue identique à l'utilisateur, concision)
    # - user: contient la question réelle
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
    # Semaphore = limite le nombre de requêtes simultanées
    # (utile pour la stabilité Ollama / ressources machine)

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

    # On crée une tâche par modèle, puis on attend toutes les réponses
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
    Stage 2: chaque modèle "reviewer" évalue et classe (ranking) les réponses anonymisées.
     Principe :
      - On anonymise les réponses du Stage 1 en les étiquetant A, B, C...
      - Chaque reviewer reçoit toutes les réponses et doit produire :
          * une critique (forces/faiblesses)
          * un classement final strictement formaté (FINAL RANKING)
    Args:
        user_query: question originale
        stage1_results: résultats du Stage 1 (réponses + erreurs)

    Returns:
        (stage2_results, label_to_model)
        - stage2_results : [
            {"model": "...", "ranking": "...", "parsed_ranking": [...], "error": None|"..."},
            ...
          ]
        - label_to_model : mapping permettant de "désanonymiser"
            {"Response A": "llama3", "Response B": "mistral", ...}
    """

    # On ne garde que les réponses valides
    valid_stage1 = [r for r in stage1_results if r.get("response")]

    # S'il n'y a pas au moins 2 réponses, on ne peut pas faire de ranking
    if len(valid_stage1) < 2:
        return [], {}

    # Création des labels A, B, C... pour anonymiser les réponses
    labels = [chr(65 + i) for i in range(len(valid_stage1))]  # A, B, C...
    label_to_model = {
        f"Response {label}": result["model"]
        for label, result in zip(labels, valid_stage1)
    }

    # Bloc texte contenant toutes les réponses anonymisées
    responses_text = "\n\n".join(
        [
            f"Response {label}:\n{result['response']}"
            for label, result in zip(labels, valid_stage1)
        ]
    )

    # On génère le format EXACT attendu pour le ranking final
    # Exemple:
    # 1. Response A
    # 2. Response B
    expected_lines = "\n".join([f"{i}. Response {label}" for i, label in enumerate(labels, start=1)])

    # Prompt envoyé aux reviewers : évaluation + ranking final strict
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
    # Limite de concurrence (évite surcharge)
    sem = asyncio.Semaphore(MAX_PARALLEL_REQUESTS)

    # Choix : seuls les modèles qui ont répondu au Stage 1 font les reviewers
    # (sinon un modèle qui a crashé Stage 1 pourrait être instable au Stage 2 aussi)
    reviewer_models = [r["model"] for r in valid_stage1]

    # Lance l'évaluation / ranking par un reviewer.
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

# On normalise la structure de sortie et on parse le ranking
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
    Stage 3 : le Chairman produit une réponse finale.

    Le Chairman voit :
      - la question originale
      - les réponses brutes (Stage 1)
      - les classements/évaluations (Stage 2)

    Il doit :
      - écrire UNE seule réponse finale structurée
      - arbitrer les divergences (ou signaler une incertitude)

    Returns:
        {"model": CHAIRMAN_MODEL, "response": "..."}

    """
     # On filtre les réponses valides
    valid_stage1 = [r for r in stage1_results if r.get("response")]
    valid_stage2 = [r for r in stage2_results if r.get("ranking")]

    # Texte regroupant Stage 1 (réponses)
    stage1_text = "\n\n".join(
        [f"Model: {r['model']}\nResponse: {r['response']}" for r in valid_stage1]
    )

    # Texte regroupant Stage 2 (rankings)
    stage2_text = "\n\n".join(
        [f"Model: {r['model']}\nRanking: {r['ranking']}" for r in valid_stage2]
    )

    # Prompt du Chairman : il reçoit le contexte complet
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
        #  # Appel au modèle "Chairman" sur l'autre machine
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

    # Sécurité : si le modèle renvoie vide
    if not final_text:
        return {"model": CHAIRMAN_MODEL, "response": "Error: Chairman returned empty response."}

    return {"model": CHAIRMAN_MODEL, "response": final_text}


# Extrait la liste ordonnée des labels "Response X" depuis la section "FINAL RANKING".
# Tolère :
      # - "FINAL RANKING:" ou "Final Ranking:"
      # - du texte additionnel après le label (ex: "1. Response A - best answer")
# Args : ranking_text: texte complet produit par le reviewer
# Returns : Liste de labels, ex: ["Response A", "Response C", "Response B"] (dédupliquée et dans l'ordre)

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


# Agrège les rankings du Stage 2 pour produire un classement global.
# Méthode :
# - Pour chaque reviewer, on récupère parsed_ranking (liste ordonnée de labels)
# - On transforme les labels (Response A, ...) en noms de modèles via label_to_model
# - On stocke la position (1 = meilleur) dans model_positions
# - On calcule ensuite un rang moyen par modèle (average_rank)
# Args: stage2_results: résultats des reviewers (ranking texte + parsed_ranking), label_to_model: mapping "Response A" -> "nom_modèle"
# Returns : Liste triée par average croissant

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

    # On calcule la moyenne des positions observées pour chaque modèle
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
    # Plus average_rank est petit, mieux c'est
    aggregate.sort(key=lambda x: x["average_rank"])
    return aggregate

# Génère un titre très court (3-5 mots) résumant la question.
# Utilité :
# - nommer automatiquement une conversation sauvegardée
# - améliorer la navigation dans l'historique
# Fallback :
# - si échec ou réponse vide -> "New Conversation"
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

# Lance le pipeline complet du Council.
# Steps:
# 1) Stage 1 : réponses individuelles
# 2) Stage 2 : rankings (si possible)
# 3) Agrégation : score moyen par modèle (optionnel mais utile)
# 4) Stage 3 : synthèse finale par le Chairman
# Returns : 
# (stage1_results, stage2_results, stage3_result, metadata)
# - metadata contient:* label_to_model (désanonymisation), * aggregate_rankings (classement global)

async def run_full_council(user_query: str) -> Tuple[List, List, Dict, Dict]:
    stage1_results = await stage1_collect_responses(user_query)

    # Si aucun modèle n'a répondu, on stoppe proprement
    valid_stage1 = [r for r in stage1_results if r.get("response")]
    if not valid_stage1:
        return stage1_results, [], {"model": "error", "response": "All models failed to respond."}, {}

    stage2_results, label_to_model = await stage2_collect_rankings(user_query, stage1_results)
     # Ranking global utile pour logs/analytics (même si le Chairman fait sa synthèse)
    aggregate_rankings = calculate_aggregate_rankings(stage2_results, label_to_model)

    stage3_result = await stage3_synthesize_final(user_query, stage1_results, stage2_results)

    metadata = {"label_to_model": label_to_model, "aggregate_rankings": aggregate_rankings}
    return stage1_results, stage2_results, stage3_result, metadata
