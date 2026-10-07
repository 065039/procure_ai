import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# Load .env from the project root
load_dotenv()

PROMPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "prompts"
    / "procurement_prompt.txt"
)


def _get_api_key():
    """
    Get OpenRouter API key from Streamlit secrets first,
    then fall back to the local .env file.
    """
    try:
        import streamlit as st

        if "OPENROUTER_API_KEY" in st.secrets:
            return st.secrets["OPENROUTER_API_KEY"]
    except Exception:
        pass

    return os.getenv("OPENROUTER_API_KEY")


def _build_prompt(top_vendor, runner_up, weights):
    """
    Build a structured prompt using deterministic ranking results.

    IMPORTANT:
    The LLM is only responsible for explanation.
    It must NOT recalculate or change the ranking.
    """

    try:
        system_instructions = PROMPT_PATH.read_text(
            encoding="utf-8"
        )
    except Exception:
        system_instructions = """
You are a procurement decision-support assistant.

Explain the vendor recommendation using ONLY the supplied
deterministic ranking results.

Do not recalculate scores.
Do not change the ranking.
Do not invent vendor data.

Discuss:
1. Why the top vendor ranked first
2. The key trade-offs
3. Comparison with the runner-up
4. Procurement risks or considerations
5. A concise recommendation

Clearly state that the final procurement decision remains with
the human procurement professional.
"""

    def clean_value(value):
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass

        if isinstance(value, dict):
            return {
                str(k): clean_value(v)
                for k, v in value.items()
            }

        return value

    payload = {
        "top_vendor": clean_value(top_vendor),
        "runner_up": clean_value(runner_up),
        "weights": clean_value(weights),
    }

    return f"""
{system_instructions}

--------------------------------------------------
DETERMINISTIC PROCUREMENT ANALYSIS
--------------------------------------------------

The following results were calculated by the Python
multi-criteria scoring engine.

These values are authoritative.

{json.dumps(payload, indent=2, default=str)}

--------------------------------------------------
IMPORTANT RULES
--------------------------------------------------

- Do NOT recalculate the ranking.
- Do NOT modify the scores.
- Do NOT invent missing information.
- Do NOT introduce a different vendor as the recommendation.
- Explain the supplied results only.
- Clearly distinguish facts from interpretation.
- The final procurement decision belongs to the human user.

Provide a concise, professional procurement analysis.
"""


def generate_ai_explanation(top_vendor, runner_up, weights):
    """
    Generate an AI explanation for the deterministic
    vendor recommendation.

    Returns:
        {
            "success": bool,
            "message": str | None,
            "text": str | None
        }
    """

    api_key = _get_api_key()

    if not api_key:
        return {
            "success": False,
            "message": (
                "OpenRouter API key is not configured. "
                "Add OPENROUTER_API_KEY to Streamlit Secrets "
                "or your .env file."
            ),
            "text": None,
        }

    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "http://localhost:8501",
                "X-Title": "ProcureAI",
            },
        )

        prompt = _build_prompt(
            top_vendor=top_vendor,
            runner_up=runner_up,
            weights=weights,
        )

        response = client.chat.completions.create(
            model=os.getenv(
                "OPENROUTER_MODEL",
                "openrouter/free"
            ),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a procurement decision-support "
                        "explainer. Explain deterministic "
                        "vendor-ranking results without "
                        "recalculating or altering them."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.2,
        )

        text = response.choices[0].message.content

        if not text:
            return {
                "success": False,
                "message": (
                    "OpenRouter returned an empty AI response."
                ),
                "text": None,
            }

        return {
            "success": True,
            "message": None,
            "text": text.strip(),
        }

    except Exception as exc:
        return {
            "success": False,
            "message": (
                "AI explanation is currently unavailable. "
                "The deterministic vendor ranking remains available. "
                f"Technical reason: {type(exc).__name__}: {exc}"
            ),
            "text": None,
        }