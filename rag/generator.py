# rag/generator.py
import os, textwrap
from typing import List, Dict
from openai import OpenAI, APIConnectionError, APIStatusError


def template_answer(query: str, hits: List[Dict]) -> str:
    """Offline fallback answer."""
    bullets = "\n".join(
        [f"- [{i+1}] ({h['source']}#{h['chunk']}) {h['text'][:160]}..." for i, h in enumerate(hits)]
    )
    return textwrap.dedent(f"""
    Answer grounded in retrieved context (no LLM mode).

    Question: {query}

    Top references:
    {bullets}

    (Enable OpenAI in .env for model-generated answers.)
    """).strip()


def openai_answer(query: str, hits: List[Dict]) -> str:
    """Generate a grounded answer using OpenAI chat completions."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return "⚠️ OPENAI_API_KEY not set. Falling back to template answer."

    try:
        client = OpenAI(api_key=api_key)

        # Build context from retrieved chunks
        context = "\n\n".join(
            [f"[{i+1}] ({h['source']}#{h['chunk']}) {h['text']}" for i, h in enumerate(hits)]
        )

        # Construct grounded prompt
        prompt = textwrap.dedent(f"""
        You are a helpful assistant answering questions based only on the provided context.
        Use citations [1], [2], etc., to reference the relevant parts.

        Context:
        {context}

        Question: {query}
        """)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=500,
        )

        return response.choices[0].message.content.strip()

    except (APIConnectionError, APIStatusError) as e:
        return f"⚠️ OpenAI API connection failed: {e}"
    except Exception as e:
        return f"⚠️ OpenAI generation failed: {e}"
