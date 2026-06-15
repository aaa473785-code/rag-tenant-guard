"""
LLM helper for rag-tenant-guard.

- Anthropic
- OpenAI

モデルIDはAPI提供状況で変わるため、画面上で編集できるようにしてあります。
"""

from __future__ import annotations


PROVIDER_MODELS = {
    "Anthropic": {
        "Claude Haiku": "claude-haiku-4-5-20251001",
        "Claude Sonnet": "claude-sonnet-4-6",
        "Claude Opus": "claude-opus-4-8",
        "カスタム": "",
    },
    "OpenAI": {
        "GPT mini": "gpt-5.5-mini",
        "GPT": "gpt-5.5",
        "カスタム": "",
    },
}


def estimate_tokens(text: str) -> int:
    """
    PoC用の簡易トークン見積もり。
    実API利用時はレスポンスのusageを使う。
    """
    if not text:
        return 0
    return max(1, len(text) // 2)


def estimate_cost_usd(
    input_tokens: int,
    output_tokens: int,
    input_per_mtok: float,
    output_per_mtok: float,
) -> float:
    return (input_tokens / 1_000_000 * input_per_mtok) + (
        output_tokens / 1_000_000 * output_per_mtok
    )


def build_prompt(query: str, results: list[dict]) -> str:
    context = "\n\n".join(
        f"[{r['doc_id']}] {r['title']}\n{r['content']}"
        for r in results
    )

    return f"""あなたは社内向けAIアシスタントです。
以下の参照文書だけを使って、簡潔に回答してください。

ルール:
- 参照文書にない情報は推測しない
- 他テナントの情報は使わない
- 最後に参照した文書IDを明記する

【質問】
{query}

【参照文書】
{context}
"""


def generate_answer(
    *,
    provider: str,
    model_id: str,
    api_key: str,
    query: str,
    results: list[dict],
) -> dict:
    """
    返却形式:
    {
        "answer": str,
        "input_tokens": int,
        "output_tokens": int,
        "note": str,
    }
    """
    prompt = build_prompt(query, results)

    if not api_key:
        raise ValueError("APIキーを入力してください。")

    if not model_id:
        raise ValueError("モデルIDを入力してください。")

    if provider == "Anthropic":
        return _generate_with_anthropic(
            api_key=api_key,
            model_id=model_id,
            prompt=prompt,
        )

    if provider == "OpenAI":
        return _generate_with_openai(
            api_key=api_key,
            model_id=model_id,
            prompt=prompt,
        )

    raise ValueError(f"未対応のプロバイダーです: {provider}")


def _generate_with_anthropic(*, api_key: str, model_id: str, prompt: str) -> dict:
    try:
        import anthropic
    except ImportError as e:
        raise ImportError(
            "anthropic パッケージがありません。python -m pip install -r requirements.txt を実行してください。"
        ) from e

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model_id,
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )

    answer = message.content[0].text
    usage = getattr(message, "usage", None)

    return {
        "answer": answer,
        "input_tokens": getattr(usage, "input_tokens", estimate_tokens(prompt)),
        "output_tokens": getattr(usage, "output_tokens", estimate_tokens(answer)),
        "note": "Anthropic API生成",
    }


def _generate_with_openai(*, api_key: str, model_id: str, prompt: str) -> dict:
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError(
            "openai パッケージがありません。python -m pip install -r requirements.txt を実行してください。"
        ) from e

    client = OpenAI(api_key=api_key)

    # Chat Completions形式。利用モデルによりResponses APIへ変更してください。
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
    )

    answer = response.choices[0].message.content or ""
    usage = getattr(response, "usage", None)

    return {
        "answer": answer,
        "input_tokens": getattr(usage, "prompt_tokens", estimate_tokens(prompt)),
        "output_tokens": getattr(usage, "completion_tokens", estimate_tokens(answer)),
        "note": "OpenAI API生成",
    }
