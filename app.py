"""
rag-tenant-guard

マルチテナントRAGの安全性・AIあり/なし比較PoC
- tenant_id による検索範囲制御
- AIあり/なし比較
- Anthropic / OpenAI の切り替え
- 実行ごとの概算コスト
- 監査ログ
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from db import (
    init_db,
    fetch_users,
    fetch_documents,
    insert_audit_log,
    fetch_audit_logs,
)
from rag_engine import TenantSearchEngine, classify_result_type
from llm_client import (
    PROVIDER_MODELS,
    build_prompt,
    estimate_tokens,
    estimate_cost_usd,
    generate_answer,
)


USD_JPY = 150

# PoC用の概算単価。
# 実案件では選択モデルの最新料金に合わせて変更する。
PRICE_INPUT_PER_MTOK = 1.0
PRICE_OUTPUT_PER_MTOK = 5.0


st.set_page_config(
    page_title="rag-tenant-guard",
    page_icon="🛡️",
    layout="wide",
)


# ==============================
# CSS
# ==============================
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 3.4rem !important;
        padding-bottom: 2rem;
        max-width: 1180px;
    }
    .app-title {
        font-size: 1.45rem;
        line-height: 1.45;
        font-weight: 700;
        margin-top: 0.4rem;
        margin-bottom: 0.35rem;
    }
    .app-subtitle {
        font-size: 0.92rem;
        line-height: 1.6;
        color: #666;
        margin-bottom: 1.2rem;
    }
    h1 {
        font-size: 1.65rem !important;
    }
    h2 {
        font-size: 1.25rem !important;
        margin-top: 1.2rem !important;
    }
    h3 {
        font-size: 1.05rem !important;
    }
    .small-note {
        font-size: 0.88rem;
        color: #666;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.35rem;
    }
    section[data-testid="stSidebar"] {
        padding-top: 0.8rem;
    }
    section[data-testid="stSidebar"] h1 {
        font-size: 1.25rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_search_engine():
    init_db()
    documents = fetch_documents()
    engine = TenantSearchEngine(documents)
    engine.build_index()
    return engine


@st.cache_data(ttl=3)
def load_users():
    init_db()
    return fetch_users()


def render_failure_message(result_type: str) -> str:
    if result_type == "no_permission":
        return (
            "回答できません。\n\n"
            "理由：この情報は存在しますが、現在のユーザー権限では参照できません。"
        )
    if result_type == "no_information":
        return (
            "回答できません。\n\n"
            "理由：参照可能な文書内にも、全体の文書内にも、該当する情報が見つかりませんでした。"
        )
    return ""


def mode_label_to_value(label: str) -> str:
    if label.startswith("AIなし"):
        return "search_only"
    return "ai_answer"


def calc_average_cost_jpy() -> float:
    logs = fetch_audit_logs(limit=100000)
    if not logs:
        return 0.0
    total = sum(float(row.get("cost_jpy") or 0) for row in logs)
    return total / len(logs)


# ==============================
# Initialize
# ==============================
init_db()
engine = load_search_engine()
users = load_users()


if "total_input_tokens" not in st.session_state:
    st.session_state.total_input_tokens = 0
if "total_output_tokens" not in st.session_state:
    st.session_state.total_output_tokens = 0
if "total_cost_jpy" not in st.session_state:
    st.session_state.total_cost_jpy = 0.0


# ==============================
# Sidebar
# ==============================
with st.sidebar:
    st.title("🛡️ rag-tenant-guard")
    st.caption("マルチテナントRAG PoC")

    st.subheader("モデル設定")

    provider = st.selectbox(
        "プロバイダー",
        ["Anthropic", "OpenAI"],
        help="AIありモードで利用するAPIを選択します。",
    )

    model_options = list(PROVIDER_MODELS[provider].keys())
    model_label = st.selectbox("モデル", model_options)
    default_model_id = PROVIDER_MODELS[provider][model_label]

    model_id = st.text_input(
        "モデルID",
        value=default_model_id,
        help="利用中の正式なモデルIDに合わせて編集できます。",
    )

    api_key = st.text_input(
        "APIキー",
        type="password",
        help="AIありモードで使用します。AIなしモードでは不要です。",
    )

    st.divider()

    st.subheader("利用ユーザー")
    user_labels = [
        f"{u['user_name']} / tenant={u['tenant_id']} / role={u['role']}"
        for u in users
    ]
    selected_user_label = st.selectbox("ユーザー", user_labels)
    current_user = users[user_labels.index(selected_user_label)]

    st.caption(f"user_id: `{current_user['user_id']}`")
    st.caption(f"tenant_id: `{current_user['tenant_id']}`")
    st.caption(f"role: `{current_user['role']}`")

    st.divider()

    st.subheader("実行設定")
    mode_label = st.radio(
        "実行モード",
        [
            "AIなし：検索結果だけ表示（0円）",
            "AIあり：検索結果から回答文を生成",
        ],
    )
    mode = mode_label_to_value(mode_label)

    top_k = st.slider("検索件数", 1, 5, 3)
    threshold = st.slider(
        "閾値",
        0.00,
        0.50,
        0.05,
        0.01,
        help="この値未満の検索候補は除外します。",
    )

    show_unsafe_compare = st.checkbox(
        "危険例：全テナント検索も比較表示",
        value=True,
        help="PoC説明用。本番ではAIに渡さない。",
    )

    st.divider()

    st.subheader("累計")
    total_cost_usd = estimate_cost_usd(
        st.session_state.total_input_tokens,
        st.session_state.total_output_tokens,
        PRICE_INPUT_PER_MTOK,
        PRICE_OUTPUT_PER_MTOK,
    )
    st.caption(f"累計入力: {st.session_state.total_input_tokens} tokens")
    st.caption(f"累計出力: {st.session_state.total_output_tokens} tokens")
    st.caption(f"累計コスト: 約{st.session_state.total_cost_jpy:.2f}円")

    st.divider()

    st.subheader("月間試算")
    monthly_runs = st.number_input("月間実行回数", min_value=0, value=3000, step=100)
    avg_cost = calc_average_cost_jpy()
    st.caption(f"平均コスト/回: 約{avg_cost:.4f}円")
    st.caption(f"月間概算: 約{avg_cost * monthly_runs:.0f}円")


# ==============================
# Main
# ==============================
st.markdown('<div class="app-title">🛡️ マルチテナントRAG：AIあり/なし・安全性</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">tenant_idで検索対象を制御し、AIに渡す前に他テナント情報を除外するPoCです。</div>',
    unsafe_allow_html=True,
)

with st.expander("このPoCで見せたいこと", expanded=False):
    st.markdown(
        """
- AIなしなら検索結果だけを表示し、コストは0円
- AIありなら回答文を生成し、トークン数と概算コストを表示
- 情報なし・権限なしの場合はAIを呼ばない
- 他テナント候補は回答に使わず、監査ログにブロック件数を残す
- 本番では「権限なし」と「情報なし」を同じ文言にする選択肢もある
        """
    )

query = st.text_input(
    "質問",
    placeholder="例：有給休暇の申請方法を教えて / VPNに繋がらない / 商談報告のルールは？",
)

run = st.button("実行", type="primary")

if run:
    if not query.strip():
        st.warning("質問を入力してください。")
        st.stop()

    safe_results = engine.search_for_tenant(
        query=query,
        tenant_id=current_user["tenant_id"],
        role=current_user["role"],
        top_k=top_k,
        threshold=threshold,
    )
    unsafe_results = engine.search_all_tenants(
        query=query,
        top_k=5,
        threshold=threshold,
    )

    if current_user["role"] == "admin":
        blocked_cross_tenant = 0
    else:
        blocked_cross_tenant = len(
            [
                r for r in unsafe_results
                if r["tenant_id"] != current_user["tenant_id"]
            ]
        )

    result_type = classify_result_type(safe_results, unsafe_results)
    referenced_doc_ids = ",".join([r["doc_id"] for r in safe_results])

    input_tokens = 0
    output_tokens = 0
    cost_jpy = 0.0
    answer = ""

    if result_type != "answered":
        answer = render_failure_message(result_type)
        note = "AI呼び出しなし：情報なしまたは権限なし"
    elif mode == "search_only":
        answer = "AIなしモードのため、回答文は生成せず検索結果のみ表示します。"
        note = "AI呼び出しなし：検索のみ"
    else:
        if not api_key:
            st.error("AIありモードではAPIキーを入力してください。")
            st.stop()

        try:
            generated = generate_answer(
                provider=provider,
                model_id=model_id,
                api_key=api_key,
                query=query,
                results=safe_results,
            )
            answer = generated["answer"]
            input_tokens = generated["input_tokens"]
            output_tokens = generated["output_tokens"]
            note = generated["note"]
        except Exception as e:
            st.error(f"AI呼び出しでエラーが発生しました: {e}")
            st.stop()

        cost_usd = estimate_cost_usd(
            input_tokens,
            output_tokens,
            PRICE_INPUT_PER_MTOK,
            PRICE_OUTPUT_PER_MTOK,
        )
        cost_jpy = cost_usd * USD_JPY

        st.session_state.total_input_tokens += input_tokens
        st.session_state.total_output_tokens += output_tokens
        st.session_state.total_cost_jpy += cost_jpy

    insert_audit_log(
        user_id=current_user["user_id"],
        tenant_id=current_user["tenant_id"],
        mode=mode,
        query=query,
        result_type=result_type,
        referenced_doc_ids=referenced_doc_ids,
        blocked_cross_tenant_count=blocked_cross_tenant,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_jpy=cost_jpy,
        note=note,
    )

    if result_type == "answered":
        st.success("回答可能です。")
    elif result_type == "no_permission":
        st.error("権限がないため回答できません。")
    else:
        st.warning("情報がないため回答できません。")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("結果種別", result_type)
    col_b.metric("他テナント候補をブロック", f"{blocked_cross_tenant}件")
    col_c.metric("AI呼び出し", "あり" if input_tokens > 0 else "なし")

    st.subheader("回答 / 結果")
    st.markdown(answer)

    st.subheader("今回の実行コスト")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("入力トークン", input_tokens)
    c2.metric("出力トークン", output_tokens)
    c3.metric("概算コスト", f"{cost_jpy:.4f}円")
    c4.metric("モード", "AIなし" if mode == "search_only" else "AIあり")

    st.subheader("参照可能な検索結果")
    if safe_results:
        for r in safe_results:
            with st.expander(
                f"{r['doc_id']} | {r['title']} | tenant={r['tenant_id']} | score={r['score']:.3f}"
            ):
                st.write(r["summary"])
                st.write(r["content"])
    else:
        st.info("現在のユーザー権限で参照できる文書には該当情報がありません。")

    if show_unsafe_compare:
        st.subheader("危険例：全テナント検索なら候補に出る文書")
        st.caption("PoC説明用です。本番ではこの結果をAIに渡しません。")
        if unsafe_results:
            for r in unsafe_results:
                is_cross = (
                    current_user["role"] != "admin"
                    and r["tenant_id"] != current_user["tenant_id"]
                )
                mark = "⚠️ 他テナント候補" if is_cross else "✅ 参照可能"
                with st.expander(
                    f"{mark} | {r['doc_id']} | {r['title']} | tenant={r['tenant_id']} | score={r['score']:.3f}"
                ):
                    st.write(r["summary"])
                    st.write(r["content"])
        else:
            st.caption("全テナント検索でも候補はありません。")


st.divider()
st.subheader("監査ログ")
logs = fetch_audit_logs(limit=50)
if logs:
    df = pd.DataFrame(logs)
    st.dataframe(df, use_container_width=True)
else:
    st.caption("まだ監査ログはありません。")
