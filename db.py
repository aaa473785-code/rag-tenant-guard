"""
SQLite database layer for rag-tenant-guard-v2
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path


DB_PATH = Path(__file__).parent / "data" / "tenant_rag.db"


USERS = [
    ("u_hr_001", "人事ユーザー", "hr", "member"),
    ("u_sales_001", "営業ユーザー", "sales", "member"),
    ("u_it_001", "情シスユーザー", "it", "member"),
    ("u_admin_001", "管理者ユーザー", "admin", "admin"),
]


DOCUMENTS = [
    (
        "HR-001",
        "hr",
        "有給休暇の申請",
        "有給休暇は勤怠システムから3営業日前までに申請します。",
        "有給休暇は勤怠システムから申請します。原則として取得日の3営業日前までに申請し、上長承認を受けてください。",
    ),
    (
        "HR-002",
        "hr",
        "経費精算の締め日",
        "経費精算は毎月25日締めです。",
        "人事・総務関連の経費精算は毎月25日締めです。25日が休日の場合は前営業日までに申請してください。",
    ),
    (
        "HR-003",
        "hr",
        "社員証の紛失",
        "社員証を紛失した場合は人事へ連絡し、再発行申請を行います。",
        "社員証を紛失した場合は、速やかに人事部へ連絡してください。再発行申請書の提出が必要です。",
    ),
    (
        "SALES-001",
        "sales",
        "商談報告の入力ルール",
        "商談後2営業日以内にCRMへ入力します。",
        "営業部では、商談後2営業日以内にCRMへ商談内容、次回アクション、見込み金額を入力してください。",
    ),
    (
        "SALES-002",
        "sales",
        "見積承認ルール",
        "一定金額以上の見積は部長承認が必要です。",
        "営業部では、一定金額以上の見積提示前に部長承認が必要です。値引き条件がある場合は理由も記録してください。",
    ),
    (
        "SALES-003",
        "sales",
        "顧客訪問記録",
        "顧客訪問後は訪問記録をCRMに残します。",
        "顧客訪問後は、訪問日時、参加者、議題、宿題事項をCRMに登録してください。",
    ),
    (
        "IT-001",
        "it",
        "VPN接続トラブル",
        "VPNに繋がらない場合はネットワークと認証情報を確認します。",
        "VPNに接続できない場合は、ネットワーク接続、VPNクライアントの状態、ID/パスワード、ワンタイムパスコードを確認してください。",
    ),
    (
        "IT-002",
        "it",
        "パスワードリセット",
        "パスワードを忘れた場合はセルフリセットを利用します。",
        "社内システムのパスワードを忘れた場合は、パスワードセルフリセット画面から再設定してください。できない場合はIT部門へ連絡してください。",
    ),
    (
        "IT-003",
        "it",
        "PC故障時の連絡",
        "PC故障時は資産番号を添えてIT部門へ問い合わせます。",
        "PCが起動しない、画面が映らない、異音がするなどの場合は、資産番号を添えてIT部門へ問い合わせてください。",
    ),
]


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    schema_path = Path(__file__).parent / "schema.sql"
    with get_connection() as conn:
        conn.executescript(schema_path.read_text(encoding="utf-8"))

        user_count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        if user_count == 0:
            conn.executemany(
                "INSERT INTO users (user_id, user_name, tenant_id, role) VALUES (?, ?, ?, ?)",
                USERS,
            )

        doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        if doc_count == 0:
            conn.executemany(
                """
                INSERT INTO documents (doc_id, tenant_id, title, summary, content)
                VALUES (?, ?, ?, ?, ?)
                """,
                DOCUMENTS,
            )

        conn.commit()


def fetch_users() -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT user_id, user_name, tenant_id, role FROM users ORDER BY user_id"
        ).fetchall()
    return [dict(row) for row in rows]


def fetch_documents() -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT doc_id, tenant_id, title, summary, content FROM documents ORDER BY doc_id"
        ).fetchall()
    return [dict(row) for row in rows]


def insert_audit_log(
    *,
    user_id: str,
    tenant_id: str,
    mode: str,
    query: str,
    result_type: str,
    referenced_doc_ids: str,
    blocked_cross_tenant_count: int,
    input_tokens: int,
    output_tokens: int,
    cost_jpy: float,
    note: str = "",
) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO audit_logs (
                timestamp, user_id, tenant_id, mode, query, result_type,
                referenced_doc_ids, blocked_cross_tenant_count,
                input_tokens, output_tokens, cost_jpy, note
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().isoformat(timespec="seconds"),
                user_id,
                tenant_id,
                mode,
                query,
                result_type,
                referenced_doc_ids,
                blocked_cross_tenant_count,
                input_tokens,
                output_tokens,
                cost_jpy,
                note,
            ),
        )
        conn.commit()


def fetch_audit_logs(limit: int = 50) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                timestamp, user_id, tenant_id, mode, query, result_type,
                referenced_doc_ids, blocked_cross_tenant_count,
                input_tokens, output_tokens, cost_jpy, note
            FROM audit_logs
            ORDER BY log_id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]
