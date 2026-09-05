"""
FraudSentry - Case Tracking & Alert Management
=================================================
A lightweight alert/case-management layer sitting on top of the model
scores. This is the piece that turns "a model produces a probability"
into "an analyst has a queue of structured, actionable cases" -- the
gap between a notebook and a usable fraud-ops tool.

Schema:
  alerts(alert_id, transaction_id, customer_id, score, model_used,
         top_factors_json, status, opened_at, assigned_analyst,
         resolution, resolved_at, notes)

Status lifecycle: open -> investigating -> confirmed_fraud | false_positive

NOTE: This is a from-scratch, project-scale implementation of the
CONCEPT behind commercial case-management modules (e.g. those bundled
with Actimize, SAS Fraud Management, Feedzai). It demonstrates
understanding of alert lifecycle, triage, and audit-trail requirements --
it is not a substitute for hands-on experience with those specific
commercial platforms, which this project does not claim.
"""
import json
import sqlite3
from datetime import datetime, timezone

def _now():
    return datetime.now(timezone.utc).isoformat()

DB_PATH = "/home/claude/fraudsentry/results/fraudsentry.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS alerts (
    alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id TEXT NOT NULL,
    customer_id TEXT NOT NULL,
    score REAL NOT NULL,
    model_used TEXT NOT NULL,
    top_factors_json TEXT,
    status TEXT NOT NULL DEFAULT 'open',
    opened_at TEXT NOT NULL,
    assigned_analyst TEXT,
    resolution TEXT,
    resolved_at TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS audit_log (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_id INTEGER NOT NULL,
    action TEXT NOT NULL,
    actor TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    detail TEXT,
    FOREIGN KEY(alert_id) REFERENCES alerts(alert_id)
);
"""


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    return conn


def open_alert(conn, transaction_id, customer_id, score, model_used, top_factors, threshold=0.5):
    if score < threshold:
        return None
    now = _now()
    cur = conn.execute(
        "INSERT INTO alerts (transaction_id, customer_id, score, model_used, "
        "top_factors_json, status, opened_at) VALUES (?, ?, ?, ?, ?, 'open', ?)",
        (transaction_id, customer_id, score, model_used, json.dumps(top_factors), now),
    )
    alert_id = cur.lastrowid
    _log(conn, alert_id, "opened", "system", f"score={score:.3f} threshold={threshold}")
    conn.commit()
    return alert_id


def assign_analyst(conn, alert_id, analyst_name):
    conn.execute(
        "UPDATE alerts SET status='investigating', assigned_analyst=? WHERE alert_id=?",
        (analyst_name, alert_id),
    )
    _log(conn, alert_id, "assigned", analyst_name, "moved to investigating")
    conn.commit()


def resolve_alert(conn, alert_id, resolution, analyst_name, notes=""):
    assert resolution in ("confirmed_fraud", "false_positive"), "invalid resolution"
    now = _now()
    conn.execute(
        "UPDATE alerts SET status=?, resolution=?, resolved_at=?, notes=? WHERE alert_id=?",
        (resolution, resolution, now, notes, alert_id),
    )
    _log(conn, alert_id, "resolved", analyst_name, f"resolution={resolution}: {notes}")
    conn.commit()


def _log(conn, alert_id, action, actor, detail):
    conn.execute(
        "INSERT INTO audit_log (alert_id, action, actor, timestamp, detail) VALUES (?, ?, ?, ?, ?)",
        (alert_id, action, actor, _now(), detail),
    )


def get_open_queue(conn):
    return conn.execute(
        "SELECT alert_id, transaction_id, customer_id, score, status, opened_at "
        "FROM alerts WHERE status IN ('open','investigating') ORDER BY score DESC"
    ).fetchall()


if __name__ == "__main__":
    import pandas as pd
    from explainability import compute_population_stats, local_explanation

    scored = pd.read_csv("/home/claude/fraudsentry/results/test_scored.csv")
    features = pd.read_csv("/home/claude/fraudsentry/data/features.csv", parse_dates=["timestamp"])
    stats = compute_population_stats(features)

    conn = get_conn()
    # Clear prior demo run for idempotent re-runs
    conn.executescript("DELETE FROM alerts; DELETE FROM audit_log;")
    conn.commit()

    THRESHOLD = scored["rf_score"].quantile(0.97)  # top ~3% by score become alerts
    to_alert = scored[scored["rf_score"] >= THRESHOLD]

    opened = 0
    for _, row in to_alert.iterrows():
        full_row = features[features["transaction_id"] == row["transaction_id"]].iloc[0]
        factors = local_explanation(full_row, stats)
        aid = open_alert(conn, row["transaction_id"], row["customer_id"],
                          row["rf_score"], "random_forest", factors, threshold=THRESHOLD)
        if aid:
            opened += 1
            # Simulate triage: auto-assign and resolve a few for demo purposes
            if opened % 4 == 0:
                assign_analyst(conn, aid, "analyst_demo")
                resolution = "confirmed_fraud" if row["is_fraud"] == 1 else "false_positive"
                resolve_alert(conn, aid, resolution, "analyst_demo",
                              notes="Triaged during FraudSentry demo run.")

    print(f"Opened {opened} alerts at threshold={THRESHOLD:.4f}")
    queue = get_open_queue(conn)
    print(f"Open/investigating queue size: {len(queue)}")
    print("Sample queue entries:")
    for row in queue[:5]:
        print(" ", row)
    conn.close()
