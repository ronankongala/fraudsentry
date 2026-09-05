"""
FraudSentry - Data Subject Rights Operationalization
=======================================================
Implements two GDPR data subject rights against the FraudSentry
alert/case database:

  - Right of Access (Article 15): export everything the system holds
    about a given customer_id.
  - Right to Erasure (Article 17): remove a customer's records, WITH
    an explicit conflict check against the retention policy described
    in dpia.md. Right to erasure is not absolute -- Article 17(3)(b)
    permits refusal where processing is necessary for compliance with
    a legal obligation. This script does not silently honor every
    erasure request; it surfaces the conflict for a human decision,
    which is the honest and legally correct behavior.

This only touches the case_tracking.db alert/case data (the system's
own records), not the underlying raw transactions.csv, which is
treated in this project as being subject to separate, longer-mandated
financial recordkeeping retention (see dpia.md Section 4).
"""
import json
import sqlite3
from datetime import datetime, timezone

import sys
sys.path.insert(0, "/home/claude/fraudsentry/src")
from case_tracking import DB_PATH, get_conn


# Simulated retention flag: in a real system this would check whether
# any of the customer's alerts are tied to an open regulatory/legal
# retention hold. Here we simulate it as: any CONFIRMED_FRAUD case is
# assumed to carry a legal retention obligation and cannot be erased
# on request alone.
RETENTION_PROTECTED_STATUSES = {"confirmed_fraud"}


def access_request(customer_id: str):
    """Article 15 - Right of Access. Returns everything held about this customer."""
    conn = get_conn()
    alerts = conn.execute(
        "SELECT * FROM alerts WHERE customer_id = ?", (customer_id,)
    ).fetchall()
    columns = [d[0] for d in conn.execute("SELECT * FROM alerts LIMIT 0").description]
    alert_records = [dict(zip(columns, row)) for row in alerts]

    audit_records = []
    for a in alert_records:
        logs = conn.execute(
            "SELECT * FROM audit_log WHERE alert_id = ?", (a["alert_id"],)
        ).fetchall()
        log_cols = [d[0] for d in conn.execute("SELECT * FROM audit_log LIMIT 0").description]
        audit_records.extend(dict(zip(log_cols, row)) for row in logs)

    conn.close()
    return {
        "customer_id": customer_id,
        "request_type": "access",
        "fulfilled_at": datetime.now(timezone.utc).isoformat(),
        "alert_count": len(alert_records),
        "alerts": alert_records,
        "audit_log_entries": audit_records,
    }


def erasure_request(customer_id: str, requester_note: str = ""):
    """Article 17 - Right to Erasure, WITH a retention-conflict check.

    Returns a dict describing what was erased and what was withheld
    (and why), rather than either (a) blindly deleting everything, or
    (b) blindly refusing. Both of those would be wrong; the correct
    behavior is case-by-case with a documented reason.
    """
    conn = get_conn()
    alerts = conn.execute(
        "SELECT alert_id, resolution, status FROM alerts WHERE customer_id = ?",
        (customer_id,),
    ).fetchall()

    erasable_ids = []
    withheld_ids = []
    for alert_id, resolution, status in alerts:
        if resolution in RETENTION_PROTECTED_STATUSES:
            withheld_ids.append(alert_id)
        else:
            erasable_ids.append(alert_id)

    for alert_id in erasable_ids:
        conn.execute("DELETE FROM audit_log WHERE alert_id = ?", (alert_id,))
        conn.execute("DELETE FROM alerts WHERE alert_id = ?", (alert_id,))
    conn.commit()
    conn.close()

    result = {
        "customer_id": customer_id,
        "request_type": "erasure",
        "fulfilled_at": datetime.now(timezone.utc).isoformat(),
        "requester_note": requester_note,
        "erased_alert_ids": erasable_ids,
        "withheld_alert_ids": withheld_ids,
        "withheld_reason": (
            "GDPR Art. 17(3)(b): processing necessary for compliance with a "
            "legal obligation (confirmed-fraud recordkeeping retention). "
            "These records are retained; the data subject may still request "
            "restriction of further processing under Art. 18."
        ) if withheld_ids else None,
    }
    return result


if __name__ == "__main__":
    # Demo against whatever customer has the most alerts in the current DB
    conn = get_conn()
    row = conn.execute(
        "SELECT customer_id, COUNT(*) c FROM alerts GROUP BY customer_id ORDER BY c DESC LIMIT 1"
    ).fetchone()
    conn.close()

    if row is None:
        print("No alerts in database yet -- run case_tracking.py first.")
    else:
        demo_customer = row[0]
        print(f"=== Access request demo for {demo_customer} ===")
        access_result = access_request(demo_customer)
        print(json.dumps(access_result, indent=2, default=str)[:1500], "...\n")

        print(f"=== Erasure request demo for {demo_customer} ===")
        erasure_result = erasure_request(demo_customer, requester_note="Customer requested account data deletion.")
        print(json.dumps(erasure_result, indent=2))
