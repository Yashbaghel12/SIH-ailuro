"""
Case worker / authority notification module.

Inspired by CaiTI (Nie et al., ACM ToCH 2026) and ChatThero (Wang et
al., arXiv:2508.20996): both systems keep the AI as a flagging layer
only, never an autonomous decision-maker for high-risk cases -- a
human (case worker, therapist, NGO staff) is always notified and
makes the actual call. This module is that human-handoff step.

Configure via environment variables for real deployment:
    ALERT_SMTP_HOST, ALERT_SMTP_PORT, ALERT_SMTP_USER, ALERT_SMTP_PASS
    ALERT_FROM_EMAIL, ALERT_TO_EMAIL

If SMTP isn't configured, alerts are appended to alerts_log.jsonl
and printed to console -- safe default for a demo, same interface
you'd use once real case-worker emails/Slack/SMS are wired in.
"""

import json
import os
import smtplib
from datetime import datetime, timezone
from email.mime.text import MIMEText

ALERT_LOG_PATH = os.path.join(os.path.dirname(__file__), "alerts_log.jsonl")


class CaseWorkerNotifier:
    def __init__(self):
        self.smtp_host = os.environ.get("ALERT_SMTP_HOST")
        self.smtp_port = int(os.environ.get("ALERT_SMTP_PORT", "587"))
        self.smtp_user = os.environ.get("ALERT_SMTP_USER")
        self.smtp_pass = os.environ.get("ALERT_SMTP_PASS")
        self.from_email = os.environ.get("ALERT_FROM_EMAIL")
        self.to_email = os.environ.get("ALERT_TO_EMAIL")
        self.smtp_configured = all([
            self.smtp_host, self.smtp_user, self.smtp_pass,
            self.from_email, self.to_email,
        ])

    def notify(self, case_id: str, score: float, risk_level: str,
               reasons: list[str], threat_reported: bool) -> dict:
        timestamp = datetime.now(timezone.utc).isoformat()
        record = {
            "timestamp": timestamp,
            "case_id": case_id,
            "score": score,
            "risk_level": risk_level,
            "reasons": reasons,
            "threat_reported": threat_reported,
        }

        if self.smtp_configured:
            sent = self._send_email(record)
            record["channel"] = "email"
            record["delivered"] = sent
        else:
            record["channel"] = "log-fallback"
            record["delivered"] = True
            print(f"[ALERT] High-risk check-in flagged for case worker: {record}")

        self._append_log(record)
        return record

    def _send_email(self, record: dict) -> bool:
        subject = f"[URGENT] High-risk check-in - case {record['case_id']}"
        body = (
            f"A check-in has been flagged as HIGH risk.\n\n"
            f"Case ID: {record['case_id']}\n"
            f"Score: {record['score']}\n"
            f"Threat reported: {record['threat_reported']}\n"
            f"Reasons: {', '.join(record['reasons'])}\n"
            f"Time (UTC): {record['timestamp']}\n\n"
            f"Please follow up per your organization's escalation protocol."
        )
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = self.from_email
        msg["To"] = self.to_email

        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_pass)
                server.send_message(msg)
            return True
        except Exception as e:
            print(f"[ALERT] Email send failed, falling back to log: {e}")
            return False

    def _append_log(self, record: dict):
        with open(ALERT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")