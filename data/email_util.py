"""
Email utility for ProjectSoccer.
Used by watchdog alerts and password reset.
"""
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import config

logger = logging.getLogger(__name__)


def send_email(to_email, subject, body_html, body_text=None):
    """Send an email via SMTP."""
    if not config.SMTP_USER or not config.SMTP_PASSWORD:
        logger.warning("SMTP not configured — skipping email")
        return False

    msg = MIMEMultipart("alternative")
    msg["From"] = config.SMTP_FROM or config.SMTP_USER
    msg["To"] = to_email
    msg["Subject"] = subject

    if body_text:
        msg.attach(MIMEText(body_text, "plain"))
    msg.attach(MIMEText(body_html, "html"))

    try:
        with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT) as server:
            server.starttls()
            server.login(config.SMTP_USER, config.SMTP_PASSWORD)
            server.send_message(msg)
        logger.info("Email sent to %s: %s", to_email, subject)
        return True
    except Exception as e:
        logger.error("Failed to send email: %s", e)
        return False


def send_watchdog_alert(status, checks, summary):
    """Send watchdog alert email when issues are found."""
    if not config.ADMIN_EMAIL:
        return False

    failed = [c for c in checks if c["status"] in ("warn", "critical")]
    if not failed:
        return False

    subject = f"ProjectSoccer Watchdog: {status.upper()} — {len(failed)} issue(s)"

    rows = ""
    for c in failed:
        color = "#dc2626" if c["status"] == "critical" else "#d97706"
        icon = "&#10008;" if c["status"] == "critical" else "&#9888;"
        rows += f'<tr><td style="padding:6px;color:{color};font-weight:bold">{icon}</td><td style="padding:6px">{c["category"]}</td><td style="padding:6px">{c["name"]}</td><td style="padding:6px">{c["message"]}</td></tr>'

    body_html = f"""
    <div style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto">
        <div style="background:{'#dc2626' if status=='critical' else '#d97706'};color:white;padding:16px;border-radius:8px 8px 0 0">
            <h2 style="margin:0">ProjectSoccer Watchdog Alert</h2>
            <p style="margin:4px 0 0;opacity:0.9">{summary}</p>
        </div>
        <div style="border:1px solid #e5e7eb;border-top:none;padding:16px;border-radius:0 0 8px 8px">
            <table style="width:100%;border-collapse:collapse;font-size:14px">
                <tr style="border-bottom:1px solid #e5e7eb;color:#6b7280"><th style="padding:6px;text-align:left"></th><th style="padding:6px;text-align:left">Category</th><th style="padding:6px;text-align:left">Check</th><th style="padding:6px;text-align:left">Issue</th></tr>
                {rows}
            </table>
            <p style="color:#9ca3af;font-size:12px;margin-top:16px">Check the Watchdog tab for full details.</p>
        </div>
    </div>
    """

    return send_email(config.ADMIN_EMAIL, subject, body_html)


def send_password_reset(to_email, username, new_password):
    """Send password reset email."""
    subject = "ProjectSoccer — Your password has been reset"

    body_html = f"""
    <div style="font-family:Arial,sans-serif;max-width:500px;margin:0 auto">
        <div style="background:linear-gradient(135deg,#16a34a,#059669);color:white;padding:20px;border-radius:8px 8px 0 0;text-align:center">
            <h2 style="margin:0">ProjectSoccer</h2>
            <p style="margin:4px 0 0;opacity:0.9">Password Reset</p>
        </div>
        <div style="border:1px solid #e5e7eb;border-top:none;padding:24px;border-radius:0 0 8px 8px">
            <p>Hi <strong>{username}</strong>,</p>
            <p>Your password has been reset. Here are your new credentials:</p>
            <div style="background:#f1f5f9;padding:16px;border-radius:8px;margin:16px 0;font-family:monospace">
                <p style="margin:0">Username: <strong>{username}</strong></p>
                <p style="margin:4px 0 0">Password: <strong>{new_password}</strong></p>
            </div>
            <p>Please log in and change your password in Settings.</p>
            <p style="color:#9ca3af;font-size:12px;margin-top:16px">If you didn't request this, contact your admin immediately.</p>
        </div>
    </div>
    """

    return send_email(to_email, subject, body_html)
