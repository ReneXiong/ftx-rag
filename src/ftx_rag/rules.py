"""Title rules that route bulk filings away from the expensive path.

About 70% of the FTX docket is mass-produced paperwork: claim transfers,
filing-fee receipts, certificates of service. Their titles say everything there
is to know, so they are classified here by prefix and never parsed, sent to the
LLM, or put in the graph.

This module is the only place these rules live. Every stage that needs to know
whether a filing is bulk asks `match_bulk_rule`; a second copy elsewhere is how
two stages end up disagreeing about the same document.
"""

from __future__ import annotations

# Title prefix -> (category, subcategory)
BULK_RULES: dict[str, tuple[str, str]] = {
    "Transfer/Assignment of Claim": ("Claim Transfers", "Transfer"),
    "Receipt of filing fee": ("Administrative", "Fee Receipt"),
    "Affidavit/Declaration of Mailing": ("Administrative", "Affidavit"),
    "Affidavit of Mailing": ("Administrative", "Affidavit"),
    "Certificate of No Objection": ("Administrative", "Certificate"),
    "Certificate of Service": ("Administrative", "Certificate"),
    "Certificate of Mailing": ("Administrative", "Certificate"),
    "Notice of Appearance": ("Administrative", "Notice of Appearance"),
    "Schedules/Statements filed": ("Administrative", "Schedules"),
    "Amended Schedules/Statements": ("Administrative", "Amended Schedules"),
    "Application for Compensation": ("Administrative", "Compensation App"),
    "Exhibit": ("Exhibits", "Exhibit"),
}

EMPTY_TITLE_RULE = ("Administrative", "Empty Title")


def match_bulk_rule(title: str) -> tuple[str, str] | None:
    """(category, subcategory) if the title marks a bulk filing, else None."""
    t = " ".join((title or "").split())
    if not t:
        return EMPTY_TITLE_RULE
    for prefix, rule in BULK_RULES.items():
        if t.startswith(prefix):
            return rule
    return None


def is_bulk(title: str) -> bool:
    return match_bulk_rule(title) is not None
