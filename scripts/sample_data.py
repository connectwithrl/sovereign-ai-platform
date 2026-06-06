"""A tiny bilingual-context government corpus + labelled eval cases used by the
seed and eval scripts. Deliberately small and self-contained for the demo."""

SAMPLE_DOCS = {
    "leave-policy": (
        "Annual leave policy. Full-time employees accrue thirty calendar days of paid annual "
        "leave each year. Leave requests must be submitted at least seven days in advance "
        "through the HR portal. Unused leave may be carried over up to a maximum of ten days "
        "into the following year."
    ),
    "data-classification": (
        "Data classification standard. Government data is classified as Public, Internal, "
        "Confidential, or Secret. Confidential and Secret data must remain within the sovereign "
        "cloud boundary and may never be sent to external model providers. All access to Secret "
        "data requires multi-factor authentication and is logged for audit."
    ),
    "procurement": (
        "Procurement guidelines. Any purchase above fifty thousand dirhams requires three "
        "competitive quotations and approval from the department director. Contracts above one "
        "million dirhams are reviewed by the procurement committee before award."
    ),
    "incident-response": (
        "Incident response runbook. A severity-one incident must be acknowledged within fifteen "
        "minutes and a postmortem published within five working days. The on-call engineer owns "
        "triage and declares severity. Customer-facing status updates are posted every thirty "
        "minutes until resolution."
    ),
}

# (question, expected source doc_id, expected substrings in a grounded answer)
EVAL_CASES = [
    ("How many days of annual leave do employees accrue each year?", "leave-policy", ["thirty"]),
    ("Can Secret data be sent to external model providers?", "data-classification", []),
    ("What approval is needed for a purchase above fifty thousand dirhams?", "procurement", []),
    ("How quickly must a severity-one incident be acknowledged?", "incident-response", ["fifteen"]),
]