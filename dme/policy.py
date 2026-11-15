"""Every threshold the system acts on lives here. No LLM reads or sets these."""

HCPCS = "K0001"
EQUIPMENT = "standard manual wheelchair"

# Supplier sourcing
MAX_MILES = 15            # beyond this a supplier is not called at all
FIRST_WAVE = 3            # parallel calls in the opening wave
TARGET_QUALIFIED = 2      # hold a primary AND a backup until delivery is committed
MAX_CALL_ATTEMPTS = 3     # unanswered / unclear calls before giving up on a supplier
RETRY_HOURS = 2           # gap before re-calling a no-answer

# "Says yes, then goes silent" watchdog
COMMIT_CHECK_BDAYS = 1    # business days after a commitment before we verify it
GHOST_ATTEMPTS = 2        # unanswered verification calls before we fail over
GHOST_RETRY_HOURS = 3

# PCP written order
ORDER_SLA_BDAYS = 1       # business days we wait for a signed order before nudging
MAX_PCP_NUDGES = 3        # status/correction calls before a human takes over

# Patient
PATIENT_REPLY_BDAYS = 1
MAX_PATIENT_ATTEMPTS = 3

MAX_CASE_DAYS = 30        # hard stop for the simulation

BUSINESS_START, BUSINESS_END = 9, 17
