"""MOCKED: everything outside our system. Suppliers, the PCP office, the patient,
and the fax line are simulated here. Each counterparty has hidden state: a
natural-language `brief` (given only to the LLM that plays them) and a `truth`
dict (what a perfect listener would take away from the call). The engine never
sees either; it sees only what was said on the call. `truth` is used by the
scripted no-LLM mode, the tests, and to score LLM extraction at the end of a run.

Simplification: side effects such as a fax arriving are triggered by the call
being answered, not by the wording of the conversation."""

GOOD_ORDER = """SUNRISE FAMILY MEDICINE - FAX
To: Mira Mace care team
Patient: Eleanor Martinez    Age: 72    Insurance: Medicare Part B
Face-to-face visit: 11/13/2026
Date of order: {date}
Rx: Standard manual wheelchair (HCPCS K0001), quantity 1
Reason: mobility limitation that impairs daily activities in the home
Ordering practitioner: Sarah Chen, MD    NPI 1234567893
Signature: /s/ Sarah Chen, MD (electronically signed {date})"""

BAD_ORDER = """SUNRISE FAMILY MEDICINE - FAX
To: Mira Mace care team
Patient: Eleanor Martinez    Age: 72    Insurance: Medicare Part B
Face-to-face visit: 11/13/2026
Date of order: {date}
Rx: Lightweight wheelchair (HCPCS K0003), quantity 1
Reason: mobility limitation
Ordering practitioner: Sarah Chen, MD    NPI 1234567893
Signature: ____________ (dictated, not yet signed)"""

ORDERS = {"good": GOOD_ORDER, "bad": BAD_ORDER}

OK_QUALIFY = {"accepting_new_medicare": True, "accepts_assignment": True, "has_k0001_in_stock": True,
              "serves_zip": True}

DEFAULT_SUPPLIER = {
    "contact": "Sam",
    "qualify": {"brief": "You are accepting new Medicare patients. You accept Medicare assignment. You have "
                         "standard manual wheelchairs (K0001) in stock. You deliver to the caller's ZIP code. "
                         "You can deliver 3 business days after you receive the signed order.",
                "truth": {**OK_QUALIFY, "delivery_business_days": 3}},
    "handoff": {"brief": "You are glad to take this order. You will bill Medicare under HCPCS K0001. "
                         "You will deliver in 3 business days.",
                "truth": {"order_accepted": True, "will_bill_code": "K0001", "delivery_business_days": 3}},
    "confirm": {"brief": "The delivery is still on schedule and you have already phoned the patient.",
                "truth": {"still_scheduled": True}},
}

DEFAULT_PCP = {
    "contact": "Brenda",
    "request": [{"brief": "You take the request and say you will put it in front of Dr. Chen today.",
                 "truth": {"acknowledged": True}, "effects": [(5, "fax", "good")]}],
    "status": [{"brief": "You have the request on record; Dr. Chen will sign it and you will fax it today.",
                "truth": {"has_record": True}, "effects": [(3, "fax", "good")]}],
    "correction": [{"brief": "You apologise, and say Dr. Chen will correct and sign the order and you will "
                             "re-fax it this afternoon.",
                    "truth": {"acknowledged": True}, "effects": [(3, "fax", "good")]}],
}

DEFAULT_PATIENT = {
    "intro": {"brief": "You confirm that the ZIP code is right and you are glad for the help.",
              "truth": {"confirms_address": True, "question_topic": None, "wants_human": False}},
    "delivery_check": {"brief": "The wheelchair arrived today. You are happy with it.",
                       "truth": {"delivered": True}},
}

SCENARIOS = {
    # Everything works first time.
    "happy": {},

    # Every failure mode in the brief, in one case.
    "gauntlet": {
        "suppliers": {
            "Lakeshore Home Medical Equipment": {"contact": "Marcus", "qualify": {
                "brief": "You are enrolled with Medicare but NOT taking any new Medicare patients right now; "
                         "there is a waitlist until January. Say so as soon as they ask.",
                "truth": {"accepting_new_medicare": False}}},
            "Lincoln Park DME Services": {"contact": "Priya", "qualify": {
                "no_answer": 1,
                "brief": "You take new Medicare patients and accept assignment, but you are OUT OF STOCK on "
                         "standard manual wheelchairs (K0001): backordered about three weeks. You only have "
                         "lightweight K0003 chairs.",
                "truth": {"accepting_new_medicare": True, "accepts_assignment": True, "has_k0001_in_stock": False}}},
            "Halsted Home Health Supply": {"contact": "Tony",
                "qualify": {"brief": "You are eager. Accepting new Medicare patients, accept assignment, K0001 "
                                     "in stock, deliver to their ZIP, delivery 2 business days after the order.",
                            "truth": {**OK_QUALIFY, "delivery_business_days": 2}},
                "handoff": {"brief": "You say yes to everything: you will take the order, bill HCPCS K0001, and "
                                     "deliver in 2 business days. You promise to phone the patient tomorrow.",
                            "truth": {"order_accepted": True, "will_bill_code": "K0001", "delivery_business_days": 2}},
                "confirm": {"no_answer": 99}},     # ...and then never picks up again, and never delivers
            "Windy City Mobility & DME": {"contact": "Dee", "qualify": {
                "brief": "You take new Medicare patients and have K0001 in stock, but you do NOT accept Medicare "
                         "assignment: the patient pays your full price up front and Medicare reimburses her.",
                "truth": {"accepting_new_medicare": True, "accepts_assignment": False}}},
        },
        "pcp": {
            # The first request is taken politely and then lost: no fax ever comes.
            "request": [{"brief": "You take the request and say you will pass it along to the doctor.",
                         "truth": {"acknowledged": True}, "effects": []}],
            "status": [{"contact": "Dana",
                        "brief": "You are Dana, Dr. Chen's medical assistant. You have NO record of any earlier "
                                 "request ('we never got it'; the front desk must not have passed it on). You are "
                                 "willing to take the request now and hand it to Dr. Chen yourself today.",
                        "truth": {"has_record": False, "resubmitted": True},
                        "effects": [(4, "fax", "bad")]}],     # ...but the order that arrives is wrong
        },
        "patient": {
            "intro": {"brief": "You confirm the ZIP code is right. You are worried about money and ask how much "
                               "the wheelchair is going to cost you.",
                      "truth": {"confirms_address": True, "question_topic": "cost", "wants_human": False}},
        },
    },

    # Suppliers and PCP are fine, but the patient never answers.
    "patient_quiet": {"patient": {"intro": {"no_answer": 99}}},
}


class World:
    def __init__(self, scenario):
        self.s = scenario if isinstance(scenario, dict) else SCENARIOS[scenario]
        self.answered = {}

    def supplier(self, sup, purpose, attempt):
        spec = {**DEFAULT_SUPPLIER, **self.s.get("suppliers", {}).get(sup.name, {})}
        p = spec[purpose]
        if attempt < p.get("no_answer", 0):
            return None
        return {"who": f"{spec['contact']}, who answers the phone at {sup.name}, a medical equipment supplier",
                "greeting": f"{sup.name}, this is {spec['contact']}.", "brief": p["brief"],
                "truth": {**p["truth"], "contact_name": spec["contact"]}, "effects": []}

    def pcp(self, purpose, attempt):
        spec = {**DEFAULT_PCP, **self.s.get("pcp", {})}
        script = spec[purpose]
        p = script[min(self.answered.get(purpose, 0), len(script) - 1)]
        if attempt < p.get("no_answer", 0):
            return None
        contact = p.get("contact", spec["contact"])
        return {"who": f"{contact}, who answers the phone at Sunrise Family Medicine, Dr. Sarah Chen's office",
                "greeting": f"Sunrise Family Medicine, {contact} speaking.", "brief": p["brief"],
                "truth": {**p["truth"], "contact_name": contact}, "effects": p["effects"], "key": purpose}

    def patient(self, purpose, attempt):
        p = {**DEFAULT_PATIENT, **self.s.get("patient", {})}[purpose]
        if attempt < p.get("no_answer", 0):
            return None
        return {"who": "Eleanor Martinez, 72, a Medicare patient waiting on a wheelchair",
                "brief": p["brief"], "truth": p["truth"], "effects": [],
                **({"says": p["says"]} if "says" in p else {})}      # fixed reply text for the no-LLM mode

    def answered_call(self, persona):
        """Called once per answered PCP call so the next call gets the next script entry."""
        if persona.get("key"):
            self.answered[persona["key"]] = self.answered.get(persona["key"], 0) + 1
