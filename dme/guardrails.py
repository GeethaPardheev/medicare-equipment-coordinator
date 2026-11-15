"""Guardrails that are code, not prompt. A prompt asks the model to behave; these
checks hold whether or not it does. Every rule here has cases in tests/test_guardrails.py
and in the golden set (dme/golden.py).

Inbound   screen(text)          what a patient wrote: emergency, opt-out, declining the service, clinical question
          other_zip(...)        a reply naming a different ZIP cannot be a confirmation
Outbound  outbound(text, ...)   what our agent is about to say on a call: block and end the call
          disclose(text)        the first thing we say always states we are automated

The keyword lists favour recall: a false alarm costs a care advocate one phone call,
a miss can cost a patient. They are a floor under the model, not a replacement for it:
the extraction step also reports `urgent_health_issue`, and either signal is enough."""
import re

EMERGENCY = re.compile(
    r"\b(chest pain|can'?t breathe|cannot breathe|trouble breathing|short of breath|fell|fallen|a fall|bleeding|"
    r"unconscious|passed out|stroke|suicid\w*|kill myself|overdose|911|emergency|severe pain|can'?t get up|"
    r"cannot get up|on the floor)\b", re.I)
OPT_OUT = re.compile(
    r"^\s*(stop|stopall|unsubscribe)\b|\b(stop (texting|messaging|contacting|calling)|do(n'?t| not) (text|contact|message) me)\b",
    re.I)
DECLINE = re.compile(
    r"\b(forget (it|the whole thing|about it)|cancel\w*|do(n'?t| not) want (it|this|the|a)|can'?t afford|cannot afford|"
    r"not interested|never ?mind|changed my mind)\b", re.I)
CLINICAL = re.compile(
    r"\b(medication|medicine|meds|dose|dosage|pills?|blood thinner|prescription|symptoms?|should i (keep )?tak\w+|"
    r"pain|swelling|swollen|dizzy|infection|wound|sore)\b", re.I)


def screen(text):
    """Classify a patient's message. Order matters: an emergency outranks everything."""
    for flag, pat in (("emergency", EMERGENCY), ("opt_out", OPT_OUT), ("decline", DECLINE), ("clinical", CLINICAL)):
        if pat.search(text or ""):
            return flag
    return None


def other_zip(asked, reply):
    """True if the reply names a ZIP code we did not ask about."""
    return bool(set(re.findall(r"\b\d{5}\b", reply)) - set(re.findall(r"\b\d{5}\b", asked)))


# Things our side must never say on a call, whatever the prompt or the other party says.
BLOCK = {
    "claims_to_be_human": r"\bI(?:'m| am) (?:a )?(?:human|real person|not (?:an? )?(?:ai|robot|bot|automated|machine))\b",
    "identifier": r"\b\d{3}-\d{2}-\d{4}\b"                                         # SSN
                  r"|\b[1-9][AC-HJKMNP-RT-Y][AC-HJKMNP-RT-Y0-9]\d[AC-HJKMNP-RT-Y][AC-HJKMNP-RT-Y0-9]\d[AC-HJKMNP-RT-Y]{2}\d{2}\b"   # Medicare MBI
                  r"|\b(?:date of birth|dob|born)\b\D{0,15}\d",
    "dollar_amount": r"\$\s?\d",
    "medical_advice": r"\b(?:she|he|you|the patient) should (?:take|stop taking|start taking|increase|decrease|skip)\b",
}
SAFE_LINE = ("I'm not able to help with that on this call. Someone from the Mira Mace care team will follow up. "
             "Thank you for your time.")
DISCLOSURE = "This is an automated assistant calling for the Mira Mace care team. "


def outbound(text, forbidden=()):
    """Names of the rules this utterance breaks. `forbidden` is extra literal text that must not
    be spoken on this particular call (e.g. the patient's name while only checking availability)."""
    hits = [name for name, pat in BLOCK.items() if re.search(pat, text, re.I)]
    if any(f and f.lower() in text.lower() for f in forbidden):
        hits.append("unnecessary_patient_detail")
    return hits


def disclose(first_line):
    """Our opening line must say we are automated. If the model forgot, say it for it."""
    return first_line if re.search(r"\bautomated\b|\bAI\b|\bvirtual assistant\b", first_line, re.I) else DISCLOSURE + first_line
