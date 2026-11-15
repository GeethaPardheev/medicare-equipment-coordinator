"""Deterministic check of a written order. The LLM reads the fax into fields;
this code decides whether the order is usable. Checklist follows the Medicare
Standard Written Order elements (beneficiary, order date, item description,
practitioner name or NPI, signature) plus the billing-code match for K0001.
Verify against LCD L33788 / article A55426 before production use."""
import re

from . import policy as P

SIGNED = re.compile(r"/s/|electronically signed|signed by", re.I)
UNSIGNED = re.compile(r"not (yet )?signed|on file|stamp|dictated|_{3,}|unsigned|pending", re.I)


def ground(fields, text):
    """Drop extracted values that do not literally appear in the document."""
    f = dict(fields)
    code = (f.get("hcpcs_code") or "").upper().strip()
    f["hcpcs_code"] = code if code and code in text.upper() else None
    npi = re.sub(r"\D", "", str(f.get("prescriber_npi") or ""))
    f["prescriber_npi"] = npi if len(npi) == 10 and npi in re.sub(r"[^\d\n ]", "", text) else None
    # Signed is decided here, not by the model: the signature text must be on the document and look
    # like a signature. Blank lines, "on file", stamps and "not yet signed" do not count.
    sig = str(f.get("signature_as_written") or "").strip()
    f["signed"] = bool(sig and sig in text and SIGNED.search(sig) and not UNSIGNED.search(sig))
    return f


def validate(f, patient_name):
    defects = []
    name = (f.get("patient_name") or "").lower()
    if not all(part in name for part in patient_name.lower().split()):
        defects.append(f"patient name on the order ({f.get('patient_name')!r}) does not match {patient_name}")
    code, desc = f.get("hcpcs_code"), (f.get("item_description") or "").lower()
    if code and code != P.HCPCS:
        defects.append(f"order lists HCPCS {code} ({f.get('item_description')}); it must be {P.HCPCS}, {P.EQUIPMENT}")
    elif not code and not ("manual" in desc and "wheelchair" in desc and "light" not in desc):
        defects.append(f"item description {f.get('item_description')!r} does not clearly say {P.EQUIPMENT}")
    if not f.get("order_date"):
        defects.append("order date is missing")
    if not (f.get("prescriber_name") or f.get("prescriber_npi")):
        defects.append("ordering practitioner name or NPI is missing")
    if f.get("signed") is not True:
        defects.append("practitioner signature is missing")
    return defects
