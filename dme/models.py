"""Case state, business-time helpers, and supplier ranking. Pure stdlib, no LLM."""
import csv
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from . import policy as P

# Static ZIP centroids (approximate). Production: a geocoder. The directory gives
# addresses only, so distance is the one ranking signal we have before calling.
ZIP_LATLON = {
    "60614": (41.922, -87.653), "60654": (41.892, -87.637), "60618": (41.946, -87.704),
    "60640": (41.972, -87.663), "60605": (41.867, -87.619), "60608": (41.850, -87.670),
    "60657": (41.940, -87.653), "60714": (42.031, -87.811), "60201": (42.055, -87.694),
}


def miles(zip_a, zip_b):
    if zip_a not in ZIP_LATLON or zip_b not in ZIP_LATLON:
        return float(P.MAX_MILES)      # unknown ZIP: keep it callable, but rank it last
    (la1, lo1), (la2, lo2) = ZIP_LATLON[zip_a], ZIP_LATLON[zip_b]
    p1, p2 = math.radians(la1), math.radians(la2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lo2 - lo1) / 2) ** 2
    return 3959 * 2 * math.asin(math.sqrt(a))


def clamp(t):
    """Next moment inside business hours (Mon-Fri 9-17). Nobody answers at 2am."""
    while True:
        if t.weekday() >= 5 or t.hour >= P.BUSINESS_END:
            t = (t + timedelta(days=1)).replace(hour=P.BUSINESS_START, minute=0)
            while t.weekday() >= 5:
                t += timedelta(days=1)
            continue
        if t.hour < P.BUSINESS_START:
            t = t.replace(hour=P.BUSINESS_START, minute=0)
        return t


def add_bdays(t, n):
    t = clamp(t)
    while n > 0:
        t += timedelta(days=1)
        if t.weekday() < 5:
            n -= 1
    return t


@dataclass
class Supplier:
    id: int
    name: str
    phone: str
    address: str
    zip: str
    miles: float
    # NEW -> CALLING -> QUALIFIED | DISQUALIFIED | UNREACHABLE
    # QUALIFIED -> HANDOFF -> COMMITTED -> DELIVERED | GHOSTED
    status: str = "NEW"
    attempts: dict = field(default_factory=dict)   # unanswered attempts per call purpose
    facts: dict = field(default_factory=dict)
    reason: str = ""
    delivery_date: str = ""

    @property
    def days(self):
        return self.facts.get("delivery_business_days") or 99


@dataclass
class Order:
    # NOT_REQUESTED -> REQUESTED -> (INVALID -> CORRECTION_REQUESTED ->) VALID
    status: str = "NOT_REQUESTED"
    gen: int = 0            # bumps on every change; stale deadlines compare against it
    nudges: int = 0
    contact: str = ""
    fields: dict = field(default_factory=dict)
    defects: list = field(default_factory=list)


@dataclass
class Case:
    patient_name: str
    patient_age: int
    patient_phone: str
    home_zip: str
    home_zip_assumed: bool
    insurance: str
    pcp_name: str
    pcp_practice: str
    pcp_phone: str
    visit_date: str
    opened_at: datetime
    suppliers: list
    order: Order = field(default_factory=Order)
    status: str = "OPEN"              # OPEN | CLOSED_DELIVERED | NEEDS_HUMAN
    address_confirmed: bool = False
    opted_out: bool = False           # patient said STOP: no more automated texts, a human takes over contact
    patient_attempts: dict = field(default_factory=dict)
    committed: int = None             # supplier id currently holding the delivery
    sourcing_started: bool = False
    escalations: list = field(default_factory=list)

    def supplier(self, sid):
        return next(s for s in self.suppliers if s.id == sid)


def load_suppliers(path, home_zip):
    out = []
    with open(path) as f:
        for i, row in enumerate(csv.DictReader(f)):
            z = re.search(r"(\d{5})\s*$", row["address"]).group(1)
            out.append(Supplier(i, row["supplier_name"], row["phone"], row["address"], z,
                                round(miles(home_zip, z), 1)))
    out.sort(key=lambda s: (s.miles, s.id))   # nearest first; directory order breaks ties
    for s in out:
        if s.miles > P.MAX_MILES:
            s.status, s.reason = "DISQUALIFIED", f"{s.miles} mi is outside the {P.MAX_MILES} mi range"
    return out


def eleanor(suppliers_csv):
    """The case from the brief. Intake has no home address, so the ZIP is an
    assumption the system confirms with the patient before any delivery is booked."""
    opened = datetime(2026, 11, 16, 21, 0)    # Monday 9pm: the care advocate logs off
    home_zip = "60614"
    return Case(
        patient_name="Eleanor Martinez", patient_age=72, patient_phone="(312) 555-0100 [mock]",
        home_zip=home_zip, home_zip_assumed=True,
        insurance="Original Medicare Part B, no supplemental plan",
        pcp_name="Dr. Sarah Chen", pcp_practice="Sunrise Family Medicine", pcp_phone="(312) 555-0198",
        visit_date="2026-11-13", opened_at=opened, suppliers=load_suppliers(suppliers_csv, home_zip),
    )
