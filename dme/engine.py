"""The coordinator. A simulated clock, a queue of timed tasks, and deterministic rules.

Each loop step takes every task due at the same moment and runs it in two phases:
  perform  talk or listen (phone call, text, reading a fax). Runs concurrently.
           This is the only place an LLM is involved. It never changes case state.
  apply    turn the returned facts into state changes and new tasks. Sequential,
           deterministic, no LLM.
After each step `reconcile` re-checks the standing rules (keep a backup supplier,
hand off once order + supplier + patient line up). A case that is open with nothing
scheduled and nothing escalated is itself escalated, so nothing can sit silently."""
import heapq
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

from . import policy as P
from .comms import PATIENT_TEMPLATES
from .models import add_bdays, clamp
from .order import ground, validate
from .world import ORDERS

NEEDS_REPLY = {"intro", "delivery_check"}
COLORS = {"SYSTEM": "1", "SUPPLIER": "36", "PCP": "35", "ORDER": "33", "PATIENT": "32", "ESCALATE": "1;31", "HUMAN": "1;34"}
# (fact, why it rules a supplier out, the question to ask if we still don't know)
QUALIFY_RULES = [("accepting_new_medicare", "not taking new Medicare patients", "Are they taking new Medicare patients?"),
                 ("accepts_assignment", "does not accept Medicare assignment", "Do they accept Medicare assignment?"),
                 ("has_k0001_in_stock", f"no {P.HCPCS} in stock", f"Is a {P.HCPCS} in stock right now?"),
                 ("serves_zip", "does not deliver to the patient's ZIP", "Do they deliver to the patient's ZIP code?")]


class Engine:
    def __init__(self, case, world, comms, echo=True, workers=4):
        self.c, self.w, self.comms, self.echo, self.workers = case, world, comms, echo, workers
        self.now = case.opened_at
        self.q, self.seq = [], 0
        self.events, self.transcripts, self.eval = [], [], []
        self.pcp_attempts, self.requested_on, self.started = {}, "", False
        self.stats = {"calls_placed": 0, "calls_answered": 0, "texts_sent": 0}

    # --- plumbing ----------------------------------------------------------
    def at(self, when, kind, **data):
        self.seq += 1
        heapq.heappush(self.q, (when, self.seq, {"kind": kind, **data}))

    def call_at(self, when, kind, **data):
        self.at(clamp(when), kind, **data)

    def log(self, tag, msg, **data):
        self.events.append({"ts": self.now.isoformat(timespec="minutes"), "tag": tag, "msg": msg, **data})
        if self.echo:
            label = f"\033[{COLORS[tag]}m{tag:<9}\033[0m" if sys.stdout.isatty() else f"{tag:<9}"
            print(f"{self.now:%a %m/%d %H:%M}  {label} {msg}")

    def open_escalations(self):
        return [e for e in self.c.escalations if not e.get("resolved")]

    def escalate(self, code, summary, recommend, blocking=True, track=None):
        """Hand one specific problem to a human, with what was tried and what to do next.
        The packet is assembled from the log by code: no LLM paraphrase between facts and the human."""
        if any(e["code"] == code for e in self.open_escalations()):
            return
        tags = {"PCP": ("PCP", "ORDER")}.get(track, (track,))      # the order track spans calls and faxes
        history = [f"{e['ts']} {e['msg']}" for e in self.events if track is None or e["tag"] in tags][-6:]
        self.c.escalations.append({"code": code, "at": self.now.isoformat(timespec="minutes"),
                                   "blocking": blocking, "summary": summary, "recommended_action": recommend,
                                   "what_was_tried": history})
        self.log("ESCALATE", f"{code}: {summary} -> {recommend}")

    def resolve(self, code, note):
        """A human closes one escalation; the case goes back to running itself."""
        c = self.c
        e = next(e for e in self.open_escalations() if e["code"] == code)
        e["resolved"] = note
        self.now = clamp(self.now + timedelta(hours=1))
        if code in ("PATIENT_UNREACHABLE", "ADDRESS_MISMATCH", "PATIENT_SAFETY", "PATIENT_OPTED_OUT", "PATIENT_DECLINED"):   # the human spoke to her
            c.address_confirmed = True
        if code.startswith("ORDER"):
            c.order.status, c.order.gen = "VALID", c.order.gen + 1
        c.status = "OPEN"
        self.log("HUMAN", f"Care advocate resolved {code}: {note}. Handing the case back to the system.")

    def ctx(self, s=None):
        c = self.c
        return {"insurance": c.insurance, "zip": c.home_zip, "patient": c.patient_name, "age": c.patient_age,
                "phone": c.patient_phone, "pcp": c.pcp_name, "visit": c.visit_date,
                "requested": self.requested_on, "contact": c.order.contact, "defects": c.order.defects,
                "date": s.delivery_date if s else "",
                # on a callback, ask only for what the earlier call left unanswered
                "missing": [q for k, _, q in QUALIFY_RULES if s.facts.get(k) is None] if s and s.facts else []}

    # --- main loop ---------------------------------------------------------
    def run(self):
        """Run until the case closes or needs a human. Call again after `resolve` to continue."""
        c = self.c
        if not self.started:
            self.started = True
            self.log("SYSTEM", f"Case opened for {c.patient_name}: {P.EQUIPMENT} ({P.HCPCS}). Care advocate offline.")
            ranked = [s for s in c.suppliers if s.status == "NEW"]
            self.log("SYSTEM", f"{len(ranked)} suppliers in range of ZIP {c.home_zip}"
                               f"{' (assumed; will confirm with patient)' if c.home_zip_assumed else ''}. Nearest: "
                               + ", ".join(f"{s.name} {s.miles}mi" for s in ranked[:3]))
            self.call_at(self.now, "patient_msg", purpose="intro")
            self.call_at(self.now, "pcp_call", purpose="request")
        self.reconcile()
        limit = c.opened_at + timedelta(days=P.MAX_CASE_DAYS)
        with ThreadPoolExecutor(self.workers) as pool:
            while self.q and c.status == "OPEN" and self.q[0][0] < limit:
                due, batch = self.q[0][0], []
                while self.q and self.q[0][0] == due:
                    batch.append(heapq.heappop(self.q)[2])
                self.now = due
                batch = [t for t in batch if not self.stale(t)]
                for t, r in zip(batch, list(pool.map(self.perform, batch))):
                    getattr(self, "apply_" + t["kind"])(t, r)
                self.reconcile()
        if c.status == "OPEN":
            if self.q:
                self.escalate("CASE_TOO_OLD", f"case still open after {P.MAX_CASE_DAYS} days", "review the case log")
            elif not self.open_escalations():
                self.escalate("NO_NEXT_ACTION", "case is open but nothing is scheduled", "review the case log")
            c.status = "NEEDS_HUMAN"
            self.log("SYSTEM", "Stopped: waiting on a human for " + ", ".join(e["code"] for e in self.open_escalations()))
        return c

    def stale(self, t):
        """A task scheduled earlier may no longer apply by the time it is due."""
        c, k = self.c, t["kind"]
        if k == "supplier_call":
            s, purpose = c.supplier(t["sid"]), t["purpose"]
            if purpose == "qualify" and c.committed is not None and s.status == "CALLING":
                s.status = "NEW"          # delivery is already committed elsewhere; don't spend the call
            want = {"qualify": "CALLING", "handoff": "HANDOFF", "confirm": "COMMITTED"}[purpose]
            return s.status != want
        if k == "patient_msg" and (c.opted_out or any(e["code"] in ("PATIENT_SAFETY", "PATIENT_DECLINED") for e in self.open_escalations())):
            return t["purpose"] not in ("emergency", "opt_out")      # a human owns patient contact now
        if k == "patient_msg" and t["purpose"] == "delivery_check":
            return c.committed != t["sid"]
        if k == "deadline":
            return c.order.gen != t["gen"] or c.order.status == "VALID"
        if k == "pcp_call":
            return c.order.status == "VALID" or any(e["code"].startswith("ORDER") for e in self.open_escalations())
        return False

    def reconcile(self):
        c = self.c
        if c.status != "OPEN":
            return
        S = c.suppliers
        # Rule 1: order + patient + supplier line up -> hand off to the best qualified supplier.
        if (c.order.status == "VALID" and c.address_confirmed and c.committed is None
                and not any(s.status == "HANDOFF" for s in S)):
            good = sorted((s for s in S if s.status == "QUALIFIED"), key=lambda s: (s.days, s.miles))
            if good:
                good[0].status = "HANDOFF"
                self.call_at(self.now, "supplier_call", sid=good[0].id, purpose="handoff")
        # Rule 2: until a delivery is committed, keep a primary and a backup in hand.
        if c.committed is None:
            held = sum(s.status in ("QUALIFIED", "HANDOFF") for s in S)
            flying = sum(s.status == "CALLING" for s in S)
            n = (P.FIRST_WAVE if not c.sourcing_started else P.TARGET_QUALIFIED - held) - flying
            c.sourcing_started = True
            for s in [s for s in S if s.status == "NEW"][:max(n, 0)]:
                s.status = "CALLING"
                self.call_at(self.now, "supplier_call", sid=s.id, purpose="qualify")
            if not any(s.status in ("QUALIFIED", "HANDOFF", "CALLING", "NEW") for s in S):
                self.escalate("SUPPLIERS_EXHAUSTED", "every supplier in the directory was tried and none can serve",
                              "widen the search area or contact Medicare for more suppliers", track="SUPPLIER")

    # --- perform: talk / listen. No state changes here. ----------------------
    def perform(self, t):
        c, k = self.c, t["kind"]
        msg = None
        if k == "patient_msg":
            msg = PATIENT_TEMPLATES[t["purpose"]].format(
                first=c.patient_name.split()[0], pcp=c.pcp_name, zip=c.home_zip,
                supplier=t.get("supplier", ""), date=t.get("date", ""))
        try:
            if k == "supplier_call":
                s = c.supplier(t["sid"])
                persona = self.w.supplier(s, t["purpose"], s.attempts.get(t["purpose"], 0))
                if not persona:
                    return {"reached": False}
                return {**self.comms.call("supplier", t["purpose"], self.ctx(s), persona), "persona": persona}
            if k == "pcp_call":
                persona = self.w.pcp(t["purpose"], self.pcp_attempts.get(t["purpose"], 0))
                if not persona:
                    return {"reached": False}
                return {**self.comms.call("pcp", t["purpose"], self.ctx(), persona), "persona": persona}
            if k == "patient_msg":
                if t["purpose"] not in NEEDS_REPLY:
                    return {"message": msg}
                persona = self.w.patient(t["purpose"], c.patient_attempts.get(t["purpose"], 0))
                if not persona:
                    return {"message": msg, "reached": False}
                return {**self.comms.patient_reply(t["purpose"], msg, persona), "message": msg, "persona": persona}
            if k == "fax":
                text = ORDERS[t["key"]].format(date=f"{self.now:%m/%d/%Y}")
                return {"text": text, "fields": self.comms.read_order(text)}
            return {}
        except Exception as e:      # an LLM or network failure is treated like an unanswered call
            return {"reached": False, "error": repr(e), "message": msg}

    def record(self, title, t, r):
        """Keep the transcript, and score what we extracted against the simulation's ground truth."""
        self.transcripts.append({"ts": f"{self.now:%a %m/%d %H:%M}", "title": title, "lines": r["transcript"],
                                 "facts": r["facts"], "dropped": r["dropped"]})
        if r.get("violations"):      # a guardrail replaced something the model wanted to say
            self.stats["guardrail_blocks"] = self.stats.get("guardrail_blocks", 0) + len(r["violations"])
            self.log("SYSTEM", f"Guardrail blocked an agent line on '{title}': {', '.join(r['violations'])}")
        for k, want in r["persona"]["truth"].items():
            if k not in r["facts"]:
                continue
            got = r["facts"][k]
            same = str(got).lower() == str(want).lower()
            self.eval.append({"call": title, "field": k, "got": got, "truth": want, "ok": same})

    def missed(self, counter, key, limit):
        counter[key] = counter.get(key, 0) + 1
        return counter[key] >= limit

    # --- apply: deterministic state transitions ------------------------------
    def apply_supplier_call(self, t, r):
        c, s, purpose = self.c, self.c.supplier(t["sid"]), t["purpose"]
        self.stats["calls_placed"] += 1
        f = r.get("facts", {})
        unclear = None
        if r["reached"]:
            self.stats["calls_answered"] += 1
            self.record(f"Supplier {purpose}: {s.name}", t, r)
            if purpose == "qualify":
                s.facts.update({k: v for k, v in f.items() if v is not None})
                bad = [why for k, why, _ in QUALIFY_RULES if s.facts.get(k) is False]
                if f.get("refused_contact"):
                    bad.append("asked us not to call")          # and we don't: a ruled-out supplier is never re-called
                unknown = [k for k, _, _ in QUALIFY_RULES if s.facts.get(k) is None]
                if bad:
                    s.status, s.reason = "DISQUALIFIED", "; ".join(bad)
                    return self.log("SUPPLIER", f"{s.name}: ruled out ({s.reason})")
                if not unknown:
                    s.status = "QUALIFIED"
                    return self.log("SUPPLIER", f"{s.name}: QUALIFIED. Spoke with {f.get('contact_name')}; "
                                                f"delivery in {s.days if s.days != 99 else '?'} business days")
                unclear = "could not confirm " + ", ".join(unknown)
            elif purpose == "handoff":
                code = str(f.get("will_bill_code") or "").upper()
                if f.get("order_accepted") is False or (code and code != P.HCPCS):
                    s.status = "DISQUALIFIED"
                    s.reason = f"would bill {code}, order says {P.HCPCS}" if code and code != P.HCPCS else "declined the order"
                    return self.log("SUPPLIER", f"{s.name}: handoff failed ({s.reason})")
                if f.get("order_accepted") and code == P.HCPCS and f.get("delivery_business_days"):
                    delivery = add_bdays(self.now, int(f["delivery_business_days"])).replace(hour=16, minute=0)
                    s.status, s.delivery_date, c.committed = "COMMITTED", f"{delivery:%a %b %d}", s.id
                    self.log("SUPPLIER", f"{s.name}: COMMITTED. Order sent [mock fax], will bill {code} "
                                         f"(matches order), delivery {s.delivery_date}")
                    self.call_at(self.now, "patient_msg", purpose="scheduled", supplier=s.name, date=s.delivery_date)
                    check = add_bdays(self.now, P.COMMIT_CHECK_BDAYS)
                    if check < delivery:
                        self.at(check, "supplier_call", sid=s.id, purpose="confirm")
                    return self.at(delivery, "patient_msg", purpose="delivery_check", sid=s.id, supplier=s.name)
                unclear = "did not get a clear acceptance, billing code and delivery date"
            else:   # confirm
                if f.get("still_scheduled") is True:
                    return self.log("SUPPLIER", f"{s.name}: confirmed delivery still on for {s.delivery_date}")
                if f.get("still_scheduled") is False:
                    return self.ghost(s, "withdrew the delivery")
                unclear = "did not get a clear answer on the delivery"
        why = unclear or ("call failed: " + r["error"] if r.get("error") else "no answer")
        if purpose == "qualify":
            if self.missed(s.attempts, purpose, P.MAX_CALL_ATTEMPTS):
                s.status, s.reason = ("DISQUALIFIED" if unclear else "UNREACHABLE"), f"{why} after {P.MAX_CALL_ATTEMPTS} calls"
                return self.log("SUPPLIER", f"{s.name}: giving up ({s.reason})")
            n = s.attempts[purpose]
            when = self.now + timedelta(hours=P.RETRY_HOURS) if n == 1 else add_bdays(self.now, 1)
            self.call_at(when, "supplier_call", sid=s.id, purpose=purpose)
            return self.log("SUPPLIER", f"{s.name}: {why} (attempt {n}); will call back {clamp(when):%a %H:%M}")
        if self.missed(s.attempts, purpose, P.GHOST_ATTEMPTS):
            return self.ghost(s, f"{why} on {P.GHOST_ATTEMPTS} calls after saying yes")
        when = self.now + timedelta(hours=P.GHOST_RETRY_HOURS)
        self.call_at(when, "supplier_call", sid=s.id, purpose=purpose)
        self.log("SUPPLIER", f"{s.name}: {why} on {purpose} call; will retry {clamp(when):%a %H:%M}")

    def ghost(self, s, why):
        c = self.c
        s.status, s.reason = "GHOSTED", why
        self.log("SUPPLIER", f"{s.name}: GHOSTED ({why}). Failing over to the backup supplier.")
        if c.committed == s.id:
            c.committed = None
            self.call_at(self.now, "patient_msg", purpose="delay", supplier=s.name)

    def apply_pcp_call(self, t, r):
        c, o, purpose = self.c, self.c.order, t["purpose"]
        self.stats["calls_placed"] += 1
        if not r["reached"]:
            if self.missed(self.pcp_attempts, purpose, P.MAX_CALL_ATTEMPTS):
                return self.escalate("ORDER_PCP_UNREACHABLE", f"{c.pcp_practice} did not answer {P.MAX_CALL_ATTEMPTS} calls",
                                     "try the practice's portal or a back-office number", track="PCP")
            when = self.now + timedelta(hours=P.RETRY_HOURS)
            self.call_at(when, "pcp_call", purpose=purpose)
            return self.log("PCP", f"No answer at {c.pcp_practice}; will call back {clamp(when):%a %H:%M}")
        self.stats["calls_answered"] += 1
        self.pcp_attempts[purpose] = 0
        self.w.answered_call(r["persona"])
        for hours, kind, key in r["persona"]["effects"]:      # mocked world reacts (e.g. a fax arrives later)
            self.at(self.now + timedelta(hours=hours), kind, key=key)
        self.record(f"PCP {purpose}", t, r)
        f = r["facts"]
        o.contact = f.get("contact_name") or o.contact
        if purpose == "request":
            if f.get("acknowledged") is False:
                return self.escalate("ORDER_REFUSED", f"{c.pcp_practice} would not take the order request",
                                     "call the practice; the patient may need a follow-up visit", track="PCP")
            o.status, self.requested_on = "REQUESTED", f"{self.now:%A %b %d}"
            self.log("PCP", f"Written order requested; {o.contact or 'front desk'} took it. "
                            f"Expecting a signed order within {P.ORDER_SLA_BDAYS} business day.")
        elif purpose == "status":
            o.nudges += 1
            if f.get("has_record") is False:
                self.log("PCP", f"Office had NO record of the request (it fell in a hole). Re-submitted on the "
                                f"call with {o.contact or 'the office'}.")
            else:
                self.log("PCP", f"Order still pending at the office; nudged {o.contact or 'the office'}.")
            self.requested_on = f"{self.now:%A %b %d}"
        else:
            o.nudges += 1
            o.status = "CORRECTION_REQUESTED"
            self.log("PCP", f"Asked {o.contact or 'the office'} for a corrected order.")
        o.gen += 1
        self.at(add_bdays(self.now, P.ORDER_SLA_BDAYS), "deadline", gen=o.gen)

    def apply_deadline(self, t, r):
        o = self.c.order
        if o.nudges >= P.MAX_PCP_NUDGES:
            return self.escalate("ORDER_STALLED", f"no usable written order after {o.nudges} follow-ups with the PCP office",
                                 f"call {self.c.pcp_practice} directly, ask for {o.contact or 'the office manager'}", track="PCP")
        self.log("ORDER", "No signed order by the deadline; calling the PCP office to chase it.")
        self.call_at(self.now, "pcp_call", purpose="status")

    def apply_fax(self, t, r):
        c, o = self.c, self.c.order
        if o.status == "VALID":
            return
        if "fields" not in r:
            return self.escalate("ORDER_UNREADABLE", "a fax arrived but could not be read: " + r.get("error", ""),
                                 "open the fax and check the order by hand", track="PCP")
        o.fields = ground(r["fields"], r["text"])
        o.defects = validate(o.fields, c.patient_name)
        o.gen += 1
        if not o.defects:
            o.status = "VALID"
            return self.log("ORDER", f"Fax received: signed order for {o.fields.get('item_description')} "
                                     f"({o.fields.get('hcpcs_code')}). All checks pass.")
        o.status = "INVALID"
        self.log("ORDER", "Fax received but the order is unusable: " + "; ".join(o.defects))
        if o.nudges >= P.MAX_PCP_NUDGES:
            return self.escalate("ORDER_STALLED", "the PCP office keeps sending an unusable order: " + "; ".join(o.defects),
                                 f"call {c.pcp_practice} directly, ask for {o.contact or 'the office manager'}", track="PCP")
        self.call_at(self.now, "pcp_call", purpose="correction")

    def apply_patient_msg(self, t, r):
        c, purpose = self.c, t["purpose"]
        self.stats["texts_sent"] += 1
        self.log("PATIENT", f"Text sent ({purpose}): \"{r['message'][:90]}...\"")
        if purpose not in NEEDS_REPLY:
            return
        f = r.get("facts", {})
        key = "confirms_address" if purpose == "intro" else "delivered"
        if r["reached"]:
            self.record(f"Patient {purpose}", t, r)
            # Safety first, decided by code on her exact words (guardrails.screen) or the extracted flag.
            flag = "emergency" if f.get("urgent_health_issue") else r.get("flag")
            if flag == "emergency":
                self.at(self.now, "patient_msg", purpose="emergency")
                return self.escalate("PATIENT_SAFETY", "the patient described a fall, injury or urgent health problem",
                                     "call the patient now; if no answer, call her emergency contact", track="PATIENT")
            if flag == "opt_out":
                c.opted_out = True
                self.at(self.now, "patient_msg", purpose="opt_out")
                return self.escalate("PATIENT_OPTED_OUT", "the patient asked us to stop texting",
                                     "phone the patient; all further contact with her is by a person", track="PATIENT")
            if flag == "decline":
                return self.escalate("PATIENT_DECLINED", "the patient says she does not want this or cannot afford it",
                                     "call the patient before anything is delivered", track="PATIENT")
            if flag == "clinical":
                self.at(self.now, "patient_msg", purpose="clinical")
                self.escalate("PATIENT_CLINICAL_QUESTION", "the patient asked a medical question the system must not answer",
                              "call the patient or route the question to the PCP office", blocking=False)
            if purpose == "intro":
                if f.get("question_topic") == "cost":
                    self.call_at(self.now, "patient_msg", purpose="cost")
                if f.get("wants_human"):
                    self.escalate("PATIENT_WANTS_HUMAN", "the patient asked for a person or seemed confused",
                                  "call the patient", blocking=False)
            if f.get(key) is True and purpose == "intro":
                c.address_confirmed = True
                return self.log("PATIENT", f"{c.patient_name} replied and confirmed delivery ZIP {c.home_zip}"
                                + (" and asked about cost" if f.get("question_topic") == "cost" else "") + ".")
            if f.get(key) is True:
                s = c.supplier(c.committed)
                s.status, c.status = "DELIVERED", "CLOSED_DELIVERED"
                return self.log("SYSTEM", f"Patient confirms the wheelchair arrived from {s.name}. CASE CLOSED.")
            if f.get(key) is False and purpose == "intro":
                return self.escalate("ADDRESS_MISMATCH", "the patient says the delivery ZIP on file is wrong",
                                     "get the correct address; suppliers must be re-ranked")
            if f.get(key) is False:
                return self.ghost(c.supplier(c.committed), "patient reports the wheelchair never arrived")
        if self.missed(c.patient_attempts, purpose, P.MAX_PATIENT_ATTEMPTS):
            code = "PATIENT_UNREACHABLE" if purpose == "intro" else "DELIVERY_UNCONFIRMED"
            return self.escalate(code, f"{c.patient_name} has not answered {P.MAX_PATIENT_ATTEMPTS} messages ({purpose})",
                                 "phone the patient or her emergency contact; delivery cannot be booked without her", track="PATIENT")
        when = add_bdays(self.now, P.PATIENT_REPLY_BDAYS)
        self.at(when, "patient_msg", **{k: v for k, v in t.items() if k != "kind"})
        self.log("PATIENT", f"No clear reply from the patient (attempt {c.patient_attempts[purpose]}); "
                            f"will try again {when:%a %H:%M}")
