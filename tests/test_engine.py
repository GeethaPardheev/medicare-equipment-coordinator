"""Deterministic tests: no LLM, no network.  Run: python3 -m unittest -v"""
import unittest
from datetime import datetime
from pathlib import Path

from dme.comms import LLMComms, ScriptedComms
from dme.engine import Engine
from dme.models import add_bdays, clamp, eleanor
from dme.order import ground, validate
from dme.world import BAD_ORDER, GOOD_ORDER, World

CSV = Path(__file__).resolve().parent.parent / "data" / "suppliers.csv"


def run(scenario):
    eng = Engine(eleanor(CSV), World(scenario), ScriptedComms(), echo=False)
    return eng.run(), eng


def status(case, name):
    return next(s for s in case.suppliers if s.name.startswith(name)).status


class Scenarios(unittest.TestCase):
    def test_happy_path_closes_with_no_human(self):
        case, _ = run("happy")
        self.assertEqual(case.status, "CLOSED_DELIVERED")
        self.assertEqual(case.escalations, [])
        self.assertEqual(case.order.nudges, 0)

    def test_gauntlet_survives_every_failure_mode(self):
        case, eng = run("gauntlet")
        self.assertEqual(case.status, "CLOSED_DELIVERED")
        self.assertEqual(case.escalations, [])
        self.assertEqual(status(case, "Lakeshore"), "DISQUALIFIED")      # not taking new patients
        self.assertEqual(status(case, "Lincoln Park"), "DISQUALIFIED")   # out of stock
        self.assertEqual(status(case, "Windy City"), "DISQUALIFIED")     # no assignment
        self.assertEqual(status(case, "Halsted"), "GHOSTED")             # said yes, went silent
        self.assertEqual(status(case, "Prairie State"), "DELIVERED")     # the backup delivered
        self.assertEqual(case.order.nudges, 2)                            # lost request + wrong order
        msgs = " ".join(e["msg"] for e in eng.events)
        self.assertIn("fell in a hole", msgs)
        self.assertIn("K0003", msgs)

    def test_quiet_patient_is_escalated_not_delivered_to(self):
        case, _ = run("patient_quiet")
        self.assertEqual(case.status, "NEEDS_HUMAN")
        self.assertEqual([e["code"] for e in case.escalations], ["PATIENT_UNREACHABLE"])
        self.assertIsNone(case.committed)          # no delivery booked to an unconfirmed address
        self.assertEqual(case.order.status, "VALID")   # the other tracks still finished

    def test_no_calls_outside_business_hours(self):
        _, eng = run("gauntlet")
        for e in eng.events[2:]:
            t = datetime.fromisoformat(e["ts"])
            self.assertTrue(t.weekday() < 5 and 9 <= t.hour < 17, e)


class Rules(unittest.TestCase):
    def test_business_clock(self):
        self.assertEqual(clamp(datetime(2026, 11, 16, 21, 0)), datetime(2026, 11, 17, 9, 0))   # Mon night -> Tue
        self.assertEqual(clamp(datetime(2026, 11, 21, 10, 0)), datetime(2026, 11, 23, 9, 0))   # Sat -> Mon
        self.assertEqual(add_bdays(datetime(2026, 11, 20, 9, 0), 1), datetime(2026, 11, 23, 9, 0))  # Fri+1 -> Mon

    def test_order_validation(self):
        read = ScriptedComms().read_order
        self.assertEqual(validate(ground(read(GOOD_ORDER), GOOD_ORDER), "Eleanor Martinez"), [])
        defects = validate(ground(read(BAD_ORDER), BAD_ORDER), "Eleanor Martinez")
        self.assertEqual(len(defects), 2)
        self.assertIn("K0003", defects[0])
        self.assertIn("signature", defects[1])

    def test_code_not_on_the_document_is_dropped(self):
        f = ground({"hcpcs_code": "K0001", "prescriber_npi": "9999999999"}, BAD_ORDER)
        self.assertIsNone(f["hcpcs_code"])        # the model "saw" K0001; the fax says K0003
        self.assertIsNone(f["prescriber_npi"])

    def test_extraction_without_a_real_quote_is_treated_as_unknown(self):
        class FakeLLM:
            def json(self, system, user, max_tokens=500):
                return {"delivered": {"value": True, "evidence": "yes it came this morning"}}
        heard = [("us", "Did it arrive? Reply yes it came this morning."), ("them", "Who is this?")]
        facts, dropped = LLMComms(FakeLLM())._extract({"delivered": "bool"}, heard, "a text exchange")
        self.assertIsNone(facts["delivered"])       # the quote exists only in OUR words, so it proves nothing
        self.assertEqual(dropped, ["delivered"])


if __name__ == "__main__":
    unittest.main()
