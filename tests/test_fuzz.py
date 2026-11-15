"""Randomised worlds. Each seed builds a different mix of failures (suppliers that
refuse, vanish, or bill the wrong code; a PCP office that loses or botches the
order; a patient who goes quiet) and a listener that randomly misses facts.
Whatever the world does, the engine must keep these promises.

Run: python3 -m unittest tests.test_fuzz -v      (no LLM, about a second)"""
import random
import unittest
from datetime import datetime
from pathlib import Path

from dme.comms import ScriptedComms
from dme.engine import Engine
from dme.models import eleanor
from dme.world import OK_QUALIFY, World

CSV = Path(__file__).resolve().parent.parent / "data" / "suppliers.csv"
SEEDS = 400


def random_world(rng, names):
    suppliers = {}
    for name in names:
        truth = {k: rng.random() > 0.25 for k in OK_QUALIFY}
        spec = {"contact": "X", "qualify": {"no_answer": rng.choice([0, 0, 0, 1, 2, 99]), "brief": "",
                                            "truth": {**truth, "delivery_business_days": rng.randint(1, 6)}}}
        if rng.random() < 0.4:
            spec["handoff"] = {"no_answer": rng.choice([0, 0, 1, 99]), "brief": "", "truth": {
                "order_accepted": rng.random() > 0.3, "will_bill_code": rng.choice(["K0001", "K0001", "K0003"]),
                "delivery_business_days": rng.randint(1, 5)}}
        if rng.random() < 0.4:
            spec["confirm"] = rng.choice([{"no_answer": 99}, {"brief": "", "truth": {"still_scheduled": False}}])
        suppliers[name] = spec

    def pcp_step(truth):
        fax = rng.choice([[], [], [(rng.randint(1, 30), "fax", rng.choice(["good", "bad"]))]])
        return {"no_answer": rng.choice([0, 0, 0, 1, 99]), "brief": "", "truth": truth, "effects": fax}

    pcp = {"request": [pcp_step({"acknowledged": rng.random() > 0.1})],
           "status": [pcp_step({"has_record": rng.random() > 0.5, "resubmitted": True}) for _ in range(rng.randint(1, 3))],
           "correction": [pcp_step({"acknowledged": True}) for _ in range(rng.randint(1, 2))]}
    patient = {"intro": {"no_answer": rng.choice([0, 0, 0, 1, 2, 99]), "brief": "", "truth": {
                   "confirms_address": rng.random() > 0.1, "question_topic": rng.choice([None, "cost"]),
                   "wants_human": rng.random() < 0.1}},
               "delivery_check": {"no_answer": rng.choice([0, 0, 0, 1, 99]), "brief": "",
                                  "truth": {"delivered": rng.random() > 0.2}}}
    return {"suppliers": suppliers, "pcp": pcp, "patient": patient}


class Invariants(unittest.TestCase):
    def test_engine_keeps_its_promises_in_any_world(self):
        outcomes = {}
        for seed in range(SEEDS):
            rng = random.Random(seed)
            case = eleanor(CSV)
            world = random_world(rng, [s.name for s in case.suppliers])
            # one worker: the noise draws come from a shared seeded rng, so call order must be fixed
            eng = Engine(case, World(world), ScriptedComms(noise=rng.choice([0, 0.15, 0.3]), rng=rng),
                         echo=False, workers=1)
            eng.run()
            ctx = f"seed {seed}"
            outcomes[case.status] = outcomes.get(case.status, 0) + 1

            # 1. It always terminates in a definite state, and never by running forever.
            self.assertIn(case.status, ("CLOSED_DELIVERED", "NEEDS_HUMAN"), ctx)
            self.assertLess(len(eng.events), 400, ctx)
            # 2. A case that is not closed always has a human on the hook.
            if case.status == "NEEDS_HUMAN":
                self.assertTrue(eng.open_escalations(), ctx)
            # 3. No supplier was ever given the order unless it truly qualifies and bills the right code,
            #    and never before a valid order and a confirmed address existed.
            for s in case.suppliers:
                if s.delivery_date:
                    spec = world["suppliers"][s.name]
                    self.assertTrue(all(spec["qualify"]["truth"][k] for k in OK_QUALIFY), f"{ctx} {s.name}")
                    self.assertEqual(spec.get("handoff", {"truth": {"will_bill_code": "K0001"}})["truth"]["will_bill_code"],
                                     "K0001", f"{ctx} {s.name}")
                    self.assertEqual(case.order.status, "VALID", ctx)
                    self.assertTrue(case.address_confirmed, ctx)
            # 4. Closed means the patient really has the chair.
            if case.status == "CLOSED_DELIVERED":
                self.assertTrue(world["patient"]["delivery_check"]["truth"]["delivered"], ctx)
                self.assertEqual(sum(s.status == "DELIVERED" for s in case.suppliers), 1, ctx)
            # 5. Nobody is phoned or texted outside business hours.
            for e in eng.events:
                if e["tag"] in ("SUPPLIER", "PCP", "PATIENT"):
                    t = datetime.fromisoformat(e["ts"])
                    self.assertTrue(t.weekday() < 5 and 9 <= t.hour < 17, f"{ctx} {e}")
            # 6. Retry budgets hold: no party is called without limit.
            self.assertLessEqual(case.order.nudges, 4, ctx)
        self.assertGreater(outcomes.get("CLOSED_DELIVERED", 0), 0)
        self.assertGreater(outcomes.get("NEEDS_HUMAN", 0), 0)
        print(f"\n  {SEEDS} random worlds: {outcomes}")


if __name__ == "__main__":
    unittest.main()
