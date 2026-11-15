"""Run a scenario:  python3 -m dme.run --scenario gauntlet"""
import argparse
import dataclasses
import json
import time
from pathlib import Path

from .comms import LLMComms, ScriptedComms
from .engine import Engine
from .llm import LLM
from .models import eleanor
from .world import SCENARIOS, World

ROOT = Path(__file__).resolve().parent.parent
HUMAN_NOTES = {"PATIENT_UNREACHABLE": "phoned the patient, address confirmed",
               "ADDRESS_MISMATCH": "corrected the delivery address with the patient"}


def main():
    ap = argparse.ArgumentParser(description="DME back-end coordination prototype")
    ap.add_argument("--scenario", default="gauntlet", choices=list(SCENARIOS))
    ap.add_argument("--comms", default="llm", choices=["llm", "scripted"],
                    help="llm = real LLM conversations; scripted = no LLM, engine only")
    ap.add_argument("--llm", help="provider:model, e.g. anthropic:claude-haiku-5-5 or ollama:qwen3.5:9b")
    ap.add_argument("--no-cache", action="store_true", help="ignore cached LLM responses; make fresh calls")
    ap.add_argument("--human", action="store_true",
                    help="after the system stops, a simulated care advocate resolves each blocking escalation "
                         "and the system carries on")
    ap.add_argument("--transcripts", action="store_true", help="print every conversation after the timeline")
    a = ap.parse_args()

    llm = LLM(a.llm, cache_dir=ROOT / ".llm_cache", use_cache=not a.no_cache) if a.comms == "llm" else None
    comms = LLMComms(llm) if llm else ScriptedComms()
    case = eleanor(ROOT / "data" / "suppliers.csv")
    eng = Engine(case, World(a.scenario), comms)
    print(f"=== scenario: {a.scenario} | comms: " + (f"{llm.provider}:{llm.model}" if llm else "scripted (no LLM)")
          + " | counterparties, fax and clock are simulated ===\n")
    t0 = time.time()
    eng.run()
    if a.human and case.status == "NEEDS_HUMAN":
        print("\n--- a care advocate comes online and works the escalation queue [simulated human] ---")
        for e in eng.open_escalations():
            eng.resolve(e["code"], HUMAN_NOTES.get(e["code"], "handled by phone"))
        eng.run()

    out = ROOT / "out" / a.scenario
    out.mkdir(parents=True, exist_ok=True)
    (out / "events.jsonl").write_text("\n".join(json.dumps(e) for e in eng.events) + "\n")
    (out / "case.json").write_text(json.dumps(dataclasses.asdict(case), indent=2, default=str))
    md = []
    for tr in eng.transcripts:
        md.append(f"## {tr['ts']}  {tr['title']}\n")
        md += [f"**{'Mira Mace assistant' if w == 'us' else 'Them'}:** {text}\n" for w, text in tr["lines"]]
        md.append(f"Extracted: `{json.dumps(tr['facts'])}`" + (f"  dropped (no evidence): {tr['dropped']}" if tr["dropped"] else "") + "\n")
    (out / "transcripts.md").write_text("\n".join(md))
    if a.transcripts:
        print("\n" + "\n".join(md))

    days = (eng.now - case.opened_at).total_seconds() / 86400
    print(f"\n=== RESULT: {case.status} after {days:.1f} simulated days ===")
    for s in case.suppliers:
        if s.status not in ("NEW", "CALLING"):
            print(f"  {s.name:<34} {s.status:<13} {s.reason or s.delivery_date}")
    print(f"  Written order: {case.order.status} after {case.order.nudges} follow-up(s)")
    print(f"  Calls placed {eng.stats['calls_placed']}, answered {eng.stats['calls_answered']}, "
          f"texts sent {eng.stats['texts_sent']}, human escalations {len(case.escalations)}")
    for e in case.escalations:
        state = "resolved: " + e["resolved"] if e.get("resolved") else "OPEN, blocking" if e["blocking"] else "OPEN, fyi"
        print(f"  ESCALATION [{state}] {e['code']}: {e['summary']}\n"
              f"      recommended: {e['recommended_action']}")
        for line in e["what_was_tried"]:
            print(f"      tried: {line}")
    if llm:
        ok = sum(e["ok"] for e in eng.eval)
        print(f"  LLM: {llm.calls} live calls, {llm.hits} cache hits, {time.time() - t0:.0f}s wall time")
        print(f"  Extraction vs simulation ground truth: {ok}/{len(eng.eval)} fields correct")
        for e in eng.eval:
            if not e["ok"]:
                print(f"      miss: {e['call']} / {e['field']}: got {e['got']!r}, truth {e['truth']!r}")
    print(f"  Wrote {out}/events.jsonl, transcripts.md, case.json")


if __name__ == "__main__":
    main()
