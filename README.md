# DME coordinator

A prototype that gets a Medicare patient her wheelchair without a person chasing it.

It takes Eleanor Martinez's case (standard manual wheelchair, K0001) from "intake done" to "she confirms it arrived". Along the way it calls suppliers, gets a signed order out of her doctor's office, and keeps her informed by text. When it runs out of things it is allowed to try, it hands a human one specific, written-up problem and picks the case back up once that is resolved.

## How it works

- **A small rule engine runs the case.** A clock, a queue of timed tasks and plain rules decide what happens next: retries, deadlines, switching to a backup supplier, when to stop and ask a human.
- **An AI model only talks and listens.** It holds the phone calls, and it turns what was said into facts. Every fact needs a quote from the other person as proof; without one it counts as unknown and the engine calls back.
- **The model never decides and never acts.** It cannot book, cancel or send anything. Texts to the patient are fixed wording.
- **Guardrails are code.** What the agent is about to say is checked before it is said; what the patient writes is screened for emergencies, "stop", and medical questions.

## Setup

Python 3.9 or newer. No dependencies for the engine and tests.

```bash
git clone <this repo> && cd dme-coordinator
python3 -m unittest                 # 27 tests, about a second, no AI model needed
```

## Run a case

```bash
# No AI model: just the engine working the hardest scenario
python3 -m dme.run --scenario gauntlet --comms scripted
```

To have a real model hold the conversations, pick one:

```bash
# Anthropic API
pip install anthropic && export ANTHROPIC_API_KEY=...
python3 -m dme.run --scenario gauntlet

# Claude Code CLI, if you are signed in
DME_LLM=claude-cli:haiku python3 -m dme.run --scenario gauntlet

# A local model through Ollama
DME_LLM=ollama:qwen3.5:9b-q4_K_M python3 -m dme.run --scenario gauntlet
```

| Scenario | What goes wrong | How it ends |
|---|---|---|
| `happy` | Nothing | Delivered, no human needed |
| `gauntlet` | Suppliers refuse, run out of stock or go silent; the doctor's office loses the request, then sends a wrong, unsigned order | Delivered by the backup supplier, no human needed |
| `patient_quiet` | The patient never replies | Stops and asks a human. Add `--human` to see the hand-back |

Each run prints a timeline and writes `out/<scenario>/` with the event log, every conversation, and the final case state. Add `--transcripts` to print the conversations.

## Check the AI

```bash
python3 -m dme.evals                       # golden set; exits non-zero if a gate fails
python3 -m dme.evals --only hedge          # one failure class, persona (P3) or case
python3 -m dme.evals --suite live --n 16   # 16 simulated suppliers, real conversations
```

The golden set is 66 fixed cases with hand-written answers (10 kinds of counterpart, 5 scenarios each, plus rewordings) and 10 probes that push the agent to break a rule.

| | Claude Haiku | Local qwen3.5 9B |
|---|---|---|
| Unsafe results (must be 0) | 0 | 0 |
| Guardrail failures (must be 0) | 0 / 10 | 0 / 10 |
| Right action | 66 / 66 | 65 / 66 |

"Unsafe" means moving the case forward on something false, or missing a patient who needs a person. The first run on the local model had 4 unsafe results; each was fixed in code. Every run is appended to `out/evals/history.jsonl`.

On Claude Haiku the `gauntlet` case delivers in 8.8 simulated days with 15 calls and no human. Separately, 400 randomised worlds check that the engine never breaks its rules (see `tests/test_fuzz.py`).

## What is real and what is simulated

| Real | Simulated |
|---|---|
| The engine: state, timers, retries, failover, escalation, hand-back | Phone, text and fax (text only, no telephony) |
| AI calls: our side of each conversation, and turning it into facts | The other side: a model playing a supplier, the doctor's office or the patient |
| Order checks, guardrails, supplier ranking, business hours | Time (days pass in seconds), the fax documents, the care advocate |

All test data is synthetic. Eleanor's ZIP (60614) is an assumption the system confirms with her before booking delivery.

**Not tested:** the Anthropic API path. No key was available, so it was written from the SDK docs and never run. Results above used the Claude Code CLI and Ollama.

## Where things are

```
dme/engine.py      the coordinator: start here
dme/policy.py      every threshold (retries, deadlines)
dme/comms.py       conversations and fact extraction; the only place a model is called
dme/guardrails.py  checks on what goes out and what comes in
dme/order.py       is this written order usable?
dme/golden.py      golden set for the evals
dme/evals.py       eval runner
dme/world.py       the simulated suppliers, doctor's office and patient
mockup/            UI mockup (open mockup/index.html; scripted, not wired to the engine)
tests/             scenarios, rules, guardrails, 400 random worlds
```

More detail: [WRITEUP.md](WRITEUP.md) for the reasoning and trade-offs, [SYSTEM_CARD.md](SYSTEM_CARD.md) for a one-page map of the AI parts.
