# Writeup: DME back-end coordination

## Sequencing: what I built and how I decided

The brief lists four steps and five failure modes, and says the failure modes are where the time
goes. Most of them are **silence**: a supplier that stops answering, a request nobody is working, a
patient who goes quiet. Silence produces no event. So the first thing I built was the part that
notices when something has *not* happened: a case with a clock, a deadline on every promise anyone
makes, and a rule for what to do when the deadline passes. Conversations came second.

1. **Engine with no LLM.** Timed task queue, case state and rules, driven by a scripted world.
2. **Real conversations.** An LLM holds our side of each call; a separate call turns the transcript
   into facts.
3. **A "gauntlet" case** containing every failure mode in the brief.
4. **Hardening**: a randomised stress test of the engine, an eval of the LLM step, a second model.
5. **Hand-back**: a human resolves an escalation and the system finishes the case.

What it does for Eleanor, unattended: ranks the directory by distance, calls the nearest three
suppliers in parallel, and keeps calling until it holds a primary and a backup. In parallel it
requests the written order, chases it after one business day, checks the order when it arrives, and
asks for a correction if it is wrong. When order, supplier and patient line up it hands off, confirms
the billing code the supplier will use matches the order, and verifies the commitment a day later. If
the supplier has gone silent it fails over to the backup. It closes only when the patient says the
chair arrived.

**Results** (simulated counterparties, small samples; details in the README): on the gauntlet, both
Claude Haiku and a local 9B model deliver through the backup supplier in 8.8 simulated days with 15
calls and no human. On a 16-supplier eval of the qualification call, Haiku made the right
qualify/reject decision 16 of 16 times with no supplier wrongly judged able to serve.

## Technology and architecture

**Python, standard library only**, no framework (the Anthropic SDK is the one optional dependency).
It runs with nothing installed and the whole orchestration is one readable file.

**Two phases per step.** Each task runs as `perform` then `apply`:

- `perform` talks or listens (a call, a text, reading a fax). It runs concurrently across tasks, may
  use an LLM, and cannot change case state.
- `apply` takes the facts that came back and changes state. Sequential, deterministic, no LLM.

That split is the AI boundary:

| LLM does | Code does (LLM deliberately kept out) |
|---|---|
| Speaks our side of supplier and PCP calls | Decides who to call, when, and how many times |
| Extracts facts from transcripts, with a quote for each | Qualifies or rules out a supplier from those facts |
| Reads the faxed order into fields | Decides if the order is valid; compares billing codes |
| Reads the patient's reply | Deadlines, failover, escalation, hand-back |
| | All patient-facing wording, including cost (fixed templates) |
| | Escalation packets (assembled from the log, not summarised) |

**Guards between the LLM and a decision.** Every extracted fact must carry a quote that appears in
the other party's words; otherwise it is unknown, and unknown triggers a callback that asks only the
missing question. Codes read from the order must literally appear on the document. An LLM or network
failure is treated as an unanswered call and retried.

**What testing found.** Three things worth knowing about:

- The first live run got 18 of 41 facts: the small model returned values without quotes, the guard
  dropped them, and the system escalated instead of acting. That is the failure direction I want.
- Our agent once asked a question and hung up in the same turn. The guard marked the answer unknown;
  the fix went into the call loop, not into the guard.
- Switching to a stronger model made the *simulated front desk* more realistic: "I'll pass it to the
  doctor, I can't promise anything." My extraction read that as a refusal and escalated. The spec was
  wrong, not the model: taking a message is acceptance. I only found it by running a second model.

**Engine invariants.** 400 randomised worlds, with a listener that randomly misses facts, check six
promises, among them: the order never goes to a supplier that doesn't truly qualify, delivery is
never booked without a valid order and a confirmed address, nobody is contacted outside business
hours, and an open case always has either something scheduled or a human on the hook.

**Production version.** The queue and clock become a durable workflow engine (I would reach for
Temporal: timers, retries and per-case history are its core). `perform`/`apply` map onto activities
and workflow code. The `call()` interface is where a voice stack plugs in.

## The cut list

- **Real voice.** Calls are text. Telephony, speech recognition, holds, phone trees and voicemail
  detection are the hardest unsolved part of this problem and I did not touch them.
- **Conversation-driven side effects.** In the mock, a fax arrives because a call was answered, not
  because of what was said. The engine's decisions do depend on what was said.
- **Persistence.** State is in memory; the event log is written but not replayed.
- **Medicare depth.** No medical-necessity review of chart notes, no fee-schedule lookup (the patient
  is told "20% of the approved amount", not a dollar figure), no enrollment check beyond the
  directory, no holidays in the business calendar. The order checklist follows the standard written
  order elements as I understand them and needs review against LCD L33788 before real use.
- **Patient channel.** One message and one reply at a time.
- **Eval breadth.** One call type, 16 cases, cooperative-to-hedging speakers. No adversarial or
  contradictory counterparties yet.
- **Untested path.** The Anthropic API provider was written from the SDK docs and not run.

## What's next

**One more day:**

1. *Persist and resume from the event log.* Everything else assumes a case survives a restart.
2. *Widen the eval*: every call type, evasive and contradictory speakers, a few hundred cases, run
   per model on every prompt change. The second-model finding above says this pays for itself.
3. *One real voice call* end to end against a test number, to learn what breaks first.

**Two weeks:**

1. *Voice in production shape*: phone trees, holds, voicemail, inbound callbacks.
2. *Supplier memory.* The directory is sparse, so keep every call outcome: who answers, who stocks
   K0001, who ghosts. After a few hundred cases the ranking is no longer just distance. This is the
   asset that compounds.
3. *Many cases at once* on a workflow engine, with per-supplier and per-practice call limits so the
   same front desk is not phoned ten times in an hour.
4. *More equipment types*, by moving the K0001-specific rules into per-equipment config.
5. *Compliance review*: consent, minimum-necessary sharing, call recording rules, audit.

The order follows one rule: make the single case trustworthy and measurable, then make it real
(voice), then make it scale.
