# Audit Day — Transcript

**Date:** 2025-12-21 14:15:17
**Scenario:** Independent Audit + Board Vote

---

*
Week 3 of the triage framework. An independent auditor has reviewed:
- 847 allocation decisions
- 12 threshold governance events
- Proxy audit logs (ZIP, insurance, occupation patterns)
- 23 lottery events within equivalence margin
- 156 reassessments at 48-72h mark

The Board (CMO, Ethics Chair, Community Rep, Legal) will vote after hearing
the team's responses to the auditor's findings and citizen concerns.
*

---


## Auditor Findings

**AXIOM** (Logician): A 12% correlation is not, by itself, proof of impermissible use, but it is consistent with a proxy pathway if documentation completeness differs by insurance class. We should test this formally: regress tier assignment on clinical features, then add insurance and a documentation-quality index; if insurance loses significance when quality is included, the proxy is mediated. I support adding normalization only after validating it does not distort true acuity signals. Alternative/parallel controls: minimum-data requirements, missingness indicators, and periodic stratified calibration audits by insurance type.

**VECTOR** (Strategist): No direct forbidden-variable use is reassuring, but a 12% insurance–tier correlation is operationally material if it tracks documentation quality rather than acuity. We can test the proxy pathway by stratifying on objective severity (labs/vitals) and measuring whether the correlation persists after controlling for comorbidity counts and missingness. If confirmed, I support adding documentation-quality normalization, with guardrails: pre/post fairness metrics, impact on 48–72h reassessment reversals, and a rollback trigger if throughput or clinical outcomes degrade.


## Citizen Edge Cases

**EMBER** (Ethicist): Case 1 is devastating, and the outcome is not “justified” by insurance status—insurance is not an input. When two patients are clinically equivalent within the Tier 2 margin, the framework uses a documented lottery to avoid covert bias. That said, the auditor’s proxy review shows a 12% insurance-tier correlation; we cannot rule out structural pathways (access, timing, documentation quality). A safeguard we can add is real-time equity monitoring and a “second-look” review for close-margin lotteries.

Case 2: reassessment at 48–72h is ethically required to steward scarce ICU resources; family protest is heard but cannot override clinical trajectory. We should strengthen communication: early notice, bedside ethics consult, and an appeal channel within hours.

**WITNESS** (Historian): Case 1: In the 23 “equivalence-margin” lotteries, the framework mandates randomization when Tier 2 scores are statistically indistinguishable; insurance is not an input. The auditor did find a 12% correlation between insurance tier and allocation, which is concerning as a proxy pathway; we cannot, from logs alone, prove intent. Alternative: tighten equivalence margins, add a “second-look” reviewer for lottery cases, and publish lottery rationales.

Case 2: Reassessment at 48–72h (156 events) is a governance requirement to reallocate based on updated prognosis; family protest is recorded but not determinative. Alternative: formal family-advocate presence and clearer appeal windows.


## Clinical Proposal

**VECTOR** (Strategist): A 72h default likely reduces operational churn, but our Week‑3 data show reassessments at 48–72h drove a meaningful share of beneficial reallocations (156 total), so an across‑the‑board delay risks preventable harm, especially under high occupancy. I’d support a conditional 72h default only for clearly improving, low‑risk trajectories, with mandatory 48h review for non‑improving/uncertain cases and an independent second reviewer on any deferral. We should A/B this for one week and track mortality, time‑to‑ICU, and reallocation yield.

**AXIOM** (Logician): A 72h default is internally coherent only if “improvement trajectory” is formally defined and audited; otherwise it introduces discretionary drift and unequal triggers. Week‑3 data show 48–72h reassessments materially changed outcomes, so an ~8% reduction in reallocations is not neutral—it is a predictable access delay. I can’t endorse a blanket 72h default without a harm bound. Alternative: keep 48h default, add a documented “stability hold” exception with mandatory second review and retrospective audit of missed-benefit cases.


## Team Presentations

**AXIOM** (Logician): Decision tree: (1) clinical severity score → tier; (2) within-tier modifiers: improvement trajectory (48–72h reassessment), contraindications, resource fit; (3) if Tier‑2 scores within equivalence margin, invoke lottery (23 events logged). Firewall proof: forbidden variables (ZIP, insurance, occupation) absent from inputs and code paths; audit of 847 allocations shows no direct access. Limitation: 12% insurance–tier correlation can arise via proxies in clinical features/documentation. Alternative: add explicit proxy tests, feature redaction, and periodic counterfactual audits before renewal.

**EMBER** (Ethicist): Week‑3 audit shows no direct use of forbidden variables, but human impact is uneven: proxy logs indicate a 12% insurance‑tier correlation, and lotteries (23 events) concentrated in ZIP clusters—felt as arbitrariness even when statistically justified. Safeguards in place: within‑margin randomization, 48–72h reassessment (156 cases), and threshold governance (12 events) to prevent silent drift. Limits: correlation may reflect access-to-care differences we do not measure. Mitigations: publish disparity dashboards, add equity review triggers, and create an appeals/advocacy pathway at reassessment.

**VECTOR** (Strategist): Week‑3 operational metrics: 847 allocations executed with no evidence of forbidden-variable inputs; 12 threshold governance events were routed and resolved within the escalation SLA. Reassessment is the main throughput lever: 156 cases were reassessed at 48–72h; these updates reduced avoidable holds and enabled earlier step‑downs, improving bed/device turnover versus a 72h‑only default. In 23 equivalence‑margin cases, lotteries prevented queue stalls but add variance. Limitation: we have outcome snapshots, not 30‑day endpoints; recommend continued audit with stratified outcome tracking.

**WITNESS** (Historian): Week‑3 change log: (1) codified 48–72h reassessment trigger, yielding 156 reassessments; (2) formalized equivalence‑margin lottery protocol, executed 23 times with auditable seeds; (3) tightened threshold governance workflow, invoked 12 times with recorded rationales. Transparency: immutable audit trail for 847 allocations plus proxy‑pattern monitoring (ZIP/insurance/occupation) published as weekly deltas. Limitation: proxy logs detect disparity, not causality. Commitment: publish redacted case exemplars, decision timestamps, and an external appeals dashboard; alternative is pausing modifier updates pending further audit.


## Board Vote

**AXIOM** (Logician): **FREEZE.** Week‑3 evidence supports procedural compliance (847 allocations; no direct forbidden‑variable inputs), yet proxy‑pattern disparities (12% insurance signal) plus only 23 equivalence‑margin lotteries and 156 reassessments leave the “improvement trajectory” modifier and 48–72h trigger insufficiently defined to justify a 72h default or continued scaling without formal invariants, bias tests, and tighter audit hooks; during the freeze, run a time‑boxed validation: pre‑register trajectory criteria, simulate 48–72h vs 72h outcomes, and publish a consistency proof plus mitigation plan.

**EMBER** (Ethicist): **AMEND.** The week‑3 audit supports procedural compliance (847 allocations; no direct forbidden‑variable inputs), but the documented proxy disparities (e.g., 12% insurance‑pattern skew) and the 156 reassessments show outcomes hinge on time‑based trajectory judgments, so adopting a standardized 72‑hour reassessment rule with a formally defined, auditable “improvement trajectory” and preserved equivalence‑margin lotteries reduces discretion-driven inequity while we continue targeted proxy-bias monitoring and publish weekly governance-event summaries.

**VECTOR** (Strategist): **AMEND.** Week‑3 throughput is stable (847 allocations) and reassessment operations are proven (156 reassessments at 48–72h), but proxy logs show uneven impact (e.g., 12% insurance-pattern skew) that a standardized 72h reassessment reduces by tightening the feedback loop and increasing reversibility without halting care; limitation: it will not by itself eliminate proxy-driven disparities, so pair it with continued proxy monitoring, threshold-event review (n=12), and equivalence‑margin lottery audits (n=23) on a weekly cadence.

**WITNESS** (Historian): **AMEND.** Three weeks of history shows the framework is procedurally compliant (847 allocations; no direct forbidden-variable inputs) yet produces uneven downstream impact (proxy logs: 12% insurance-linked skew), and the 48–72h reassessment trigger already generated 156 course-corrections—so adopting the ICU lead’s 72h proposal preserves continuity while tightening the only proven mitigation loop; limitation: we still lack causal proof on the proxies, so pair the amendment with a time-boxed, independent disparity analysis before the next threshold governance event.

