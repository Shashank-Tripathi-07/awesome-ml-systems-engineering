# Changelog

All notable changes to this list are documented here.

## Release 1.0.0 - 2026-07-04

First tagged release.

### Added

- Eight collapsible topic groups (Get Started, Hardware Compilers & Kernels, Training at Scale, Inference & Efficiency, LLM & Agentic Systems, Data & Experimentation, Production Governance & Cost, Papers Community & Case Studies) so the list reads as a short, browsable index instead of one long page of 49 flat sections.
- "Learning Paths by Level" section with beginner, intermediate, and advanced tracks through the list, plus a skill-level legend (green/yellow/red).
- New sections: Reasoning and Test-Time Compute, On-Device and Mobile LLM Inference, Data Labeling and Annotation.
- New entries in Agentic Systems Infrastructure (Model Context Protocol), RLHF Infrastructure (Gymnasium, BrowserGym), Memory-Efficient Training (ZeRO-Infinity), Inference and Serving (FlexGen), Fault Tolerance and Reliability (Chaos Mesh), Cost Engineering (Karpenter), LLMOps Platforms (Portkey), and Experiment Management (Sacred).
- Link from README to ROADMAP.md.
- README badges: CI status, last commit, contributors, license.
- Issue templates for resource suggestions, new section proposals, and broken links.
- Pull request template with a contribution checklist.
- Weekly scheduled link-check run in CI, independent of PR activity.
- Table of contents, ~40 hyperlinked resources, and inline plain-language
  glosses for jargon (RDMA, GEMM, KV cache, and about 20 more) throughout
  ROADMAP.md, so it reads start to finish with no prior ML background required.

### Fixed

- Removed duplicate entries (Langfuse, Evidently AI, DVC, MLflow, Scaling Laws, Chinchilla) that appeared verbatim or near-verbatim in two sections; each resource now has a single canonical entry.
- Removed anchor links that pointed at the same in-page target more than once (the list only allows each internal link to appear once), replacing repeat mentions with plain text pointers.
- Dead link: `phoenix.arize.com` no longer resolves; repointed to `docs.arize.com/phoenix`.
- ROADMAP.md: a leftover mid-edit artifact in the KV cache memory math, a
  tensor-parallelism code example that contradicted its own explanation
  (`reduce_scatter` vs the correct `all_reduce`), and an H100 L2 cache size
  that was actually the A100 figure.
- Hardened the link-check CI job (accept 415, more retries, capped
  concurrency, excluded a handful of bot-hostile domains) after it produced
  several false-positive failures on unrelated, healthy links.
