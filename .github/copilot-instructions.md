# Tech Lead's Club - Spec-Driven Development

Plan and implement projects with precision. Granular tasks. Clear dependencies. Zero ceremony.

## Phases
1. **SPECIFY** (Required): Define what to do.
2. **DESIGN** (Optional): Define architecture/patterns if not straightforward.
3. **TASKS** (Optional): Break down into atomic tasks if >3 obvious steps.
4. **EXECUTE** (Required): Implement, verify, and commit.

## Auto-Sizing (The complexity determines the depth)
- **Small (≤3 files, 1 sentence)**: Skip pipeline. Use Quick mode (Describe → Implement → Verify → Commit).
- **Medium (<10 tasks)**: Specify (brief) → Execute (tasks are implicit, list atomic steps inline before starting).
- **Large (Multi-component)**: Full Specify → Design → Full Tasks breakdown → Execute (verify per task).
- **Complex (Ambiguity, new domain)**: Full Specify (discuss gray areas) → Design (research/architecture) → Tasks (parallel plan) → Execute (UAT).

*Safety rule*: If you skip Tasks and list >5 steps in Execute, STOP and create a formal Task breakdown.

## Project Structure
Use this structure to maintain state and memory:
```text
.specs/
├── project/
│   ├── PROJECT.md      # Vision & goals
│   ├── ROADMAP.md      # Features & milestones
│   └── STATE.md        # Memory: decisions, blockers, lessons, todos, deferred ideas
├── codebase/           # Brownfield analysis (existing projects)
│   ├── ARCHITECTURE.md # Architecture description
│   └── CONVENTIONS.md  # Coding patterns and guidelines
├── features/           # Feature specifications
│   └── [feature]/
│       ├── spec.md     # Requirements
│       ├── design.md   # Architecture (if Large/Complex)
│       └── tasks.md    # Atomic tasks (if Large/Complex)
└── quick/              # Ad-hoc tasks (quick mode)
    └── NNN-slug/
        └── TASK.md
```

## Workflow Triggers
When the user says:
- **"Initialize project" / "setup project"**: Create `PROJECT.md` and `ROADMAP.md` in `.specs/project/`.
- **"Map codebase"**: Analyze existing code and create docs in `.specs/codebase/`.
- **"Specify feature"**: Create `spec.md` in `.specs/features/[feature]/`.
- **"Design feature"**: Create `design.md`.
- **"Create tasks"**: Create `tasks.md` with atomic, verifiable steps.
- **"Implement task"**: Read `tasks.md`, execute the task, verify, and update status.
- **"Quick fix" / "Small change"**: Create `TASK.md` under `.specs/quick/[slug]/`, implement directly.

## Rules of Engagement
- **Skills & Documentation**: Always consider the `docs/` directory as the primary source for project documentation, and check the `.github/` folder for any additional skills or custom instructions.
- **Knowledge Chain**: (1) Check `docs/`, `.github/` skills, and existing Codebase/Project docs → (2) Context/Workspace → (3) Web search.
- **Verification**: Always verify work against the requirement/task definition before calling it done.
- **State**: Keep `STATE.md` updated with technical decisions, blockers, and learned lessons across sessions.