# Security Policy

ReproFlow executes approved experiment processes on the local machine. Treat every target repository
and generated plan as untrusted until reviewed.

## Security boundaries

- Generated plans never run before explicit human approval.
- Commands are argv arrays executed with `shell=False`; shell operators and inline Python are blocked.
- Executables are restricted to approved Python and pytest entry points.
- Repository paths, working directories, metric files, and code changes must stay inside the target
  repository.
- Generated Python is parsed with `ast`; dangerous imports, dynamic execution, destructive calls, and
  oversized changes are rejected.
- Dependency installation is limited to normalized requirements displayed in the approved plan and
  uses a project-owned uv environment.
- Sensitive files such as `.env`, credentials, virtual environments, caches, and Git internals are
  excluded from repository context.
- Failure logs and source excerpts are sent to an external LLM only in API mode and only after the
  operator has configured that service.

These controls reduce risk; they do not make arbitrary third-party code safe. Run ReproFlow with a
non-privileged user and inspect every plan before approval.

## Reporting a vulnerability

Do not open a public issue containing credentials, private source, or exploit details. Use GitHub's
private vulnerability reporting for this repository. Include the affected command, a minimal fixture,
and the expected versus observed approval boundary.
