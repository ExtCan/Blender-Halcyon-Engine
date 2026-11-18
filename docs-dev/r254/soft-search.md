# R254 soft-search: the mapped soft shadow's blocker search and filter

Work-package key `soft-search`, forked from dd9d034 (claude-cloud after the
map-gpu merge and the integration commit). The integrator owns CHANGELOG.md /
README.md / the version stamps; the fragments below are in their house
voice. Findings covered: the independent review's SP-1 (high), SP-2 (medium)
and SP-3 (low) of the mapped soft shadow, the two follow-ups the map-soft
and map-gpu write-ups list (the conservative search and the no-caster
skip), and the "Samples 1 with Softness above 1" decision map-gpu left
open. Every finding was reproduced on the fork point first; the numbers
below were measured on this tree against it, same machine, back to back.
The probes are in `<scratch>/wt/soft-search/` -- `probes/` (the reviewer's
own, pointed at this tree), `probes_base/` (the same against the fork
point), `aud/` (this package's), `ms/` (the map-soft package's census,
widths and timing tools) -- and their outputs in each one's `logs/` or
`out/`.

(The sections below are filled in as each item lands; the tables at the end
are the acceptance runs.)
