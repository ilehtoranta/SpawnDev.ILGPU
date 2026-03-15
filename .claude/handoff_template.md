# Session Handoff Template

Use this with `/compact` to preserve the mental model across context compaction or new sessions. Copy the template below, fill in the current state, and paste it as the compaction summary.

---

## Active Task
<!-- One sentence: what are you doing RIGHT NOW? -->


## Mental Model: Transpiler State
<!-- Which backend(s) are you working in? What layer (IR → codegen → dispatch → worker)? -->

**Backend**: <!-- WebGPU / Wasm / WebGL / ILGPU core / Algorithms -->
**Layer**: <!-- IR type system / kernel codegen / dispatch/serialization / worker execution / buffer management -->
**Key files open**: <!-- The 2-3 files you're actively editing -->

## What's Working
<!-- Tests that pass, things you've verified. Anchor points so you don't re-verify. -->


## What's Broken
<!-- The specific failure: error message, wrong output, WAT line numbers, etc. -->


## Root Cause Hypothesis
<!-- Your current theory for WHY it's broken. Include evidence. -->


## Last Thing Tried
<!-- The exact code change or diagnostic you ran last, and its result. -->


## Next Step
<!-- The single next action to take. Be specific: which file, which function, what change. -->


## Eliminated Theories
<!-- Things you checked that are NOT the problem. Saves re-investigation. -->


## Key Discoveries This Session
<!-- Non-obvious findings that should be in memory if not already saved. -->


## Diagnostic Code In Place
<!-- Any temporary logging/debug code currently in the codebase that needs cleanup. -->


## Files Modified (uncommitted)
<!-- `git diff --stat HEAD` snapshot. Helps the next session know what's dirty. -->

