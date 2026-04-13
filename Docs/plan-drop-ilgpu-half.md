# Plan: Drop ILGPU.Half, Use System.Half Exclusively

**Date:** 2026-04-12
**Status:** Approved direction, not yet scheduled
**Target:** .NET 10+ only (System.Half available since .NET 5)

---

## What
Replace all `ILGPU.Half` with `System.Half`. Remove the custom Half type entirely.

## Why
- One Half type instead of two
- No bridging/conversion operators needed
- System.Half is runtime-optimized with full math operations
- Frontend IL processing handles System.Half natively
- Eliminates HalfConversion.tt maintenance

## Scope
- Large refactor across the entire ILGPU codebase
- T4 templates (.tt files) need updating
- HalfExtensions intrinsic registrations need updating
- All consuming projects (SpawnDev.ILGPU.ML, AubsCraft, etc.) need System.Half

## When
After Int16/Float16 sub-word support is fully landed and tested. Not today.
