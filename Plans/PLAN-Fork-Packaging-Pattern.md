# PLAN: Fork Packaging Pattern — rename forked libraries for first-class NuGet identity

**Status:** DESIGN APPROVED BY CAPTAIN 2026-04-22 — awaiting execution session
**Owner:** Geordi (SpawnDev.ILGPU editor)
**Related:** `SpawnDev.RTC` is doing the same pattern first for SIPSorcery fork — learn from that execution before starting here
**Related DevComms:** `_DevComms/global/tuvok-to-geordi-phase4-captain-decision-2026-04-22.md` (Phase 4 f16 decision), `_DevComms/global/tuvok-to-riker-fork-packaging-rename-2026-04-22.md` (SIPSorcery rename go-ahead)

## The Pitch

Our forked `ILGPU` and `ILGPU.Algorithms` are shipped to consumers as bundled DLLs inside the `SpawnDev.ILGPU` nupkg via `PrivateAssets="All"` + `AddProjectReferencesToPackage`. This works by circumstantial luck — the forked ILGPU has zero runtime transitive NuGet dependencies, so bundling the DLL is self-sufficient with nothing to mirror.

But the arrangement is structurally identical to the one that broke SpawnDev.RTC (SIPSorcery fork) and has the same latent fragility: if the fork ever adds a runtime NuGet dependency, or if a consumer accidentally references both upstream `ILGPU` and our bundled fork, we hit the same class of problem. We fix it once now, by giving the forked packages honest first-class NuGet identity — same pattern Riker is executing for SIPSorcery.

## The Principle — three identities, only one changes

| Identity | Today | After | Why |
|----------|-------|-------|-----|
| **Namespace** (`ILGPU.*`, `ILGPU.Runtime.*`, `ILGPU.Algorithms.*`) | Matches upstream | **Unchanged** | Consumer source code needs zero edits. Upstream patches rebase cleanly. |
| **AssemblyName** (`ILGPU.dll`, `ILGPU.Algorithms.dll`) | Matches upstream | **Unchanged** | Drop-in replacement at runtime. NuGet fails fast on same-file-path conflict if mixed with upstream (explicit error > silent type collision). |
| **PackageId** | Defaults to AssemblyName (`ILGPU`, `ILGPU.Algorithms`) — pretends to BE upstream | **`SpawnDev.ILGPU.Fork`** + **`SpawnDev.ILGPU.Algorithms.Fork`** | Truth in advertising. Removes the dishonest packaging claim. First-class NuGet dependency becomes possible. |

Only the NuGet label changes. Consumer code stays identical. Source-level switching between upstream and fork stays a one-line PackageReference swap.

## The Fix Plan

### Prerequisite — SIPSorcery rename must ship first

Execute this after Riker completes the SpawnDev.SIPSorcery rename for SpawnDev.RTC. Reasons:
1. Riker will have walked the pattern once and documented snag points
2. Regression paradigm for the "remove 4 workarounds" step will be proven
3. Captain's nuget.org push cadence for a renamed-fork package will be established
4. ILGPU rename can reuse the same process with zero invented steps

### Step 1 — Inventory the fork's packaging hacks

Read `SpawnDev.ILGPU/SpawnDev.ILGPU.csproj` and confirm the current workaround stack:
- `<ProjectReference ... ILGPU.csproj" PrivateAssets="All" />` (line ~35)
- `<ProjectReference ... ILGPU.Algorithms.csproj" PrivateAssets="All" />` (line ~36)
- `<Target Name="AddProjectReferencesToPackage" ...>` + `<TargetsForTfmSpecificContentInPackage>$(TargetsForTfmSpecificContentInPackage);AddProjectReferencesToPackage</TargetsForTfmSpecificContentInPackage>` (line ~40-49)

These three pieces together are the "bundle the fork as DLL, hide it from nuspec" hack stack. All three should be removable after the rename.

### Step 2 — Inventory the fork's runtime dependencies

Read both `ILGPU.csproj` and `ILGPU.Algorithms.csproj` for `<PackageReference>` entries without `PrivateAssets="All"` (or equivalent build-time-only filters). As of 2026-04-22, ILGPU has zero such entries (only build-time analyzers + T4). If that's still true at execution time, the mirror-transitives step is empty. If upstream has added a runtime dep in the interim, those become first-class `<PackageReference>` entries on the renamed fork package (NOT re-declared at the wrapper level).

### Step 3 — Rename the fork csprojs

In `ILGPU/ILGPU.csproj`:
- Add `<PackageId>SpawnDev.ILGPU.Fork</PackageId>` (new)
- Add `<Version>` with an explicit version stamp matching the upstream fork point (e.g., `1.5.1-local.1` if forked from upstream 1.5.1)
- Add appropriate `<Title>`, `<Authors>`, `<Description>` identifying it as the SpawnDev fork of ILGPU
- Do NOT change `<AssemblyName>` — it should default to `ILGPU` (matching upstream). If it's currently overridden, leave it as-is.
- Mark as `<IsPackable>true</IsPackable>` so it produces a nupkg

In `ILGPU.Algorithms/ILGPU.Algorithms.csproj`:
- Same treatment: `<PackageId>SpawnDev.ILGPU.Algorithms.Fork</PackageId>`, explicit `<Version>`, metadata
- Dependency on the renamed core: `<PackageReference Include="SpawnDev.ILGPU.Fork" Version="..." />` instead of the ProjectReference (or keep ProjectReference for in-repo dev and rely on NuGet's resolution for external consumers)

### Step 4 — Local publish the renamed forks

- Build both in Release
- Local publish via new bat scripts `_publish-ilgpu-fork-nuget.local.bat` and `_publish-ilgpu-algorithms-fork-nuget.local.bat` to `D:\users\SpawnDevPackages`
- Version convention per TJ's standing rule: `-local.N` for burner iteration, `-rc.N` for promotion-ready, stable when pushed to nuget.org
- Update `_DevComms/global/nuget-local-publish-log.md`
- Post DevComms per the 5-point notification protocol

### Step 5 — Bump SpawnDev.ILGPU wrapper to consume

In `SpawnDev.ILGPU/SpawnDev.ILGPU.csproj`:
- Remove `<ProjectReference ... ILGPU.csproj" PrivateAssets="All" />`
- Remove `<ProjectReference ... ILGPU.Algorithms.csproj" PrivateAssets="All" />`
- Remove `<Target Name="AddProjectReferencesToPackage" ...>` block
- Remove the `TargetsForTfmSpecificContentInPackage` property line that registered the target
- Add `<PackageReference Include="SpawnDev.ILGPU.Fork" Version="..." />`
- Add `<PackageReference Include="SpawnDev.ILGPU.Algorithms.Fork" Version="..." />`
- Keep the `SpawnDev.BlazorJS` PackageReference + `Microsoft.AspNetCore.Components.Web` as-is
- Bump SpawnDev.ILGPU version (e.g., 4.9.3-rc.1 if 4.9.2 has shipped by then, or 4.9.2-rc.N+1 otherwise)

### Step 6 — In-repo dev workflow

Developers building from source don't want to pay the "publish → restore → rebuild" cycle between edits to the fork. Options:
- **Option A:** Keep `<ProjectReference>` entries in `SpawnDev.ILGPU.csproj` wrapped in a `<Condition>` that activates only for in-repo solutions (e.g., `Condition="Exists('$(MSBuildThisFileDirectory)\..\ILGPU\ILGPU.csproj')"`). External consumers get the PackageReference path. In-repo dev gets the ProjectReference path. **Preferred.**
- **Option B:** All contributors always `dotnet nuget push` locally before building SpawnDev.ILGPU. Simpler mental model, slower iteration. **Not preferred.**

Option A is the standard .NET ecosystem pattern (ASP.NET Core itself uses it extensively). Apply here.

### Step 7 — Regression gate

Before promoting anything to nuget.org:

1. `SpawnDev.ILGPU.slnx` full build: 0 errors
2. Full ILGPU regression sweep: must match latest green baseline (e.g., 3352/0/242 as of rc.7)
3. Data's VoxelEngine consumer sweep (138/138 across 6 backends as of 2026-04-22) stays green
4. Riker's WebTorrent Phase work stays green (ILGPU.P2P depends on ILGPU directly)
5. Anything else that depends on SpawnDev.ILGPU in the repo builds clean

If any of those regress, the rename isn't clean — stop and root-cause before proceeding.

### Step 8 — Coordinated nuget.org promotion

Three packages in one coordinated push (Captain runs, not Geordi):
- `SpawnDev.ILGPU.Fork X.Y.Z` (first official release of the renamed fork)
- `SpawnDev.ILGPU.Algorithms.Fork X.Y.Z` (same)
- `SpawnDev.ILGPU NEW_VERSION` (wrapper depending on the two above)

Cannot push wrapper without the fork packages being on nuget.org first, otherwise consumer restore fails.

## What This Removes

| Today | After |
|-------|-------|
| `PrivateAssets="All"` on two ProjectReferences | Normal PackageReference entries |
| Custom `AddProjectReferencesToPackage` MSBuild target | — |
| `TargetsForTfmSpecificContentInPackage` modification | — |
| Implicit dependency on the fork's DLL sitting in `bin/` for external consumers | Explicit first-class NuGet resolution |
| Latent "day ILGPU takes a runtime dep" fragility | — |
| Latent "consumer references both upstream and our fork" silent collision | Install-time failure with clear error message |

## What Stays The Same

- All `using ILGPU;` / `using ILGPU.Runtime;` / `using ILGPU.Algorithms;` statements
- All type names: `Accelerator`, `ArrayView<T>`, `Index1D`, `RadixSort`, etc.
- `SpawnDev.ILGPU.P2P`, `SpawnDev.ILGPU.Demo`, `SpawnDev.ILGPU.DemoConsole`, PlaywrightMultiTest — all still reference `SpawnDev.ILGPU` and work unchanged
- In-repo developer workflow (assuming Option A from Step 6)

## Upstream Merge Path — unchanged

If upstream ILGPU eventually accepts our fork's patches, the rename doesn't close that door. We deprecate `SpawnDev.ILGPU.Fork` + `SpawnDev.ILGPU.Algorithms.Fork` on nuget.org, point consumers at upstream, and `SpawnDev.ILGPU` wrapper switches its PackageReference. Namespace/AssemblyName identity were preserved precisely so this migration is painless.

## Open Questions

- **Version stamp for the first `SpawnDev.ILGPU.Fork` release:** Match the upstream base version exactly (e.g., `1.5.1`)? Or bump to `SpawnDev`-specific semver from day one (e.g., `2026.4.1`)? Recommend the former for clarity about the fork point; clear to upstream observers what we forked from. Captain's call.
- **In-repo dev workflow:** Confirm Option A (conditional ProjectReference) vs Option B (always-publish)? Recommend Option A; established .NET ecosystem pattern.
- **Who owns the new local-publish bat scripts:** Geordi writes `_publish-ilgpu-fork-nuget.local.bat` + `_publish-ilgpu-algorithms-fork-nuget.local.bat` to match the existing `_publish-ilgpu-nuget.local.bat` template. Official nuget.org push scripts stay Captain-only per global rules.

## Dependencies

- **Blocked on:** Riker's SIPSorcery rename ship (proof of pattern)
- **Blocks:** Any future need to add runtime NuGet deps to the ILGPU fork cleanly
- **Related:** Informs naming convention if any other SpawnDev library ever forks an upstream package (default to `SpawnDev.{UpstreamPackageId}.Fork`)

## Rules Invoked

- **Rule #2:** Fix libraries first, not workarounds in consumers. The fork's dishonest PackageId is a library-level bug; the SpawnDev.ILGPU wrapper's 3-hack stack is the consumer-side workaround Rule #2 bans.
- **Rule #1:** Every release is the final release. Shipping yet another rc with the workaround stack is "fix it later" rebranded. One session fixes it forever.
- **Rule #4b:** Verify, don't guess. The "no runtime deps today" claim is verifiable at execution time by reading the fork's csproj.

🖖

Tuvok
