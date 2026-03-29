# Plan: Swarm Ownership & Role-Based Access Control

**Author:** TJ (Captain) + Tuvok (Claude CLI #3)
**Date:** 2026-03-28
**Status:** Approved Design — Implementation Pending
**Priority:** HIGH — Required for trusted P2P compute

---

## Problem

The P2P backend currently conflates **ownership** (who controls the swarm) with **coordination** (who dispatches work). Coordinator role transfers based on TFLOPS scoring, and any peer can potentially become coordinator. There is no persistent identity, no cryptographic authority, and no way for the swarm creator to maintain control across devices.

## Solution: Separate Ownership from Coordination

### Role Hierarchy

```
Owner (cryptographic identity, persistent across devices)
│
├── Can join from ANY device with their key (YubiKey, passkey, etc.)
├── Can promote/demote coordinators
├── Can add/revoke other owners (multi-owner orgs)
├── Can manage role assignments (admin, coordinator, worker)
├── Can kick/block peers
├── Survives coordinator loss — authority lives in the KEY, not the device
│
Admin (delegated authority, optional)
│
├── Can manage coordinators and workers
├── Cannot add/remove owners
├── Authority granted by owner via signed role assignment
│
Coordinator (delegated role, transferable)
│
├── Dispatches work, manages peer scoring, handles fault tolerance
├── Authority granted BY owner or admin
├── Can be revoked at any time
├── Elected among peers ONLY if no owner/admin is present
│
Worker (default role)
│
└── Executes kernels, reports results
    No management authority
```

### Core Principle

**Ownership is a cryptographic identity, not a device.** A YubiKey holder IS the owner regardless of which device they use. The swarm recognizes the key, not the IP, browser tab, or peer ID.

---

## Architecture

### SwarmIdentity

Manages the owner's cryptographic identity. Generates, imports, exports, and binds to hardware keys.

```csharp
public class SwarmIdentity
{
    // The owner's ECDSA key pair (via SpawnDev.BlazorJS.Cryptography)
    public PortableECDSAKey OwnerKey { get; }

    // Public key bytes for verification by peers
    public byte[] PublicKeySpki { get; }

    // Create new identity (generates fresh ECDSA key)
    public static Task<SwarmIdentity> CreateAsync(PortableCrypto crypto);

    // Import from hardware key (WebAuthn / FIDO2)
    public static Task<SwarmIdentity> FromHardwareKeyAsync(PortableCrypto crypto);

    // Import from stored key material
    public static Task<SwarmIdentity> ImportAsync(PortableCrypto crypto,
        byte[] publicKey, byte[] privateKey);

    // Sign a message as this identity
    public Task<byte[]> SignAsync(byte[] data);

    // Verify a message from any identity
    public static Task<bool> VerifyAsync(PortableCrypto crypto,
        byte[] publicKey, byte[] data, byte[] signature);
}
```

### RoleAssignment

Signed message: "Owner X grants Role Y to Peer Z at Time T."

```csharp
public class RoleAssignment
{
    public string PeerId { get; set; }           // Who receives the role
    public SwarmRole Role { get; set; }           // Owner, Admin, Coordinator, Worker
    public byte[] GranterPublicKey { get; set; }  // Who granted it
    public byte[] Signature { get; set; }         // ECDSA signature of the assignment
    public long Timestamp { get; set; }           // When granted (UTC ticks)
    public long? ExpiresAt { get; set; }          // Optional expiration

    // Verify this assignment is authentic
    public Task<bool> VerifyAsync(PortableCrypto crypto);
}

public enum SwarmRole
{
    Worker = 0,       // Default — execute kernels only
    Coordinator = 1,  // Dispatch work, manage peers
    Admin = 2,        // Manage coordinators and workers
    Owner = 3,        // Full control, manage everything including other owners
}
```

### KeyRegistry

Owner-signed list of authorized keys and their roles. Published via BEP 46 so all peers can verify authority without trusting the coordinator.

```csharp
public class KeyRegistry
{
    public List<AuthorizedKey> Keys { get; set; }
    public List<RevokedKey> Revocations { get; set; }
    public long Sequence { get; set; }            // Monotonic, prevents replay
    public byte[] OwnerSignature { get; set; }    // Signed by swarm owner

    // Verify the entire registry is authentic
    public Task<bool> VerifyAsync(PortableCrypto crypto, byte[] ownerPublicKey);

    // Check if a specific key has a specific role
    public bool HasRole(byte[] publicKey, SwarmRole minimumRole);

    // Check if a key has been revoked
    public bool IsRevoked(byte[] publicKey);
}

public class AuthorizedKey
{
    public byte[] PublicKey { get; set; }
    public SwarmRole Role { get; set; }
    public string Label { get; set; }             // "TJ's YubiKey", "Lab Desktop", etc.
    public long AddedAt { get; set; }
}

public class RevokedKey
{
    public byte[] PublicKey { get; set; }
    public string Reason { get; set; }
    public long RevokedAt { get; set; }
}
```

### WebAuthn / Hardware Key Integration

```csharp
// Browser: uses navigator.credentials (WebAuthn / FIDO2)
// Desktop: uses platform authenticator (Windows Hello, Touch ID)
// YubiKeys, passkeys, biometrics — all produce ECDSA signatures

public class HardwareKeyProvider
{
    // Register a new hardware key as swarm owner
    public Task<SwarmIdentity> RegisterAsync();

    // Authenticate with an existing hardware key
    public Task<SwarmIdentity> AuthenticateAsync();
}
```

---

## Data Flow

### Creating a Swarm

```
1. Owner generates or imports SwarmIdentity (ECDSA key pair)
2. Owner creates swarm → P2PSwarmCoordinator.CreateSwarmAsync()
3. Owner publishes KeyRegistry via BEP 46 (signed by owner key)
   - Registry contains owner's public key as the sole Owner
4. Join link includes the owner's public key hash for verification
```

### Owner Joins from New Device

```
1. Owner opens app on new device
2. Owner authenticates via hardware key (YubiKey/passkey)
   → SwarmIdentity restored from key material
3. Owner joins swarm via join link or magnet
4. Peer sends CapabilityResponse including owner's public key
5. Current coordinator verifies key against KeyRegistry
6. Owner is recognized — full control restored
```

### Coordinator Election (Fallback)

```
If no Owner or Admin is present in the swarm:
1. Peers check KeyRegistry for any assigned Coordinators
2. If assigned Coordinator is present → they become coordinator
3. If no assigned Coordinator → TFLOPS-based election (existing logic)
4. Elected coordinator operates with limited authority until Owner returns

When Owner rejoins:
5. Owner's key is verified against registry
6. Owner can confirm or replace the elected coordinator
```

### Role Management

```
Owner actions (signed and published via BEP 46):
- AddOwner(publicKey, label)      → requires existing owner signature
- RemoveOwner(publicKey)          → requires owner signature, cannot remove last owner
- AssignRole(peerId, role)        → grants coordinator/admin/worker role
- RevokeRole(peerId)              → revokes any role, peer becomes worker
- RevokeKey(publicKey, reason)    → adds to revocation list
```

### Authority Verification (Every Peer)

```
When peer receives a command (kick, block, dispatch, role change):
1. Extract sender's public key from signed message
2. Look up key in cached KeyRegistry
3. Verify signature using SpawnDev.BlazorJS.Cryptography
4. Check role: does sender have authority for this action?
5. Accept or reject based on role hierarchy

KeyRegistry updates via BEP 46 subscription:
- All peers subscribe to the swarm's BEP 46 channel
- When owner publishes updated registry, all peers receive it
- Peers cache the registry locally for fast verification
```

---

## Solving the Split-Brain Problem

The current election system has no distributed agreement — each peer independently calculates who should be coordinator, potentially electing different peers.

With ownership, the split-brain is resolved:

1. **Owner's signed RoleAssignment is the tiebreaker.** If peer A has a signed Coordinator assignment from the Owner, and peer B elected itself via TFLOPS, peer A wins.

2. **KeyRegistry is the authority.** Published via BEP 46 with monotonic sequence numbers. All peers converge on the same registry. No quorum needed — the owner's signature IS the authority.

3. **No owner present?** Fall back to existing TFLOPS election, but the elected coordinator operates with limited authority (cannot kick owners, cannot modify registry).

---

## Dependencies

### Already Built
- **SpawnDev.BlazorJS.Cryptography** — ECDSA sign/verify, cross-platform (browser + desktop)
- **BEP 46 mutable items** — signed state publication via DHT (needs crypto stub replacement)
- **P2PSwarmCoordinator** — kick/block/transfer already implemented
- **AgentChannel** — pub/sub for state updates
- **P2PProtocol** — message types for role management (some exist, some need adding)

### Reference Code (SpawnDev project — NOT a dependency)
- **WebAuthn/FIDO2 reference implementation** — `SpawnDev.AccountsServer` + `SpawnDev.AccountsShared` (21 files)
  - Shows browser-side `navigator.credentials.create/get` via BlazorJS
  - Shows attestation/assertion verification patterns
  - Shows credential management UI flows
  - TJ's YubiKey tested and working with this code
  - **Use as reference for HOW to do WebAuthn calls — but the P2P model is serverless**
  - In P2P: no central server stores credentials. The owner-signed KeyRegistry (published via BEP 46) replaces the server. Every peer verifies independently using the public key from the registry.

### Needs Building
- **SwarmIdentity** — owner key management class (extract WebAuthn pattern from SpawnDev.AccountsShared)
- **RoleAssignment** — signed role grant/revoke messages
- **KeyRegistry** — authorized keys + revocations, signed by owner, published via BEP 46
- **HardwareKeyProvider** — adapt existing WebAuthn flow from SpawnDev.AccountsShared/AccountService.cs
- **Authority verification in P2PWorker** — verify sender's role before accepting commands
- **Role-aware P2PSwarmCoordinator** — check authority before kick/block/transfer

### Crypto Prerequisite
Replace the stub `IDhtSigner` implementations in SpawnDev.WebTorrent with real `PortableCrypto.Sign/Verify` calls (see `_DevComms/global/tuvok-crypto-solution-2026-03-28.md`). This MUST happen before ownership can be trusted.

---

## Implementation Order

1. **Replace WebTorrent crypto stubs** with SpawnDev.BlazorJS.Cryptography (Data's territory)
2. **SwarmIdentity** — key generation, import/export, sign/verify
3. **KeyRegistry** — data structure, serialization, BEP 46 publish/subscribe
4. **RoleAssignment** — signed role messages, verification
5. **Authority checks** — integrate into P2PWorker, P2PSwarmCoordinator, P2PTransport
6. **WebAuthn/HardwareKey** — browser integration for YubiKey/passkey support
7. **Tests** — key management, role assignment, authority verification, split-brain resolution

---

## Security Considerations

- **Last-owner protection:** Cannot remove the last owner key from the registry.
- **Key rotation:** Owner can add a new key and revoke the old one in a single signed registry update.
- **Offline owner:** Swarm continues operating under delegated authority. Owner regains control on rejoin.
- **Compromised key:** Owner revokes via any other owner key, or if sole owner, the swarm must be recreated.
- **Replay protection:** KeyRegistry has monotonic sequence numbers. Older registries rejected.
- **Hardware key loss:** If sole owner loses their YubiKey, swarm authority is lost. Multi-owner setup mitigates this (recommended for production swarms).
