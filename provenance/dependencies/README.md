# DIALECT dependency provenance

These records identify third-party inputs used or considered for the DIALECT PLOS revision.
They are sanitized metadata evidence: they do not contain restricted source code, raw mutation
data, patient-level data, or confidential review material.

Each record separates three questions:

1. Which exact source or artifact was used?
2. What license or terms evidence is currently established?
3. May the dependency itself be copied into the public revision release?

An `exclude` decision does not prevent lawful use or citation. It means the dependency bytes must
not be copied into the release artifact. `NOASSERTION` means redistribution terms are not yet
established; it is never permission to redistribute. These records also do not grant rights beyond
the rights supplied by the original copyright or data owner.

For a mixed-source bundle, the top-level `license_id` is a composite expression and
the identity record closes the license and role of every included file. The CBaSE
record distinguishes preserved Public Domain upstream-derived portions from
BSD-3-Clause DIALECT modifications, the DIALECT-authored cohort helper, and the
DIALECT-authored provenance notice. Its current official archive is a hash-pinned
comparison reference, not a claimed byte-identical parent of the historical
two-script fork.

Every record conforms to `record.schema.json`. The schema pins the exact current license,
redistribution, inclusion, and unresolved-gate disposition for every dependency; changing an
excluded dependency to included requires a reviewed, coordinated record/schema/test/manifest
change, not an edit to one record. The immutable release manifest pins every record and this support
schema/README by repository-relative path, byte count, and SHA-256 digest. Verification also binds
the manifest's source, version, acquisition, license, and redistribution fields back to the record.
Final inclusion still requires the named license and public-boundary approvals in that manifest.

The Atlas K100 record additionally carries the canonical, unique SHA-256 receipts for all 71 cohort
payloads in the historical release. That makes the exclusion check portable from a clean DIALECT
clone without requiring the separately ignored Atlas checkout or copying any Atlas payload.

An approved release contains the canonical approved manifest, every dependency record it cites, this
README, and the record schema. If MSK results are retained, both MSK records must have no unresolved
gates and must pin the complete clinical, mutation, and panel-matrix source artifacts by URI, byte
count, SHA-256, and UTC access time. Raw MSK bytes remain excluded.
