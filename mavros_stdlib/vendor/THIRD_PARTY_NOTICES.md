# ECDSA secp256r1 Noir arithmetic

`ecdsa_secp256r1.nr` contains an adapted subset of:

- `zkpassport/noir-bignum` v0.10.0-1 at commit
  `675378b1622a2b125ae6927803c79e6bd7ed2b61`.
- `zkpassport/noir_bigcurve` v0.14.0-1 at commit
  `3c3c82619a6ef397fc77fc28a3419ab2681ec2f4`.

The ECDSA verification flow is adapted from `zkpassport/noir-ecdsa` v0.4.0 at commit
`6ec6dc9458211587cb3059a38a1dac2300702273`.

The subset retains only the bignum operations, secp256r1 curve arithmetic, and transcript checks
reachable from ECDSA verification. It removes tests, benchmarks, unrelated curves and fields,
seed derivation, square roots, integer division, standalone curve-operation wrappers, upstream API
documentation, and the upstream bignum debugging assertion. Paths are adjusted so the code can live
inside Mavros' injected Noir standard-library module. The upstream `noir-bignum` and `noir-ecdsa`
releases carry the Apache License 2.0; Mavros itself is distributed under the same license.
