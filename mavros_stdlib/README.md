# Mavros Standard Library for Noir

Contains additional black-box functionality, implemented in Noir, to support Mavros-specific
operations.

The injected library compiles under BN254 and Goldilocks. AES is shared. SHA-256 uses packed spread
arithmetic on BN254 and ordinary wrapping integer arithmetic on Goldilocks. BLAKE's Goldilocks
implementation also keeps its carries in integers, avoiding native-field bit decomposition.

The vendored bignum and curve gadgets require BN254's field headroom and are field-gated, along with
the embedded-curve and Poseidon2 implementations. Calls to ECDSA or Poseidon2 on Goldilocks produce
unresolved-item compile-time errors. Porting those algorithms requires suitable limb layouts or
parameters; reducing their BN254 constants modulo Goldilocks would not preserve their meaning.

[Issue #435](https://github.com/reilabs/mavros/issues/435) tracks replacing the vendored bignum and
curve gadgets with native wide integers and portable implementations.
