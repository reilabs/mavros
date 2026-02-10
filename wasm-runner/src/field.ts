/**
 * BN254 Field Element Utilities
 */

export const MODULUS = 21888242871839275222246405745257275088548364400416034343698204186575808495617n;
const R = 6350874878119819312338956282401532410528162663560392320966563075034087161851n;
const R_INV = 9915499612839321149637521777990102151350674507940716049588462388200839649614n;
const MASK_64 = (1n << 64n) - 1n;

export type FieldElement = [bigint, bigint, bigint, bigint];

export function toMontgomery(value: bigint): FieldElement {
  let v = value % MODULUS;
  if (v < 0n) v += MODULUS;
  const mont = (v * R) % MODULUS;
  return [
    mont & MASK_64,
    (mont >> 64n) & MASK_64,
    (mont >> 128n) & MASK_64,
    (mont >> 192n) & MASK_64,
  ];
}

export function fromMontgomery(limbs: FieldElement): bigint {
  const mont = limbs[0] | (limbs[1] << 64n) | (limbs[2] << 128n) | (limbs[3] << 192n);
  return (mont * R_INV) % MODULUS;
}

export function fieldToHex(limbs: FieldElement): string {
  return '0x' + fromMontgomery(limbs).toString(16).padStart(64, '0');
}

export function parseFieldElement(str: string): FieldElement {
  return toMontgomery(BigInt(str));
}
