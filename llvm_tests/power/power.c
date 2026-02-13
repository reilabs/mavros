#include <stdint.h>

// Injected "special" field type for C->LLVM flow: opaque handle in addrspace(7).
// This is a Clang extension and maps to LLVM `ptr addrspace(7)`.
typedef struct FieldImpl __attribute__((address_space(7))) *Field;

// Injected "special" field operations.
extern Field __field_mul(Field a, Field b);
extern Field __field_from_u64(uint64_t x);
extern void __assert_eq(Field a, Field b);

void mavros_main(Field x, Field y) {
  Field r = __field_from_u64(1ULL);

  for (uint32_t i = 0; i < 1000000; i++) {
    r = __field_mul(r, x);
  }

  __assert_eq(r, y);
}
