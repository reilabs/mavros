# LLSSA Design

Low-Level SSA: a target-independent IR between HLSSA and codegen backends (LLVM/WASM, eventually VM).

## Motivation

HLSSA lowers directly to WASM and the custom VM. Both targets must implement a complex runtime:
reference counting, copy-on-write arrays, layout-descriptor-driven traversal, etc. Adding LLVM as
a target would mean reimplementing all of this a third time.

LLSSA eliminates the runtime by making everything explicit. By the time code reaches LLSSA:
- RC is just struct fields manipulated with Load/Store/Add/Sub
- CoW is just branches that compare the RC and conditionally copy
- Drop is a generated function that decrements RC and frees if zero
- Arrays and tuples are just pointers to structs with known field layouts

Each target backend only needs to implement: memory allocation/deallocation, field arithmetic
intrinsics, and straightforward instruction selection. No runtime semantics.

## Pipeline Position

```
HLSSA
  -> ... existing passes ...
  -> WitnessWriteToVoid -> DCE -> RCInsertion -> FixDoubleJumps
  -> HLSSA-to-LLSSA lowering      <-- new
  -> LLSSA target codegen          <-- new (LLVM, WASM, eventually VM)
```

RC insertion runs on HLSSA (reusing the existing pass). The HLSSA-to-LLSSA lowering translates
`MemOp::Bump`/`MemOp::Drop` into explicit load-add-store / call-to-drop-fn sequences.

## SSA Infrastructure

LLSSA reuses the generic parametric SSA structure:

```rust
pub type LLSSA = SSA<LLOp, LLType>;
```

Blocks, block parameters, terminators (`Jmp`, `JmpIf`, `Return`), value IDs, function IDs — all
unchanged from HLSSA. Only the opcode enum and type enum are new.

## Type System

### SSA Value Types

What a `ValueId` can hold:

```rust
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LLType {
    /// Sized unsigned integer. Int(1) = bool, Int(8) = byte, Int(32), Int(64), etc.
    Int(u32),
    /// Opaque pointer. Target decides width (8 bytes native, 4 bytes wasm32).
    Ptr,
    /// Multi-word aggregate, by value. Can only be used for "value-safe" structs
    /// (those whose fields are all Int, Ptr, or Inline — no InlineArray/FlexArray).
    Struct(LLStruct),
}
```

### Struct Definitions

```rust
/// Struct layout, owned inline. Structural equality.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LLStruct {
    pub fields: Vec<LLFieldType>,
}

/// What a struct field / memory slot holds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LLFieldType {
    Int(u32),
    Ptr,
    /// Nested struct embedded in place (not behind a pointer).
    Inline(LLStruct),
    /// Fixed-count contiguous array of identical structs.
    /// Accessible via ArrayElemPtr. Memory-only (cannot ExtractField).
    InlineArray(LLStruct, usize),
    /// Variable-length trailing array (like C99 flexible array member).
    /// Must be the last field. Length known at allocation time, not in the type.
    FlexArray(LLStruct),
}
```

### Value-Safe vs Memory-Only Structs (Convention)

A struct is **value-safe** if all its fields are `Int`, `Ptr`, or `Inline(value_safe_struct)`.
Value-safe structs can be used as SSA values (`LLType::Struct`) and manipulated with
`MkStruct`/`ExtractField`/`InsertField`.

A struct with `InlineArray` or `FlexArray` fields is **memory-only** — it always lives behind a
`Ptr` and is accessed via `StructFieldPtr`/`ArrayElemPtr`/`Load`/`Store`.

This is a convention enforced by the lowering, not a type system constraint.

### Standard Struct Definitions

The HLSSA-to-LLSSA lowering generates these as needed:

```
// BN254 field element: 4 limbs in Montgomery form
FieldElem = { Int(64), Int(64), Int(64), Int(64) }

// RC headers
RcHeader    = { Int(64) }                  // refcount only (fixed-size arrays, tuples)
SliceHeader = { Int(64), Int(32) }         // refcount + element count (slices)

// Heap RC'd fixed array of 5 field elements (memory-only)
RcArray5Field = { Inline(RcHeader), InlineArray(FieldElem, 5) }

// Heap RC'd dynamic slice of field elements (memory-only)
DynSlice_Field = { Inline(SliceHeader), FlexArray(FieldElem) }

// Heap RC'd tuple { Field, u32 } (memory-only due to RC header)
RcTuple_F_U32 = { Inline(RcHeader), Inline(FieldElem), Int(32) }

// Stack tuple { Field, u32 } (value-safe, no RC)
Tuple_F_U32 = { Inline(FieldElem), Int(32) }
```

No layout descriptor word. Fixed-size arrays don't store their length (the generated drop function
knows it statically). Slices store their element count in `SliceHeader` to bound iteration in the
generated drop function.

## Opcodes

```rust
pub enum LLOp {
    // ═══════════════════════════════════════════════════════════════════════
    // Constants
    // ═══════════════════════════════════════════════════════════════════════

    /// Integer constant. Result type: Int(n) determined by context.
    IntConst { result: ValueId, value: u64 },

    /// Null pointer constant.
    NullPtr { result: ValueId },

    // ═══════════════════════════════════════════════════════════════════════
    // Integer Arithmetic — polymorphic over Int(n)
    // ═══════════════════════════════════════════════════════════════════════
    // lhs and rhs must be same Int(n). Result is Int(n).

    IntArith { kind: IntArithOp, result: ValueId, a: ValueId, b: ValueId },
    // IntArithOp: Add, Sub, Mul, UDiv, URem, And, Or, Xor, Shl, UShr

    /// Bitwise NOT. Int(n) -> Int(n).
    Not { result: ValueId, value: ValueId },

    // ═══════════════════════════════════════════════════════════════════════
    // Integer Comparison — polymorphic over Int(n)
    // ═══════════════════════════════════════════════════════════════════════
    // a and b must be same Int(n). Result is always Int(1).

    IntCmp { kind: IntCmpOp, result: ValueId, a: ValueId, b: ValueId },
    // IntCmpOp: Eq, ULt

    /// Truncate: Int(m) -> Int(n) where n < m. Drops high bits.
    Truncate { result: ValueId, value: ValueId, to_bits: u32 },
    /// Zero-extend: Int(n) -> Int(m) where m > n. Fills high bits with 0.
    ZExt { result: ValueId, value: ValueId, to_bits: u32 },

    // ═══════════════════════════════════════════════════════════════════════
    // Field Arithmetic — operates on Struct(FieldElem) values
    // ═══════════════════════════════════════════════════════════════════════
    // All take/produce Struct(FieldElem) SSA values (4 x Int(64) by value).
    // Target implements the actual multi-limb BN254 Montgomery arithmetic.

    FieldArith { kind: FieldArithOp, result: ValueId, a: ValueId, b: ValueId },
    // FieldArithOp: Add, Sub, Mul, Div

    /// Field negation (unary). Struct(FieldElem) -> Struct(FieldElem).
    FieldNeg { result: ValueId, src: ValueId },

    /// Field equality. a, b: Struct(FieldElem). Result: Int(1).
    FieldEq { result: ValueId, a: ValueId, b: ValueId },

    /// Convert field element from Montgomery form to 4 plain limbs.
    /// Struct(FieldElem) -> (Int(64), Int(64), Int(64), Int(64)).
    FieldToLimbs { result: ValueId, src: ValueId },
    /// Convert 4 plain limbs into a field element in Montgomery form.
    /// (Int(64), Int(64), Int(64), Int(64)) -> Struct(FieldElem).
    FieldFromLimbs { result: ValueId, limbs: ValueId },

    // ═══════════════════════════════════════════════════════════════════════
    // Aggregate Value Ops — SSA-level struct manipulation
    // ═══════════════════════════════════════════════════════════════════════
    // Only valid for value-safe structs (no InlineArray/FlexArray fields).

    /// Construct a struct value from its fields.
    MkStruct {
        result: ValueId,
        struct_type: LLStruct,
        fields: Vec<ValueId>,
    },

    /// Extract a single field from a struct value.
    /// Result type: Int(n) for Int fields, Ptr for Ptr fields,
    ///              Struct(inner) for Inline(inner) fields.
    ExtractField {
        result: ValueId,
        value: ValueId,
        struct_type: LLStruct,
        field: usize,
    },

    /// Functional update: produce new struct with one field replaced.
    InsertField {
        result: ValueId,
        base: ValueId,
        struct_type: LLStruct,
        field: usize,
        value: ValueId,
    },

    // ═══════════════════════════════════════════════════════════════════════
    // Memory
    // ═══════════════════════════════════════════════════════════════════════

    /// Heap allocate. For fixed structs: sizeof(struct). For structs with
    /// FlexArray tail: sizeof(struct) + flex_count * sizeof(flex_elem).
    /// Memory is uninitialized. Caller must init all fields (including RC).
    HeapAlloc {
        result: ValueId,       // Ptr
        struct_type: LLStruct,
        flex_count: Option<ValueId>,  // None = no flex tail, Some(n) = n flex elements
    },

    /// Free heap memory. Pointer must have been returned by HeapAlloc.
    Free { ptr: ValueId },

    /// Load a value from memory. The `ty` field specifies what to load:
    /// Int(n) loads n/8 bytes, Ptr loads a pointer, Struct loads sizeof(struct) bytes.
    Load {
        result: ValueId,
        ptr: ValueId,
        ty: LLType,
    },

    /// Store a value to memory. Size determined by the value's type.
    Store {
        ptr: ValueId,
        value: ValueId,
    },

    /// Compute pointer to a field within a struct in memory.
    /// ptr must point to an instance of struct_type.
    /// Result: Ptr to the field (offset computed by target from struct layout).
    StructFieldPtr {
        result: ValueId,       // Ptr
        ptr: ValueId,          // Ptr to struct
        struct_type: LLStruct,
        field: usize,
    },

    /// Compute pointer to an array element. ptr must point to a contiguous
    /// array of elem_type structs. Result: ptr + index * sizeof(elem_type).
    ArrayElemPtr {
        result: ValueId,       // Ptr
        ptr: ValueId,          // Ptr to array start
        elem_type: LLStruct,
        index: ValueId,        // Int
    },

    /// Copy memory. Copies `count` instances of struct_type from src to dst.
    /// count=None means copy 1 instance.
    Memcpy {
        dst: ValueId,          // Ptr
        src: ValueId,          // Ptr
        struct_type: LLStruct,
        count: Option<ValueId>,
    },

    // ═══════════════════════════════════════════════════════════════════════
    // Selection
    // ═══════════════════════════════════════════════════════════════════════

    /// Conditional select. cond: Int(1). if_t and if_f must have same type.
    /// Works on any type including Struct (target emits per-field cmov).
    Select {
        result: ValueId,
        cond: ValueId,
        if_t: ValueId,
        if_f: ValueId,
    },

    // ═══════════════════════════════════════════════════════════════════════
    // Calls — static only, multi-return
    // ═══════════════════════════════════════════════════════════════════════

    /// Static function call. Dynamic dispatch is resolved before LLSSA
    /// (by defunctionalization). Multiple return values supported.
    Call {
        results: Vec<ValueId>,
        func: FunctionId,
        args: Vec<ValueId>,
    },

    // ═══════════════════════════════════════════════════════════════════════
    // Globals
    // ═══════════════════════════════════════════════════════════════════════

    /// Get pointer to a global variable slot. Target maps this to a global
    /// variable (LLVM) or VM globals array offset (VM).
    GlobalAddr {
        result: ValueId,       // Ptr
        global_id: usize,
    },

    // ═══════════════════════════════════════════════════════════════════════
    // Trap
    // ═══════════════════════════════════════════════════════════════════════

    /// Abort execution. Used for assertion failures.
    /// This is a terminator (no successors).
    Trap,
}
```

## HLSSA-to-LLSSA Lowering

### Type Mapping

| HLSSA Type | LLSSA Type | Notes |
|---|---|---|
| `Field` | `Struct(FieldElem)` | 4 x Int(64) by value |
| `U(n)` | `Int(n)` | Direct mapping |
| `Array(T, N)` | `Ptr` | Ptr to `RcArrayN_T` struct |
| `Slice(T)` | `Ptr` | Ptr to `DynSlice_T` struct |
| `Ref(T)` | `Ptr` | Ptr to heap-allocated T |
| `Tuple(...)` | `Struct(...)` or `Ptr` | Small tuples: value. Heap/RC'd: Ptr to struct. |
| `WitnessOf(T)` | same as `T` | Absent in witgen path (stripped). **TODO**: AD path still has WitnessOf — lowering should ICE if encountered until AD design is done. |
| `Function` | never reaches LLSSA | Eliminated by defunctionalization |

### Operation Lowering

#### Constants

```
HLSSA:  %x = Const(42, Field)
LLSSA:  %limbs = MkStruct(FieldElem, [IntConst(42), IntConst(0), IntConst(0), IntConst(0)])
        %x     = FieldFromLimbs(%limbs)
```

#### Arithmetic

```
HLSSA:  %c = BinaryArithOp(Add, %a, %b)   // a, b : Field
LLSSA:  %c = FieldArith(Add, %a, %b)      // Struct(FieldElem) values

HLSSA:  %c = BinaryArithOp(Add, %a, %b)   // a, b : U(32)
LLSSA:  %c = IntArith(Add, %a, %b)        // Int(32) values
```

#### Comparisons

```
HLSSA:  %r = Cmp(Eq, %a, %b)    // a, b : Field
LLSSA:  %r = FieldEq(%a, %b)    // r : Int(1)

HLSSA:  %r = Cmp(Lt, %a, %b)    // a, b : U(32)
LLSSA:  %r = IntCmp(ULt, %a, %b)   // r : Int(1)
```

#### Cast

```
HLSSA:  %y = Cast(%x, target=U(32))   // x : Field
LLSSA:  %limbs = FieldToLimbs(%x)          // Struct(FieldElem) — plain limbs
        %lo    = ExtractField(%limbs, FieldElem, 0)  // Int(64)
        %y     = Truncate(%lo, 32)         // Int(32)

HLSSA:  %y = Cast(%x, target=Field)    // x : U(32)
LLSSA:  %ext   = ZExt(%x, 64)                        // Int(64)
        %limbs = MkStruct(FieldElem, [%ext, IntConst(0), IntConst(0), IntConst(0)])
        %y     = FieldFromLimbs(%limbs)               // Struct(FieldElem)
```

#### Tuple Construction (small, stack-resident)

```
HLSSA:  %t = MkTuple([%f, %x], [Field, U(32)])

LLSSA:  %t = MkStruct(Tuple_F_U32, [%f, %x])
        // %f : Struct(FieldElem), %x : Int(32)
        // %t : Struct(Tuple_F_U32)
```

#### Tuple Projection

```
HLSSA:  %f = TupleProj(%t, Static(0))   // t : Tuple(Field, U(32))

LLSSA:  %f = ExtractField(%t, Tuple_F_U32, 0)
        // %f : Struct(FieldElem)
```

#### Array Literal

```
HLSSA:  %arr = MkSeq([%a, %b, %c], Array(3), Field)

LLSSA:
  // RcArray3Field = { Inline(RcHeader), InlineArray(FieldElem, 3) }
  %arr     = HeapAlloc(RcArray3Field)
  %rc_ptr  = StructFieldPtr(%arr, RcArray3Field, 0)    // -> Ptr to RcHeader
  %rc_word = StructFieldPtr(%rc_ptr, RcHeader, 0)      // -> Ptr to Int(64)
  Store(%rc_word, IntConst(1))

  %data    = StructFieldPtr(%arr, RcArray3Field, 1)    // -> Ptr to InlineArray start
  %e0      = ArrayElemPtr(%data, IntConst(0), FieldElem)
  Store(%e0, %a)                                        // stores 4 words
  %e1      = ArrayElemPtr(%data, IntConst(1), FieldElem)
  Store(%e1, %b)
  %e2      = ArrayElemPtr(%data, IntConst(2), FieldElem)
  Store(%e2, %c)
```

#### Array Get

```
HLSSA:  %val = ArrayGet(%arr, %idx)   // arr : Array(Field, 3)

LLSSA:
  %data     = StructFieldPtr(%arr, RcArray3Field, 1)    // InlineArray start
  %elem_ptr = ArrayElemPtr(%data, %idx, FieldElem)
  %val      = Load(%elem_ptr, Struct(FieldElem))         // loads 4 words
```

#### Array Set (Copy-on-Write)

```
HLSSA:  %new = ArraySet(%arr, %idx, %val)

LLSSA:
  // Check RC
  %hdr     = StructFieldPtr(%arr, DynSlice_Field, 0)
  %rc_ptr  = StructFieldPtr(%hdr, SliceHeader, 0)
  %rc      = Load(%rc_ptr, Int(64))
  %unique  = IntCmp(Eq, %rc, IntConst(1))
  JmpIf(%unique, mutate_blk, copy_blk)

mutate_blk:
  // Mutate in place
  %data    = StructFieldPtr(%arr, DynSlice_Field, 1)
  %slot    = ArrayElemPtr(%data, %idx, FieldElem)
  Store(%slot, %val)
  Jmp(merge, %arr)

copy_blk:
  // Decrement old RC
  %new_rc  = IntArith(Sub, %rc, IntConst(1))
  Store(%rc_ptr, %new_rc)
  // Allocate new array
  %len_ptr = StructFieldPtr(%hdr, SliceHeader, 1)
  %len     = Load(%len_ptr, Int(32))
  %new_arr = HeapAlloc(DynSlice_Field, flex_count = %len)
  // Init new header
  %new_hdr     = StructFieldPtr(%new_arr, DynSlice_Field, 0)
  %new_rc_ptr  = StructFieldPtr(%new_hdr, SliceHeader, 0)
  Store(%new_rc_ptr, IntConst(1))
  %new_len_ptr = StructFieldPtr(%new_hdr, SliceHeader, 1)
  Store(%new_len_ptr, %len)
  // Copy all data
  %old_data = StructFieldPtr(%arr, DynSlice_Field, 1)
  %new_data = StructFieldPtr(%new_arr, DynSlice_Field, 1)
  %len64    = ZExt(%len, 64)
  Memcpy(%new_data, %old_data, FieldElem, count = %len64)
  // Overwrite element at index
  %new_slot = ArrayElemPtr(%new_data, %idx, FieldElem)
  Store(%new_slot, %val)
  Jmp(merge, %new_arr)

merge(%result: Ptr):
  ...
```

#### RC Bump

```
HLSSA:  MemOp::Bump(2, %arr)

LLSSA:
  %hdr    = StructFieldPtr(%arr, DynSlice_Field, 0)
  %rc_ptr = StructFieldPtr(%hdr, SliceHeader, 0)
  %rc     = Load(%rc_ptr, Int(64))
  %new_rc = IntArith(Add, %rc, IntConst(2))
  Store(%rc_ptr, %new_rc)
```

#### RC Drop (generated type-specific function)

```
HLSSA:  MemOp::Drop(%arr)    // arr : Slice<Field>

LLSSA:  Call(drop_DynSlice_Field, [%arr])
```

Generated `drop_DynSlice_Field`:

```
fn drop_DynSlice_Field(%ptr: Ptr) -> ():
entry:
  %hdr     = StructFieldPtr(%ptr, DynSlice_Field, 0)
  %rc_ptr  = StructFieldPtr(%hdr, SliceHeader, 0)
  %rc      = Load(%rc_ptr, Int(64))
  %new_rc  = IntArith(Sub, %rc, IntConst(1))
  Store(%rc_ptr, %new_rc)
  %dead    = IntCmp(Eq, %new_rc, IntConst(0))
  JmpIf(%dead, free_blk, done)

free_blk:
  // Field elements contain no pointers, so no recursive drops needed.
  // For types with Ptr fields, generate a loop here:
  //   %len = Load(StructFieldPtr(%hdr, SliceHeader, 1), Int(32))
  //   for i in 0..len: drop_ElemType(ArrayElemPtr(data, i, ElemType))
  Free(%ptr)
  Jmp(done)

done:
  Return()
```

#### Alloc (mutable local)

```
HLSSA:  %ref = Alloc(elem_type: Field)

LLSSA:  %ref = HeapAlloc(RcFieldRef)    // RcFieldRef = { Inline(RcHeader), Inline(FieldElem) }
        // init RC, then Store/Load through %ref
```

#### Load / Store (through references)

```
HLSSA:  Store(%ref, %val)          // ref : Ref(Field), val : Field
LLSSA:  Store(%ref, %val)          // ref : Ptr, val : Struct(FieldElem)

HLSSA:  %val = Load(%ref)          // ref : Ref(Field)
LLSSA:  %val = Load(%ref, Struct(FieldElem))
```

#### Select

```
HLSSA:  %r = Select(%cond, %a, %b)   // a, b : Field

LLSSA:  %r = Select(%cond, %a, %b)
        // cond : Int(1), a/b/r : Struct(FieldElem)
        // Target emits 4 conditional moves (one per limb)
```

#### AssertEq

```
HLSSA:  AssertEq(%a, %b)    // a, b : Field

LLSSA:
  %eq = FieldEq(%a, %b)
  JmpIf(%eq, ok, fail)
fail:
  Trap
ok:
  ...
```

#### Globals

```
HLSSA:  %val = ReadGlobal(offset=3, result_type=Field)

LLSSA:
  %gptr = GlobalAddr(global_id=3)
  %val  = Load(%gptr, Struct(FieldElem))
```

#### Builtin Operations

`ToBits`, `ToRadix`, `Rangecheck`, `Lookup` — lowered to calls to generated LLSSA helper
functions. These are complex enough that they deserve their own generated routines rather than
being inlined at every use site.

## Target Codegen Mapping

### LLVM

| LLSSA | LLVM IR |
|---|---|
| `Int(1)` | `i1` |
| `Int(8)` | `i8` |
| `Int(32)` | `i32` |
| `Int(64)` | `i64` |
| `Ptr` | `ptr` (opaque pointer) |
| `Struct(FieldElem)` | `[4 x i64]` or `{ i64, i64, i64, i64 }` |
| `MkStruct` | aggregate literal / `insertvalue` chain |
| `ExtractField` | `extractvalue` |
| `InsertField` | `insertvalue` |
| `StructFieldPtr` | `getelementptr %struct_type, ptr %p, i32 0, i32 field` |
| `ArrayElemPtr` | `getelementptr %elem_type, ptr %p, i64 %index` |
| `HeapAlloc` | `call @malloc(size)` — size computed from LLVM's `sizeof` |
| `Free` | `call @free(ptr)` |
| `FieldArith` | `call @bn254_{add,sub,mul,div}(...)` |
| `Select` | `select` |
| `Trap` | `call @llvm.trap()` + `unreachable` |

### WASM (via LLVM)

Same as LLVM, but `Int(1)` and `Int(8)` promote to `i32`. Pointer type becomes `i32` on wasm32.
LLVM handles this automatically via the target triple.

### VM (future migration)

| LLSSA | VM Bytecode |
|---|---|
| `Int(n)` (n <= 64) | 1 frame slot (u64) |
| `Ptr` | 1 frame slot (host pointer as u64) |
| `Struct(FieldElem)` | 4 frame slots (LIMBS) |
| `StructFieldPtr` | `ptr + precomputed_offset` |
| `HeapAlloc` | `call vm.alloc(layout)` |
| `Free` | `call vm.dealloc(ptr)` |
| `FieldArith` | `field_{add,sub,mul,div}` bytecode op |

## Open Questions / Future Work

- **AD pipeline**: Deferred. AD values (ADConst, ADWitness, ADSum, ADMulConst) and derivative
  operations (BumpD, NextDCoeff) will need their own LLSSA representation when we get to it.
  The key challenge is that AD currently uses backpropagation during RC drop — in LLSSA this
  would become explicit code in the generated drop functions.

- **R1CS path**: Currently LLSSA targets the witgen (execution) path. The R1CS constraint
  generation path stays on HLSSA for now. Long-term, constraint generation might also go
  through LLSSA or a parallel lowering.

- **Optimization passes on LLSSA**: Once LLSSA exists, we can run target-independent
  optimizations: dead code elimination, common subexpression elimination on pointer arithmetic,
  etc. These are simpler than HLSSA-level optimizations because the
  IR is lower-level and more uniform.

- **Slice capacity vs length**: Currently slices store `len` (number of elements). If we want
  amortized O(1) push, we may need separate `len` and `capacity` fields. For now, every push
  reallocates (matching current HLSSA semantics).
