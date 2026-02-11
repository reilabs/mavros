# Monomorphization and Cast Insertion

## 1. Overview

The refactored monomorphization pass and a new **WitnessCastInsertion** pass replace the
current combination of Monomorphization + WitnessToRef:

1. **Monomorphization** (modified): specializes functions by their `WitnessOf` type
   signatures. Only does function cloning/specialization — no cast insertion.
2. **WitnessCastInsertion** (new, separate pass): inserts explicit `Cast` operations at
   boundaries where `X` is used as `WitnessOf(X)`, including deep casts (array conversion
   loops, tuple field-by-field conversion).

Separating these concerns keeps monomorphization simple and puts the complex cast emission
logic in a dedicated pass.

## 2. Type Signatures

### Current system
```rust
struct Signature {
    cfg_taint: Taint,
    param_taints: Vec<TaintType>,
    return_taints: Vec<TaintType>,
}
```

### New system
```rust
struct Signature {
    param_types: Vec<Type<Empty>>,    // Types with WitnessOf
    return_types: Vec<Type<Empty>>,   // Types with WitnessOf
    // No cfg_taint — that's handled by UntaintControlFlow
}
```

Signatures now use actual types (which may contain `WitnessOf`) instead of a separate
taint overlay. This is more direct and eliminates the need for `typify_taint()`.

## 3. Specialization Algorithm (Monomorphization)

The algorithm is essentially the same as the current queue-based approach:

```
function run(ssa, witness_analysis):
    // 1. Create main's specialized signature
    main_sig = compute_main_signature(ssa)
    request_specialization(ssa, main_id, main_sig)

    // 2. Process work queue
    while queue.not_empty():
        work_item = queue.pop()
        specialize_function(ssa, witness_analysis, work_item)

    // 3. Remove unspecialized originals
    remove_originals(ssa)
```

### Computing Main Signature

Main parameters come from `PrepareEntryPoint`, which wraps them with `WriteWitness`.
So main parameters have `WitnessOf(X)` types (from the inference). Returns are also
`WitnessOf` because they flow from witness computations.

```
main_signature = Signature {
    param_types: [WitnessOf(param1_type), WitnessOf(param2_type), ...],
    return_types: [inferred_return1, inferred_return2, ...],
}
```

### Specialization of a Function

For each function + signature pair:

1. Clone the function from the original
2. Use the constraint solver to resolve type variables:
   - Add equations: `inferred_param_type == specialized_param_type`
   - Add equations: `inferred_return_type == specialized_return_type`
   - Solve remaining constraints
3. Update the function's value types based on solved constraints
4. Scan call instructions → compute callee signatures → request specialization

**No cast insertion here** — that's handled by the separate WitnessCastInsertion pass.

## 4. WitnessCastInsertion Pass (New, Separate)

### Purpose

Runs after monomorphization. Walks through each specialized function and inserts explicit
`Cast` operations where the inferred types require subtyping coercions (X → WitnessOf(X)).

### When casts are needed

A cast from `X` to `WitnessOf(X)` is needed when a value of type `X` flows into a
context that expects `WitnessOf(X)`. This happens at:

1. **Function call arguments**: caller passes `Field`, callee expects `WitnessOf(Field)`
2. **Array write**: storing `Field` into `Array<WitnessOf(Field), n>`
3. **Tuple construction**: building `Tuple<WitnessOf(Field), ...>` from `Field` elements
4. **Return**: function declared to return `WitnessOf(X)` but body computes `X`
5. **Block parameter (phi)**: merge point typed `WitnessOf(X)` but one branch produces `X`

### Cast representation

We extend the existing `Cast` instruction with a `WitnessOf` target:

```rust
enum CastTarget {
    Field,
    U(usize),
    Nop,
    WitnessOf,   // NEW: cast X to WitnessOf(X)
}
```

The `Cast` with `CastTarget::WitnessOf` is a **type-level** operation during this phase:
- It marks the subtyping coercion explicitly in the IR
- It has no operational effect yet (the value is the same bits)
- Later passes (WitnessLowering / ExplicitWitness) will lower this to actual witness
  operations where needed

### Algorithm

```
function run(ssa, witness_analysis):
    for each function in ssa:
        let type_map = witness_analysis.get_value_types(function_id)
        insert_casts(function, type_map)

function insert_casts(function, type_map):
    for each block in function:
        for each instruction in block:
            match instruction:
                Call(f, args):
                    callee_params = get_param_types(f)
                    for (arg, expected) in zip(args, callee_params):
                        if needs_witness_cast(type_of(arg), expected):
                            new_arg = emit_cast(arg, expected)
                            replace arg with new_arg

                ArraySet(arr, idx, value):
                    elem_type = type_of(arr).element_type()
                    if needs_witness_cast(type_of(value), elem_type):
                        new_value = emit_cast(value, elem_type)
                        replace value with new_value

                MkTuple(elems, elem_types):
                    for (elem, expected) in zip(elems, elem_types):
                        if needs_witness_cast(type_of(elem), expected):
                            // cast each element

                // Similarly for Return, Jmp (block parameter passing)

        // Handle terminator:
        match terminator:
            Return(values):
                for (val, ret_type) in zip(values, function.return_types):
                    if needs_witness_cast(type_of(val), ret_type):
                        insert cast before return

            Jmp(target, args):
                for (arg, param_type) in zip(args, block_param_types(target)):
                    if needs_witness_cast(type_of(arg), param_type):
                        insert cast before jump
```

### needs_witness_cast function

```rust
fn needs_witness_cast(actual: &Type, expected: &Type) -> bool {
    match (&expected.expr, &actual.expr) {
        // Scalar: X vs WitnessOf(X)
        (TypeExpr::WitnessOf(inner), _) if !actual.is_witness_of() => true,

        // Array: Array<X,n> vs Array<WitnessOf(X),n> — element mismatch
        (TypeExpr::Array(exp_inner, _), TypeExpr::Array(act_inner, _)) =>
            needs_witness_cast(act_inner, exp_inner),

        // Tuple: field-by-field check
        (TypeExpr::Tuple(exp_fields), TypeExpr::Tuple(act_fields)) =>
            exp_fields.iter().zip(act_fields.iter())
                .any(|(e, a)| needs_witness_cast(a, e)),

        _ => false,
    }
}
```

### Deep cast for compound types

**Scalar cast:**
```
Cast { result: new_val, value: old_val, target: CastTarget::WitnessOf }
```

**Array cast** (`Array<Field, 5>` → `Array<WitnessOf(Field), 5>`):

Emits a conversion loop (same pattern as current `WitnessToRef::emit_array_conversion_loop`):
```
// Create loop: for i in 0..5
//   elem = array_get(src, i)
//   cast_elem = cast(elem, WitnessOf)
//   dst = array_set(dst, i, cast_elem)
// Result: dst array with converted elements
```

This requires creating new blocks (loop header, loop body, loop exit) and is the main
source of complexity in this pass.

**Tuple cast:**
```
// For Tuple<Field, U(32)> → Tuple<WitnessOf(Field), WitnessOf(U(32))>
field0 = tuple_proj(src, 0)          // Field
field0_cast = cast(field0, WitnessOf) // WitnessOf(Field)
field1 = tuple_proj(src, 1)          // U(32)
field1_cast = cast(field1, WitnessOf) // WitnessOf(U(32))
result = mk_tuple(field0_cast, field1_cast)
```

### WitnessOf(Array) — Not Handled

`WitnessOf(Array<X, n>)` (a witnessed array as a whole, distinct from an array of
witnessed elements) does not arise in practice in the current Noir compilation model.
Arrays are always built element-by-element, so the witness-ness ends up at the element
level (`Array<WitnessOf(X), n>`), not at the array level.

If a `WitnessOf(Array<...>)` type is encountered during cast insertion, we **panic** with
a clear error message. This can be revisited later if needed.

## 5. Interaction with Other Passes

### Before monomorphization: WitnessTypeInference
- Provides `WitnessType` for each value in each unspecialized function
- Provides function signatures (parameter and return types with WitnessOf)

### Between monomorphization and cast insertion
- SSA has WitnessOf types in value type map
- But no explicit cast operations yet
- Values may be used at boundaries with type mismatches (implicit subtyping)

### After cast insertion: UntaintControlFlow
- Receives `SSA<Empty>` with WitnessOf types and explicit casts
- Determines CFG taint, converts to `SSA<ConstantTaint>`
- Verifies no witness JmpIf

### Relationship to current WitnessToRef
- Current WitnessToRef does cast insertion + instruction lowering
- New system: cast insertion is in WitnessCastInsertion, instruction lowering stays in
  WitnessLowering (reworked WitnessToRef)

## 6. Example

Given:
```noir
fn add_one(x: Field) -> Field { x + 1 }
fn main(a: Field) { assert_eq(add_one(a), a + 1); }
```

After WriteWitness in PrepareEntryPoint:
```
main: a = WriteWitness(input)  // a: WitnessOf(Field)
      result = call add_one(a) // add_one expects Field, but a is WitnessOf(Field)
```

After type inference:
```
add_one called with WitnessOf(Field) → inferred: WitnessOf(Field) -> WitnessOf(Field)
```

After monomorphization (specialization only):
```
add_one_witness(x: WitnessOf(Field)) -> WitnessOf(Field):
    one = const(1)              // one: Field
    result = add(x, one)        // types don't match yet (WitnessOf(Field) + Field)
    return result
```

After WitnessCastInsertion:
```
add_one_witness(x: WitnessOf(Field)) -> WitnessOf(Field):
    one = const(1)                           // one: Field
    one_cast = cast(one, WitnessOf)          // one_cast: WitnessOf(Field)
    result = add(x, one_cast)                // WitnessOf(Field) + WitnessOf(Field) = WitnessOf(Field)
    return result

main:
    a = WriteWitness(input)                 // a: WitnessOf(Field)
    result = call add_one_witness(a)        // types match, no cast needed
    ...
```

## 7. Handling Recursion in Monomorphization

The current system already handles recursion in monomorphization via the `signature_map`:
if a (function_id, signature) pair has already been requested, it returns the existing
specialized function id. This prevents infinite specialization.

The same mechanism works for the new system. The Mycroft fixpoint has already converged on
the function signatures before monomorphization runs, so each recursive call has a known
signature.
