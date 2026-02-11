# WitnessOf Type System Design

## 1. TypeExpr Changes

### Eliminating the `V` annotation parameter

The generic annotation parameter `V` on `Type<V>` and `TypeExpr<V>` is **removed entirely**.
Types become plain structs with no generic parameter. The `CommutativeMonoid` trait, `Empty`,
and `ConstantTaint` types are all removed.

Before:
```rust
struct Type<V> { expr: TypeExpr<V>, annotation: V }
enum TypeExpr<V> { Field, U(usize), WitnessRef, Array(Box<Type<V>>, usize), ... }
```

After:
```rust
struct Type { expr: TypeExpr }
enum TypeExpr { Field, U(usize), WitnessOf(Box<Type>), Array(Box<Type>, usize), ... }
```

This is a large mechanical change (every `Type<V>`, `SSA<V>`, `Function<V>`, `Block<V>`,
`OpCode<V>` loses its type parameter) but simplifies the codebase significantly.

### New variant

```rust
enum TypeExpr {
    Field,
    U(usize),
    WitnessOf(Box<Type>),   // NEW: replaces WitnessRef
    Array(Box<Type>, usize),
    Slice(Box<Type>),
    Ref(Box<Type>),
    Tuple(Vec<Type>),
    Function,
    // WitnessRef — REMOVED
}
```

`WitnessOf(X)` wraps any type `X` and denotes a witness representation of `X`.
After monomorphization + cast insertion, `WitnessOf(X)` is a fully independent type from
`X` — it represents a distinct runtime kind (witness tape reference), exactly as `WitnessRef`
does today.

### Idempotency

`WitnessOf(WitnessOf(X)) = WitnessOf(X)` — enforced by a normalizing constructor:

```rust
impl Type {
    fn witness_of(inner: Type) -> Type {
        match inner.expr {
            TypeExpr::WitnessOf(_) => inner,  // idempotent
            _ => Type {
                expr: TypeExpr::WitnessOf(Box::new(inner)),
            },
        }
    }
}
```

### Removing WitnessRef

Every occurrence of `TypeExpr::WitnessRef` is replaced:
- `Type::witness_ref()` → `Type::witness_of(Type::field())`
- `Const::WitnessRef(fr)` → `Const::Witness(fr)` (renamed for clarity)

### Impact of removing `V`

All generic type parameters are removed:
- `SSA<V>` → `SSA`
- `Function<V>` → `Function`
- `Block<V>` → `Block`
- `OpCode<V>` → `OpCode`
- `Type<V>` → `Type`
- `TypeExpr<V>` → `TypeExpr`
- `prepare_rebuild::<ConstantTaint>()` → just `prepare_rebuild()`
- The `CommutativeMonoid` trait is deleted
- `get_arithmetic_result_type`, `combine_with_annotation`, etc. — simplified or removed

## 2. Subtyping Rules (Pre-Monomorphization Only)

**Important:** The subtyping relation `X <: WitnessOf(X)` only exists during type inference
and monomorphization. After `WitnessCastInsertion` inserts explicit `Cast(WitnessOf)` ops,
subtyping disappears. From that point on, `WitnessOf(X)` and `X` are fully independent types
with no implicit conversion — just like `Field` and `U(32)` are independent today.

`Cast(WitnessOf)` is a **real runtime conversion** (like current `PureToWitnessRef`), not
a type-level no-op.

The subtyping relation `<:` is defined as follows:

### Base rule
```
X <: WitnessOf(X)              for any type X
```

### Structural rules (covariance)
```
Array<X, n>   <: Array<Y, n>         if X <: Y
Slice<X>      <: Slice<Y>            if X <: Y
Tuple<X1..Xn> <: Tuple<Y1..Yn>       if Xi <: Yi for all i
WitnessOf(X)  <: WitnessOf(Y)        if X <: Y
```

### Ref rule (special — see section 4)
```
Ref<X> is NOT covariant in the usual sense.
Ref types are handled specially during type inference.
```

### Idempotency rule
```
WitnessOf(WitnessOf(X)) = WitnessOf(X)
```

### What is NOT a subtype
```
WitnessOf(Array<X, n>)  and  Array<WitnessOf(X), n>  are INCOMPARABLE.
```

These are distinct types:
- `Array<WitnessOf(Field), 5>` = a concrete array of 5 witness field elements
- `WitnessOf(Array<Field, 5>)` = a witness observation of an entire array of 5 field elements

Both are supertypes of `Array<Field, 5>`:
```
                    WitnessOf(Array<WitnessOf(Field), 5>)
                   /                                     \
  Array<WitnessOf(Field), 5>               WitnessOf(Array<Field, 5>)
                   \                                     /
                           Array<Field, 5>
```

Their join (least upper bound) is `WitnessOf(Array<WitnessOf(Field), 5>)`.

## 3. Join (Least Upper Bound) Operation

The join operation computes the least common supertype. It is needed for:
- Merge points (phi nodes) where two branches produce different types
- Mycroft fixpoint iteration when combining type estimates

### Definition

```
join(X, X) = X                                          (reflexive)
join(X, WitnessOf(Y)) = WitnessOf(join(X, Y))          (WitnessOf absorbs)
join(WitnessOf(X), Y) = WitnessOf(join(X, Y))          (symmetric)
join(WitnessOf(X), WitnessOf(Y)) = WitnessOf(join(X,Y)) (covariant)

join(Field, Field) = Field
join(U(n), U(n)) = U(n)
join(Array<X,n>, Array<Y,n>) = Array<join(X,Y), n>
join(Slice<X>, Slice<Y>) = Slice<join(X,Y)>
join(Tuple<Xs>, Tuple<Ys>) = Tuple<join(Xi,Yi) for each i>
join(Function, Function) = Function
```

### Properties

- **Commutative**: `join(X, Y) = join(Y, X)`
- **Associative**: `join(X, join(Y, Z)) = join(join(X, Y), Z)`
- **Idempotent**: `join(X, X) = X`
- **Monotone**: if `X <: Y` then `join(X, Z) <: join(Y, Z)`
- **Finite height**: for any given type shape, the lattice height is bounded by
  `2 * nesting_depth` (each structural level can independently be WitnessOf or not).

### Examples

```
join(Field, WitnessOf(Field)) = WitnessOf(Field)

join(Array<Field, 5>, Array<WitnessOf(Field), 5>)
  = Array<join(Field, WitnessOf(Field)), 5>
  = Array<WitnessOf(Field), 5>

join(Array<WitnessOf(Field), 5>, WitnessOf(Array<Field, 5>))
  = WitnessOf(join(Array<WitnessOf(Field), 5>, Array<Field, 5>))
  = WitnessOf(Array<join(WitnessOf(Field), Field), 5>)
  = WitnessOf(Array<WitnessOf(Field), 5>)
```

## 4. Ref Types — Special Handling

Mutable references (`Ref<X>`) cannot be covariant because writes through the reference could
violate the type invariant. Instead, Ref types get special treatment during type inference.

### Strategy

During type inference, each `Ref<X>` gets a **type variable** for its element type.
Constraints are generated:

1. **Store constraint**: if `store(ref, value)` where `ref: Ref<T>` and `value: V`,
   then `V <: T` (the stored value must be a subtype of the ref's element type).

2. **Load constraint**: if `result = load(ref)` where `ref: Ref<T>`,
   then `result: T` (the loaded value has the ref's element type).

3. **Allocation constraint**: `alloc` creates a Ref with a fresh type variable as element type.

The constraint solver ensures `T` is wide enough to accommodate all stores, while loads get
the widened type.

### Example

```
let ref = alloc<Field>();       // ref: Ref<α>  where α is fresh
store(ref, pure_field);         // constraint: Field <: α
store(ref, witness_field);      // constraint: WitnessOf(Field) <: α
let x = load(ref);              // x: α

// Solving: α = join(Field, WitnessOf(Field)) = WitnessOf(Field)
// So: ref: Ref<WitnessOf(Field)>, x: WitnessOf(Field)
```

## 5. WriteWitness Typing

```
WriteWitness: X -> WitnessOf(X)
```

Given a value of type `X`, `WriteWitness` produces a value of type `WitnessOf(X)`.
This is the **only** way to introduce `WitnessOf` types (besides propagation through
operations).

There is **no inverse** operation (`WitnessOf(X) -> X`). Witness values can only be
"consumed" by:
- Constraint operations (`Constrain`, `AssertEq`)
- Being used as operands in arithmetic (which produces `WitnessOf` results)
- Implicit subtyping coercion (inserting a `Cast` during monomorphization)

## 6. Operation Typing Rules

For binary arithmetic/comparison operations:

```
op(X, Y) : join(X, Y)
```

Specifically:
```
add(Field, Field) : Field
add(Field, WitnessOf(Field)) : WitnessOf(Field)
add(WitnessOf(Field), WitnessOf(Field)) : WitnessOf(Field)

eq(Field, WitnessOf(Field)) : WitnessOf(U(1))
```

For array operations:
```
array_get(Array<X, n>, idx: Y) : join(X, scalar_part(Y))
  — where scalar_part(Y) propagates witness-ness from the index

array_set(Array<X, n>, idx: Y, val: Z) : Array<join(X, Z, scalar_part(Y)), n>
  — the result array element type widens to accommodate the new value
```

For tuple operations:
```
mk_tuple(x1: T1, ..., xn: Tn) : Tuple<T1, ..., Tn>
tuple_proj(t: Tuple<T1,...,Tn>, i) : Ti
```

For select (ternary):
```
select(cond: C, if_t: T, if_f: F) : join(C_scalar, T, F)
  — condition's witness-ness propagates to top level of result
  — may produce WitnessOf(Array<...>) — panics downstream (acceptable for now)
```

## 7. Const Values

```rust
enum Const {
    U(usize, u128),
    Field(Fr),
    Witness(Fr),       // renamed from WitnessRef — represents a witness value at runtime
    FnPtr(FunctionId),
}
```

All `Const` values are pure (not `WitnessOf`) at the type level. `Const::Witness` is
only produced after lowering and represents a VM-level witness reference.

## 8. Display / Debug Format

Types are displayed as:
```
Field                         — pure field
WitnessOf(Field)              — witness field
Array<WitnessOf(Field), 5>    — array of 5 witness fields
WitnessOf(Array<Field, 5>)    — witness array of 5 pure fields
Tuple<Field, WitnessOf(U(32))> — tuple with pure field and witness u32
```
