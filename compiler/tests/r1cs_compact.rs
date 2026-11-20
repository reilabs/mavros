use mavros_artifacts::{Field, FieldConfig, Goldilocks, MavrosField};
use mavros_compiler::{
    api,
    compiler::{
        codegen::r1cs_compact,
        ssa::hlssa::{HLSSA, Type, builder::HLEmitter as _},
    },
};

mod common;

#[test]
fn protected_columns_survive_compaction() {
    for (source, removable) in [
        ("fn main() -> pub u8 { 42 }", 0),
        ("fn main(unused: [u8; 3]) -> pub u8 { 42 }", 0),
        (
            "fn main(unused: u8, y: pub u8, z: u8) -> pub u16 { (y as u16) * (z as u16) }",
            1,
        ),
    ] {
        let dir = common::noir_project(source);
        let (driver, r1cs) =
            api::compile_to_r1cs::<Field>(dir.path().to_path_buf(), false).unwrap();
        let stats = r1cs_compact::analyze(&r1cs, driver.protected_r1cs_cols());
        assert_eq!(stats.removable_cols, removable, "{source}: {stats:?}");
        assert_eq!(stats.removable_rows, removable, "{source}: {stats:?}");
    }
}

/// [`protected_columns_survive_compaction`] on hand-built SSA, whose `Field` parameters nothing
/// reads, not even a range check, so the private ones are elided.
#[test]
fn protected_columns_of_hand_built_entry_points_survive_compaction() {
    type Case = (&'static str, fn(FieldConfig) -> HLSSA, Vec<bool>, usize);
    fn check<F: MavrosField>() {
        let cases: [Case; 4] = [
            (
                "main() -> pub Field { 42 }",
                |field| {
                    common::hand_built_main(field, &[], Type::field(), |e, _| e.field_const(42u64))
                },
                vec![],
                0,
            ),
            (
                "main(unused: [Field; 3]) -> pub Field { 42 }",
                |field| {
                    common::hand_built_main(
                        field,
                        &[Type::field().array_of(3)],
                        Type::field(),
                        |e, _| e.field_const(42u64),
                    )
                },
                vec![false],
                0,
            ),
            (
                "main(unused: Field, y: pub Field, z: Field) -> pub Field { y * z }",
                |field| {
                    common::hand_built_main(
                        field,
                        &[Type::field(), Type::field(), Type::field()],
                        Type::field(),
                        |e, params| e.umul(params[1], params[2]),
                    )
                },
                vec![false, true, false],
                1,
            ),
            // The sum is a linear relation, which would solve for a parameter column that
            // compaction did not protect.
            (
                "main(y: pub Field, z: Field) -> pub Field { y + z }",
                |field| {
                    common::hand_built_main(
                        field,
                        &[Type::field(), Type::field()],
                        Type::field(),
                        |e, params| e.uadd(params[0], params[1]),
                    )
                },
                vec![true, false],
                0,
            ),
        ];
        for (program, build, param_is_public, removable) in cases {
            let seeded = common::compile_seeded::<F>(build(F::CONFIG), param_is_public);
            let stats = r1cs_compact::analyze(&seeded.r1cs, seeded.driver.protected_r1cs_cols());
            assert_eq!(
                stats.removable_cols,
                removable,
                "{:?} {program}: {stats:?}",
                F::ID
            );
            assert_eq!(
                stats.removable_rows,
                removable,
                "{:?} {program}: {stats:?}",
                F::ID
            );
        }
    }
    check::<Field>();
    check::<Goldilocks>();
}
