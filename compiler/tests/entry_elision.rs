//! Entry-point parameters nothing reads, on hand-built SSA: a Noir `main` takes no `Field`, and
//! every integer it does take is read by its range check.

mod common;

use mavros_artifacts::{Bn254, Goldilocks, InputValueOrdered, MavrosField};
use mavros_compiler::{
    api,
    compiler::ssa::hlssa::{Type, builder::HLEmitter as _},
};

/// An unread private parameter gets no column, and the private parameters after it follow the
/// public one. An unread integer keeps its column, as its range check reads it.
#[test]
fn an_unread_private_parameter_gets_no_column() {
    fn check<F: MavrosField>() {
        // `main(unused: Field, unused_u8: u8, y: pub Field, z: Field) -> pub Field { y + z }`
        let ssa = common::hand_built_main(
            F::CONFIG,
            &[Type::field(), Type::int(8), Type::field(), Type::field()],
            Type::field(),
            |e, params| e.uadd(params[2], params[3]),
        );
        let mut built = common::compile_hand_built::<F>(ssa, vec![false, false, true, false]);
        let columns = built.driver.entry_columns().unwrap().clone();
        assert_eq!(
            columns.params,
            [None, Some(2), Some(1), Some(3)],
            "{:?}",
            F::ID
        );

        let params: Vec<InputValueOrdered<F>> = [5u64, 9, 6, 7]
            .map(|value| InputValueOrdered::Field(F::from(value)))
            .to_vec();
        let result = api::run_witgen_from_binary(&mut built.binary, &built.r1cs, &params, None)
            .unwrap_or_else(|error| panic!("{:?}: {error:?}", F::ID));
        assert!(api::check_witgen(&built.r1cs, &result), "{:?}", F::ID);
        let witness = &result.out_wit_pre_comm;
        assert_eq!(
            [witness[1], witness[2], witness[3]],
            [6u64, 9, 7].map(F::from),
            "{:?}",
            F::ID
        );
        assert_eq!(
            witness[columns.returns.unwrap()],
            F::from(13u64),
            "{:?}",
            F::ID
        );
    }
    check::<Bn254>();
    check::<Goldilocks>();
}
