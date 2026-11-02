use std::rc::Rc;

use noirc_errors::Location;
use noirc_frontend::monomorphization::ast::{
    ArrayLiteral, Call, Ident, IdentId, Let, Literal, LocalId, Type as AstType,
};

use super::*;
use crate::compiler::{
    analysis::{flow_analysis::FlowAnalysis, types::Types},
    diagnostic::{Diagnostic, render_all},
    pass_manager::PassManager,
    passes::{defunctionalize::Defunctionalize, elide_tuples::ElideTuples},
    ssa::{
        Terminator,
        hlssa::{CallTarget, Constant, OpCode, Type},
    },
};

fn unit() -> Expression {
    Expression::Literal(Literal::Unit)
}

fn ident(definition: Definition, typ: AstType) -> Expression {
    Expression::Ident(Ident {
        location: None,
        definition,
        mutable: false,
        name: "value".into(),
        typ: Rc::new(typ),
        id: IdentId(0),
    })
}

fn function(id: u32, body: Expression, return_type: AstType) -> AstFunction {
    AstFunction {
        id: AstFuncId(id),
        name: format!("function_{id}"),
        parameters: vec![],
        body,
        return_type,
        return_visibility: Default::default(),
        unconstrained: false,
        inline_type: Default::default(),
        is_entry_point: id == 0,
        allow_constant_return: false,
    }
}

fn try_lower(functions: Vec<AstFunction>) -> Result<(HLSSA, bool), Vec<Diagnostic>> {
    HLSSA::from_program(&Program {
        functions,
        ..Default::default()
    })
}

fn lower(functions: Vec<AstFunction>) -> HLSSA {
    try_lower(functions).unwrap().0
}

fn assert_refused(functions: Vec<AstFunction>, message: &str) {
    let Err(diagnostics) = try_lower(functions) else {
        panic!("expected lowering to refuse the program");
    };
    let rendered = render_all(&diagnostics);
    assert!(
        rendered.contains(message),
        "missing {message:?}: {rendered}"
    );
}

#[test]
fn oracle_identifiers_are_refused_without_a_call() {
    assert_refused(
        vec![function(
            0,
            ident(
                Definition::Oracle {
                    name: "barnacle".into(),
                    pure: true,
                },
                AstType::Unit,
            ),
            AstType::Unit,
        )],
        "error: oracle functions are not supported by Mavros: `barnacle`",
    );
}

/// An oracle bound as a function value is refused, and lowering carries on past it, whether the
/// oracle is typed as one function or as its constrained/unconstrained pair.
#[test]
fn oracle_function_values_are_refused() {
    let oracle_function = |unconstrained| {
        AstType::Function(
            vec![AstType::Field],
            Rc::new(AstType::Field),
            Rc::new(AstType::Unit),
            unconstrained,
        )
    };
    for oracle_type in [
        oracle_function(true),
        AstType::Tuple(vec![oracle_function(false), oracle_function(true)]),
    ] {
        let binding = Expression::Let(Let {
            id: LocalId(0),
            mutable: false,
            name: "f".into(),
            expression: Box::new(ident(
                Definition::Oracle {
                    name: "barnacle".into(),
                    pure: true,
                },
                oracle_type,
            )),
        });
        assert_refused(
            vec![function(
                0,
                Expression::Block(vec![binding, unit()]),
                AstType::Unit,
            )],
            "error: oracle functions are not supported by Mavros: `barnacle`",
        );
    }
}

#[test]
fn empty_folded_functions_are_refused_without_source_locations() {
    let mut folded = function(0, unit(), AstType::Unit);
    folded.inline_type = InlineType::Fold;
    assert_refused(
        vec![folded],
        "error: #[fold] functions are not supported by Mavros",
    );
}

fn assert_return_shape(ssa: &HLSSA, expected: &[Type]) {
    let types = Types::new().run(ssa, &FlowAnalysis::run(ssa));
    for (fid, function) in ssa.iter_functions() {
        assert_eq!(function.get_returns(), expected);
        let mut returns = 0;
        for (_, block) in function.get_blocks() {
            if let Some(Terminator::Return(values)) = block.get_terminator() {
                returns += 1;
                let actual: Vec<_> = values
                    .iter()
                    .map(|value| types.get_function(*fid).get_value_type(*value).clone())
                    .collect();
                assert_eq!(actual, expected, "return values must match the signature");
            }
        }
        assert!(returns > 0);
    }
}

#[test]
fn unit_bodies_and_empty_structs_have_one_typed_result_until_tuple_elision() {
    let binding = |mutable| {
        Expression::Let(Let {
            id: LocalId(0),
            mutable,
            name: "value".into(),
            expression: Box::new(unit()),
        })
    };
    let read = || ident(Definition::Local(LocalId(0)), AstType::Unit);
    let cases = [
        ("literal", unit(), AstType::Unit, vec![]),
        (
            "empty block",
            Expression::Block(vec![]),
            AstType::Unit,
            vec![],
        ),
        ("binding statement", binding(false), AstType::Unit, vec![]),
        (
            "immutable read",
            Expression::Block(vec![binding(false), read()]),
            AstType::Unit,
            vec![],
        ),
        (
            "mutable read",
            Expression::Block(vec![binding(true), read()]),
            AstType::Unit,
            vec![],
        ),
        (
            "projection",
            Expression::ExtractTupleField(Box::new(Expression::Tuple(vec![unit()])), 0),
            AstType::Unit,
            vec![],
        ),
        (
            "empty struct",
            Expression::Tuple(vec![]),
            AstType::Tuple(vec![]),
            vec![],
        ),
        (
            "scalar",
            Expression::Literal(Literal::Bool(true)),
            AstType::Bool,
            vec![Type::bool()],
        ),
    ];
    for (name, body, return_type, after) in cases {
        let expected = TypeConverter::new().convert_type(&return_type);
        let mut ssa = lower(vec![function(0, body, return_type)]);
        assert_return_shape(&ssa, &[expected.clone()]);
        PassManager::new(name.into(), false, vec![Box::new(ElideTuples::new())]).run(&mut ssa);
        assert_return_shape(&ssa, &after);
    }
}

fn unit_call(indirect: bool) -> Expression {
    let target = ident(
        Definition::Function(AstFuncId(1)),
        AstType::Function(
            vec![AstType::Unit],
            Rc::new(AstType::Unit),
            Rc::new(AstType::Unit),
            false,
        ),
    );
    Expression::Call(Call {
        func: Box::new(if indirect {
            Expression::Block(vec![target])
        } else {
            target
        }),
        arguments: vec![unit()],
        return_type: AstType::Unit,
        location: Location::dummy(),
    })
}

fn identity() -> AstFunction {
    let mut f = function(
        1,
        ident(Definition::Local(LocalId(0)), AstType::Unit),
        AstType::Unit,
    );
    f.parameters.push((
        LocalId(0),
        false,
        "value".into(),
        Rc::new(AstType::Unit),
        Default::default(),
    ));
    f
}

#[test]
fn nested_statement_aggregate_types_match_their_values() {
    use noirc_frontend::token::FmtStrFragment;

    let nested = || {
        Expression::Tuple(vec![
            Expression::Literal(Literal::Bool(false)),
            Expression::Block(vec![Expression::Semi(Box::new(unit()))]),
        ])
    };
    let nested_type = AstType::Tuple(vec![AstType::Bool, AstType::Unit]);
    let cases = [
        (nested(), nested_type.clone()),
        (Expression::Block(vec![nested()]), nested_type.clone()),
        (
            Expression::ExtractTupleField(Box::new(Expression::Tuple(vec![nested()])), 0),
            nested_type.clone(),
        ),
        (
            Expression::Literal(Literal::FmtStr(
                vec![FmtStrFragment::String("x".into())],
                1,
                Box::new(Expression::Tuple(vec![nested()])),
            )),
            AstType::FmtString(1, Rc::new(AstType::Tuple(vec![nested_type]))),
        ),
    ];
    for (value, typ) in cases {
        let body = Expression::Tuple(vec![value, Expression::Literal(Literal::Bool(true))]);
        let typ = AstType::Tuple(vec![typ, AstType::Bool]);
        let ssa = lower(vec![function(0, body, typ.clone())]);
        assert_return_shape(&ssa, &[TypeConverter::new().convert_type(&typ)]);
        let types = Types::new().run(&ssa, &FlowAnalysis::run(&ssa));
        for (fid, f) in ssa.iter_functions() {
            let fti = types.get_function(*fid);
            for (_, block) in f.get_blocks() {
                for op in block.get_instructions() {
                    if let OpCode::MkTuple {
                        elems,
                        element_types,
                        ..
                    } = op
                    {
                        for (value, declared) in elems.iter().zip(element_types) {
                            assert_eq!(fti.get_value_type(*value), declared);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn direct_and_indirect_unit_calls_match_their_signatures() {
    for indirect in [false, true] {
        let mut ssa = lower(vec![
            function(0, unit_call(indirect), AstType::Unit),
            identity(),
        ]);
        let unit_type = Type::tuple_of(vec![]);
        assert_return_shape(&ssa, &[unit_type.clone()]);
        let types = Types::new().run(&ssa, &FlowAnalysis::run(&ssa));
        let main = ssa.get_unique_entrypoint_id();
        let calls: Vec<_> = ssa
            .get_function(main)
            .get_entry()
            .get_instructions()
            .filter_map(|op| {
                if let OpCode::Call {
                    results,
                    function,
                    args,
                    ..
                } = op
                {
                    Some((results, function, args))
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(calls.len(), 1);
        let (results, target, args) = calls[0];
        assert_eq!(results.len(), 1);
        assert_eq!(args.len(), 1);
        assert_eq!(types.get_function(main).get_value_type(args[0]), &unit_type);
        assert_eq!(matches!(target, CallTarget::Dynamic(_)), indirect);
        PassManager::new(
            "unit_calls".into(),
            false,
            vec![
                Box::new(Defunctionalize::new()),
                Box::new(ElideTuples::new()),
            ],
        )
        .run(&mut ssa);
        assert_return_shape(&ssa, &[]);
        for (_, function) in ssa.iter_functions() {
            for (_, block) in function.get_blocks() {
                for op in block.get_instructions() {
                    if let OpCode::Call {
                        results,
                        args,
                        function: CallTarget::Static(target),
                        ..
                    } = op
                    {
                        assert_eq!(results.len(), ssa.get_function(*target).get_returns().len());
                        assert_eq!(
                            args.len(),
                            ssa.get_function(*target).get_param_types().len()
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn unit_sequence_elements_are_typed_values_and_calls_are_not_duplicated() {
    let array = AstType::Array(2, Rc::new(AstType::Unit));
    let vector = AstType::Vector(Rc::new(AstType::Unit));
    let repeated = AstType::Array(3, Rc::new(AstType::Unit));
    let body = Expression::Tuple(vec![
        Expression::Literal(Literal::Array(ArrayLiteral {
            contents: vec![unit(), unit_call(false)],
            typ: array.clone(),
        })),
        Expression::Literal(Literal::Vector(ArrayLiteral {
            contents: vec![unit(), Expression::Block(vec![])],
            typ: vector.clone(),
        })),
        Expression::Literal(Literal::Repeated {
            element: Box::new(unit_call(false)),
            length: 3,
            is_vector: false,
            typ: repeated.clone(),
        }),
    ]);
    let ssa = lower(vec![
        function(0, body, AstType::Tuple(vec![array, vector, repeated])),
        identity(),
    ]);
    let main = ssa.get_unique_entrypoint_id();
    let types = Types::new().run(&ssa, &FlowAnalysis::run(&ssa));
    let unit_type = Type::tuple_of(vec![]);
    let fti = types.get_function(main);
    let mut calls = 0;
    let mut sequences = 0;
    for op in ssa.get_function(main).get_entry().get_instructions() {
        match op {
            OpCode::MkSeq {
                elems, elem_type, ..
            } => {
                sequences += 1;
                assert_eq!(elems.len(), 2);
                assert_eq!(elem_type, &unit_type);
                for elem in elems {
                    assert_eq!(fti.get_value_type(*elem), &unit_type);
                }
            }
            OpCode::MkRepeated {
                element,
                count,
                elem_type,
                ..
            } => {
                sequences += 1;
                assert_eq!(*count, 3);
                assert_eq!(elem_type, &unit_type);
                assert_eq!(fti.get_value_type(*element), &unit_type);
            }
            OpCode::Call { .. } => calls += 1,
            _ => {}
        }
    }
    assert_eq!(sequences, 3);
    assert_eq!(calls, 2, "each source call must be evaluated exactly once");
    let Some(Terminator::Return(values)) = ssa.get_function(main).get_entry().get_terminator()
    else {
        panic!("missing return")
    };
    assert_eq!(values.len(), 1);
    assert_eq!(
        fti.get_value_type(values[0]),
        &Type::tuple_of(vec![
            unit_type.clone().array_of(2),
            unit_type.clone().slice_of(),
            unit_type.array_of(3),
        ])
    );
}

/// The Noir `Field` literal `value`, which the frontend hands over as the signed integer written.
fn field_literal(value: i128) -> Expression {
    Expression::Literal(Literal::Integer(
        noir_bigint::BigInt::from(value),
        AstType::Field,
        Location::dummy(),
    ))
}

/// The constant `main` returns, lowered over `field`.
fn returned_field_constant(field: FieldConfig, body: Expression) -> crate::compiler::Field {
    let program = Program {
        functions: vec![function(0, body, AstType::Field)],
        ..Default::default()
    };
    let (ssa, _) = HLSSA::from_program_with_file_manager(&program, None, field).unwrap();
    assert_eq!(ssa.field(), field);
    let main = ssa.get_unique_entrypoint();
    let returned = main
        .get_blocks()
        .find_map(|(_, block)| match block.get_terminator() {
            Some(Terminator::Return(values)) => Some(values[0]),
            _ => None,
        })
        .expect("main returns");
    match ssa.get_const(returned).as_deref() {
        Some(Constant::Field(value)) => *value,
        other => panic!("main returns {other:?}, not a field constant"),
    }
}

#[test]
fn a_field_literal_below_the_goldilocks_modulus_keeps_its_integer() {
    let goldilocks = FieldConfig::goldilocks();
    let top = i128::from(mavros_artifacts::field::goldilocks::MODULUS - 1);
    for value in [0, 7, top] {
        assert_eq!(
            returned_field_constant(goldilocks, field_literal(value)),
            goldilocks.constant(value)
        );
    }
}

/// A negative literal is the negation of its magnitude in the configured field.
#[test]
fn a_negative_field_literal_negates_in_goldilocks() {
    let goldilocks = FieldConfig::goldilocks();
    assert_eq!(
        returned_field_constant(goldilocks, field_literal(-1)),
        -goldilocks.one()
    );
    let top = i128::from(mavros_artifacts::field::goldilocks::MODULUS - 1);
    assert_eq!(
        returned_field_constant(goldilocks, field_literal(-top)),
        goldilocks.one()
    );
}

/// Lowering `main` returning `body` over goldilocks is refused because a literal does not fit.
fn assert_goldilocks_literal_refused(body: Expression) {
    let program = Program {
        functions: vec![function(0, body, AstType::Field)],
        ..Default::default()
    };
    let Err(diagnostics) =
        HLSSA::from_program_with_file_manager(&program, None, FieldConfig::goldilocks())
    else {
        panic!("expected lowering to refuse the literal");
    };
    let rendered = render_all(&diagnostics);
    assert!(rendered.contains("does not fit the field"), "{rendered}");
}

#[test]
fn a_field_literal_at_the_goldilocks_modulus_is_refused() {
    let modulus = i128::from(mavros_artifacts::field::goldilocks::MODULUS);
    assert_goldilocks_literal_refused(field_literal(modulus));
}

#[test]
fn a_negative_field_literal_of_the_goldilocks_modulus_is_refused() {
    let modulus = i128::from(mavros_artifacts::field::goldilocks::MODULUS);
    assert_goldilocks_literal_refused(field_literal(-modulus));
}

#[test]
fn a_field_literal_keeps_its_bn254_value() {
    let bn254 = FieldConfig::bn254();
    assert_eq!(
        returned_field_constant(bn254, field_literal(-1)),
        -bn254.one()
    );
    assert_eq!(
        returned_field_constant(bn254, field_literal(i128::MAX)),
        bn254.constant(i128::MAX)
    );
}
