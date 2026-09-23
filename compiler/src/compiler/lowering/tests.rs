use std::rc::Rc;

use noirc_errors::Location;
use noirc_frontend::monomorphization::ast::{
    ArrayLiteral, Call, Ident, IdentId, Let, Literal, LocalId, Type as AstType,
};

use super::*;
use crate::compiler::{
    analysis::{flow_analysis::FlowAnalysis, types::Types},
    pass_manager::PassManager,
    passes::{defunctionalize::Defunctionalize, elide_tuples::ElideTuples},
    ssa::{
        Terminator,
        hlssa::{CallTarget, OpCode, Type},
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

fn lower(functions: Vec<AstFunction>) -> HLSSA {
    HLSSA::from_program(&Program {
        functions,
        ..Default::default()
    })
    .0
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
        ("literal", unit(), AstType::Unit),
        ("empty block", Expression::Block(vec![]), AstType::Unit),
        ("binding statement", binding(false), AstType::Unit),
        (
            "immutable read",
            Expression::Block(vec![binding(false), read()]),
            AstType::Unit,
        ),
        (
            "mutable read",
            Expression::Block(vec![binding(true), read()]),
            AstType::Unit,
        ),
        (
            "projection",
            Expression::ExtractTupleField(Box::new(Expression::Tuple(vec![unit()])), 0),
            AstType::Unit,
        ),
        (
            "empty struct",
            Expression::Tuple(vec![]),
            AstType::Tuple(vec![]),
        ),
        (
            "scalar",
            Expression::Literal(Literal::Bool(true)),
            AstType::Bool,
        ),
    ];
    for (name, body, return_type) in cases {
        let expected = TypeConverter::new().convert_type(&return_type);
        let mut ssa = lower(vec![function(0, body, return_type)]);
        assert_return_shape(&ssa, &[expected.clone()]);
        PassManager::new(name.into(), false, vec![Box::new(ElideTuples::new())]).run(&mut ssa);
        let after = if expected == Type::bool() {
            vec![expected]
        } else {
            vec![]
        };
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
