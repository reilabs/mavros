struct Payload {
    scalar: Field,
    values: [Field; 2],
}

fn main(x: Field) {
    unsafe {
        assert(identity_scalar(x) == x);

        let values = identity_array([x, x]);
        assert(values[0] == x);
        assert(values[1] == x);

        let payload = identity_struct(Payload { scalar: x, values: [x, x] });
        assert(payload.scalar == x);
        assert(payload.values[0] == x);
        assert(payload.values[1] == x);
    }
}

unconstrained fn identity_scalar(value: Field) -> Field {
    std::hint::black_box(value)
}

unconstrained fn identity_array(value: [Field; 2]) -> [Field; 2] {
    std::hint::black_box(value)
}

unconstrained fn identity_struct(value: Payload) -> Payload {
    std::hint::black_box(value)
}
