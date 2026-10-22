use mavros_compiler::{api, compiler::codegen::r1cs_compact};

mod common;

#[test]
fn protected_columns_survive_compaction() {
    for (source, removable) in [
        ("fn main() -> pub Field { 42 }", 0),
        ("fn main(unused: [Field; 3]) -> pub Field { 42 }", 0),
        (
            "fn main(unused: Field, y: pub Field, z: Field) -> pub Field { y * z }",
            1,
        ),
    ] {
        let dir = common::noir_project(source);
        let (driver, r1cs) = api::compile_to_r1cs(dir.path().to_path_buf(), false).unwrap();
        let stats = r1cs_compact::analyze(&r1cs, driver.protected_r1cs_cols());
        assert_eq!(stats.removable_cols, removable, "{source}: {stats:?}");
        assert_eq!(stats.removable_rows, removable, "{source}: {stats:?}");
    }
}
