use mavros_compiler::{Project, driver::Driver};

#[test]
fn unit_only_slice_operands_lower_to_ssa() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    std::fs::create_dir(root.join("src")).unwrap();
    std::fs::write(
        root.join("Nargo.toml"),
        "[package]\nname = \"unit_slice_operands\"\ntype = \"bin\"\nauthors = []\n",
    )
    .unwrap();
    std::fs::write(
        root.join("src/main.nr"),
        r#"
        fn unit() {}

        fn main(index: u32) {
            let mut values = @[(), unit()];
            values[index] = unit();
            values = values.push_back(unit());
            values = values.push_front(());
            values = values.insert(index, unit());
            assert_eq(values.len(), 5);
            let repeated: [(); 2] = [unit(); 2];
            assert_eq(repeated[index], values[index]);
        }
        "#,
    )
    .unwrap();

    let mut driver = Driver::new(Project::new(root).unwrap(), false);
    // Exercise AST-to-SSA lowering independently of the later tuple-elision pass,
    // which does not yet preserve lengths for slices with no scalar fields.
    driver.run_noir_compiler().unwrap();
}
