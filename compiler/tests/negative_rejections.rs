use mavros_compiler::{Project, driver::Driver};

#[test]
fn source_checks_survive_unused_results_and_respect_inactive_branches() {
    let cases = [
        ("let a = [1, 2, 3]; let _ = a[index()];", false),
        ("let a = [1, 2, 3].as_vector(); let _ = a[10];", false),
        ("let _: [bool; 1] = (2 as Field).to_be_bits();", false),
        ("let _: [bool; 1] = (2 as Field).to_le_bits();", false),
        ("let _: [u8; 1] = (256 as Field).to_be_bytes();", false),
        ("let _: [u8; 1] = (256 as Field).to_le_bytes();", false),
        ("let _: [u8; 1] = (255 as Field).to_be_bytes();", true),
        ("let _: [u8; 2] = (256 as Field).to_be_bytes();", true),
        ("let _: [u8; 2] = (256 as Field).to_le_bytes();", true),
        ("let _: [bool; 1] = (1 as Field).to_be_bits();", true),
        ("let _: [u8; 0] = (0 as Field).to_be_bytes();", true),
        (
            "if false { let a = [1, 2, 3].as_vector(); let _ = a[10]; }",
            true,
        ),
        (
            "if false { let _: [bool; 1] = (2 as Field).to_be_bits(); }",
            true,
        ),
    ];
    for (body, accepted) in cases {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("src")).unwrap();
        std::fs::write(
            dir.path().join("Nargo.toml"),
            "[package]\nname = 'negative_rejections'\ntype = 'bin'\nauthors = []\n",
        )
        .unwrap();
        std::fs::write(
            dir.path().join("src/main.nr"),
            format!("fn index() -> u32 {{ 10 }} fn main() {{ {body} }}"),
        )
        .unwrap();
        let mut driver = Driver::new(Project::new(dir.path().to_path_buf()).unwrap(), false);
        driver.run_noir_compiler().unwrap();
        let result = driver
            .make_struct_access_static()
            .and_then(|_| driver.monomorphize())
            .and_then(|_| driver.spill_witness())
            .and_then(|_| driver.generate_r1cs().map(|_| ()));
        assert_eq!(result.is_ok(), accepted, "{body}: {result:?}");
        if !accepted {
            assert!(
                matches!(
                    result,
                    Err(mavros_compiler::driver::Error::UnsatisfiableProgram(_))
                ),
                "{body}: {result:?}"
            );
        }
    }
}
