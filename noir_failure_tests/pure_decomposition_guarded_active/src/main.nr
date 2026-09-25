// The fit check for an unused byte conversion must follow the runtime branch guard.
fn main(enabled: bool) {
    if enabled {
        let _: [u8; 1] = (256 as Field).to_be_bytes();
    }
}
