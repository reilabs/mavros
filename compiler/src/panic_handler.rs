//! A panic handler for internal compiler errors, modelled on the one in `rustc`.
//!
//! This hook keeps the panic message, the backtrace if asked, and appends a note saying that this
//! is our bug along with where to report it.

use std::{
    io::{self, Write},
    panic,
};

const BUG_REPORT_URL: &str = "https://github.com/reilabs/mavros/issues/new";

pub fn install() {
    let default_hook = panic::take_hook();
    panic::set_hook(Box::new(move |info| {
        default_hook(info);
        let _ = writeln!(io::stderr(), "{}", ice_note());
    }));
}

fn ice_note() -> String {
    let args = std::env::args_os()
        .skip(1)
        .map(|arg| arg.to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join(" ");
    format!(
        "\nerror: the compiler unexpectedly panicked. this is a bug.\n\n\
         note: we would appreciate a bug report: {BUG_REPORT_URL}\n\
         note: mavros {version} running on {arch}-{os}\n\
         note: compiler flags: {args}",
        version = env!("CARGO_PKG_VERSION"),
        arch = std::env::consts::ARCH,
        os = std::env::consts::OS,
    )
}
