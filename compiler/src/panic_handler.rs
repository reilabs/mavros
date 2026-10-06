//! A panic handler for internal compiler errors, modelled on the one in `rustc`.
//!
//! This hook keeps the panic message, the backtrace if asked, and appends a note saying that this
//! is our bug along with where to report it.
//!
//! [`ice_usr!`](crate::ice_usr) is the exception. It prints [`USER_ERROR_PREFIX`] followed by the
//! diagnostic alone.

use std::{
    any::Any,
    io::{self, Write},
    panic,
};

const BUG_REPORT_URL: &str = "https://github.com/reilabs/mavros/issues/new";

pub const USER_ERROR_PREFIX: &str = "Unhandled error from user input";

pub fn install() {
    let default_hook = panic::take_hook();
    panic::set_hook(Box::new(move |info| {
        if let Some(diagnostic) = user_error(info.payload()) {
            let _ = writeln!(io::stderr(), "{diagnostic}");
        } else {
            default_hook(info);
            let _ = writeln!(io::stderr(), "{}", ice_note());
        }
    }));
}

fn user_error(payload: &(dyn Any + Send)) -> Option<String> {
    let message = payload
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| payload.downcast_ref::<&str>().copied())?;

    if message == USER_ERROR_PREFIX {
        return Some(message.to_string());
    }
    message
        .strip_prefix(&format!("{USER_ERROR_PREFIX}: "))
        .map(str::to_string)
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
