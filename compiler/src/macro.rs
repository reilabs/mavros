//! Internal compiler error reporting.

/// Panic with an internal compiler error.
#[macro_export]
macro_rules! ice {
    () => {
        ::std::panic!("ICE: internal compiler error")
    };
    ($($arg:tt)+) => {
        ::std::panic!("ICE: {}", ::std::format_args!($($arg)+))
    };
}

/// Panic with an internal compiler error on a code path that is supposed to be unreachable.
#[macro_export]
macro_rules! ice_unreachable {
    () => {
        ::std::panic!("ICE::Unreachable")
    };
    ($($arg:tt)+) => {
        ::std::panic!(
            "ICE::Unreachable: {}",
            ::std::format_args!($($arg)+)
        )
    };
}

/// Panic on an error that a user's program or environment can trigger.
#[macro_export]
macro_rules! ice_usr {
    () => {
        ::std::panic!("Unhandled error from user input")
    };
    ($($arg:tt)+) => {
        ::std::panic!(
            "Unhandled error from user input: {}",
            ::std::format_args!($($arg)+)
        )
    };
}
