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
        ::std::panic!("ICE: entered unreachable code")
    };
    ($($arg:tt)+) => {
        ::std::panic!(
            "ICE: entered unreachable code: {}",
            ::std::format_args!($($arg)+)
        )
    };
}
