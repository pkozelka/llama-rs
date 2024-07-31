/// Debug macro that prints the given expression to the console, using same syntax as `println!`.
/// The goal is to make it easy to turn on/off debug output in the code.
/// It will only print if the program is compiled with the `debug_assertions` flag set.

/// In debug mode, we want to see the output and compare it with the corresponding changes on llama2.c's side.
/// In release mode, we want to compare performance, and eliminate all that output.
#[macro_export]
macro_rules! dirty_dbg {
    ($($arg:tt)*) => {
        if cfg!(debug_assertions) {
            eprintln!($($arg)*);
        }
    }
}
