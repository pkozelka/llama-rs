use std::time::{SystemTime, UNIX_EPOCH};

pub fn safe_printf(piece: &str) {
    // piece might be a raw byte token, and we only want to print printable chars or whitespace
    // because some of the other bytes can be various control codes, backspace, etc.
    if piece.is_empty() { return; }
    if piece.len() == 1 {
        let byte_val = piece.as_bytes()[0];
        if !(byte_val.is_ascii_graphic() || byte_val.is_ascii_whitespace()) {
            return; // bad byte, don't print it
        }
    }
    println!("{}", piece);
}

pub fn time_in_ms() -> i64 {
    // return time in milliseconds, for benchmarking the model speed
    let time = SystemTime::now();
    let since_the_epoch = time.duration_since(UNIX_EPOCH).expect("Time went backwards");
    since_the_epoch.as_millis() as i64
}
