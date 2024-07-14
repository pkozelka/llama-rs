all: build

build:
	cargo build

debug: build
	RUST_LOG=debug cargo run --package llama2-rs --bin llama2-rs -- ../llama2.c/stories42M.bin -t 0.8 -n 256 -i "One day, Lily met a Shoggoth" -s 100

run: build
	RUST_LOG=info cargo run --package llama2-rs --bin llama2-rs -- ../llama2.c/stories42M.bin -t 0.8 -n 15 -i "One day, Lily met a Shoggoth" -s 100