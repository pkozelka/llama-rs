all: build

build:
	cargo build

debug: build
	RUST_LOG=debug cargo run --package llama2-rs --bin llama2-rs -- ../llama2.c/stories42M.bin -t 0.8 -n 1 -i "One day, Lily met a Shoggoth" -s 100

run: build
	RUST_LOG=info cargo run --package llama2-rs --bin llama2-rs -- ../llama2.c/stories42M.bin -t 0.8 -n 15 -i "One day, Lily met a Shoggoth" -s 100

c-run:
	cd ../llama2.c \
	&& make \
	&& ./run stories42M.bin -t 0.8 -i "One day, Lily met a Shoggoth" -s 100 -n 1
