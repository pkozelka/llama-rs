TARGET=$(abspath target)
LLAMA_RS=target/debug/llama-rs
LLAMA_RS_RELEASE=target/release/llama-rs

all: build

$(LLAMA_RS): Cargo.toml $(shell find src -type f)
	cargo build

$(LLAMA_RS_RELEASE): Cargo.toml $(shell find src -type f)
	cargo build --release

build: $(LLAMA_RS)

# run my code for comparison, just a few steps, more verbose
llama2-rs-trace: $(LLAMA_RS) downloads/stories42M.bin
	RUST_LOG=trace $(LLAMA_RS) downloads/stories42M.bin -t 0.8 -n 12 -i "One day, Lily met a Shoggoth" -s 100 2>$(TARGET)/rust-debug.out

# run my code for comparison, just a few steps
llama2-rs-debug: $(LLAMA_RS) downloads/stories42M.bin
	RUST_LOG=debug $(LLAMA_RS) downloads/stories42M.bin -t 0.8 -n 12 -i "One day, Lily met a Shoggoth" -s 100 2>$(TARGET)/rust-debug.out

# run my code
llama2-rs-run: $(LLAMA_RS) downloads/stories42M.bin
	RUST_LOG=info $(LLAMA_RS) downloads/stories42M.bin -t 0.8 -n 256 -i "One day, Lily met a Shoggoth" -s 100 2>/dev/null

llama2-rs-test: $(LLAMA_RS_RELEASE) downloads/stories15M.bin
	$(LLAMA_RS_RELEASE) downloads/stories15M.bin -t 0.8 -n 256 -i "One day, Lily met a Shoggoth" -s 100 | tee target/llama2-rs-test.txt
	diff -u4 tests/llama2-rs-test.expected.txt target/llama2-rs-test.txt

# run my code in release mode, just for speed comparison
llama2-rs-run-generate: downloads/stories42M.bin
	cargo build --release
	RUST_LOG=info $(LLAMA_RS_RELEASE) downloads/stories42M.bin -t 0.8 -n 256 -i "One day, Lily met a Shoggoth" -s 100 2>/dev/null

# run original code for comparison, just a few steps
llama2-c-debug: downloads/stories42M.bin
	mkdir -p target
	gcc -O3 -o target/llama2c-debug-run c/llama2-debug-run.c -lm
	target/llama2c-debug-run downloads/stories42M.bin -z ../llama2.c/tokenizer.bin -t 0.8 -i "One day, Lily met a Shoggoth" -s 100 -n 12 2>$(TARGET)/llama2-c-debug.out

# run original code for comparison
llama2-c-run: downloads/stories42M.bin
	mkdir -p target
	gcc -Ofast -o target/llama2c-run c/llama2-run.c -lm
	target/llama2c-run downloads/stories42M.bin -z ../llama2.c/tokenizer.bin -t 0.8 -i "One day, Lily met a Shoggoth" -s 100 -n 256

# If you want to share downloaded projects with other local, projects create directory "~/Downloads/BIG" before running make here.
downloads:
	if [ -d $(HOME)/Downloads/BIG ]; then ln -s $(HOME)/Downloads/BIG downloads; fi
	mkdir -pv downloads

downloads/stories15M.bin: downloads
	curl -L "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin" --output downloads/stories15M.bin

downloads/stories42M.bin: downloads
	curl -L "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.bin" --output downloads/stories42M.bin

download-all: downloads/stories15M.bin downloads/stories42M.bin

test: llama2-rs-test
