TARGET=$(abspath target)
LLAMA2_RS=target/debug/llama2-rs

all: build

$(LLAMA2_RS):
	cargo build

build: $(LLAMA2_RS)

# run my code for comparison, just a few steps, more verbose
trace: build downloads/stories42M.bin
	RUST_LOG=trace $(LLAMA2_RS) downloads/stories42M.bin -t 0.8 -n 12 -i "One day, Lily met a Shoggoth" -s 100 2>$(TARGET)/rust-debug.out

# run my code for comparison, just a few steps
debug: build downloads/stories42M.bin
	RUST_LOG=debug $(LLAMA2_RS) downloads/stories42M.bin -t 0.8 -n 12 -i "One day, Lily met a Shoggoth" -s 100 2>$(TARGET)/rust-debug.out

# run my code
llama2-rs-run: build downloads/stories42M.bin
	RUST_LOG=info $(LLAMA2_RS) downloads/stories42M.bin -t 0.8 -n 256 -i "One day, Lily met a Shoggoth" -s 100 2>/dev/null

# run my code in release mode, just for speed comparison
llama2-rs-run-generate: downloads/stories42M.bin
	cargo build --release
	RUST_LOG=info target/release/llama2-rs downloads/stories42M.bin -t 0.8 -n 256 -i "One day, Lily met a Shoggoth" -s 100 2>/dev/null

# run original code for comparison, just a few steps
llama2-c-debug: downloads/stories42M.bin
	cd ../llama2.c \
	&& make \
	&& ./run stories42M.bin -t 0.8 -i "One day, Lily met a Shoggoth" -s 100 -n 12 2>$(TARGET)/llama2-c-debug.out

# run original code for comparison
llama2-c-run: downloads/stories42M.bin
	mkdir -p target
	gcc -O3 -o target/llama2_c ../llama2.c/run.c -lm
	target/llama2_c downloads/stories42M.bin -z ../llama2.c/tokenizer.bin -t 0.8 -i "One day, Lily met a Shoggoth" -s 100 -n 256

# If you want to share downloaded projects with other local, projects create directory "~/Downloads/BIG" before running make here.
downloads:
	if [ -d $(HOME)/Downloads/BIG ]; then ln -s $(HOME)/Downloads/BIG downloads; fi
	mkdir -pv downloads

downloads/stories15M.bin: downloads
	mkdir -p data
	curl -L "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin" --output downloads/stories15M.bin

downloads/stories42M.bin: downloads
	mkdir -p data
	curl -L "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.bin" --output downloads/stories42M.bin

download-all: downloads/stories15M.bin downloads/stories42M.bin

