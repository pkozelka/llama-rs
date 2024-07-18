TARGET=$(abspath target)
LLAMA2_RS=target/debug/llama2-rs

all: build

$(LLAMA2_RS):
	cargo build

build: $(LLAMA2_RS)

# run my code for comparison, just a few steps, more verbose
trace: build data/stories42M.bin
	RUST_LOG=trace $(LLAMA2_RS) data/stories42M.bin -t 0.8 -n 12 -i "One day, Lily met a Shoggoth" -s 100 2>$(TARGET)/rust-debug.out

# run my code for comparison, just a few steps
debug: build data/stories42M.bin
	RUST_LOG=debug $(LLAMA2_RS) data/stories42M.bin -t 0.8 -n 12 -i "One day, Lily met a Shoggoth" -s 100 2>$(TARGET)/rust-debug.out

# run my code
run: build data/stories42M.bin
	RUST_LOG=info $(LLAMA2_RS) data/stories42M.bin -t 0.8 -n 256 -i "One day, Lily met a Shoggoth" -s 100 2>/dev/null

# run original code for comparison, just a few steps
c-debug:
	cd ../llama2.c \
	&& make \
	&& ./run stories42M.bin -t 0.8 -i "One day, Lily met a Shoggoth" -s 100 -n 12 2>$(TARGET)/c-debug.out

# run original code for comparison
c-run:
	cd ../llama2.c \
	&& make \
	&& ./run stories42M.bin -t 0.8 -i "One day, Lily met a Shoggoth" -s 100 -n 256

data/stories15M.bin:
	mkdir -p data
	curl -L "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin" --output data/stories15M.bin

data/stories42M.bin:
	mkdir -p data
	curl -L "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.bin" --output data/stories42M.bin

download: data/stories15M.bin data/stories42M.bin