# LLAMA2 - Rewrite from C to Rust

This is a naive rewrite of [A.Karpathy's llama2.c](https://github.com/karpathy/llama2.c/blob/master/run.c) to Rust.
I can't express enough gratitude to Andrey for sharing this code. BTW he also has couple of great videos on Youtube, be sure to check them out.

The primary goal of this work is for me to get better understanding of the algorithms used, and have a starting point for getting LLM knowledge in general.

As a Rust enthusiast, rewriting something to Rust makes a lot sense for me; not necessarily for anyone else.
However, keep in mind that it is just a plain rewrite; not an example of best code practices in Rust. Actually quite the opposite; translating C's pointer arithmetics into Rust comes with a performance penalty. 
Written from scratch, code is typically designed very differently in Rust than in any other languages.

I hope to be able to refactor this project into a beauty ... soon.

Feel free to read, fork, send PR or issues and discuss anything relevant.

## Usage

The best start is to just execute `make run`.

and it will download data, compile the project, and run it with test parameters.
The output should be exactly this:

```
One day, Lily met a Shoggoth. He was very curious and wanted to explore the world. Lily said to him, "Would you like to come with me and see the world?"
The Shoggoth was very excited and replied, "Yes! I want to see the world!"
So, Lily and Shoggle set off on an adventure. They walked through the forest, over the mountains, and  across the rivers. They were amazed by all the sights and sounds they saw.
Soon, they came to a clearing. In the middle of the clearing, there was a big, dark tree. "What is this place?" asked Lily.
Shoggle looked around and said, "I don't know, but let's find out!"
They climbed the tree and discovered a magical cave at the top. Inside the cave, Lily and Shoggle discovered a whole world of amazing things. They had a wonderful adventure and were so happy that they were able to explore the world together.
```

To achieve reproducible results, a RNG seed is set to `100`.

All the commandline arguments are compatible with the original C version so you can try to compare with the C code results.

You will definitely notice that the speed is much worse than the original C.

There reasons include:
- we are running the debug variant, with nearly no optimizations and a some extras in the binary
- the code is not well-adjusted to Rust's strengths - doing so might be the next steps here
- there is a lot of debug output that was used to track the differences during initial development; this will gradually go away

## Performance comparison

_Note: not very precise measurement, as it currently includes the build phase; but second invocation gives some idea_
Rust debug version:
```
$ time make run
...
real    1m42,896s
user    1m29,443s
sys     0m13,332s
```

Rust release version:
```
$ time make generate 
...
real    0m37,726s
user    0m24,268s
sys     0m13,442s
```

C version:
```
$ time make c-run
...
achieved tok/s: 23.348694
9.79user 0.02system 0:09.83elapsed 99%CPU (0avgtext+0avgdata 174432maxresident)k
0inputs+0outputs (0major+5119minor)pagefaults 0swaps

real    0m10,381s
user    0m10,317s
sys     0m0,056s
```
