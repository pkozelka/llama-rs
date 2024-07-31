# export.py analysis


## version 1 export

Export the model weights in full float32 .bin file to be read from C.
This is same as legacy_export, but with a proper header.

```
# 1.) magic, u8[4] of "ak42" in ASCII, reversed
`32 34 6b 61`       

# 2.) version, u32 of 1
`00 00 00 01`       

# 3.) header - 7 ints (i32/u32)
`?? ?? ?? ??`           # p.dim
`?? ?? ?? ??`           # hidden_dim
`?? ?? ?? ??`           # p.n_layers
`?? ?? ?? ??`           # p.n_heads,
`?? ?? ?? ??`           # n_kv_heads
`?? ?? ?? ??`           # p.vocab_size
`?? ?? ?? ??`           # p.max_seq_len

# 4.) some other flags
`??`                    # u8 =  torch.equal(model.tok_embeddings.weight, model.output.weight)

# 5.) padding to 256 bytes

# 6.) params:
model.tok_embeddings.weight
```

    # now let's write out all the params
    weights = [
        *[layer.attention_norm.weight for layer in model.layers],
        *[layer.ffn_norm.weight for layer in model.layers],
        model.norm.weight,
        model.tok_embeddings.weight,
        *[layer.attention.wq.weight for layer in model.layers],
        *[layer.attention.wk.weight for layer in model.layers],
        *[layer.attention.wv.weight for layer in model.layers],
        *[layer.attention.wo.weight for layer in model.layers],
        *[layer.feed_forward.w1.weight for layer in model.layers],
        *[layer.feed_forward.w2.weight for layer in model.layers],
        *[layer.feed_forward.w3.weight for layer in model.layers],
    ]
    if not shared_classifier:
        weights.append(model.output.weight)
    for w in weights:
        serialize_fp32(out_file, w)



