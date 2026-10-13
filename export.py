import torch
from download_weights import model
import struct
with open("gpt2.bin","wb") as f:
    f.write(b"CGPT")
    f.write(struct.pack("<7i", 1, 12, 12, 768, 50257, 1024, 3072))

    for name, p in model.named_parameters():
        if len(p.shape)  == 2:
            p = p.T
        f.write(p.detach().numpy().astype("float32").tobytes())
