# %%
from generator import Generator
# %%
model4 = Generator(nz=256, ngf=64, img_size=32, nc=3)


# %%
param_size = 0
for param in model4.parameters():
    param_size += param.nelement() * param.element_size()
buffer_size = 0
for buffer in model4.buffers():
    buffer_size += buffer.nelement() * buffer.element_size()

size_all_mb = (param_size + buffer_size) / 1024**2
print("abcd")
print('model size: {:.3f}MB'.format(size_all_mb))
