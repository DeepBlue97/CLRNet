

p = 'work_dirs\\clr\\r18_culane\\20230113_113829_lr_6e-04_b_24\\ckpt\\10.pth'


import torch


content = torch.load(p)
print(content.keys())   # keys()
# 之后有其他需求比如要看 key 为 model 的内容有啥
# print(content['model'])

print()
