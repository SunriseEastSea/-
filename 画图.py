import torch
import torch.nn.functional as F

# a = torch.ones([4, 1, 255, 255])
# print('a =',a.size())
# b = torch.ones([4, 1, 240, 240])
# print('b =',b.size())
# c = a+b
# print('c =',c.size())

a = torch.ones([4, 1, 240, 240])
print('a =',a.size())
a_max = F.interpolate(a, size=[255, 255])
print('a_max =',a_max.size())