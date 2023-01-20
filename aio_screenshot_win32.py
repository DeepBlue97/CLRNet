import time

from grabscreen import grab_screen

t0 = time.time()
for i in range(10):
    img = grab_screen(region=(0, 0, 1280, 720))

print(time.time()-t0)
