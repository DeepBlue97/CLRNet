
# import gtk.gdk
 
# w = gtk.gdk.get_default_root_window()
# sz = w.get_size()
# print "The size of the window is %d x %d" % sz
# pb = gtk.gdk.Pixbuf(gtk.gdk.COLORSPACE_RGB,False,8,sz[0],sz[1])
# pb = pb.get_from_drawable(w,w.get_colormap(),0,0,0,0,sz[0],sz[1])
# if (pb != None):
#     pb.save("screenshot.png","png")
#     print "Screenshot saved to screenshot.png."
# else:
#     print "Unable to get the screenshot."

import numpy as np
import pyautogui
import time 

t0 = time.time()

for i in range(10):
    im = pyautogui.screenshot(region=(0, 0, 1280 ,720))
# im.save("python截图.jpg")

# np.array(im.getdata()).reshape(im.size[0], im.size[1], 3)

print(time.time()-t0)
