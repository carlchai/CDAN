import os

from PIL import Image

root = "./dataset/ChangeSim/dark"

mask = root+"/mask"
t0 = root+"/t0"
t1 = root+"/t1"

mask_dirs =  os.listdir(mask)
t0_dirs =  os.listdir(t0)
t1_dirs =  os.listdir(t1)
num = 0
for dir in t1_dirs:
    np = os.path.join(t1 ,dir)
    pics = os.listdir(np)
    for pic in pics:
        picd = os.path.join(np,pic)
        picture = Image.open(picd)
        nn = t1+"/"+str(num)+".png"
        print(nn)
        picture.save(nn)
        num = num+1




