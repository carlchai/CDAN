from PIL import Image
import os
Image.MAX_IMAGE_PIXELS = 2800000000

# 打开原始图片
image1 = Image.open('WHU/t0/after.jpg')
image2 = Image.open('WHU/t1/before.jpg')
image3 = Image.open('WHU/mask/change_label.jpg')

# 获取图片的尺寸
width, height = image1.size

# 计算每个子图的尺寸
tile_width = width // 100
tile_height = height // 100

# 创建输出目录
output_dir1 = 'output_tiles1'
output_dir2 = 'output_tiles2'
output_dir3 = 'output_tiles3'
os.makedirs(output_dir1, exist_ok=True)
os.makedirs(output_dir2, exist_ok=True)
os.makedirs(output_dir3, exist_ok=True)
i = 0
# 裁剪并保存子图
for row in range(100):
    for col in range(100):
        left = col * tile_width
        upper = row * tile_height
        right = (col + 1) * tile_width
        lower = (row + 1) * tile_height
        # 裁剪子图
        tile1 = image1.crop((left, upper, right, lower))
        tile2 = image2.crop((left, upper, right, lower))
        tile3 = image3.crop((left, upper, right, lower))
        # 保存子图
        tile1.save(os.path.join(output_dir1, f'{i}.jpg'))
        tile2.save(os.path.join(output_dir2, f'{i}.jpg'))
        tile3.save(os.path.join(output_dir3, f'{i}.jpg'))
        i=i+1

print('裁剪完成')