import cv2
import os

# 图片文件夹路径和输出视频路径
image_folder = '/home/ennheng/ufld/result/hdu/'  # 图片文件夹路径
output_video = '/home/ennheng/ufld/result/hdu.mp4'  #


# 获取图片文件夹中的所有图片文件名
images = [img for img in os.listdir(image_folder) if img.endswith(".png") or img.endswith(".jpg")]

# 根据图片的生成时间对图片进行排序
images = sorted(images, key=lambda x: os.path.getmtime(os.path.join(image_folder, x)))

# 获取第一张图片的宽度和高度
first_image = cv2.imread(os.path.join(image_folder, images[0]))
height, width, _ = first_image.shape
# 定义视频编码器和输出视频对象
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 编码器
video = cv2.VideoWriter(output_video, fourcc, 30, (width, height))  # 输出视频对象

# 逐帧写入视频
for image_name in images:
    image_path = os.path.join(image_folder, image_name)
    frame = cv2.imread(image_path)
    video.write(frame)  # 将帧写入视频

# 释放视频编码器对象和关闭视频文件
video.release()
cv2.destroyAllWindows()



