import cv2

# 打开视频文件
video_capture = cv2.VideoCapture('/home/ennheng/视频/xuelinjie2.mp4')

# 确定输出图片的文件夹
output_folder = '/home/ennheng/ufld/test/xuelinjie2/'

# 逐帧读取视频并保存为图片
frame_count = 0
while True:
    # 读取视频的每一帧
    ret, frame = video_capture.read()

    if not ret:
        break

    # 保存当前帧为图片
    output_path = output_folder + 'frame_{:04d}.jpg'.format(frame_count)
    cv2.imwrite(output_path, frame)

    frame_count += 1

# 释放视频捕获对象
video_capture.release()