import cv2

# 打开原始视频
video_capture = cv2.VideoCapture('/home/ennheng/视频/xuelinjie2.mp4')

# 获取原始视频的帧率和尺寸
fps = video_capture.get(cv2.CAP_PROP_FPS)
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 创建输出视频的编码器和写入器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter('/home/ennheng/视频/xuelinjie2-1.mp4', fourcc, fps, (1640, 590))

while True:
    # 读取视频的每一帧
    ret, frame = video_capture.read()

    if not ret:
        break

    # 调整帧的尺寸
    resized_frame = cv2.resize(frame, (1640, 590))

    # 写入调整后的帧到输出视频
    output_video.write(resized_frame)

    # 显示调整后的帧（可选）
    cv2.imshow('Resized Video', resized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放视频捕获和写入对象
video_capture.release()
output_video.release()

# 关闭窗口（如果有）
cv2.destroyAllWindows()