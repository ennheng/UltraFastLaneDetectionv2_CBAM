import cv2
image = Image.open(image_path)
label = Image.open(label_path)
w, h = label.size

# 将行采样点按输入图像高度放缩
if h != 288:
    scale_f = lambda x : int((x * 1.0/288) * h)
    sample_tmp = list(map(scale_f,culane_row_anchor))
    # 根据提供的函数对指定序列做映射

lines = ""
for i,r in enumerate(sample_tmp):
    label_r = np.asarray(label)[int(round(r))]
    # 取出label图像中行坐标为int(round(r))的一行
    for lane_idx in range(1, 5):
        line = ""
        pos = np.where(label_r == lane_idx)[0]
        if len(pos) == 0:
            continue
        pos = np.mean(pos)
        line = line + str(pos) + " " + str(r) + " "
        # print(line)
        # cv2.circle(image, (int(round(pos)), int(round(r))), 1, (0,0,255),2)
    lines = lines + line + "\n"
with open(lines_file_path, 'w') as f:
    f.write(lines)