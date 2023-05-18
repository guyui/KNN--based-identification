import cv2 as cv
import numpy as np
import train


# def show(window_name, image):
#     cv.namedWindow(window_name, 0)
#     cv.imshow(window_name, image)
#     resized_img = cv.resize(image, (509, 321), interpolation=cv.INTER_AREA)  # 添加该行代码
#     cv.imshow(window_name, resized_img)
#     cv.waitKey(0)
#     cv.destroyAllWindows()

def show(window_name,image):
    cv.namedWindow(window_name, 0)
    cv.moveWindow(window_name,200,200)
    cv.resizeWindow(window_name,400,300)
    cv.imshow(window_name, image)
   # cv.resizeWindow("image",509, 321)
    cv.waitKey(0)
    cv.destroyAllWindows()


# 读取图片和身份证号位置模板
idimg = cv.imread("test2.jpg")
# show('orpic',idimg)

# idimg = cv.resize(idimg, (509, 321), interpolation=cv.INTER_CUBIC)
template = cv.imread("position3.jpg", 1)
template=cv.resize(template, (280, 18), interpolation=cv.INTER_AREA)
# show('template',template)

# 身份证转灰度图
gray = cv.cvtColor(idimg, cv.COLOR_BGR2GRAY)
# gray_resized = cv.resize(gray, (509, 321))
show('gray',gray)

# 中值滤波
blur = cv.medianBlur(gray, 7)
show("blur",blur)

#二值处理，用cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
# THRESH_OTSU模式只适用于单通道的8位或16位灰度图像
threshold = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
show("threshold",threshold)

#边缘检测
edges = cv.Canny(threshold, 100, 150)#后两个为阈值
show("canny",edges)


#边缘膨胀
kernel = np.ones((3, 3), np.uint8)
dilate = cv.dilate(edges, kernel, iterations=5)
show("dilate",dilate)

#轮廓检测
contours, hierarchy = cv.findContours(dilate, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
image_copy = idimg.copy()
res = cv.drawContours(image_copy, contours, -1, (255, 0, 0), 20)
show("res",res)

#轮廓面积排序
contours = sorted(contours, key=cv.contourArea, reverse=True)[0]#从大到小第一个
image_copy = idimg.copy()
res = cv.drawContours(image_copy, contours, -1, (255, 0, 0), 20)
show("contours",res)
# 首先，使用cv.arcLength函数计算轮廓的周长，乘以0.02因子得到一个较小的epsilon值。
# 然后将该值用于 cv.approxPolyDP函数中，以获得近似的轮廓。
# 接着，将近似轮廓的点按x坐标进行排序，并将前两个点按y坐标排序，后两个点按y坐标逆序排序。
# 最后，将四个角点存储在 p1数组中，并创建一个目标矩阵 pts2，其中包含矩形的四个角点。
epsilon = 0.02 * cv.arcLength(contours, True)
approx = cv.approxPolyDP(contours, epsilon, True)
n = []
for x, y in zip(approx[:, 0, 0], approx[:, 0, 1]):
    n.append((x, y))
n = sorted(n)
sort_point = []
n_point1 = n[:2]
n_point1.sort(key=lambda x: x[1])
sort_point.extend(n_point1)
n_point2 = n[2:4]
n_point2.sort(key=lambda x: x[1])
n_point2.reverse()
sort_point.extend(n_point2)
p1 = np.array(sort_point, dtype=np.float32)
h = sort_point[1][1] - sort_point[0][1]
w = sort_point[2][0] - sort_point[1][0]
pts2 = np.array([[0, 0], [0, h], [w, h], [w, 0]], dtype=np.float32)

# 生成变换矩阵
M = cv.getPerspectiveTransform(p1, pts2)
# 进行透视变换
dst = cv.warpPerspective(idimg, M, (w, h))
# print(dst.shape)
show("dst",dst)

#固定大小
if w < h:
    dst = np.rot90(dst)
resize = cv.resize(dst, (509, 321), interpolation=cv.INTER_AREA)
show("resize",resize)



# 黑帽运算闭运算的卷积核
kernel1 = np.ones((15, 15), np.uint8)
# kernel2 = np.ones((1,1),np.uint8)
# 黑帽运算，提取轮廓
cvblackhat = cv.morphologyEx(resize, cv.MORPH_BLACKHAT, kernel1)
# print(cvblackhat.shape)

# 闭运算
cvclose1 = cv.morphologyEx(cvblackhat, cv.MORPH_CLOSE, kernel1)

cvclose1=cv.cvtColor(cvclose1,cv.COLOR_BGR2GRAY)
 # 原图像二值化
ref = cv.threshold(cvclose1, 0, 255, cv.THRESH_OTSU)[1]
show('ref',ref)
# 身份证号码区域二值化
cvblackhat = cv.cvtColor(cvblackhat,cv.COLOR_BGR2GRAY) # 转换为灰度图
thresh,img_bin = cv.threshold(cvblackhat,127,255, cv.THRESH_BINARY|cv.THRESH_OTSU)
twoimg = cv.threshold(cvblackhat, 0, 255, cv.THRESH_OTSU)[1]
# print(twoimg.shape)

# 为了模板匹配
cv.imwrite("ref.jpg", ref)
ref = cv.imread("ref.jpg", 1)
#print(twoimg.shape)
# 获取模板高和宽
h, w = template.shape[:2]
# 模板匹配（相关匹配）找身份证号码位置
res = cv.matchTemplate(ref, template, cv.TM_CCORR)
# 获得最匹配地方的左上角坐标
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
top_left = max_loc
# 计算最匹配地方的右下角坐标
bottom_right = (top_left[0] + w, top_left[1] + h)
# 框出身份证号区域并展示
cv.rectangle(idimg, top_left, bottom_right, (0, 255, 0), 2)
# 展示身份证号码的二值图像
rectangleid = cv.resize(twoimg[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]], (432, 32),
                        interpolation=cv.INTER_CUBIC)
# rectangleid = cv.erode(rectangleid,kernel2)
cv.imshow("rectangleid", rectangleid)

# 划分获得每一个数字的图像
cells = [np.hsplit(row, 18) for row in np.vsplit(rectangleid, 1)]

# 转换成np.array类型
x = np.array(cells)
# cv.imshow("cell9", x[0][9])
# cv.imshow("cell10", x[0][10])
# 图像数据转换为特征矩阵
test = x[:, :].reshape(-1, 768).astype(np.float32)
# 获得训练好的knn模型
knn,_,_ = train.KNN()
# 测试
ret, result, neighbours, dist = knn.findNearest(test, 3)
# result：表示每个测试数据集样本数据的预测结果。即测试数据集中每个样本数据所属的类别或数值
# 回归问题 result数组中的每个元素表示对应测试数据集样本数据的数值预测结果。
# 输出预测结果
result = np.uint8(result).reshape(-1, 18)[0]
id = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "X"]
idstr = ""
for i in result:
    idstr += id[i]
print(idstr)

cv.waitKey(0)

