# coding:utf-8
from PIL import Image
import numpy as np
import math
import time
import os  # 用于查找目录下的文件
import copy
import sys

np.set_printoptions(threshold=np.inf)

# 输出图片的位置
ImageWidth = 512
ImageHeight = 512
FILE_PATH = r"D:\image hiding\MPM22-main\%d_%d\Output626" % (ImageWidth, ImageHeight)
if not os.path.exists(FILE_PATH):
    os.makedirs(FILE_PATH)

# 写入文件
def SaveResult(str):
    # 将str写入结果文件中
    try:
        fname = time.strftime("%Y%m%d", time.localtime())
        f2 = open(FILE_PATH + '\\0_result' + fname + '.txt', 'a+')
        f2.read()
        # f2.write('--------------------------------------------------')
        # f2.write('\n')
        # timestr = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        # f2.write(timestr)
        f2.write('\n')
        f2.write(str)
        f2.write('\n')
    finally:
        if f2:
            f2.close()
    return 0

# PSNR
def PSNR(image_array1, image_array2):
    # 输入为两个图像数组，一维，大小相同
    assert (np.size(image_array1) == np.size(image_array2))
    n = np.size(image_array1)
    assert (n > 0)
    MSE = 0.0
    for i in range(0, n):
        MSE += math.pow(int(image_array1[i]) - int(image_array2[i]), 2)
    MSE = MSE / n
    if MSE > 0:
        rtnPSNR = 10 * math.log10(255 * 255 / MSE)
    else:
        rtnPSNR = 100
    return rtnPSNR
def convert_to_ksqu(num,k):
    """
    将十进制数转换为k**2进制
    """
    digits = []
    while num > 0:
        digits.append(num % (k**2))
        num //= (k**2)
    # digits.reverse()
    return digits
def convert_to_decimal(digits, k):
    """
    将k**2进制数转换为十进制数
    """
    decimal_num = 0
    for i in range(len(digits)):
        decimal_num += digits[i] * (k**2)**i
    return decimal_num

# ALGORITHM: 按分组的方法,每个pixel分别找模数，具有高位最大值的那个pixel嵌入多一个bit（k+1），其他组嵌入k个bit
def MPM22(image_array, secret_string, n_input=6, k_input=3, image_file_name=''):
    # image_array:输入的一维图像数组
    # image_file_name:传入的图像文件名（带全路径）
    # n为一组像素的数量,在本算法中，固定为3
    n = int(n_input)
    k = int(k_input)  # 每组pixels嵌入n * k 个bit
    k2 = k + 1
    moshu0 = 2 ** (k + 1)  # 模数的底数
    moshu1 = 2 ** k  # 模数的底数

    # 分成n个像素一组,保证整数组，不足的补零
    numPG = math.ceil(image_array.size / n)
    PGoriginalValue = np.zeros((numPG, n), dtype="int")
    PGhighPartValue = np.zeros((numPG, n), dtype="int")
    PGlowerMask = 2 ** k2 - 1  # used for spliting lower and higher bits
    for i in range(0, numPG, 1):
        for j2 in range(0, n, 1):
            if i * n + j2 < image_array.size:
                PGoriginalValue[i, j2] = image_array[i * n + j2]  # 原像素值
                PGhighPartValue[i, j2] = image_array[i * n + j2] & (
                            255 - PGlowerMask)  # higher part, used for searching the highest pixel value

    numBitsPerPG = n * k + 1  # 每组pixcel嵌入的bit数
    numSecretGroups = math.ceil(secret_string.size / numBitsPerPG)
    secretGroup = np.zeros((numSecretGroups, numBitsPerPG), dtype="uint8")
    secret_string_copy = secret_string.copy()
    for i in range(0, numSecretGroups, 1):
        for j2 in range(0, numBitsPerPG, 1):
            if i * numBitsPerPG + j2 < secret_string.size:
                secretGroup[i, j2] = secret_string_copy[i * numBitsPerPG + j2]

    secret_d_array = np.zeros((numSecretGroups), dtype="int")  # 待嵌入的secret值
    for i in range(0, numSecretGroups, 1):
        for j2 in range(0, numBitsPerPG, 1):
            # secret_d_array[i]+=(2 ** j) * secret_group[i,j] #低位在前
            secret_d_array[i] += (2 ** (numBitsPerPG - 1 - j2)) * secretGroup[i, j2]  # 高位在前,decimal value

    assert (numPG > numSecretGroups)
    PGembedded = copy.deepcopy(PGoriginalValue)

    # embedding ---------------------------------------------
    for i in range(0, numSecretGroups, 1):
        # 找出pixel组中的一个pixel，用来嵌入最前面的k+1 bit
        selIndex = i % n

        firstk1BitsDvalue = int(secret_d_array[i] / (2 ** ((n - 1) * k)))  # 取出k+1 bits
        kBitsGroup = np.zeros(n, dtype=int)  # get k bits group in the rest
        # 第selIndex个不用，保持为0
        lastBitsDvalue = int(secret_d_array[i]) & (2 ** ((n - 1) * k) - 1)
        for j in range(0, selIndex, 1):
            kBitsGroup[j] = int(lastBitsDvalue / (2 ** ((n - j - 2) * k))) & (2 ** k - 1)
        for j in range(selIndex + 1, n, 1):
            kBitsGroup[j] = int(lastBitsDvalue / (2 ** ((n - j - 1) * k))) & (2 ** k - 1)

        # 确保转换没有错误
        v1 = firstk1BitsDvalue
        v2 = v1 * (2 ** ((n - 1) * k))
        tmpValue = 0
        for j2 in range(0, selIndex, 1):
            v3 = kBitsGroup[j2]
            tmpValue += v3 * (2 ** ((n - j2 - 2) * k))
        for j2 in range(selIndex + 1, n, 1):
            v3 = kBitsGroup[j2]
            tmpValue += v3 * (2 ** ((n - j2 - 1) * k))
        assert (v2 + tmpValue == secret_d_array[i])

        # search x for firstk1BitsDvalue
        diffMin = 99999999
        xSel = -9999
        for x in range(-moshu0, moshu0 + 1, 1):
            t = PGoriginalValue[i, selIndex]
            if ((t + x) % moshu0 == firstk1BitsDvalue) and ((t + x) >= 0) and ((t + x) <= 255):
                tDiff = x ** 2
                if tDiff < diffMin:
                    diffMin = tDiff
                    xSel = x
        PGembedded[i, selIndex] = PGoriginalValue[i, selIndex] + xSel

        # search x for kBitsGroup
        for j1 in range(0, selIndex, 1):
            diffMin = 99999999
            xSel = -9999
            for x in range(-moshu1, moshu1 + 1, 1):
                t = PGoriginalValue[i, j1]
                if ((t + x) % moshu1 == kBitsGroup[j1]) and ((t + x) >= 0) and ((t + x) <= 255):
                    tDiff = x ** 2
                    if tDiff < diffMin:
                        diffMin = tDiff
                        xSel = x
            PGembedded[i, j1] = PGoriginalValue[i, j1] + xSel

        for j1 in range(selIndex + 1, n, 1):
            diffMin = 99999999
            xSel = -9999
            for x in range(-moshu1, moshu1 + 1, 1):
                t = PGoriginalValue[i, j1]
                if ((t + x) % moshu1 == kBitsGroup[j1]) and ((t + x) >= 0) and ((t + x) <= 255):
                    tDiff = x ** 2
                    if tDiff < diffMin:
                        diffMin = tDiff
                        xSel = x
            PGembedded[i, j1] = PGoriginalValue[i, j1] + xSel
    # 使用了多少pixel来进行嵌入
    num_pixels_changed = numSecretGroups * n
    # -----------------------------------------------------------------------------------
    # 恢复，提取加密数据
    recover_d_array = np.zeros((numSecretGroups), dtype="int")  # 恢复值
    for i in range(0, numSecretGroups, 1):
        # 找出pixel组中，高bits最大值的那一个pixel
        selIndex = i % n
        v1 = PGembedded[i, selIndex] % moshu0
        v2 = v1 * (2 ** ((n - 1) * k))

        # 恢复剩余的n-1 pixel
        tmpValue = 0
        for j2 in range(0, selIndex, 1):
            v3 = PGembedded[i, j2] % moshu1
            tmpValue += v3 * (2 ** ((n - j2 - 2) * k))
        for j2 in range(selIndex + 1, n, 1):
            v3 = PGembedded[i, j2] % moshu1
            tmpValue += v3 * (2 ** ((n - j2 - 1) * k))

        recover_d_array[i] = v2 + tmpValue
        assert (int((recover_d_array[i] - secret_d_array[i]).sum()) == 0)

    assert (int((recover_d_array - secret_d_array).sum()) == 0)
    # -----------------------------------------------------------------------------------
    # 输出图像
    img_out = PGembedded.flatten()
    img_out = img_out[:ImageWidth * ImageHeight]  # 取前面的pixel
    # 计算PSNR
    img_array_out = img_out.copy()
    imgpsnr1 = image_array[0:num_pixels_changed]
    imgpsnr2 = img_array_out[0:num_pixels_changed]
    # psnr = PSNR(image_array,img_array_out)
    psnr = PSNR(imgpsnr1, imgpsnr2)

    # 重组图像
    img_out = img_out.reshape(ImageWidth, ImageHeight)
    img_out = Image.fromarray(img_out)
    # img_out.show()
    img_out = img_out.convert('L')

    (filepath, tempfilename) = os.path.split(image_file_name)
    (originfilename, extension) = os.path.splitext(tempfilename)

    new_file = FILE_PATH + '\\' + originfilename + '_' + sys._getframe().f_code.co_name + "_n_" + str(n) + "_k_" + str(
        k) + ".png"
    img_out.save(new_file, 'png')

    str1 = 'Image:%30s,Method:%15s,n=%d,k=%d,pixels used: %d,bpp: %.4f ,PSNR: %.2f' % (
    originfilename, sys._getframe().f_code.co_name, n, k, num_pixels_changed,s_data.size/num_pixels_changed, psnr)
    print(str1)
    SaveResult('\n' + str1)
    # -----------------------------------------------------------------------------------
    return psnr



def k_aray_New23(image_array, secret_string, n_input=2, k_input=4, image_file_name=''):
    # image_array:输入的一维图像数组
    # image_file_name:传入的图像文件名（带全路径）
    # n为一组像素的数量,在本算法中，固定为3
    # print("像素的值为",image_array[125052:129150])
    # print("125052像素的值为", image_array[125052])
    # print("125053像素的值为", image_array[125053])
    # print("128124像素的值为", image_array[128124])
    # print("128125像素的值为", image_array[128125])
    # print("128636像素的值为", image_array[128636])
    # print("128637像素的值为", image_array[128637])
    # print("129148像素的值为", image_array[129148])
    # print("129149像素的值为", image_array[129149])
    # print("像素的值为", image_array[6])
    # print("像素的值为", image_array[7])
    # print("像素的值为", image_array[8])
    # print("像素的值为", image_array[9])
    # print("像素的值为", image_array[10])
    # print("像素的值为", image_array[11])
    # print("像素的值为", image_array[12])
    # print("像素的值为", image_array[13])
    # print("像素的值为", image_array[14])
    # print("像素的值为", image_array[15])
    # print("像素的值为", image_array[16])
    print("图片像素的长度",image_array.size)

    n = int(n_input)
    # print("n",n)
    k = int(k_input)  # 每组pixels嵌入n * k 个bit
    # print("k", k)
    # print("secret_string", secret_string[:200])
    # print("secret_string", secret_string[-200:])
    # print("secret_string[1]", secret_string[1])
    # print("secret_string[2]", secret_string[2])
    # print("secret_string[3]", secret_string[3])
    # print("secret_string[4]", secret_string[4])
    # print("secret_string[5]", secret_string[5])
    # print("secret_string[6]", secret_string[6])
    # print("secret_string[7]", secret_string[7])
    # print("secret_string[8]", secret_string[8])
    # print("secret_string[9]", secret_string[9])
    # print("secret_string[10]", secret_string[10])
    # print("secret_string[11]", secret_string[11])
    # print("secret_string[12]", secret_string[12])
    # print("secret_string[13]", secret_string[13])
    # print("secret_string[14", secret_string[14])
    # print("secret_string[15]", secret_string[15])

    mat = np.zeros((256, 256), dtype="int")
    for i in range(0, 256, 1):
        for j2 in range(0, 256, 1):
            mat[i, j2] = ((k * (i % k)) + (j2 % k)) % (k**2)
            # print("mat矩阵"+str(i)+"行"+str(j2)+"列的值为：", mat[i, j2])

    # print("矩阵",mat)

    k2 = k + 1
    # moshu0 = 2 ** (n - 1)  # 模数的底数
    # moshu1 = 2 ** k  # 模数的底数

    # 分成n个像素一组,保证整数组，不足的补零
    numPG = math.ceil(image_array.size / n)
    PGoriginalValue = np.zeros((numPG, n), dtype="int")
    PGhighPartValue = np.zeros((numPG, n), dtype="int")
    PGlowerMask = 2 ** k2 - 1  # used for spliting lower and higher bits
    for i in range(0, numPG, 1):
        for j2 in range(0, n, 1):
            if i * n + j2 < image_array.size:
                PGoriginalValue[i, j2] = image_array[i * n + j2]  # 原像素值
                PGhighPartValue[i, j2] = image_array[i * n + j2] & (255 - PGlowerMask)  # higher part, used for searching the highest pixel value
    # print("PGoriginalValue", PGoriginalValue)
    # 首先将二进制秘密信息转化为十进制数字
    # 再将十进制数字转化为25（k**2）进制数字
    # numBitsPerPG每组嵌入的bit数也就是一个25（k**2）进制数字
    # binary_seq = secret_string  # 二进制序列
    secret_string_str = ''.join([str(x) for x in secret_string])
    binary_seq = secret_string_str
    # print("二进制表示为：", binary_seq)
    decimal_num = int(binary_seq, 2)
    # print("十进制表示为：", decimal_num)

    ksqu_num = convert_to_ksqu(decimal_num,k)  # 将十进制整数转换为25进制数字
    ksqu_num_array = list(ksqu_num)
    num_digits = len(ksqu_num)
    print("K**2进制的个数为：", num_digits)
    # print(str(k**2)+"进制表示为：", ksqu_num)

    # 怎样使numBitsPerPG不为整数，嵌入的就是一个（k**2）进制数字
    # numBitsPerPG = math.log(k**2,2)  # 每组pixcel嵌入的bit数(n-1)(k+1)
    # numBitsPerPG = 1
    numSecretGroups = num_digits
    # secretGroup = np.zeros((numSecretGroups), dtype="uint8")
    secret_d_array = np.zeros((numSecretGroups), dtype="int")  # 待嵌入的secret值
    # secret_string_copy = secret_string.copy()
    for i in range(0, numSecretGroups, 1):
            # if i * numBitsPerPG + j2 < secret_string.size:
        # secretGroup[i] = ksqu_num_array[i]
        secret_d_array[i] = ksqu_num_array[i]
    # print("秘密信息序列表示为：", secret_d_array)
    # print("secretGroup的数组为：", secretGroup)

    # secret_d_array = np.zeros((numSecretGroups), dtype="int")  # 待嵌入的secret值
    # print("n", n)
    # print("k", k)
    # print("numSecretGroups", numSecretGroups)
    # print("numBitsPerPG", numBitsPerPG)
    # for i in range(0, numSecretGroups, 1):
    #     for j2 in range(0, numBitsPerPG, 1):
    #         # secret_d_array[i]+=(2 ** j) * secret_group[i,j] #低位在前
    #         secret_d_array[i] += (2 ** (numBitsPerPG - 1 - j2)) * secretGroup[i, j2]  # 高位在前,decimal value

            # print("第" + str(i) + "个secret_d_array的值为", secret_d_array[i])
    assert (numPG > numSecretGroups)
    PGembedded = copy.deepcopy(PGoriginalValue)
    # if isinstance(PGembedded, int):
    #     PGembedded = [[PGembedded]]
    # print("PGembedded",PGembedded)

    # embedding ---------------------------------------------
    # print("秘密信息序列表示为：", secret_d_array)
    for i in range(0, numSecretGroups, 1):
        # 找出pixel组中的一个pixel，用来嵌入最前面的k+1 bit
        # for j1 in range(0, n, 1):
        p1 = PGoriginalValue[i, 0]
        p2 = PGoriginalValue[i, 1]
        # print("p1",p1)
        # print("p2", p2)
        # print("该像素对对应的矩阵值",mat[p1,p2])
        # print( "要嵌入的秘密信息十进制值",secret_d_array[i])
    # if (mat[px, py] == secret_d_array[i]):
    #     p1 += p1 +1

        if(p1 < int(k / 2)):
            print("第" + str(i) + "像素为：",p1)
            print("第" + str(i) + "像素为：", p2)
            p1 += (int(k/2) - p1)
            # print("p1小于"+int(k / 2)+"p1为"+p1)
        if (p2 < int(k / 2)):
            p2 += (int(k / 2) - p2)
            # print("p2小于" + int(k / 2) + "p2为" + p2)
        if (p1 > 256 - int(k / 2)):

            p1 -= (int(k / 2) + 1)
            # print("p1大于" + 256 - int(k / 2) + "p1为" + p1)
        if (p2 > 256 - int(k / 2)):
            print("第" + str(i) + "像素为：", p2)
            p2 -= (int(k / 2) + 1)
            # print("p1大于" + 256 - int(k / 2) + "p1为" + p1)
        D1 = 9999
        for py in range (p2 - int(k/2),p2 + int(k/2) + 2,1):
            for px in range (p1 - int(k/2),p1 + int(k/2) + 2,1):
                # print("寻找信息为", mat[px, py])

                if (mat[px,py] == secret_d_array[i]):
                    # print("yes")
                    # p1 += p1 +1
                    D = abs(p1 - px) + abs(p2 - py)
                    if D < D1:
                        D1 = D
                        PGembedded[i,0] = px
                        PGembedded[i,1] = py
                    # print("秘密信息为", mat[PGembedded[i,0],PGembedded[i,1]])
                    # print("嵌入之后的像素值", PGembedded[i])
                # else:
                #     print("chucuo")



    num_pixels_changed = numSecretGroups * n
    print("改变了多少像素值", num_pixels_changed)
    # 嵌入了多少秘密信息（相同的像素个数嵌入了多少秘密信息）
    # -----------------------------------------------------------------------------------
    # 恢复，提取加密数据
    recover_d_array = np.zeros((numSecretGroups), dtype="int")  # 恢复值

    # 将十进制转化为二进制
    # binary_seq1 = bin(decimal_num)[2:]
    # print("binary_seq1：", binary_seq1)
    # 将25进制数字进制转化为十进制
    # decimal_num_conversr = convert_to_decimal(ksqu_num, k)
    # print("恢复后的十进制表示为：", decimal_num)

    for i in range(0, numSecretGroups, 1):
        recover_d_array[i] = mat[PGembedded[i,0],PGembedded[i,1]]
        # print("PGembedded[i]",PGembedded[i])
        # print("PGembedded[0]", PGembedded[i,0])
        # print("PGembedded[1]", PGembedded[i,1])
        # print("矩阵中的秘密信息", mat[PGembedded[i,0],PGembedded[i,1]])

    # print(recover_d_array)
    #     # 找出pixel组中，高bits最大值的那一个pixel
    #     selIndex = i % n
    #     v1 = PGembedded[i, selIndex] % moshu0
    #     # 将v1分为n-1组，放在B1bitGroup中
    #     B1bitGroup = np.zeros(n, dtype=int)
    #     for j3 in range(0, selIndex, 1):
    #         B1bitGroup[j3] = int(v1 / (2 ** (n - j3 - 2))) & (1)
    #     for j3 in range(selIndex + 1, n, 1):
    #         B1bitGroup[j3] = int(v1 / (2 ** (n - j3 - 1))) & (1)
    #     # v2 = v1 * (2 ** ((n - 1) * k))
    #     # print("v1", v1)
    #     # print("v2", v2)
    #     # 恢复剩余的n-1 pixel
    #     tmpValue = 0
    #     for j2 in range(0, selIndex, 1):
    #         # print("第" + str(i) + "第" + str(j2) + "个PGembedded[i, j2]的值为", PGembedded[i, j2])
    #         a1 = B1bitGroup[j2]
    #         v2 = PGembedded[i, j2] % moshu1
    #         v3 = a1 * (2 ** k) + v2
    #         tmpValue += v3 * (2 ** ((n - j2 - 2) * k2))
    #     for j2 in range(selIndex + 1, n, 1):
    #         # print("第" + str(i) + "第" + str(j2) + "个PGembedded[i, j2]的值为", PGembedded[i, j2])
    #         a1 = B1bitGroup[j2]
    #         v2 = PGembedded[i, j2] % moshu1
    #         v3 = a1 * (2 ** k) + v2
    #         tmpValue += v3 * (2 ** ((n - j2 - 1) * k2))
    #     recover_d_array[i] =  tmpValue
    #     # print("recover_d_array[i]", recover_d_array[i])
    #     # print("第" + str(i) + "个recover_d_array的值为", recover_d_array[i])
    #     assert (int((recover_d_array[i] - secret_d_array[i]).sum()) == 0)
    #
    assert (int((recover_d_array - secret_d_array).sum()) == 0)
    #
    # # -----------------------------------------------------------------------------------
    # # 输出图像
    img_out = PGembedded.flatten()
    img_out = img_out[:ImageWidth * ImageHeight]  # 取前面的pixel
    # 计算PSNR
    img_array_out = img_out.copy()
    imgpsnr1 = image_array[0:num_pixels_changed]
    imgpsnr2 = img_array_out[0:num_pixels_changed]
    # psnr = PSNR(image_array,img_array_out)
    psnr = PSNR(imgpsnr1, imgpsnr2)

    # 重组图像
    img_out = img_out.reshape(ImageWidth, ImageHeight)
    img_out = Image.fromarray(img_out)
    # img_out.show()
    img_out = img_out.convert('L')

    (filepath, tempfilename) = os.path.split(image_file_name)
    (originfilename, extension) = os.path.splitext(tempfilename)

    new_file = FILE_PATH + '\\' + originfilename + '_' + sys._getframe().f_code.co_name + "_n_" + str(n) + "_k_" + str(
        k) + ".png"
    img_out.save(new_file, 'png')

    str1 = 'Image:%30s,Method:%15s,n=%d,k=%d,pixels used: %d,bpp:%.4f, PSNR: %.2f' % (
        originfilename, sys._getframe().f_code.co_name, n, k, num_pixels_changed,s_data.size/num_pixels_changed,  psnr)
    print(str1)
    SaveResult('\n' + str1)
    # -----------------------------------------------------------------------------------
    return psnr

def FEMD11(image_array, secret_string, n_input=2, k_input=4, image_file_name=''):
    # image_array:输入的一维图像数组
    # image_file_name:传入的图像文件名（带全路径）
    # n为一组像素的数量,在本算法中，固定为3
    # print("像素的值为",image_array[0])
    # print("像素的值为", image_array[1])
    # print("像素的值为", image_array[2])
    # print("像素的值为", image_array[3])
    # print("像素的值为", image_array[4])
    # print("像素的值为", image_array[5])
    # print("像素的值为", image_array[6])
    # print("像素的值为", image_array[7])
    # print("像素的值为", image_array[8])
    # print("像素的值为", image_array[9])
    # print("像素的值为", image_array[10])
    # print("像素的值为", image_array[11])
    # print("像素的值为", image_array[12])
    # print("像素的值为", image_array[13])
    # print("像素的值为", image_array[14])
    # print("像素的值为", image_array[15])
    # print("像素的值为", image_array[16])
    print("图片像素的长度",image_array.size)

    n = int(n_input)
    # print("n",n)
    k = int(k_input)  # 每组pixels嵌入n * k 个bit
    # print("k", k)
    # print("secret_string[0]", secret_string[0])
    # print("secret_string[1]", secret_string[1])
    # print("secret_string[2]", secret_string[2])
    # print("secret_string[3]", secret_string[3])
    # print("secret_string[4]", secret_string[4])
    # print("secret_string[5]", secret_string[5])
    # print("secret_string[6]", secret_string[6])
    # print("secret_string[7]", secret_string[7])
    # print("secret_string[8]", secret_string[8])
    # print("secret_string[9]", secret_string[9])
    # print("secret_string[10]", secret_string[10])
    # print("secret_string[11]", secret_string[11])
    # print("secret_string[12]", secret_string[12])
    # print("secret_string[13]", secret_string[13])
    # print("secret_string[14", secret_string[14])
    # print("secret_string[15]", secret_string[15])

    mat = np.zeros((256, 256), dtype="int")
    for i in range(0, 256, 1):
        for j2 in range(0, 256, 1):
            # mat[i, j2] = ((k * (i % k)) + (j2 % k)) % (2**k)
            mat[i, j2] = ((k - 1) * i + k * j2) % (k ** 2)
            # print("mat矩阵"+str(i)+"行"+str(j2)+"列的值为：", mat[i, j2])

    # print("矩阵",mat)

    k2 = k + 1
    # moshu0 = 2 ** (n - 1)  # 模数的底数
    # moshu1 = 2 ** k  # 模数的底数

    # 分成n个像素一组,保证整数组，不足的补零
    numPG = math.ceil(image_array.size / n)
    PGoriginalValue = np.zeros((numPG, n), dtype="int")
    PGhighPartValue = np.zeros((numPG, n), dtype="int")
    PGlowerMask = 2 ** k2 - 1  # used for spliting lower and higher bits
    for i in range(0, numPG, 1):
        for j2 in range(0, n, 1):
            if i * n + j2 < image_array.size:
                PGoriginalValue[i, j2] = image_array[i * n + j2]  # 原像素值
                PGhighPartValue[i, j2] = image_array[i * n + j2] & (255 - PGlowerMask)  # higher part, used for searching the highest pixel value
    # print("PGoriginalValue", PGoriginalValue)
    numBitsPerPG = int(math.log(k**2,2))  # 每组pixcel嵌入的bit数(n-1)(k+1)
    numSecretGroups = math.ceil(secret_string.size / numBitsPerPG)
    secretGroup = np.zeros((numSecretGroups, numBitsPerPG), dtype="uint8")
    secret_string_copy = secret_string.copy()
    for i in range(0, numSecretGroups, 1):
        for j2 in range(0, numBitsPerPG, 1):
            if i * numBitsPerPG + j2 < secret_string.size:
                secretGroup[i, j2] = secret_string_copy[i * numBitsPerPG + j2]

    secret_d_array = np.zeros((numSecretGroups), dtype="int")  # 待嵌入的secret值
    # print("n", n)
    # print("k", k)
    # print("numSecretGroups", numSecretGroups)
    # print("numBitsPerPG", numBitsPerPG)
    for i in range(0, numSecretGroups, 1):
        for j2 in range(0, numBitsPerPG, 1):
            # secret_d_array[i]+=(2 ** j) * secret_group[i,j] #低位在前
            secret_d_array[i] += (2 ** (numBitsPerPG - 1 - j2)) * secretGroup[i, j2]  # 高位在前,decimal value

            # print("第" + str(i) + "个secret_d_array的值为", secret_d_array[i])
    assert (numPG > numSecretGroups)
    PGembedded = copy.deepcopy(PGoriginalValue)
    # if isinstance(PGembedded, int):
    #     PGembedded = [[PGembedded]]
    # print("PGembedded",PGembedded)

    # embedding ---------------------------------------------
    for i in range(0, numSecretGroups, 1):
        # 找出pixel组中的一个pixel，用来嵌入最前面的k+1 bit
        # for j1 in range(0, n, 1):
        p1 = PGoriginalValue[i, 0]
        p2 = PGoriginalValue[i, 1]
        # print("p1",p1)
        # print("p2", p2)
        # print("该像素对对应的矩阵值",mat[p1,p2])
        # print( "要嵌入的秘密信息十进制值",secret_d_array[i])
    # if (mat[px, py] == secret_d_array[i]):
        # p1 += p1 +1

        if(p1 < int(k / 2)):
            p1 += (int(k/2) - p1)
        #     print("p1小于"+(k / 2)+"p1为"+p1)
        if (p2 < int(k / 2)):
            p2 += (int(k / 2) - p2)
        #     print("p2小于" + (k / 2) + "p2为" + p2)
        if (p1 > 256 - int(k / 2)):
            p1 -= (int(k / 2) + 1)
        #     print("p1大于" + 256 - int(k / 2) + "p1为" + p1)
        if (p2 > 256 - int(k / 2)):
            p2 -= (int(k / 2) + 1)
        #     print("p1大于" + 256 - int(k / 2) + "p1为" + p1)
        D1 = 9999
        for px in range (p1 - int(k/2),p1 + int(k/2) + 2,1):
            for py in range (p2-int(k/2),p2+int(k/2) + 2,1):
                if (mat[px,py] == secret_d_array[i]):
                    D = abs(p1 - px) + abs(p2 - py)
                    if D < D1:
                        D1 = D
                        PGembedded[i,0] = px
                        PGembedded[i,1] = py
                    # print("嵌入之后的像素值", PGembedded[i])



    num_pixels_changed = numSecretGroups * n
    # 嵌入了多少秘密信息（相同的像素个数嵌入了多少秘密信息）
    # -----------------------------------------------------------------------------------
    # 恢复，提取加密数据
    recover_d_array = np.zeros((numSecretGroups), dtype="int")  # 恢复值
    for i in range(0, numSecretGroups, 1):
        recover_d_array[i] = mat[PGembedded[i,0],PGembedded[i,1]]
        # print("PGembedded[i]",PGembedded[i])
        # print("PGembedded[0]", PGembedded[i,0])
        # print("PGembedded[1]", PGembedded[i,1])
        # print("矩阵中的秘密信息", mat[PGembedded[i,0],PGembedded[i,1]])

        # print("第"+str(i)+"个恢复后的秘密信息",recover_d_array[i])
    #     # 找出pixel组中，高bits最大值的那一个pixel
    #     selIndex = i % n
    #     v1 = PGembedded[i, selIndex] % moshu0
    #     # 将v1分为n-1组，放在B1bitGroup中
    #     B1bitGroup = np.zeros(n, dtype=int)
    #     for j3 in range(0, selIndex, 1):
    #         B1bitGroup[j3] = int(v1 / (2 ** (n - j3 - 2))) & (1)
    #     for j3 in range(selIndex + 1, n, 1):
    #         B1bitGroup[j3] = int(v1 / (2 ** (n - j3 - 1))) & (1)
    #     # v2 = v1 * (2 ** ((n - 1) * k))
    #     # print("v1", v1)
    #     # print("v2", v2)
    #     # 恢复剩余的n-1 pixel
    #     tmpValue = 0
    #     for j2 in range(0, selIndex, 1):
    #         # print("第" + str(i) + "第" + str(j2) + "个PGembedded[i, j2]的值为", PGembedded[i, j2])
    #         a1 = B1bitGroup[j2]
    #         v2 = PGembedded[i, j2] % moshu1
    #         v3 = a1 * (2 ** k) + v2
    #         tmpValue += v3 * (2 ** ((n - j2 - 2) * k2))
    #     for j2 in range(selIndex + 1, n, 1):
    #         # print("第" + str(i) + "第" + str(j2) + "个PGembedded[i, j2]的值为", PGembedded[i, j2])
    #         a1 = B1bitGroup[j2]
    #         v2 = PGembedded[i, j2] % moshu1
    #         v3 = a1 * (2 ** k) + v2
    #         tmpValue += v3 * (2 ** ((n - j2 - 1) * k2))
    #     recover_d_array[i] =  tmpValue
    #     # print("recover_d_array[i]", recover_d_array[i])
    #     # print("第" + str(i) + "个recover_d_array的值为", recover_d_array[i])
    #     assert (int((recover_d_array[i] - secret_d_array[i]).sum()) == 0)
    #
    assert (int((recover_d_array - secret_d_array).sum()) == 0)
    #
    # # -----------------------------------------------------------------------------------
    # # 输出图像
    img_out = PGembedded.flatten()
    img_out = img_out[:ImageWidth * ImageHeight]  # 取前面的pixel
    # 计算PSNR
    img_array_out = img_out.copy()
    imgpsnr1 = image_array[0:num_pixels_changed]
    imgpsnr2 = img_array_out[0:num_pixels_changed]
    # psnr = PSNR(image_array,img_array_out)
    psnr = PSNR(imgpsnr1, imgpsnr2)

    # 重组图像
    img_out = img_out.reshape(ImageWidth, ImageHeight)
    img_out = Image.fromarray(img_out)
    # img_out.show()
    img_out = img_out.convert('L')

    (filepath, tempfilename) = os.path.split(image_file_name)
    (originfilename, extension) = os.path.splitext(tempfilename)

    new_file = FILE_PATH + '\\' + originfilename + '_' + sys._getframe().f_code.co_name + "_n_" + str(n) + "_k_" + str(
        k) + ".png"
    img_out.save(new_file, 'png')

    str1 = 'Image:%30s,Method:%15s,n=%d,k=%d,pixels used: %d,bpp:%.4f, PSNR: %.2f' % (
        originfilename, sys._getframe().f_code.co_name, n, k, num_pixels_changed,s_data.size/num_pixels_changed,  psnr)
    print(str1)
    SaveResult('\n' + str1)
    # -----------------------------------------------------------------------------------
    return psnr

def SB19(image_array, secret_string, k, image_file_name=''):
    # image_array:输入的一维图像数组
    # image_file_name:传入的图像文件名（带全路径）
    n = 1  # 此算法在一个像素中嵌入
    num_pixel_groups = image_array.size
    # -----------------------------------------------------------------------------------
    # 从待嵌入bit串数据中取出k个比特，作为一组
    # k = 2 #k应该是可调的
    moshu = k * k
    # 分组
    num_secret_groups = math.ceil(secret_string.size / k)
    secret_group = np.zeros((num_secret_groups, k))
    for i in range(0, num_secret_groups, 1):
        for j in range(0, k, 1):
            if (i * k + j < secret_string.size):
                secret_group[i, j] = s_data[i * k + j]

    # 一组pixels_group嵌入一组secret_group的信息，多了不能嵌入,最后一组pixel不用于嵌入以防止错误
    assert (num_pixel_groups > num_secret_groups)
    # 每一组secret_group计算得到一个d值，d为十进制的一个数
    secret_d_array = np.zeros(num_secret_groups)
    for i in range(0, num_secret_groups, 1):
        # d代表一个（2n+1）进制的一个数
        d = 0
        for j in range(0, k, 1):
            d += secret_group[i, j] * (2 ** j)  # 将secret视为低位在前
        secret_d_array[i] = d

    # -----------------------------------------------------------------------------------
    # 开始进行嵌入
    embedded_pixels_group = image_array.copy()
    pixels_group = image_array.copy()
    for i in range(0, num_secret_groups):
        x = 0
        for x in range(-1 * math.floor(moshu / 2), math.floor(moshu / 2) + 1, 1):
            f = (pixels_group[i] + x) % moshu
            if int(f) == int(secret_d_array[i]):
                if pixels_group[i] + x < 0:
                    embedded_pixels_group[i] = pixels_group[i] + x + moshu
                else:
                    embedded_pixels_group[i] = pixels_group[i] + x

                break
        tmp1 = embedded_pixels_group[i] % moshu
        tmp2 = int(secret_d_array[i])
        assert (tmp1 == tmp2)

    # -----------------------------------------------------------------------------------
    # 恢复，提取加密数据
    recover_d_array = np.zeros(num_secret_groups)
    for i in range(0, num_secret_groups):
        recover_d_array[i] = embedded_pixels_group[i] % moshu

    # 恢复出的和以前的应该是一致的
    diff_array = recover_d_array - secret_d_array
    assert (int((recover_d_array - secret_d_array).sum()) == 0)
    # 使用了多少pixel来进行嵌入
    num_pixels_changed = num_secret_groups * n
    # -----------------------------------------------------------------------------------
    # 输出图像
    img_out = embedded_pixels_group.flatten()
    img_out = img_out[:512 * 512]  # 取前面的pixel
    # 计算PSNR
    img_array_out = img_out.copy()
    #psnr = PSNR(image_array, img_array_out)
    imgpsnr1 = image_array[0:num_pixels_changed]
    imgpsnr2 = img_array_out[0:num_pixels_changed]
    psnr = PSNR(imgpsnr1, imgpsnr2)
    # print('SB19 k=%d PSNR: %.2f' % (k, psnr))
    # print('SB19 k=%d pixels used: %d' % (k, num_pixels_changed))
    # 重组图像
    img_out = img_out.reshape(512, 512)
    img_out = Image.fromarray(img_out)
    # img_out.show()
    img_out = img_out.convert('L')
    (filepath, tempfilename) = os.path.split(image_file_name)
    (originfilename, extension) = os.path.splitext(tempfilename)
    new_file = FILE_PATH + '\\' + originfilename + '_' + sys._getframe().f_code.co_name + "_n_" + str(n) + "_k_" + str(
        k) + ".png"
    img_out.save(new_file, 'png')
    str1 = 'Image:%30s,Method:%15s,n=%d,k=%d,pixels used: %d,bpp:%.4f, PSNR: %.2f' % (
        originfilename, sys._getframe().f_code.co_name, n, k, num_pixels_changed, s_data.size / num_pixels_changed,
        psnr)
    print(str1)
    SaveResult('\n' + str1)

    return psnr

def New20(image_array,secret_string,k,image_file_name=''):
    #image_array:输入的一维图像数组
    #image_file_name:传入的图像文件名（带全路径）
    n = 1 #此算法在一个像素中嵌入
    num_pixel_groups = image_array.size
    #-----------------------------------------------------------------------------------
    #从待嵌入bit串数据中取出k个比特，作为一组
    #k = 2 #k应该是可调的
    moshu = 2 ** k
    #moshu = 4 * k*k*k+10**n+6*n
    #分组
    num_secret_groups = math.ceil(secret_string.size / k)
    secret_group = np.zeros((num_secret_groups,k))
    for i in range(0,num_secret_groups,1):
        for j in range(0,k,1):
            if(i * k + j < secret_string.size):
                 secret_group[i,j] = s_data[i * k + j]

    #一组pixels_group嵌入一组secret_group的信息，多了不能嵌入,最后一组pixel不用于嵌入以防止错误
    assert(num_pixel_groups > num_secret_groups)
    #每一组secret_group计算得到一个d值，d为十进制的一个数
    secret_d_array = np.zeros(num_secret_groups)
    for i in range(0,num_secret_groups,1):
        #d代表一个（2n+1）进制的一个数
        d = 0
        for j in range(0,k,1):
            d += secret_group[i,j] * (2 ** j)  #将secret视为低位在前
        secret_d_array[i] = d

    #-----------------------------------------------------------------------------------
    #开始进行嵌入
    embedded_pixels_group = image_array.copy()
    pixels_group = image_array.copy()
    for i in range(0,num_secret_groups):
        x = 0
        for x in range(-1 * math.floor(moshu / 2),math.floor(moshu / 2) + 2,1):
            f = (pixels_group[i] + x) % moshu
            if int(f) == int(secret_d_array[i]):
                ptmp = pixels_group[i] + x
                if ptmp < 0:
                    embedded_pixels_group[i] = ptmp + moshu
                else:
                    if ptmp >255:
                        embedded_pixels_group[i] = pixels_group[i] + x - moshu
                    else:
                        embedded_pixels_group[i] = ptmp
                break
        tmp1 = embedded_pixels_group[i] % moshu
        tmp2 = int(secret_d_array[i])
        assert(tmp1 == tmp2)

    #-----------------------------------------------------------------------------------
    # 恢复，提取加密数据
    recover_d_array = np.zeros(num_secret_groups)
    for i in range(0,num_secret_groups):
        recover_d_array[i] = embedded_pixels_group[i] % moshu

    # 恢复出的和以前的应该是一致的
    diff_array = recover_d_array - secret_d_array
    assert(int((recover_d_array - secret_d_array).sum()) == 0)
    #使用了多少pixel来进行嵌入
    num_pixels_changed = num_secret_groups * n
    #-----------------------------------------------------------------------------------
    #输出图像
    img_out = embedded_pixels_group.flatten()
    img_out = img_out[:512 * 512] #取前面的pixel
    #计算PSNR
    img_array_out = img_out.copy()
    # psnr = PSNR(image_array,img_array_out)
    imgpsnr1 = image_array[0:num_pixels_changed]
    imgpsnr2 = img_array_out[0:num_pixels_changed]
    psnr = PSNR(imgpsnr1, imgpsnr2)
    # print('New20 k=%d PSNR: %.2f moshu:%d' % (k,psnr,moshu))
    # print('New20 k=%d pixels used: %d' % (k,num_pixels_changed))
    #重组图像
    img_out = img_out.reshape(512,512)
    img_out = Image.fromarray(img_out)
    #img_out.show()
    img_out = img_out.convert('L')
    (filepath,tempfilename) = os.path.split(image_file_name)
    (originfilename,extension) = os.path.splitext(tempfilename)
    new_file = FILE_PATH + '\\' + originfilename + '_' + sys._getframe().f_code.co_name + "_n_" + str(n) + "_k_" + str(
        k) + ".png"
    img_out.save(new_file, 'png')
    str1 = 'Image:%30s,Method:%15s,n=%d,k=%d,pixels used: %d,bpp:%.4f, PSNR: %.2f' % (
        originfilename, sys._getframe().f_code.co_name, n, k, num_pixels_changed, s_data.size / num_pixels_changed,
        psnr)
    print(str1)
    SaveResult('\n' + str1)

    return psnr
def RGEMD17(image_array, secret_string, n, k=99, image_file_name=''):
    # image_array:输入的一维图像数组
    # image_file_name:传入的图像文件名（带全路径）
    # n为一组像素的数量
    # k无意义，只是把函数定义为与其他函数形式一样
    # 将一个十进制数x转换为（n+1）个bit的二进制数,低位在前
    # print("secret_string",secret_string)
    def dec2bin_lower_ahead(x, n):
        b_array1 = np.zeros(n + 1)
        for i in range(0, n + 1, 1):
            b_array1[i] = int(x % 2)
            x = x // 2
        # 没有这个功能 b_array.reverse()
        # b_array2 = np.zeros(n + 1)
        # for i in range(0,n + 1,1):
        #    b_array2[i] = b_array1[n - i]
        return b_array1

    def dec2bin_higher_ahead(x, n):
        b_array2 = np.zeros(n + 1)
        for i in range(0, n + 1, 1):
            b_array2[i] = int(x % 2)
            x = x // 2
        b_array2 = b_array2[::-1]  # 使用切片反转数组
        return b_array2

    moshu = 2 ** (n + 1)  # 模数的底
    # 分成n个像素一组
    num_pixel_groups = math.ceil(image_array.size / n)
    pixels_group = np.zeros((num_pixel_groups, n))
    i = 0
    while (i < num_pixel_groups):
        for j in range(0, n):
            if (i * n + j < image_array.size):
                pixels_group[i, j] = image_array[i * n + j]
        i = i + 1
    # 每一个像素组计算出一个fGEMD值
    fGEMD_array = np.zeros(num_pixel_groups)
    for i in range(0, num_pixel_groups):
        fGEMD = 0
        for j in range(0, n):
            fGEMD += (2 ** (j + 1) - 1) * pixels_group[i, j]
        fGEMD_array[i] = fGEMD % moshu
    # -----------------------------------------------------------------------------------
    # 从待嵌入bit串数据中取出m个比特，作为一组
    m = 2 * n
    m1 = n + 1
    m2 = n - 1
    # 分组
    num_secret_groups = math.ceil(secret_string.size / m)
    secret_group = np.zeros((num_secret_groups, m))
    g_gemd = np.zeros((num_secret_groups, m1))
    g_lsb = np.zeros((num_secret_groups, m2))
    i = 0
    while (i < num_secret_groups):
        for j in range(0, m):
            if (i * m + j < s_data.size):
                secret_group[i, j] = s_data[i * m + j]
        # print("i", i)
        secret_g = secret_group[i]
        # print("secret_g", secret_group[i])
        g_gemd[i] = secret_group[i][:n + 1]
        g_lsb[i] = secret_group[i][n + 1:]
        # print("g_gemd",g_gemd[i])
        # print("g_lsb",g_lsb[i])
        i = i + 1
        # j = 2 * n * i
        # print("secret_group", secret_group[i])



    # print("secret_group", secret_group[56])
    # secret_g = secret_group[1]

    # -----------------------------------------------------------------------------------

    # 一组pixels_group嵌入一组secret_group的信息，多了不能嵌入,最后一组pixel不用于嵌入以防止错误
    assert (np.shape(secret_group)[0] <= np.shape(pixels_group)[0] - 1)
    # 每一组secret_group计算得到一个d值，d为（2n+1）进制的一个数
    d_array = np.zeros(num_secret_groups)
    s_array = np.zeros(num_secret_groups)
    for i in range(0, num_secret_groups):
        # d代表一个十进制的一个数
        d = 0
        s = 0
        for j in range(0, m1):
            d += g_gemd[i, j] * (2 ** (m1 - 1 - j))
        d_array[i] = d
        # print("d_array[i]",d_array[i])
        for j1 in range(0, m2):
            s += g_lsb[i, j1] * (2 ** (m2 - 1 - j1))
        s_array[i] = s
        # print("s_array[i]", s_array[i])
        # print("d_array[i]",d_array[i])
    # -----------------------------------------------------------------------------------
    # 开始进行嵌入
    embedded_pixels_group = pixels_group.copy()
    diff_array = np.zeros(num_secret_groups)
    for i in range(0, num_secret_groups):
        d = d_array[i]
        # print("d",d)
        fGEMD = fGEMD_array[i]
        assert (fGEMD < 33)
        diff_array[i] = int(d - fGEMD) % moshu  #diff_array[i]相当于D，（d）

    for i in range(0, num_secret_groups):
        diff = int(diff_array[i])# diff相当于D，（d）
        if (diff == 2 ** n):
            embedded_pixels_group[i, 0] = pixels_group[i, 0] + 1
            embedded_pixels_group[i, n - 1] = pixels_group[i, n - 1] + 1
        if (diff > 0) and (diff < 2 ** n):
            # 将diff转换为（n+1）个二进制数
            b_array = np.zeros(n + 1)
            b_array = dec2bin_lower_ahead(diff, n)
            for j in range(n, 0, -1):  # 倒序
                if (int(b_array[j]) == 0) and (int(b_array[j - 1]) == 1):
                    embedded_pixels_group[i, j - 1] = pixels_group[i, j - 1] + 1
                if (int(b_array[j]) == 1) and (int(b_array[j - 1]) == 0):
                    embedded_pixels_group[i, j - 1] = pixels_group[i, j - 1] - 1
        if (diff > 2 ** n) and (diff < 2 ** (n + 1)):
            # 将diff转换为（n+1）个二进制数
            b_array = np.zeros(n + 1)
            b_array = dec2bin_lower_ahead(2 ** (n + 1) - diff, n)
            for j in range(n, 0, -1):  # 倒序
                if (int(b_array[j]) == 0) and (int(b_array[j - 1]) == 1):
                    embedded_pixels_group[i, j - 1] = pixels_group[i, j - 1] - 1
                if (int(b_array[j]) == 1) and (int(b_array[j - 1]) == 0):
                    embedded_pixels_group[i, j - 1] = pixels_group[i, j - 1] + 1
    for i in range(0, num_secret_groups):
        s1 = s_array[i]
        # print("s1",s1)
        s_array1 = np.zeros(n - 1)
        s_array1 = dec2bin_lower_ahead(s1, n-2)
        # print("s_array1",s_array1)
        for j in range(n-2, -1, -1):   #倒序#为什么这个范围缩小反倒是对的，正确应该是-1
            if s_array1[j] != ( int(embedded_pixels_group[i,j+1]) & 1):
                if (embedded_pixels_group[i,j+1] >= pixels_group[i,j+1]) and (embedded_pixels_group[i,j] <= pixels_group[i,j]):
                    embedded_pixels_group[i,j+1] = embedded_pixels_group[i,j+1] - 1
                    embedded_pixels_group[i,j] = embedded_pixels_group[i,j] + 2
                    embedded_pixels_group[i, 0] = embedded_pixels_group[i, 0] + 1
                else:
                    embedded_pixels_group[i, j + 1] = embedded_pixels_group[i, j + 1] + 1
                    embedded_pixels_group[i, j ] = embedded_pixels_group[i, j] - 2
                    embedded_pixels_group[i, 0] = embedded_pixels_group[i, 0] - 1


    # -----------------------------------------------------------------------------------
    # 恢复，提取加密数据
    recover_d_array = np.zeros(num_secret_groups)
    recover_s1_array = np.zeros(num_secret_groups)
    recover_s2_array = np.zeros(num_secret_groups)
    s_array2 = np.zeros(n - 1)

    for i in range(0, num_secret_groups):
        fGEMD = 0
        for j in range(0, n):
            fGEMD += (2 ** (j + 1) - 1) * embedded_pixels_group[i, j]
        recover_s1_array[i] = fGEMD % moshu #recover_s1_array[i]为十进制数字，将其转化为二进制
        a = dec2bin_higher_ahead(recover_s1_array[i],n)
        # print("recover_m1_array",recover_s1_array[i])
        # print(a)
        # print("recover_m1_array",recover_s1_array[i])
        # recover_m1_array = dec2bin_lower_ahead(recover_s1_array[i], n )
        # print("recover_m1_array",recover_m1_array)
            # print("recover_s2_array:", value)
        s_dec = 0
        for j1 in range(n,1,-1):
            # print("embedded_pixels_group[i,j1-1]"+str(i)+str(j1),embedded_pixels_group[i,j1-1])
            s_array2[j1-2] = (int(embedded_pixels_group[i,j1-1]) & 1)
            # print("lsb" + str(i) + str(j1), (int(embedded_pixels_group[i,j1-1]) & 1))
            s_dec += (int(embedded_pixels_group[i,j1-1]) & 1) * (2 ** (j1-2))
        recover_s2_array = s_dec
        # print("lsb:i=" + str(i) + "j=" + str(j1),  recover_s2_array)
        b = dec2bin_higher_ahead(recover_s2_array,n-2)
        recover_d_array = np.concatenate((a, b))

        # print(recover_d_array)

    # 恢复出的和以前的应该是一致的
    assert (int(( recover_d_array - secret_g).sum()) == 0)
    # 使用了多少pixel来进行嵌入
    num_pixels_changed = num_secret_groups * n
    # -----------------------------------------------------------------------------------
    # 输出图像
    img_out = embedded_pixels_group.flatten()
    img_out = img_out[:512 * 512]  # 取前面的pixel
    # 计算PSNR
    img_array_out = img_out.copy()
    # psnr = PSNR(image_array,img_array_out)
    imgpsnr1 = image_array[0:num_pixels_changed]
    imgpsnr2 = img_array_out[0:num_pixels_changed]
    # print(np.size(imgpsnr1))
    # print(np.size(imgpsnr2))
    psnr = PSNR(imgpsnr1, imgpsnr2)

    # 重组图像
    img_out = img_out.reshape(512, 512)
    img_out = Image.fromarray(img_out)
    # img_out.show()
    img_out = img_out.convert('L')

    (filepath, tempfilename) = os.path.split(image_file_name)
    (originfilename, extension) = os.path.splitext(tempfilename)
    # new_file = FILE_PATH + '\\' + originfilename + "_GEMD_2013.png"
    new_file = FILE_PATH + '\\' + originfilename + '_' + sys._getframe().f_code.co_name + "_n_" + str(n) + "_k_" + str(
        k) + ".png"
    img_out.save(new_file, 'png')

    str1 = 'Image:%30s,Method:%15s,n=%d,k=%d,pixels used: %d,bpp:%.4f, PSNR: %.2f' % (
        originfilename, sys._getframe().f_code.co_name, n, k, num_pixels_changed, s_data.size / num_pixels_changed,
        psnr)
    print(str1)
    # 保存结果到文件
    SaveResult('\n' + str1)
    return 0



# 需要嵌入的信息,用整形0,1两种数值，分别表示二进制的0,1
np.random.seed(11)
kTest = 5 # k parameter value
s_data = np.random.randint(0, 2, (ImageWidth-500) * (ImageHeight) * (kTest) * 6)  # 32k,260k,1m,1.3m,1.5m
# s_data = np.random.randint(0, 2, kTest * kTest )  # 32k,260k,1m,1.3m,1.5m
print("s_data的长度为：",s_data.size)
path = os.getcwd()  # 获取当前路径
path = path + r"\OriginalPictures\%d_%d" % (ImageWidth, ImageHeight)
SaveResult('start')

for file in os.listdir(path):
    file_path = os.path.join(path, file)
    # if "Step wedge" not in file_path:
    #    continue
    # if "Tiffany.png" not in file_path:
    #    continue
    if os.path.isfile(file_path):
        print(file_path)
        # 开始仿真
        img = Image.open(file_path, "r")
        img = img.convert('L')
        # img.show()
        # SaveResult(file_path)
        # 将二维数组，转换为一维数组
        img_array1 = np.array(img)
        # print("array1行数"+str(len(img_array1))+"列数"+str(len(img_array1[0])))
        # print("img_array1.shape[0]",img_array1.shape[0])
        # print("img_array1.shape[1]]", img_array1.shape[1])
        img_array2 = img_array1.reshape(img_array1.shape[0] * img_array1.shape[1])
        # print("array2行数" + str(len(img_array2)) + "列数" + str(len(img_array2[0])))
        # print(img_array2)
        # 将二维数组，转换为一维数组
        img_array3 = img_array1.flatten()
        # print("array3行数" + str(len(img_array3)) + "列数" + str(len(img_array3[0])))
        # print(img_array3)

        # 调用函数
        # EMD06(img_array3,s_data,2,9,file_path)
        # EMD06(img_array3,s_data,4,9,file_path)
        # LWC07(img_array3,s_data,2,9,file_path)
        # GEMD13(img_array3,s_data,4,9,file_path)
        # MPM22(img_array3,s_data,2,2,file_path)

        # psnr1 = New23(img_array3, s_data, 2, kTest, file_path)
        # psnr2 = MPM22(img_array3, s_data, 2, kTest, file_path)
        # print(psnr1 - psnr2)
        # # SaveResult(str(psnr1 - psnr2))
        # psnr3 = FEMD11(img_array3, s_data, 2, 23, file_path)

        # psnr6 = RGEMD17(img_array3, s_data, 2, 3, file_path)
        # psnr5 = SB19(img_array3,s_data,2,file_path)
        # psnr4 = New20(img_array3, s_data, 4, file_path)
        # psnr2 = MPM22(img_array3, s_data, 2,4, file_path)
        psnr1 = k_aray_New23(img_array3, s_data, 2, 4, file_path)


        # print("k_aray_New23的PSNR",psnr1 )
        # print("MPM22的PSNR",psnr2)
        # SaveResult(str(psnr1 - psnr2))

        # psnr1 = New23(img_array3, s_data, 4, kTest, file_path)
        # psnr2 = MPM22(img_array3, s_data, 4, kTest, file_path)
        # print(psnr1 - psnr2)
        # # SaveResult(str(psnr1 - psnr2))
        #
        # psnr1 = New23(img_array3, s_data, 5, kTest, file_path)
        # psnr2 = MPM22(img_array3, s_data, 5, kTest, file_path)
        # print(psnr1 - psnr2)
        # SaveResult(str(psnr1 - psnr2))

        print('-----------------------')
SaveResult('end')
time.sleep(10)
