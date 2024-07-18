import pickle
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

pkl_path = '/home/mlx/affordance_handover/test_results/mask_param/0000_00_s0.pkl'

def print_pkl(pkl_path):

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    print(data)

def compare_img(path1, path2):
    image1 = Image.open(path1)
    image2 = Image.open(path2)

    # 获取图片大小
    size1 = image1.size
    size2 = image2.size

    # 检查大小是否相同
    if size1 == size2:
        print(f"图片大小相同：{size1}")
        
        # 转换为numpy数组
        image1_array = np.array(image1)
        image2_array = np.array(image2)
        
        # 计算像素差值
        difference = np.abs(image1_array - image2_array)
        # print all none-zero values
        print(f"不同像素点数量：{np.count_nonzero(difference)}")
        
        # 打印像素差值
        # print(f"像素差值:\n{difference}")
        
        # 显示图片差值
        plt.imshow(difference)
        plt.title('Pixel Difference')
        plt.show()
    else:
        print(f"图片大小不同：image1大小为{size1}，image2大小为{size2}")

if __name__ == '__main__':
    # print_pkl(pkl_path)
    
    compare_img('/home/mlx/affordance_handover/test_results/gt/0000_00_.png', '/home/mlx/affordance_handover/test_results/inp/0000_00_.png')