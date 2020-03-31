import matplotlib.pyplot as plt
import numpy as np
import random
def show_img(pic):
    plt.imshow(pic, cmap='gray')
    plt.show()

def show_100img(array):
    temp = [i for i in range(0, 10)]
    total = []
    for i in range(0, 10):
        for j in range(0, 10):
            pic = array[random.randint(0, 4999)].reshape((20, 20)).transpose()
            if j == 0:
                temp[i] = pic
            else:
                temp[i] = np.concatenate((temp[i], pic), axis=1)
        if i == 0:
            total = temp[i]
        else:
            total = np.concatenate((total, temp[i]))
        
    show_img(total)
