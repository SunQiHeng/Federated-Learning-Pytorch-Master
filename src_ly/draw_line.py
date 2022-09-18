import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family'] = 'SimHei'

def draw_line():

    x = ['20','40','60','80','100']

    val1 = [40.18, 41.78 , 42.41 ,43.17,43.63]

    val2 = [36.73, 37.62 ,38.51 ,40.84 ,41.71]

    val3 = [38.82 , 38.76, 37.49 , 40.88, 40.04]

    val4 = [34.38 ,35.05 ,35.78 , 36.11, 36.32]

    val5 = [39.48 ,41.83,42.51, 42.58 , 39.60]

    l1 = plt.plot(x, val1, marker='o',markersize=6,markeredgecolor='black',color='brown', markerfacecolor='brown',label='CGFed')
    l2 = plt.plot(x, val2, marker='*',markersize=6,markeredgecolor='black',color='gold', markerfacecolor='gold',label='Fedavg')
    l3 = plt.plot(x, val3,marker='v',markersize=6,markeredgecolor='black',color='c', markerfacecolor='c',label='S-Fedavg')
    l4 = plt.plot(x, val4, marker='p',markersize=6,markeredgecolor='black',color='navy', markerfacecolor='navy',label='Fedprox')
    l5 = plt.plot(x, val5, marker='x',markersize=6,markeredgecolor='black',color='slategray', markerfacecolor='slategray',label='Scaffold')

    plt.xlabel('训练轮次')
    plt.ylabel('准确率')
    plt.legend()
    plt.savefig('MLP-CF.png', dpi=300, bbox_inches="tight")
    plt.show()

draw_line()



