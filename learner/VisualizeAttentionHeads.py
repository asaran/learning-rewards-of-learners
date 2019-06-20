import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import sys
import seaborn as sns

frame_num = int(sys.argv[1])
rollout_num = frame_num // 35
frame_id = frame_num % 35
plt.axis('off')
ax = plt.subplot(3, 3, 1)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.set_aspect('equal')
if frame_num == 1:
    img=mpimg.imread('viz-SA/heromax_frame.png')
elif frame_num==2:
    img=mpimg.imread('viz-SA/heromin_frame.png')
ax.imshow(img)

base = (0.5,0.5)


for i in range(8):
    ax = plt.subplot(3, 3, i+2)
    a = np.zeros((16, 8))

    file_name = "viz-SA/heads/f"+str(frame_num)+"_h"+str(i)+".txt"
    print(file_name)
    fh = open(file_name, 'r')
    arrows = []
    for line in fh:
        line = line.strip()
        data = eval(line)
        a[data[1][0]][data[1][1]] += data[2]
        if len(arrows) < 10:
            arrows.append(( data[0][1], data[0][0], data[1][1], data[1][0]))


    ax = sns.heatmap(a, cmap="YlGnBu", linewidth=0.1, linecolor ='k',xticklabels=False, yticklabels=False,cbar=False)
    for arrow in arrows:
        ax.annotate("", xy=(base[0]+arrow[0],base[0]+arrow[1]), xytext=(arrow[2]+base[0],arrow[3]+base[1]), arrowprops=dict(arrowstyle="<-"),color='white')


plt.subplots_adjust(wspace=0.1, hspace=0.1)
# plt.show()
plt.savefig('viz-SA/att_heads.png', bbox_inches='tight')
