import numpy as np
import cv2

img_folder = '100_RZ_3592991_Aug-24-11-44-38/'
file = open("100_RZ_3592991_Aug-24-11-44-38.txt")
fps = 20
format="XVID"
outvid='image_video.avi'
fourcc = cv2.VideoWriter_fourcc(*format)
vid = None
size = None
is_color = True

line = file.readline()
# print(line)
i = 0
for line in file:	
	contents = line.split(',')
	# if (i==0):
		# print(contents[0])
	# i = 1
	img_name = contents[0]
	episode = contents[1]
	score = contents[2]
	duration = contents[3]
	unclipped_reward = contents[4]
	action = contents[5]
	gaze = contents[6:]

	img = cv2.imread(img_folder+img_name+'.png')
	# print('img_read')

	if vid is None:
		if size is None:
			size = img.shape[1], img.shape[0]
		vid = cv2.VideoWriter(outvid, fourcc, float(fps), size, is_color)
	# if size[0] != img.shape[1] and size[1] != img.shape[0]:
	# 	img = resize(img, size)


	# overlay gaze coordinates on img
	for j in range(0,len(gaze),2):
		if('null' not in gaze[j]):
			x = float(gaze[j])
			y = float(gaze[j+1])
		cv2.circle(img, (int(x),int(y)), 5, (0,255,0), thickness=1, lineType=8, shift=0)
	vid.write(img)
vid.release()
# return vid


# TODO: show action and return on the video
# TODO: repeat frames if the duration of gaze on a frame is >1/20 seconds. Repeat it n/20 times.
# TODO: remove blinking artefacts