import numpy as np
import cv2

img_folder = '/home/akanksha/Documents/gaze_atari/hero/195_RZ_205678_Jun-28-12-09-00/'
file = open("/home/akanksha/Documents/gaze_atari/hero/195_RZ_205678_Jun-28-12-09-00.txt")
fps = 20
format="XVID"
outvid='/home/akanksha/Documents/gaze_atari/hero/195_RZ_205678_Jun-28-12-09-00.avi'
fourcc = cv2.VideoWriter_fourcc(*format)
vid = None
size = None
is_color = True


action_name = { 
'0': 'PLAYER_A_NOOP',

'1': 'PLAYER_A_FIRE',          
'2': 'PLAYER_A_UP',             
'3': 'PLAYER_A_RIGHT',          
'4': 'PLAYER_A_LEFT',           
'5': 'PLAYER_A_DOWN',          

'6': 'PLAYER_A_UPRIGHT',        
'7': 'PLAYER_A_UPLEFT',         
'8': 'PLAYER_A_DOWNRIGHT',     
'9': 'PLAYER_A_DOWNLEFT',       

'10': 'PLAYER_A_UPFIRE',        
'11': 'PLAYER_A_RIGHTFIRE',     
'12': 'PLAYER_A_LEFTFIRE',      
'13': 'PLAYER_A_DOWNFIRE',     

'14': 'PLAYER_A_UPRIGHTFIRE',   
'15': 'PLAYER_A_UPLEFTFIRE',    
'16': 'PLAYER_A_DOWNRIGHTFIRE', 
'17': 'PLAYER_A_DOWNLEFTFIRE',
'null': 'NULL'
}

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,10)
fontScale              = 0.25
fontColor              = (255,255,255)
lineType               = 1

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

		# TODO: show action and return on the video
		print(action_name[action])
		cv2.putText(img,action_name[action], 
		    bottomLeftCornerOfText, 
		    font, 
		    fontScale,
		    fontColor,
		    lineType)


	vid.write(img)
vid.release()
# return vid



# TODO: repeat frames if the duration of gaze on a frame is >1/20 seconds. Repeat it n/20 times.
# TODO: remove blinking artefacts