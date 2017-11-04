import cv2
import skvideo.io as skio

vd = skio.vread('serve.mp4')
nb_frames, _, _, _ = vd.shape
labels = []
for i in range(0, nb_frames):
	img = vd[i, :]
	img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	ngray = gray
	cv2.normalize(gray, ngray, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
	print(str(ngray.max()) + " " + str(ngray.min()))
	print(ngray.shape)
	cv2.imshow('window', ngray)
	k = cv2.waitKey(0)
	# 1048603 is Esc on this system, don't know generalization
	if k == 1048603:
		labels.append(1)
	else:
		labels.append(0)
with open('serve.labels', mode='wt', encoding='utf-8') as f:
	f.write('\n'.join(str(label) for label in labels))


