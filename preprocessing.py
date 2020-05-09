from grabscreen import grab_screen
import cv2
import time
import numpy as np
from getkeys import key_check
import pandas as pd
controls=['W','S','A','D','E',' ', 'I', 'K']
control_data=np.zeros((1,len(controls)))
print(np.shape(control_data))
training_data=[]
dfcontrols=pd.DataFrame(control_data,columns=controls)
print(dfcontrols)

for i in list(range(4))[::-1]:
    print(i+1)
    time.sleep(1)

f=0
while(True):
		screen=grab_screen(region=(0,40,640,500))
		#last_time = time.time()
		screen = cv2.resize(screen, (256,200))
		screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

		#print('loop took {} seconds'.format(time.time()-last_time))
		#last_time = time.time()
		cv2.imshow('window',cv2.resize(screen,(640,500)))
		if cv2.waitKey(25) & 0xFF == ord('p'):
			cv2.destroyAllWindows()
			break

		keys=key_check()
		for i in keys:
			if i in controls:
				dfcontrols[str(i)]=1
		#output=keys_to_output(keys)
		output=dfcontrols.values.tolist()
		#print(' Output is ' + str(output[0]))
		training_data.append([screen,output[0]])
		for col in dfcontrols.columns:
			dfcontrols[col].values[:] = 0
		
		
		if len(training_data) % 100 == 0:
			print("Total length is " + str(len(training_data)))

		if len(training_data) == 10000:
			filename='data_save'+str(f)+'.npy'
			np.save(filename,training_data)
			
			f+=1
			
			print('SAVED'+ str(filename))


			training_data = []
			
		


		


