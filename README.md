# explict-opponent-policy-reconstruction

#### 介绍 (introduction)
code-for-paper-opponent-policy-reconstruction

License: GPL


#### 使用说明 (manual)

0. platform: windows 10/11 and anaconda 

1. unzip Ccardev.zip (especially unzip the file HandRanks.dat to Ccardev)
(Note that：must be run in windows, if the libwinrate.so in Ccardev runs wrong, please install Mingw for windows; if the HandRanks.dat is broken, please unzip from the Ccardev.zip)

2. run the '.py' file to see result

3. set 1 of the switchs condition in the '.py' file to see other results.
 (for example: if we want to see figure one, set：
```
# figure 1
if 0:
	# sub figure (b) 
	#statmodeldiffattri('res-2p-ph-LA-20000.md','LA','fps',wdin=0.01)
	statmodeldiffattrinojson('dataph\\res-2p-ph-LA-20000-msg.md','LA','fps',wdin=0.01)
```
to
```
# figure 1
if 1:
	# sub figure (b) 
	#statmodeldiffattri('res-2p-ph-LA-20000.md','LA','fps',wdin=0.01)
	statmodeldiffattrinojson('dataph\\res-2p-ph-LA-20000-msg.md','LA','fps',wdin=0.01)

```
and set condition of other switchs to 0.
) 


#### data set

1.  ACPC log data:  logs_2pn_2017.tar.bz2  download from ( http://www.computerpokercompetition.org/ )

2.  other data file: please contact me (hzzmail@163.com)  





