Introduction

Multimodal Genuine Emotion and Expression Detection (MGEED) is a facial emotional database, which consists of facial image sequences, depth maps, electroencephalography (EEG), optomyography (OMG) and electrocardiography (ECG) signals. Each facial image is annotated by the 9-level valence, arousal and the 6 basic emotions.


![Picture1](https://github.com/YMPort/MGEED/assets/82915545/d2889d2d-a48c-42a6-b8ea-1c0bc0735636)

-----------------------------------------------------------------

Data format

The dataset provides 17 subject folders. Each one contain 5 types of signal data. The signals are mainly stored in the “.mat” format. The users are suggested to use Matlab for data processing.
 
Image: The image sequence of a participant is stored in the “image” folder. The images are in 640x480 pixels jpg format at 30 frame per second. The timestamp each frame is stored in the variable “facedata” from the “ind.mat” file.

Depth map: The depth data is stored in the “dep.mat” file.

EMG: The EMG data is recorded in the “emteqAR.mat” file. The variable “AR_clock_time” is the starting of this EMG recording. The variable “time” is the timestamp starting at 0. Users could have the real timestamp by adding the two variables. We also provide the function “emg_AR.m” for EMG preprocessing. The clean 20-channel EMG data can be obtained by using this function. 

EEG: The EEG data is stored in “bp.csv”. We provide the function “eeg_pro.m” for extracting EEG data and timestamp.

[Notice]: ECG and GSR data are stored in the “emteqVR.mat” file. The data format and the timestamp follows the same settings as EMG. However, ECG and GSR contains too much noise. Normally, we don’t suggest users to use ECG and GSR data.


-----------------------------------------------------------------

Annotation:

There are three annotations of each frame. The annotations are stored in the variable “facedata” from the “ind.mat” file.

Valence: The value of valence is ranking from 1 to 9 representing emotional level from negative to positive.

Arousal: The value of arousal is ranking from 1 to 9 representing intensity level from calm to exciting.

Emotion: The emotion is one of the 7-class categorical affective states (1:neutral; 2:happy; 3:sad; 4:angry; 5:fear; 6:disgust and 7:surprise).

--------------------------------------------------------------
