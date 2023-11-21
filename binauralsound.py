
#https://www.youtube.com/watch?v=a4mpK_2koR4

import numpy as np
import matplotlib.pyplot as plt
import sys, glob
import soundfile as sf 
import sofa
import librosa
from scipy import signal

import IPython.display as ipd

source_dir = 'C:\codings\python\emsys\handclap.wav'
hrtf_dir_SOFA = 'C:\codings\python\emsys\HRTFs.zenodo\subj_A.sofa'

filenum = 'H50e056a'
filter = 'C:\codings\python\emsys\diffuse_KEMAR\elev50\H'+ filenum[1:] + '.wav'



#filter = 'C:\codings\python\emsys\diffuse_KEMAR\elev-30\H-30e030a.wav'

[HRIR, fs_H] = sf.read(filter)

print("sampling rate = "+str(fs_H))
print('Data dimensions',HRIR.shape)

# time domain visualization

# plt.plot(HRIR[:,0]) # leftchannel data
# plt.plot(HRIR[:,1])
# plt.xlabel("Time in smaples")
# plt.ylabel("Amplitude")
# plt.title('HRIR at angle: filter')
# plt.legend(['left','right'])
#plt.show()





# # Freq domain visualization

# nfft = len(HRIR)*8
# HRTF = np.fft.fft(HRIR,n=nfft, axis=0)
# HRTF_mag = (2/nfft)*np.abs(HRTF[0:int(len(HRTF)/2)+1,:])
# HRTF_mag_dB = 20*np.log10(HRTF_mag)

# f_axis = np.linspace(0,fs_H/2,len(HRTF_mag_dB))
# plt.semilogx(f_axis, HRTF_mag_dB)
# plt.title('HRTF at angle : filter')
# plt.xlabel('Frequency[Hz]')
# plt.ylabel('Magnitube[dB]')
# plt.legend(['Left','Right'])
# plt.show()




# 소스 사운드파일 불러오기

print('Source is : sourfile')
[sig, fs_s] = sf.read(source_dir)
print('Sample rate', fs_s)
print('source Data dimensions: ', sig.shape)

# 소스 사운드파일을 mono로 바꾸기

sig_mono = sig
# if sig.shape[1]>1:
#     sig_mono = np.mean(sig,axis=1)
# else:
#     sig_mono = sig

print('New data dimensions : ', sig_mono.shape)

# Listen to the mono version

#ipd.Audio(sig_mono, rate=fs_s)



# convolution 수행

s_L = np.convolve(sig_mono, HRIR[:,0])
s_R = np.convolve(sig_mono, HRIR[:,1])

Bin_Mix = np.vstack([s_L,s_R]).transpose()
print('Data Dimensions:', Bin_Mix.shape)

ipd.Audio(Bin_Mix.transpose(), rate=fs_s)

sf.write(f'Handclap_Right_{filenum}.wav', Bin_Mix, fs_s)

s2_L = np.convolve(sig_mono, HRIR[:,1])
s2_R = np.convolve(sig_mono, HRIR[:,0])
Bin2_Mix = np.vstack([s2_L,s2_R]).transpose()
sf.write(f'Handclap_Left_{filenum}.wav', Bin2_Mix, fs_s)
