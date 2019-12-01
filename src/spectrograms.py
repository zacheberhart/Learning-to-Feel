import os, io, torch, matplotlib
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from random import random, shuffle
from time import time
from PIL import Image
from PIL.ImageOps import crop
from random import randint
from math import ceil
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import pylab, librosa
from pydub import AudioSegment
import librosa.display

class Spectrogram(object):
    
    '''Build and sample (partition) spectrograms for a given audio file for use in a CNN.
    
    General Workflow
    -----------------
     - Gather .mp3 files
     - Convert to mono-channel .wav
     - Build full spectrogram
     - Get n square partitions as needed (sampled in a sliding manner or randomly)
    '''
    
    def __init__(self, aspect_ratio:int = 3, min_dpi:int = 100):
        self.aspect_ratio = aspect_ratio # the aspect ratio for the full-size spectrogram
        self.min_dpi = min_dpi           # the min dpi for the output image (increased as necesary to fit aspect ratio)
    
    def stereo2mono(self, mp3_path:str, export_path:str):
        '''Convert a stereo .mp3 file to a mono .wav'''
        sound = AudioSegment.from_mp3(mp3_path).set_channels(1)
        sound.export(export_path, format = 'wav');
    
    def load(self, fname:str, sr:int = 44100):
        '''Get the audio time series and sampling rate for a given audio file'''
        return librosa.load(fname, sr = sr)
    
    def show(self, signal:np.ndarray, sr:int, window_sz:int, n_mels:int, hop_length:int, ref_max:bool, figsize:Tuple):
        '''Display a Mel Spectrogram for a given audio file in notebook'''
        S = librosa.feature.melspectrogram(signal, sr, n_fft = window_sz, n_mels = n_mels, hop_length = hop_length)
        plt.figure(figsize = figsize)
        librosa.display.specshow(
            librosa.power_to_db(S, ref=np.max) if ref_max else librosa.power_to_db(abs(S)),
            sr = sr, hop_length = hop_length, x_axis = 'time', y_axis = 'mel'
        )
        
    def _calc_fig_size_res(self, h:int):
        '''Helper function to calculate the new figsize and image resolution (in DPI) for a given height'''
        w = h * self.aspect_ratio
        assert w % ceil(w) == 0
        for i in range(self.min_dpi, 301):
            if w % i == 0 and h % i == 0:
                dpi = i
        figsize = (w / dpi, h / dpi)
        return figsize, dpi
        
    def export(self, export_path:str, img_height:int, signal:np.ndarray, sr:int, window_sz:int,
               n_mels:int, hop_length:int, top_db:int = 80, cmap:str = 'coolwarm', to_disk:bool = True):
        '''Export a Mel Spectrogram as a .png file for a given audio file.
        
        Parameters
        ----------
        export_path : the path and file name of the created spectrogram
        img_height  : the height of the spectrogram, the width will be calculated automatically based on `dpi`
        signal      : the audio signal
        sr          : the sample rate of the audio
        window_sz   : n_fft, the number of samples used in each Fourier Transform (the width of the window)
        n_mels      : how many mel bins are used, this will determine how many pixels tall the spectrogram is
        hop_length  : the number of samples the Fourier Transform window slides (too large: compressing data, too small: blurring)
        top_db      : distance between the loudest and softest sound displayed in spectrogram
        cmap        : the color map used for the spectrogram, default for librosa: "magma"
        '''
        
        # generate spectro
        S = librosa.power_to_db(
            abs(librosa.feature.melspectrogram(signal, sr, n_fft = window_sz,
                                               n_mels = n_mels, hop_length = hop_length)),
            top_db = top_db
        )
        
        # build fig
        self.img = None
        figsize, dpi = self._calc_fig_size_res(img_height)
        fig = plt.Figure(figsize=figsize)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.imshow(torch.from_numpy(S).flip(0), cmap = cmap)
        fig.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1)
        ax.axis('tight'); ax.axis('off')
        
        # save to disk
        if to_disk:
            fig.savefig(export_path, dpi = dpi)
        
        # keep in memory
        else:
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi = dpi)
            buf.seek(0)
            self.img = deepcopy(Image.open(buf))
            buf.close()
        
        print('Successfully built spectrogram.')
        
    
    def crop_partitions(self, fname:str, how:str, n:int = 1):
        '''Crop n square partitions from a spectrogram image file for CNN.
        "how" should be one of the following:
         - "slide" : start from the left edge and slide n times slide_window distance
         - "center": crop from the center of the spectrogram (should only be used when n=1)
         - "random": take partions n times from random places of the spectrogram
        '''
        assert how == 'slide' or how == 'center' or how == 'random'
        
        if self.img: img = deepcopy(self.img)
        else       : img = Image.open(fname)
        w, h = img.size
        
        if how == 'center':
            margin = w - ((w // 2) + (h // 2))
            crop(img, border = (margin, 0, margin, 0)).save(fname.replace('.png', '_1.png'))
        
        elif how == 'slide':
            slide_window = (w - h) // (n - 1)
            for i in range(n):
                crop(img, border = (0+(slide_window*i), 0, w-h-(slide_window*i), 0)
                    ).save(fname.replace('.png', f'_{i+1}.png'))
        
        elif how == 'random':
            for i in range(n):
                left = randint(0, w-h)
                crop(img, border = (left, 0, w-(left+h), 0)).save(fname.replace('.png', f'_{i+1}.png'))
    
    def load_and_partition(self, wav_fname:str, png_fname:str, img_sz:int, window_sz:int, n_mels:int, hop_length:int,
                           top_db:int, cmap:str, how:str, n:int, del_full_spectro:bool = False, to_disk:bool = True):
        '''Load a .wav file, generate spectrogram, and partition'''
        
        sig, sr = self.load(wav_fname)
        self.export(png_fname, img_sz, sig, sr, window_sz, n_mels, hop_length, top_db, cmap, to_disk)
        self.crop_partitions(png_fname, how, n)
        if del_full_spectro and to_disk: os.remove(png_fname)