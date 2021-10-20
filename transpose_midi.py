import glob
import os
import music21
from music21 import *


# transpose each midi file to the 12 keys

for file in glob.glob('midi_files/*.mid'):
    for i in range(12):
        temp_name = file.split('\\')[-1] # get midi file name, not folder location

        # transpose midi file
        score = music21.converter.parse(file)
        newscore = score.transpose(i)
        
        # output transposed midi files
        temp_name = temp_name[:-4]
        temp_name = temp_name + '_transposed_' + str(i) + '.mid'
        newFileName = 'transposed_midi_files/' + temp_name
        newscore.write('midi', newFileName)

print('Done transposing')
