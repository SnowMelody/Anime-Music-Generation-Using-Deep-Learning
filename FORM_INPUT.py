import pickle
from natsort import os_sorted

"""
v: verse 1
v2: verse 2
p: pre-chorus 1
p2: pre-chorus 2
c: chorus 1
c2: chorus 2
ce: extension of chorus

Other form of a song, such as bridge, refrain, etc,
are not included.

Also note that in the notated pdf files,
vr refers to v2,
cr refers to c2,
c2 refers to ce (sorry for any confusion).
"""

all_form = []

# AGONY
templist = []
tempname = 'Agony'

templist = [0]*54
templist[6:14] = ['v']*8
templist[14:22] = ['v2']*8
templist[22:26] = ['p']*4
templist[26:30] = ['p2']*4
templist[30:38] = ['c']*8
templist[38:46] = ['c2']*8

#print(templist, len(templist))
all_form.append([tempname, templist])


# AKANE SASU
templist = []
tempname = 'Akane Sasu'

templist = [0]*35
templist[6:14] = ['v']*8
templist[14:18] = ['p']*4
templist[18:22] = ['p2']*4
templist[22:26] = ['c']*4
templist[26:30] = ['c2']*4
templist[30:32] = ['ce']*2

#print(templist, len(templist))
all_form.append([tempname, templist])


# AMEFURASHI NO UTA
templist = []
tempname = 'Amefurashi no Uta ~Beautiful Rain~'

templist = [0]*48
templist[8:12] = ['v']*4
templist[12:16] = ['v2']*4
templist[16:20] = ['p']*4
templist[20:25] = ['p2']*5
templist[25:33] = ['c']*8
templist[33:41] = ['c2']*8
templist[41:43] = ['ce']*2

#print(templist, len(templist))
all_form.append([tempname, templist])


# BRAVELY YOU
templist = []
tempname = 'Bravely You'

templist = [0]*55
templist[4:8] = ['v']*4
templist[8:14] = ['v2']*6
templist[14:22] = ['v']*8
templist[22:30] = ['v2']*8
templist[30:34] = ['p']*4
templist[34:38] = ['p2']*4
templist[38:46] = ['c']*8
templist[46:54] = ['c2']*8
templist[54] = 'ce'

#print(templist, len(templist))
all_form.append([tempname, templist])


# COLORFUL
templist = []
tempname = 'Colorful'

templist = [0]*57
templist[6:14] = ['v']*8
templist[14:22] = ['v2']*8
templist[22:26] = ['v']*4
templist[26:30] = ['v2']*4
templist[30:34] = ['p']*4
templist[34:38] = ['p2']*4
templist[38:46] = ['c']*8
templist[46:54] = ['c2']*8
templist[54:57] = ['ce']*3

#print(templist, len(templist))
all_form.append([tempname, templist])


# DREAM SOLISTER
templist = []
tempname = 'Dream Solister'

templist = [0]*63
templist[10:14] = ['v']*4
templist[14:18] = ['v2']*4
templist[18:22] = ['v']*4
templist[22:26] = ['v2']*4
templist[26:30] = ['p']*4
templist[30:38] = ['p2']*8
templist[38:46] = ['c']*8
templist[46:54] = ['c2']*8
templist[54:58] = ['ce']*4

#print(templist, len(templist))
all_form.append([tempname, templist])


# EBB AND FLOW
templist = []
tempname = 'Ebb and Flow'

templist = [0]*39
templist[4:12] = ['v']*8
templist[12:18] = ['v2']*6
templist[18:23] = ['p']*5
templist[23:31] = ['c']*8
templist[31:39] = ['c2']*8

#print(templist, len(templist))
all_form.append([tempname, templist])


# ETERNAL SNOW
templist = []
tempname = 'Eternal Snow'

templist = [0]*59
templist[4:12] = ['v']*8
templist[12:20] = ['v2']*8
templist[20:24] = ['p']*4
templist[24:29] = ['p2']*5
templist[29:37] = ['c']*8
templist[37:45] = ['c2']*8
templist[45:47] = ['ce']*2

#print(templist, len(templist))
all_form.append([tempname, templist])


# EVERYDAY WORLD
templist = []
tempname = 'Everyday World'

templist = [0]*163
templist[10:18] = ['v']*8
templist[18:26] = ['v2']*8
templist[26:30] = ['p']*4
templist[30:36] = ['p2']*6
templist[36:44] = ['c']*8
templist[44:52] = ['c2']*8
templist[52:54] = ['ce']*2
templist[62:70] = ['v']*8
templist[70:78] = ['v2']*8
templist[78:82] = ['p']*4
templist[82:88] = ['p2']*6
templist[88:96] = ['c']*8
templist[96:104] = ['c2']*8
templist[104:106] = ['ce']*2
templist[130:138] = ['c']*8
templist[138:146] = ['c2']*8
templist[146:148] = ['ce']*2
templist[150:154] = ['c2']*4
templist[154:156] = ['ce']*2

#print(templist, len(templist))
all_form.append([tempname, templist])


# GENTLE JENA
templist = []
tempname = 'Gentle Jena'

templist = [0]*38
templist[4:12] = ['v']*8
templist[12] = 'p'
templist[13:17] = ['c']*4
templist[17:21] = ['c2']*4
templist[21:29] = ['v']*8
templist[29] = 'p'
templist[30:34] = ['c']*4
templist[34:38] = ['c2']*4

#print(templist, len(templist))
all_form.append([tempname, templist])


# GO-TOUBUN NO KATACHI
templist = []
tempname = 'Go-toubun no Katachi'

templist = [0]*62
templist[5:13] = ['c']*8
templist[21:29] = ['v']*8
templist[29:33] = ['p']*4
templist[33:39] = ['p2']*6
templist[39:47] = ['c']*8
templist[47:55] = ['c2']*8
templist[55:57] = ['ce']*2

#print(templist, len(templist))
all_form.append([tempname, templist])


# HAREBARE FANFARE
templist = []
tempname = 'Harebare Fanfare'

templist = [0]*33
templist[0:4] = ['c']*4
templist[7:11] = ['v']*4
templist[11:15] = ['v2']*4
templist[15:17] = ['p']*2
templist[17:19] = ['p2']*2
templist[19:23] = ['c']*4
templist[23:27] = ['c2']*4

#print(templist, len(templist))
all_form.append([tempname, templist])


# IMA KOKO
templist = []
tempname = 'Ima Koko'

templist = [0]*51
templist[0:4] = ['v']*4
templist[16:20] = ['v']*4
templist[20:24] = ['v2']*4
templist[24:28] = ['p']*4
templist[28:32] = ['p2']*4
templist[32:40] = ['c']*8
templist[40:48] = ['c2']*8

#print(templist, len(templist))
all_form.append([tempname, templist])


# INNOCENCE
templist = []
tempname = 'Innocence'

templist = [0]*34
templist[1:5] = ['c']*4
templist[9:13] = ['v']*4
templist[13:17] = ['v2']*4
templist[17:19] = ['p']*2
templist[19:21] = ['p2']*2
templist[21:25] = ['c']*4
templist[25:29] = ['c2']*4
templist[29:32] = ['ce']*3

#print(templist, len(templist))
all_form.append([tempname, templist])


# KAWAKI WO AMEKU
templist = []
tempname = 'Kawaki wo Ameku'

templist = [0]*128
templist[8:16] = ['v']*8
templist[16:24] = ['v2']*8
templist[24:28] = ['p2']*4
templist[28:36] = ['c']*8
templist[36:42] = ['c2']*6
templist[50:58] = ['v']*8
templist[58:62] = ['p']*4
templist[62:66] = ['p2']*4
templist[66:74] = ['c']*8
templist[74:80] = ['c2']*6
templist[92:100] = ['c']*8
templist[100:106] = ['c2']*6

#print(templist, len(templist))
all_form.append([tempname, templist])


# LEVEL 5 - JUDGELIGHT
templist = []
tempname = 'Level 5 - Judgelight'

templist = [0]*52
templist[8:16] = ['c']*8
templist[17:25] = ['v']*8
templist[25:29] = ['p']*4
templist[29:33] = ['p2']*4
templist[33:41] = ['c']*8
templist[41:49] = ['c2']*8
templist[49:51] = ['ce']*2

#print(templist, len(templist))
all_form.append([tempname, templist])


# METAMERISM
templist = []
tempname = 'Metamerism'

templist = [0]*33
templist[1:5] = ['c2']*4
templist[9:13] = ['v']*4
templist[13:17] = ['v2']*4
templist[17:19] = ['p']*2
templist[19:21] = ['p2']*2
templist[21:25] = ['c']*4
templist[25:29] = ['c2']*4

#print(templist, len(templist))
all_form.append([tempname, templist])


# NANAIRO BIYORI
templist = []
tempname = 'Nanairo Biyori'

templist = [0]*63
templist[12:20] = ['v']*8
templist[20:28] = ['v2']*8
templist[28:32] = ['p']*4
templist[32:36] = ['p2']*4
templist[36:44] = ['c']*8
templist[44:50] = ['c2']*6

#print(templist, len(templist))
all_form.append([tempname, templist])


# NEE
templist = []
tempname = 'Nee'

templist = [0]*36
templist[0:5] = ['c2']*5
templist[7:11] = ['v']*4
templist[11:15] = ['v2']*4
templist[15:19] = ['p']*4
templist[19:22] = ['p2']*3
templist[22:26] = ['c']*4
templist[26:30] = ['c2']*4
templist[30:34] = ['ce']*4

#print(templist, len(templist))
all_form.append([tempname, templist])


# OVER AND OVER
templist = []
tempname = 'Over and Over'

templist = [0]*35
templist[1:3] = ['c']*2
templist[9:13] = ['v']*4
templist[13:17] = ['v2']*4
templist[17:21] = ['p']*4
templist[21:24] = ['p2']*3
templist[24:28] = ['c']*4
templist[28:32] = ['c2']*4
templist[32:35] = ['ce']*3

#print(templist, len(templist))
all_form.append([tempname, templist])


# ROLLING GIRL
templist = []
tempname = 'Rolling Girl'

templist = [0]*76
templist[12:20] = ['v']*8
templist[24:32] = ['v2']*8
templist[32:36] = ['c']*4
templist[36:40] = ['c2']*4
templist[44:52] = ['v']*8
templist[52:60] = ['v2']*8
templist[60:64] = ['c']*4
templist[64:68] = ['c2']*4
templist[68:72] = ['c']*4
templist[72:76] = ['c2']*4

#print(templist, len(templist))
all_form.append([tempname, templist])


# SARISHINOHARA
templist = []
tempname = 'Sarishinohara'

templist = [0]*150
templist[12:20] = ['v']*8
templist[20:28] = ['v2']*8
templist[28:30] = ['p']*2
templist[30:38] = ['c']*8
templist[38:46] = ['c2']*8
templist[48:56] = ['v']*8
templist[56:64] = ['v2']*8
templist[64:66] = ['p']*2
templist[66:74] = ['c']*8
templist[74:82] = ['c2']*8
templist[110:118] = ['c']*8
templist[118:124] = ['c2']*6
templist[124:132] = ['c2']*8
templist[132:136] = ['ce']*4

#print(templist, len(templist))
all_form.append([tempname, templist])


# SASAYAKA NA INORI
templist = []
tempname = 'Sasayaka na Inori'

templist = [0]*95
templist[5:9] = ['v']*4
templist[9:13] = ['v2']*4
templist[13:17] = ['p']*4
templist[17:22] = ['p2']*5
templist[22:26] = ['c']*4
templist[26:30] = ['c2']*4
templist[30:32] = ['ce']*2
templist[33:37] = ['v']*4
templist[37:41] = ['v2']*4
templist[41:45] = ['p']*4
templist[45:50] = ['p2']*5
templist[50:54] = ['c']*4
templist[54:58] = ['c2']*4
templist[58:60] = ['ce']*2
templist[77:81] = ['c']*4
templist[81:85] = ['c2']*4
templist[85:88] = ['ce']*3

#print(templist, len(templist))
all_form.append([tempname, templist])


# SHIRUSHI
templist = []
tempname = 'Shirushi'

templist = [0]*52
templist[10:18] = ['v']*8
templist[18:26] = ['v2']*8
templist[26:30] = ['p']*4
templist[30:35] = ['p2']*5
templist[35:43] = ['c']*8
templist[43:51] = ['c2']*8

#print(templist, len(templist))
all_form.append([tempname, templist])


# SING MY PLEASURE
templist = []
tempname = 'Sing My Pleasure'

templist = [0]*71
templist[12:20] = ['v']*8
templist[20:28] = ['v2']*8
templist[28:32] = ['p']*4
templist[32:38] = ['p2']*6
templist[38:46] = ['c']*8
templist[46:54] = ['c2']*8
templist[54:62] = ['ce']*8

#print(templist, len(templist))
all_form.append([tempname, templist])


# SORA NO TAMOTO
templist = []
tempname = 'Sora no Tamoto'

templist = [0]*93
templist[0:8] = ['v']*8
templist[8:16] = ['v2']*8
templist[16:20] = ['p']*4
templist[20:24] = ['c']*4
templist[24:28] = ['c2']*4
templist[32:40] = ['v']*8
templist[40:44] = ['p']*4
templist[44:48] = ['c']*4
templist[48:52] = ['c2']*4
templist[52:56] = ['c']*4
templist[56:60] = ['c2']*4
templist[68:72] = ['c']*4
templist[72:76] = ['c2']*4

#print(templist, len(templist))
all_form.append([tempname, templist])


# STAY ALIVE
templist = []
tempname = 'Stay Alive'

templist = [0]*30
templist[8:12] = ['v']*4
templist[12:16] = ['v2']*4
templist[16:18] = ['p']*2
templist[18:22] = ['c']*4
templist[22:26] = ['c2']*4

#print(templist, len(templist))
all_form.append([tempname, templist])


# THIS GAME
templist = []
tempname = 'This Game'

templist = [0]*73
templist[26:30] = ['v']*4
templist[30:34] = ['v2']*4
templist[34:38] = ['v']*4
templist[38:42] = ['v2']*4
templist[42:46] = ['p']*4
templist[46:51] = ['p2']*5
templist[51:59] = ['c']*8
templist[59:67] = ['c2']*8
templist[67:69] = ['ce']*2

#print(templist, len(templist))
all_form.append([tempname, templist])


# UCHIAGE HANABI
templist = []
tempname = 'Uchiage Hanabi'

templist = [0]*112
templist[8:12] = ['v']*4
templist[12:16] = ['v2']*4
templist[16:20] = ['p']*4
templist[20:24] = ['p2']*4
templist[24:28] = ['c']*4
templist[28:32] = ['c2']*4
templist[36:40] = ['v']*4
templist[40:44] = ['v2']*4
templist[44:48] = ['p']*4
templist[48:55] = ['p2']*7
templist[55:59] = ['c']*4
templist[59:63] = ['c2']*4
templist[75:79] = ['v']*4
templist[79:83] = ['v2']*4
templist[83:87] = ['c']*4
templist[87:91] = ['c2']*4

#print(templist, len(templist))
all_form.append([tempname, templist])


# YOUTHFUL BEAUTIFUL
templist = []
tempname = 'Youthful Beautiful'

templist = [0]*31
templist[4:8] = ['v']*4
templist[8:12] = ['v2']*4
templist[12:14] = ['v']*2
templist[14:16] = ['v2']*2
templist[16:18] = ['p']*2
templist[18:20] = ['p2']*2
templist[20:24] = ['c']*4
templist[24:28] = ['c2']*4
templist[28] = 'ce'

#print(templist, len(templist))
all_form.append([tempname, templist])


# YOZORA
templist = []
tempname = 'Yozora'

templist = [0]*36
templist[5:13] = ['v']*8
templist[13:15] = ['p']*2
templist[15:18] = ['p2']*3
templist[18:26] = ['c']*8
templist[26:34] = ['c2']*8

#print(templist, len(templist))
all_form.append([tempname, templist])


# ZANKOKU NA YUME TO NEMURE
templist = []
tempname = 'Zankoku na Yume to Nemure'

templist = [0]*68
templist[8:16] = ['v']*8
templist[16:24] = ['v2']*8
templist[24:32] = ['v']*8
templist[32:36] = ['p']*4
templist[36:41] = ['p2']*5
templist[41:49] = ['c']*8
templist[49:57] = ['c2']*8
templist[57:68] = ['ce']*11

#print(templist, len(templist))
all_form.append([tempname, templist])


# ZZZ
templist = []
tempname = 'ZZZ'

templist = [0]*57
templist[0:4] = ['v2']*4
templist[6:14] = ['v']*8
templist[14:22] = ['v2']*8
templist[22:26] = ['p']*4
templist[26:30] = ['p2']*4
templist[30:38] = ['c']*8
templist[38:41] = ['ce']*3
templist[42:50] = ['v2']*8
templist[50:54] = ['v2']*4

#print(templist, len(templist))
all_form.append([tempname, templist])


# SORT AND CHECK
all_form = os_sorted(all_form)
print(all_form, len(all_form))

for i in all_form:
    print(i[0])

saveObject = all_form

with open('current_form', 'wb') as f:
    pickle.dump(saveObject, f)

print('Done saving')