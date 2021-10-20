from music21 import *
from fractions import Fraction
import math
import numpy as np
import glob
import tensorflow as tf
import pickle
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from ordered_set import OrderedSet


def get_midi_files(folder='transposed_midi_files/*.mid'):
	list_all_midi = glob.glob(folder)
	return list_all_midi


def get_notes(list_all_midi):
	all_items = []

	for midi in list_all_midi:
		items = []
		print("Parsing %s" % midi)
		song = converter.parse(midi)
		song = stream.Stream(song)

		for item in song.recurse().notes:
			items.append([item.offset, item.pitches, item.duration.quarterLength])

		items = sorted(items, key=lambda x: x[0])
		all_items.append(items)

	return all_items


def process_part(part_to_process, note_info, offset):
	part = list(part_to_process)

	# remove additional holds
	if len(part) > offset*4:
		part = part[0:int(offset*4)]

	#part.append(note_info[2]) # temporary, get note pitch
	part.append(note_info[0]) # get midi value 

	# round float/fraction to nearest 0.25 duration (ceiling)
	if isinstance(note_info[1], float):
		part += ['H']*(math.ceil(note_info[1]/0.25) - 1)

	elif isinstance(note_info[1], Fraction):
		part += ['H']*(math.ceil(note_info[1]/0.25) - 1)

	else:
		print('Error converting duration value.')

	return part


def get_four_parts(all_items):
	all_melody = []
	all_rmid = []
	all_lmid = []
	all_bass = []

	# iterate each song 
	for items in all_items:
		offset = 0
		final_offset = items[-1][0]
		ind = 0
		melody = []
		right_mid = []
		left_mid = []
		bass = []

		# search offset by increments of 0.25 till end of song
		while offset <= final_offset:
			templist = []
			templist2 = []

			# add rest where notes are not present
			if len(melody) < offset*4:
				melody.append('R')

			if len(right_mid) < offset*4:
				right_mid.append('R')

			if len(left_mid) < offset*4:
				left_mid.append('R')

			if len(bass) < offset*4:
				bass.append('R')
			
			# get list of notes that matches offset
			while ind < len(items) and items[ind][0] == offset:
				templist.append(items[ind])
				ind += 1
				
			# get midi values equal to note pitches in list of notes
			if templist:
				for i in range(len(templist)):
					for p in templist[i][1]:
						templist2.append([p.midi, templist[i][2], str(p)]) # [midi value, note duration, note pitch]

				templist2.sort()

				# if 1 note, put to melody or bass part
				if len(templist2) == 1:
					if templist2[0][0] >= 60:
						melody = process_part(melody, templist2[0], offset)

					else:
						bass = process_part(bass, templist2[0], offset)

				# if 2 notes, put to melody and bass part
				elif len(templist2) == 2:
					melody = process_part(melody, templist2[-1], offset)
					bass = process_part(bass, templist2[0], offset)

				# if 3 notes, put middle note to right_mid or left_mid part
				elif len(templist2) == 3:
					melody = process_part(melody, templist2[-1], offset)
					bass = process_part(bass, templist2[0], offset)

					if templist2[1][0] >= 60:
						right_mid = process_part(right_mid, templist2[1], offset)

					else:
						left_mid = process_part(left_mid, templist2[1], offset)

				# if 4 or more notes, split to 4 parts
				else:
					melody = process_part(melody, templist2[-1], offset)
					right_mid = process_part(right_mid, templist2[-2], offset)
					bass = process_part(bass, templist2[0], offset)
					left_mid = process_part(left_mid, templist2[1], offset)

			# ignore indexes in song where offset is not in increment of 0.25
			while ind < len(items) and (float(items[ind][0]/0.25) != int(items[ind][0]/0.25)):
				ind += 1

			offset += 0.25

		# fill final bar with rests so that it's a full bar length
		if len(melody)%16 != 0:
			melody += ['R']*(16 - (len(melody)%16))

		if len(right_mid)%16 != 0:
			right_mid += ['R']*(16 - (len(right_mid)%16))

		if len(left_mid)%16 != 0:
			left_mid += ['R']*(16 - (len(left_mid)%16))

		if len(bass)%16 != 0:
			bass += ['R']*(16 - (len(bass)%16))

		# finally compare 4 parts, fill with rests if needed to make all parts equal length
		final_length = max(len(melody), len(right_mid), len(left_mid), len(bass))
		melody += ['R']*(final_length - len(melody))
		right_mid += ['R']*(final_length - len(right_mid))
		left_mid += ['R']*(final_length - len(left_mid))
		bass += ['R']*(final_length - len(bass))

		# append to respective list
		all_melody.append(melody)
		all_rmid.append(right_mid)
		all_lmid.append(left_mid)
		all_bass.append(bass)

	return all_melody, all_rmid, all_lmid, all_bass


def item_to_int(item):
	return dict(zip(item, list(range(0, len(item)))))


def int_to_item(item_dict):
	return {v: k for k, v in item_dict.items()}


def get_training_sequences(all_part, item_dict, seq_len):
	train_list = []
	target_list = []

	for s in range(len(all_part)):
		part_list = [item_dict[k] for k in all_part[s]]

		for i in range(len(part_list) - seq_len):
			train_list.append(part_list[i: i+seq_len])
			target_list.append(part_list[i+seq_len])

	return train_list, target_list


def get_bar_count(all_part):
	all_counter = []

	for part in all_part:
		if len(part)%16 == 0:
			counter = [i for i in range(0, 16)]*int(len(part)/16)

		else:
			print('Error with total bar length.')

		all_counter.append(counter)

	return all_counter


def get_motifs(all_part, i_size=16, seq_len=64, max_count=3):
	all_motifs = []

	for part in all_part:
		l = part
		l2 = [0]*len(l)
		i, j = 0, 0

		while i < len(l) - i_size:
			count = 0

			if len(l[i: i+i_size]) != (l[i: i+i_size].count('H') + l[i: i+i_size].count('R')):
				for n in range(seq_len//i_size):
					if n == 0:
						continue

					else:
						if l[i: i+i_size] == l[i + (i_size*n): i+ (i_size*(n+1))]:
							count += 1

							if count >= max_count:
								break

							if count == 1:
								l2[i: i+i_size] = [count]*i_size

							l2[i + (i_size*n): i+ (i_size*(n+1))] = [count+1]*i_size
							j = i + (i_size*n)

			i = j + i_size
			j = i

		all_motifs.append(l2)

	# check for length mismatch
	for i in range(len(all_part)):
		#print(len(all_part[i]))
		#print(len(all_motifs[i]))

		if len(all_part[i]) != len(all_motifs[i]):
			print('Length mismatch in motif.')

	return all_motifs


def get_form(list_all_midi, form_list, all_part):
	all_form = []

	# check if form sorted order same as midi files 
	'''
	for i in range(len(list_all_midi)):
		print(list_all_midi[i])
		print(form_list[i][0])
	'''

	# make form list the same length as note/melody list
	for i in range(len(form_list)):
		templist = [s for item in form_list[i][1] for s in [item]*16]
		#print(len(templist))

		templist += [templist[-1]]*(len(all_part[i*12]) - len(templist))
		#templist += [templist[-1]]*(len(all_part[i]) - len(templist))
		#print(len(all_part[i]), len(templist))
		
		# check for length mismatch
		if len(all_part[i*12]) != len(templist):
			print(len(all_part[i*12]), len(templist))
			print('Length mismatch in form for', list_all_midi[i*12])

		# append 12 times to cater to transposed midi files in 12 keys
		for j in range(12):
			all_form.append(templist)

	return all_form


# PART 1: preprocessing
# get pickled file for form
with open('current_form', 'rb') as f:
	fileout = pickle.load(f)

form_list = fileout

# load pickled file for 4 parts, else preprocess and save it
try:
	with open('current_parts', 'rb') as f:
		fileout = pickle.load(f)

	all_melody, all_rmid, all_lmid, all_bass = fileout
	print('Done loading 4 parts')

except:
	all_items = get_notes(get_midi_files())
	all_melody, all_rmid, all_lmid, all_bass = get_four_parts(all_items)
	saveObject = (all_melody, all_rmid, all_lmid, all_bass)

	with open('current_parts', 'wb') as f:
		pickle.dump(saveObject, f)

	print('Done saving 4 parts')

	file=open("output_notes.txt", "w")

	for score in all_items:
		for item in score:
			print(item, file=open("output_notes.txt", "a"))
		print(file=open("output_notes.txt", "a"))

# preprocess counter, motifs and form
all_counter = get_bar_count(all_melody)
all_melody_motifs = get_motifs(all_melody)
all_form = get_form(get_midi_files(), form_list, all_melody)
print('No issues for prepocessing')

# dict to store unique values and vice versa for each list
unique_melody = list(OrderedSet([i for s in all_melody for i in s]))
unique_rmid = list(OrderedSet([i for s in all_rmid for i in s]))
unique_lmid = list(OrderedSet([i for s in all_lmid for i in s]))
unique_bass = list(OrderedSet([i for s in all_bass for i in s]))
unique_counter = list(OrderedSet([i for s in all_counter for i in s]))
unique_melody_motifs = list(OrderedSet([i for s in all_melody_motifs for i in s]))
unique_form = list(OrderedSet([i for s in all_form for i in s]))

melody_to_int = item_to_int(unique_melody)
int_to_melody = int_to_item(melody_to_int)
rmid_to_int = item_to_int(unique_rmid)
int_to_rmid = int_to_item(rmid_to_int)
lmid_to_int = item_to_int(unique_lmid)
int_to_lmid = int_to_item(lmid_to_int)
bass_to_int = item_to_int(unique_bass)
int_to_bass = int_to_item(bass_to_int)
counter_to_int = item_to_int(unique_counter)
int_to_counter = int_to_item(counter_to_int)
melody_motifs_to_int = item_to_int(unique_melody_motifs)
int_to_melody_motifs = int_to_item(melody_motifs_to_int)
form_to_int = item_to_int(unique_form)
int_to_form = int_to_item(form_to_int)
 
# prepare sequences for training
SEQ_LEN = 128
input_melody, output_melody = get_training_sequences(all_melody, melody_to_int, SEQ_LEN)
input_rmid, output_rmid = get_training_sequences(all_rmid, rmid_to_int, SEQ_LEN)
input_lmid, output_lmid = get_training_sequences(all_lmid, lmid_to_int, SEQ_LEN)
input_bass, output_bass = get_training_sequences(all_bass, bass_to_int, SEQ_LEN)
input_counter, output_counter = get_training_sequences(all_counter, counter_to_int, SEQ_LEN)
input_motifs, output_motifs = get_training_sequences(all_melody_motifs, melody_motifs_to_int, SEQ_LEN)
input_form, output_form = get_training_sequences(all_form, form_to_int, SEQ_LEN)

# shuffle and split sequences into train/test sets
(train_input_melody, test_input_melody, train_output_melody, test_output_melody,
	train_input_rmid, test_input_rmid, train_output_rmid, test_output_rmid,
	train_input_lmid, test_input_lmid, train_output_lmid, test_output_lmid,
	train_input_bass, test_input_bass, train_output_bass, test_output_bass,
	train_input_counter, test_input_counter, train_output_counter, test_output_counter,
	train_input_motifs, test_input_motifs, train_output_motifs, test_output_motifs,
	train_input_form, test_input_form, train_output_form, test_output_form
	) = train_test_split(input_melody, output_melody,
						input_rmid, output_rmid,
						input_lmid, output_lmid,
						input_bass, output_bass,
						input_counter, output_counter,
						input_motifs, output_motifs,
						input_form, output_form,
						test_size = 0.1,
						random_state = 1996)

# check if all sequences lengths are the same
print(len(input_melody), len(output_melody))
print(len(input_rmid), len(output_rmid))
print(len(input_lmid), len(output_lmid))
print(len(input_bass), len(output_bass))
print(len(input_counter), len(output_counter))
print(len(input_motifs), len(output_motifs))
print(len(input_form), len(output_form))

# verify if it works
file=open("output_melody.txt", "w")
file=open("output_rmid.txt", "w")
file=open("output_lmid.txt", "w")
file=open("output_bass.txt", "w")

for song in all_melody:
	print(len(song), file=open("output_melody.txt", "a"))
	print(song, file=open("output_melody.txt", "a"))
	print(file=open("output_melody.txt", "a"))

for song in all_rmid:
	print(len(song), file=open("output_rmid.txt", "a"))
	print(song, file=open("output_rmid.txt", "a"))
	print(file=open("output_rmid.txt", "a"))

for song in all_lmid:
	print(len(song), file=open("output_lmid.txt", "a"))
	print(song, file=open("output_lmid.txt", "a"))
	print(file=open("output_lmid.txt", "a"))

for song in all_bass:
	print(len(song), file=open("output_bass.txt", "a"))
	print(song, file=open("output_bass.txt", "a"))
	print(file=open("output_bass.txt", "a"))

'''
print(len(unique_melody))
print(melody_to_int)
print(int_to_melody)
print(len(unique_rmid))
print(rmid_to_int)
print(int_to_rmid)
print(len(unique_lmid))
print(lmid_to_int)
print(int_to_lmid)
print(len(unique_bass))
print(bass_to_int)
print(int_to_bass)
print(len(unique_counter))
print(counter_to_int)
print(int_to_counter)
'''

# class for attention layer
class Attention(Layer):
	def __init__(self, return_sequences=True):
		self.return_sequences = return_sequences
		super(Attention,self).__init__()
		
	def build(self, input_shape):
		
		self.W=self.add_weight(name="att_weight", shape=(input_shape[-1],1),
							   initializer="normal")
		self.b=self.add_weight(name="att_bias", shape=(input_shape[1],1),
							   initializer="zeros")
		
		super(Attention,self).build(input_shape)
		
	def call(self, x):
		
		e = K.tanh(K.dot(x,self.W)+self.b)
		a = K.softmax(e, axis=1)
		output = x*a
		
		if self.return_sequences:
			return output

		return K.sum(output, axis=1)


# PART 2: build model architecture
num_melody = len(unique_melody)
train_input_melody = np.reshape(train_input_melody, (len(train_input_melody), SEQ_LEN))
train_output_melody = np.reshape(train_output_melody, (len(train_output_melody), 1))
test_input_melody = np.reshape(test_input_melody, (len(test_input_melody), SEQ_LEN))
test_output_melody = np.reshape(test_output_melody, (len(test_output_melody), 1))

num_rmid = len(unique_rmid)
train_input_rmid = np.reshape(train_input_rmid , (len(train_input_rmid ), SEQ_LEN))
train_output_rmid = np.reshape(train_output_rmid, (len(train_output_rmid), 1))
test_input_rmid = np.reshape(test_input_rmid, (len(test_input_rmid), SEQ_LEN))
test_output_rmid = np.reshape(test_output_rmid, (len(test_output_rmid), 1))

num_lmid = len(unique_lmid)
train_input_lmid = np.reshape(train_input_lmid, (len(train_input_lmid), SEQ_LEN))
train_output_lmid = np.reshape(train_output_lmid, (len(train_output_lmid), 1))
test_input_lmid = np.reshape(test_input_lmid, (len(test_input_lmid), SEQ_LEN))
test_output_lmid = np.reshape(test_output_lmid, (len(test_output_lmid), 1))

num_bass = len(unique_bass)
train_input_bass = np.reshape(train_input_bass, (len(train_input_bass), SEQ_LEN))
train_output_bass = np.reshape(train_output_bass, (len(train_output_bass), 1))
test_input_bass = np.reshape(test_input_bass, (len(test_input_bass), SEQ_LEN))
test_output_bass = np.reshape(test_output_bass, (len(test_output_bass), 1))

num_counter = len(unique_counter)
train_input_counter = np.reshape(train_input_counter, (len(train_input_counter), SEQ_LEN))
test_input_counter = np.reshape(test_input_counter, (len(test_input_counter), SEQ_LEN))

num_motifs = len(unique_melody_motifs)
train_input_motifs = np.reshape(train_input_motifs, (len(train_input_motifs), SEQ_LEN))
test_input_motifs = np.reshape(test_input_motifs, (len(test_input_motifs), SEQ_LEN))

num_form = len(unique_form)
train_input_form = np.reshape(train_input_form, (len(train_input_form), SEQ_LEN))
test_input_form = np.reshape(test_input_form, (len(test_input_form), SEQ_LEN))

print(len(unique_melody), train_input_melody.shape, train_output_melody.shape)
print(len(unique_rmid), train_input_rmid.shape, train_output_rmid.shape)
print(len(unique_lmid), train_input_lmid.shape, train_output_lmid.shape)
print(len(unique_bass), train_input_bass.shape, train_output_bass.shape)
print(len(unique_counter), train_input_counter.shape)
print(len(unique_melody_motifs), train_input_motifs.shape)
print(len(unique_form), train_input_form.shape)


def create_model(num_melody, num_rmid, num_lmid, num_bass, seq_len, dropout):
	melody_input = tf.keras.layers.Input(shape = (seq_len,))
	rmid_input = tf.keras.layers.Input(shape = (seq_len,))
	lmid_input = tf.keras.layers.Input(shape = (seq_len,))
	bass_input = tf.keras.layers.Input(shape = (seq_len,))

	melody_embedding = tf.keras.layers.Embedding(input_dim = num_melody, output_dim = 32)(melody_input)
	rmid_embedding = tf.keras.layers.Embedding(input_dim = num_rmid, output_dim = 32)(rmid_input)
	lmid_embedding = tf.keras.layers.Embedding(input_dim = num_lmid, output_dim = 32)(lmid_input)
	bass_embedding = tf.keras.layers.Embedding(input_dim = num_bass, output_dim = 32)(bass_input)
	merge_layer = tf.keras.layers.Concatenate()([melody_embedding, rmid_embedding, lmid_embedding, bass_embedding])

	bilstm_layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences = True))(merge_layer)
	attention_layer = Attention(return_sequences = True)(bilstm_layer)
	batch_norm = tf.keras.layers.BatchNormalization()(attention_layer)
	dropout_layer = tf.keras.layers.Dropout(dropout)(batch_norm)
	bilstm_layer2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences = True))(dropout_layer)
	attention_layer2 = Attention(return_sequences = False)(bilstm_layer2)
	batch_norm2 = tf.keras.layers.BatchNormalization()(attention_layer2)
	dropout_layer2 = tf.keras.layers.Dropout(dropout)(batch_norm2)
	dense_layer = tf.keras.layers.Dense(64, activation = 'relu')(dropout_layer2)

	melody_output = tf.keras.layers.Dense(num_melody, activation = 'softmax')(dense_layer)
	rmid_output = tf.keras.layers.Dense(num_rmid, activation = 'softmax')(dense_layer)
	lmid_output = tf.keras.layers.Dense(num_lmid, activation = 'softmax')(dense_layer)
	bass_output = tf.keras.layers.Dense(num_bass, activation = 'softmax')(dense_layer)
	model = tf.keras.Model(inputs = [melody_input, rmid_input, lmid_input, bass_input], 
							outputs = [melody_output, rmid_output, lmid_output, bass_output])

	return model


def train_model(model, training_input, training_output, test_input, test_output):
	checkpoint_path = "training_checkpoints_model2/weights-{epoch:02d}-{loss:.4f}.ckpt"
	checkpoint_dir = os.path.dirname(checkpoint_path)
	os.listdir(checkpoint_dir)
	latest = tf.train.latest_checkpoint(checkpoint_dir)
	print(latest)

	# resume training using weights from last epoch
	if latest != None:
		model.load_weights(latest)

	cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath = checkpoint_path, save_weights_only = True, verbose = 1) 
	history = model.fit(training_input, training_output,
						validation_data = (test_input, test_output), 
						epochs = 50, batch_size = 128, callbacks = [cp_callback], verbose = 1)

	saveObject = history.history

	with open('model2_history', 'wb') as f:
	    pickle.dump(saveObject, f)

	print('Done saving model history')
 
	print(history.history.keys())
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('Model loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Training loss', 'Test loss'], loc='upper left')
	plt.show()

	plt.plot(history.history['dense_1_sparse_categorical_accuracy'])
	plt.plot(history.history['val_dense_1_sparse_categorical_accuracy'])
	plt.plot(history.history['dense_4_sparse_categorical_accuracy'])
	plt.plot(history.history['val_dense_4_sparse_categorical_accuracy'])
	plt.title('Model accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['Melody training acc', 'Melody test acc', 'Bass training acc', 'Bass test acc'], loc='upper left')
	plt.show()

	return
	

model = create_model(num_melody, num_rmid, num_lmid, num_bass, SEQ_LEN, dropout = 0.3)
model.compile(loss = 'sparse_categorical_crossentropy', metrics = ['sparse_categorical_accuracy'], optimizer = 'adam')
model.summary()
train_model(model, 
			[train_input_melody, train_input_rmid, train_input_lmid, train_input_bass], 
			[train_output_melody, train_output_rmid, train_output_lmid, train_output_bass],
			[test_input_melody, test_input_rmid, test_input_lmid, test_input_bass], 
			[test_output_melody, test_output_rmid, test_output_lmid, test_output_bass])

model.evaluate([test_input_melody, test_input_rmid, test_input_lmid, test_input_bass],
				[test_output_melody, test_output_rmid, test_output_lmid, test_output_bass])
print('Done training model')


# PART 3: use trained model to generate notes and write midi files
def predict_notes(melody_seq, rmid_seq, lmid_seq, bass_seq):
	predicted_melody, predicted_rmid, predicted_lmid, predicted_bass = model.predict([melody_seq, rmid_seq, lmid_seq, bass_seq])

	return np.argmax(predicted_melody), np.argmax(predicted_rmid), np.argmax(predicted_lmid), np.argmax(predicted_bass)


def process_prediction_input(pred_input, predicted_note):
	pred_input = pred_input.flatten()
	pred_input = np.append(pred_input, predicted_note)[1:]
	pred_input = np.reshape(pred_input, (1, len(pred_input)))

	return pred_input


pred_input_melody = np.array([train_input_melody[30]])
pred_input_rmid = np.array([train_input_rmid[30]])
pred_input_lmid = np.array([train_input_lmid[30]])
pred_input_bass = np.array([train_input_bass[30]])
print(pred_input_melody)
print(pred_input_rmid)
print(pred_input_lmid)
print(pred_input_bass)

generated_melody = []
generated_bass = []
generated_rmid = []
generated_lmid = []

for i in range(256):
	predicted_melody, predicted_rmid, predicted_lmid, predicted_bass = predict_notes(pred_input_melody, pred_input_rmid, pred_input_lmid,
																					pred_input_bass)
	generated_melody.append(int_to_melody[predicted_melody])
	generated_rmid.append(int_to_rmid[predicted_rmid])
	generated_lmid.append(int_to_lmid[predicted_lmid])
	generated_bass.append(int_to_bass[predicted_bass])

	# reshape the prediction input for next prediction
	pred_input_melody = process_prediction_input(pred_input_melody, predicted_melody)
	pred_input_rmid = process_prediction_input(pred_input_rmid, predicted_rmid)
	pred_input_lmid = process_prediction_input(pred_input_lmid, predicted_lmid)
	pred_input_bass = process_prediction_input(pred_input_bass, predicted_bass)


def add_notes_to_stream(generated_part, output_stream):
	note_val = -1
	pos = -1
	flag_midi = False

	for i in range(len(generated_part)):
		if isinstance(generated_part[i], int):
			if i != len(generated_part)-1:
				if flag_midi == False:
					flag_midi = True

				else:
					output_stream.insert(float(pos/4), note.Note(note_val, quarterLength = float((i-pos)/4)))

				note_val = generated_part[i]
				pos = i

			else:
				if flag_midi == False:
					output_stream.insert(float(i/4), note.Note(generated_part[i], quarterLength = 0.25))

				else:
					output_stream.insert(float(pos/4), note.Note(note_val, quarterLength = float((i-pos)/4)))
					output_stream.insert(float(i/4), note.Note(generated_part[i], quarterLength = 0.25))
					
		else:
			if generated_part[i] == 'H':
				if i != len(generated_part)-1:
					pass

				else:
					if flag_midi == True:
						output_stream.insert(float(pos/4), note.Note(note_val, quarterLength = float((i-pos+1)/4)))

			elif generated_part[i] == 'R':
				if flag_midi == True:
					output_stream.insert(float(pos/4), note.Note(note_val, quarterLength = float((i-pos)/4)))

				flag_midi = False

			else:
				print('Error, undefined item in list.')

	'''
	for item in output_stream:
		print(item.offset, item.pitches, item.duration.quarterLength)
	'''

	return

 
print(generated_melody)
print(generated_rmid)
print(generated_lmid)
print(generated_bass)

output_stream = stream.Stream()
output_stream.insert(0.0, tempo.MetronomeMark(number=100))
add_notes_to_stream(generated_melody, output_stream)
add_notes_to_stream(generated_rmid, output_stream)
add_notes_to_stream(generated_lmid, output_stream)
add_notes_to_stream(generated_bass, output_stream)
output_stream.write('midi', 'model2_sample.mid')
print('Done generating midi')