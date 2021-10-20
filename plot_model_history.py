import pickle
import matplotlib.pyplot as plt

# change this to open file containing model 1, 2, or 3 history
with open('model1_history', 'rb') as f:
	fileout = pickle.load(f)

history = fileout

print(history.keys())

# plot training and test loss against epoch
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training loss', 'Test loss'], loc='upper left')
plt.show()

# plot training and test accuracy for melody and bass parts against epoch
plt.plot(history['dense_1_sparse_categorical_accuracy'])
plt.plot(history['val_dense_1_sparse_categorical_accuracy'])
plt.plot(history['dense_4_sparse_categorical_accuracy'])
plt.plot(history['val_dense_4_sparse_categorical_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Melody training acc', 'Melody test acc', 'Bass training acc', 'Bass test acc'], loc='upper left')
plt.show()

print(history['dense_1_sparse_categorical_accuracy'][49])
print(history['val_dense_1_sparse_categorical_accuracy'][49])
print(history['dense_2_sparse_categorical_accuracy'][49])
print(history['val_dense_2_sparse_categorical_accuracy'][49])
print(history['dense_3_sparse_categorical_accuracy'][49])
print(history['val_dense_3_sparse_categorical_accuracy'][49])
print(history['dense_4_sparse_categorical_accuracy'][49])
print(history['val_dense_4_sparse_categorical_accuracy'][49])