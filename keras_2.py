import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


fashion_mnist = tf.keras.datasets.fashion_mnist

(train_image, train_labels), (test_image, test_labels) = fashion_mnist.load_data()


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

train_image.shape

len(train_labels)

train_labels

test_image.shape

len(test_labels)

# preprocess of the data

plt.figure

plt.imshow(train_image[0])

plt.colorbar()

plt.grid(False)

plt.show()

train_images = train_image/255.0

test_image = test_image/255.0

plt.figure(figsize=(10,10))

for i in range(25):
  plt.subplot(5,5,i+1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.imshow(train_image[i], cmap = plt.cm.binary)
  plt.xlabel(class_names[train_labels[i]])
plt.show()

#model training

model = tf.keras.Sequential([
                             tf.keras.layers.Flatten(input_shape=(28,28)),
                             tf.keras.layers.Dense(128, activation='relu'),
                             tf.keras.layers.Dense(10)

])


#model compile

model.compile(optimizer = 'Adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits= True),
              metrics=['accuracy'])


#Train the model
#(i) - feed the training data to the model. The training data is in the train_image and train_labels 
#(ii) - model learns to associate images and labels
#(iii) - test the model
#(iv) - Verify that the predictions match the labels from the test_labels array.


model.fit(train_image, train_labels, epochs = 10)

#Evaluation of accuracy


test_loss, test_acc = model.evaluate(test_image, test_labels, verbose = 2)

print('\n Test accuracy = ', test_acc)

#Make predicitons 

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])


predictions = probability_model.predict(test_image)

predictions[0]

test_labels[0]

def plot_image(i, prediction_array, true_label, img):
  true_labels, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])


  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(prediction_array)

  predicted_label = np.argmax(prediction_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                       100*np.max(prediction_array),
                                       class_names[true_label]),
             color=color)


def plot_value_array(i, prediction_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), prediction_array, color ='#777777')
  plt.ylim([0,1])
  predicted_label = np.argmax(prediction_array)


  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

# verify predictions

i = 0 
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplots(1,2,2)
plot_value_array(i, predictions[i], test_labels)
plt.show()
