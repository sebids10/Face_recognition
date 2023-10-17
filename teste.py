import tensorflow as tf
model = tf.keras.models.load_model('C:/Users/Sebi/proiect_ps/Face Recognition System/keras_model.h5')

# Afișează arhitectura modelului
model.summary()

# Vizualizează straturile modelului
for layer in model.layers:
    print(layer.name)
    