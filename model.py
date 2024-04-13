import tensorflow as tf

#tensorflow 2.15 version used
# Convert the SavedModel to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_saved_model('mood-shift_saved_model')
tflite_model = converter.convert()

# Save the TensorFlow Lite model to a file
with open('mood-shift.tflite', 'wb') as f:
    f.write(tflite_model)
