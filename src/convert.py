import tensorflow as tf


def convert_model():
    # tensorflow 2.15 version used
    # Convert the SavedModel to TensorFlow Lite format
    converter = tf.lite.TFLiteConverter.from_saved_model('saved_model')
    tflite_model = converter.convert()

    # Save the TensorFlow Lite model to a file
    with open('model_files/mood-shift.tflite', 'wb') as f:
        f.write(tflite_model)
    return tflite_model


if __name__ == '__main__':
    convert_model()
