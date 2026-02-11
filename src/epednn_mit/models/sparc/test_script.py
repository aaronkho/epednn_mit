from pathlib import Path
import numpy as np
import tensorflow as tf


def load_epednn_mit_sparc():
    root = Path(__file__).resolve().parent
    model_list = []
    for i in range(10):
        model_list.append(tf.keras.models.load_model(root / f'epednn_mit_sparc_{i:d}.keras'))
    input_shape = tf.keras.layers.Input(shape=(9, ))
    x = [m(input_shape) for m in model_list]
    output_shape = tf.keras.layers.Average()(x)
    return tf.keras.models.Model(inputs=input_shape, outputs=output_shape)


def test_epednn_mit_sparc():
    test_in = np.atleast_2d([10.0, 9.0, 1.85, 0.57, 2.0, 0.45, 30.0, 1.2, 2.0])
    test_out = np.atleast_2d([3.19934668e+02, 4.11556657e-02])
    model = load_epednn_mit_sparc()
    model_out = model.predict(test_in)
    return model_out, test_out


def main():
    model_out, test_out = test_epednn_mit_sparc()
    print('Evaluation:', model_out)
    assert model_out.shape == test_out.shape
    assert np.all(np.isclose(model_out, test_out))


if __name__ == '__main__':
    main()
