
import io
import numpy as np
import tensorflow as tf

import sys
import matplotlib
backend = 'Agg' if sys.platform == 'linux' else 'TkAgg'
matplotlib.use(backend)
import matplotlib.pyplot as plt

def scatter_encodings_summary(x, name):
    plt.scatter(x[:,0], x[:,1])
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_sum = tf.Summary.Image(encoded_image_string=buf.getvalue())
    summary = tf.Summary(value=[
       tf.Summary.Value(tag='{}/encoding'.format(name), image=img_sum)
    ])
    plt.close()
    return summary
