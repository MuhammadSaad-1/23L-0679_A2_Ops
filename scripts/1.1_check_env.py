import os
import sys
import numpy as np
import pandas as pd
import matplotlib
import PIL
import tensorflow as tf

print("=" * 50)
print(f"Python Version     : {sys.version.split()[0]}")
print(f"TensorFlow Version : {tf.__version__}")
print(f"NumPy Version      : {np.__version__}")
print(f"Pandas Version     : {pd.__version__}")
print(f"Matplotlib Version : {matplotlib.__version__}")
print(f"Pillow Version     : {PIL.__version__}")
print("=" * 50)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU Detected ({len(gpus)} device(s)):")
    for gpu in gpus:
        print(f" - {gpu.name}")
else:
    print("No GPU detected. TensorFlow will run in CPU mode.")
print("=" * 50)
