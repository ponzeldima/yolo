import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  
import tensorflow as tf
import numpy as np

# tf.config.set_visible_devices(tf.config.list_physical_devices('CPU'))

print('hello')

# print("TF version:", tf.__version__)
print("Num GPUs:", len(tf.config.list_physical_devices('GPU')))
# просте обчислення:
# print("ten sum:", tf.reduce_sum(tf.random.normal([50000,50000])).numpy())

x = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0])
y = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0])

model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),
    tf.keras.layers.Dense(units=1)
])

model.compile(optimizer='sgd', loss='mean_squared_error')

# Train the model
model.fit(x, y, epochs=1000)

prediction10 = model.predict(np.array([10]), verbose=0)
prediction100 = model.predict(np.array([100]), verbose=0)
prediction1 = model.predict(np.array([1]), verbose=0)
print(f"prediction: {prediction1.item():.5f}")
print(f"prediction: {prediction10.item():.5f}")
print(f"prediction: {prediction100.item():.5f}")
