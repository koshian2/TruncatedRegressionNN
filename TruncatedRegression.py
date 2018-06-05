import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# 1953年～2017年の月次データ（旭川）
data = np.loadtxt("asahikawa.csv", delimiter=",", skiprows=1)
X, y = data[:, 0], data[:, 1]
X = X[:, np.newaxis] #行列にする

model = Sequential()
model.add(Dense(1, activation="relu", input_shape=X.shape[1:]))
model.compile(loss="mean_squared_error", optimizer=Adam(lr=1), metrics=["mse"])

n_epoch = 200
history = model.fit(X, y, epochs=n_epoch, batch_size=128).history

pred_X = np.arange(-20, 30, 0.1)[:, np.newaxis]
pred_y = model.predict(pred_X)
print(model.get_weights())
plt.plot(X, y, ".")
plt.plot(pred_X, pred_y)
plt.xlabel("Temperature")
plt.ylabel("Snowfall")
plt.plot()
plt.show()

plt.plot(np.arange(n_epoch), history["loss"])
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()
