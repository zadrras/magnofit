import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np
import argparse
import time

import tools.utils as utils


tf.get_logger().setLevel("ERROR")
tf.random.set_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("--start-lr", type=float, default=0.001)
parser.add_argument("--neurons", type=int, default=128)
parser.add_argument("--layers", type=int, default=2)
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--activation", type=str, default="elu")
parser.add_argument("--no-dropout", action="store_false")
args = parser.parse_args()

outflow_properties = utils.load_simulated_outflows()

X = utils.to_numpy(outflow_properties[utils.input_params])
y = utils.to_numpy(outflow_properties[utils.output_params])

train_mask, test_mask = utils.split_sets_masks(outflow_properties)

X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]

X_mean, X_stddev, y_mean, y_stddev = utils.fit_normalization(X_train, y_train)

X_train = utils.normalize(X_train, X_mean, X_stddev)
y_train = utils.normalize(y_train, y_mean, y_stddev)

X_test = utils.normalize(X_test, X_mean, X_stddev)
y_test = utils.normalize(y_test, y_mean, y_stddev)

layers = [tf.keras.layers.Input(len(utils.input_params))]

for l in range(args.layers):
    layers.append(tf.keras.layers.Dense(args.neurons, activation=args.activation))

if not args.no_dropout:
    layers.append(tf.keras.layers.Dropout(0.5))
layers.append(tf.keras.layers.Dense(len(utils.output_params)))

model = tf.keras.models.Sequential(layers)


def step_decay(epoch):
    lr = args.start_lr
    if epoch == 4:
        lr = lr / 10
    if epoch == 8:
        lr = lr / 10

    return lr


model.compile(
    loss=tf.keras.losses.MSE,
    optimizer=tf.keras.optimizers.Adam(amsgrad=True),
    metrics=["mse"],
)

start_time = time.time()
model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=12,
    batch_size=args.batch_size,
    callbacks=tf.keras.callbacks.LearningRateScheduler(step_decay),
    verbose=1,
)
end_time = time.time()
print(f"Training took {end_time - start_time:.2f} s.")

predictions = model.predict(X_test, batch_size=1024, verbose=0)

individual_mses = np.mean((predictions - y_test) ** 2, axis=0)
overall_mse = np.mean((predictions - y_test) ** 2)

print(
    f"start-lr: {args.start_lr}, neurons: {args.neurons}, "
    f"layers: {args.layers}, activation: {args.activation}, "
    f"no-dropout: {args.no_dropout}, batch-size: {args.batch_size}"
)

print(f"overall: {overall_mse:.5f}", end=" ")
for m, name in zip(individual_mses, utils.output_params):
    print(f"{name}: {m:.5f}", end=" ")

print()
print()
model.save("./outputs/model.h5")
