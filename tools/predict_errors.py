import argparse
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import tools.utils as utils


parser = argparse.ArgumentParser()
parser.add_argument("--num-random-instances", type=int, default=1000)
args = parser.parse_args()

rng = np.random.default_rng(seed=0)

outflow_properties = utils.load_simulated_outflows()

X = utils.to_numpy(outflow_properties[utils.input_params])
y = utils.to_numpy(outflow_properties[utils.output_params])

train_ids, test_ids = utils.split_sets_ids(outflow_properties)
train_mask, test_mask = utils.split_sets_masks(outflow_properties)

X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]

sample_mask = rng.integers(low=0, high=len(X_test), size=200)
X_sample, y_sample = X[test_mask][sample_mask], y[test_mask][sample_mask]

# create args.num_random_instances instances of each point - only X values change
X_sample_embiggened = X_sample.repeat(args.num_random_instances, axis=0)

weights = 10 ** rng.normal(loc=0, scale=0.15, size=X_sample_embiggened.shape)
X_sample_embiggened = np.multiply(X_sample_embiggened, weights)

X_mean, X_stddev, y_mean, y_stddev = utils.load_normalization()

X_sample_embiggened = utils.normalize(X_sample_embiggened, X_mean, X_stddev)

model = tf.keras.models.load_model("./outputs/model.h5")

y_sample_predictions = model.predict(X_sample_embiggened)
y_sample_predictions = utils.denormalize(y_sample_predictions, y_mean, y_stddev)


# now find the mean and stdev of predictions for each group of 1000 lines and reduce the prediction table back to the original size
y_sample_predictions = np.c_[
    y_sample_predictions, y_sample_predictions * 0, y_sample_predictions * 0
]
for i in range(len(y_sample)):
    for k in range(len(utils.output_params)):
        y_sample_predictions[
            i * args.num_random_instances, len(y_sample[0]) + k
        ] = np.mean(
            y_sample_predictions[
                i * args.num_random_instances : (i + 1) * args.num_random_instances - 1,
                k,
            ]
        )
        y_sample_predictions[
            i * args.num_random_instances, 2 * len(y_sample[0]) + k
        ] = np.std(
            y_sample_predictions[
                i * args.num_random_instances : (i + 1) * args.num_random_instances - 1,
                k,
            ]
        )
y_sample_predictions = y_sample_predictions[:: args.num_random_instances]
y_sample_predictions = y_sample_predictions[:, len(y_sample[0]) :]

colnames_pred = [s + "_mean" for s in utils.output_params]
colnames_pred.extend([s + "_stdev" for s in utils.output_params])
prediction_table = utils.from_numpy(y_sample_predictions, column_names=colnames_pred)
groundtruth_table = utils.from_numpy(y_sample, column_names=utils.output_params)

prediction_table["id"] = outflow_properties[test_mask][sample_mask]["id"]
groundtruth_table["id"] = outflow_properties[test_mask][sample_mask]["id"]
groundtruth_table["radius"] = outflow_properties[test_mask][sample_mask]["radius"]


utils.output_params.remove("bulge_mass")

plt.rcParams.update({"font.size": 14})
for idx, column_name in enumerate(utils.output_params):
    plt.subplot(2, int(np.ceil(len(utils.output_params) / 2)), idx + 1)

    if column_name == "duty_cycle":  # 1
        plt.xlabel(f"True duty cycle")
        plt.ylabel(f"Duty cycle uncertainty")
        ticks = [0.2, 0.4, 0.6, 0.8, 1.0]
        max_y = 0.25
    elif column_name == "quasar_activity_duration":
        plt.xlabel(f"True activity duration [kyr]")
        plt.ylabel(f"Activity duration uncertainty [kyr]")
        ticks = [100000, 200000, 300000]
        max_y = 50000
    elif column_name == "outflow_solid_angle_fraction":  # 3
        plt.xlabel(f"True solid angle fraction")
        plt.ylabel(f"Solid angle fraction uncertainty")
        ticks = [0.2, 0.4, 0.6, 0.8, 1.0]
        max_y = 0.25
    elif column_name == "bulge_gas_fraction":
        plt.xlabel(f"True bulge gas fraction")
        plt.ylabel(f"Bulge gas fraction uncertainty")
        ticks = [0.1, 0.2, 0.3]
        max_y = 0.05

    histogrammed_uncertainties = np.histogram(
        groundtruth_table[column_name],
        bins=12,
        weights=prediction_table[column_name + "_stdev"],
    )

    histogrammed_values = (
        histogrammed_uncertainties[0]
        / np.histogram(groundtruth_table[column_name], bins=12)[0]
    )

    min_x = np.min(groundtruth_table[column_name])
    max_x = np.max(groundtruth_table[column_name])
    min_y = 0

    plt.bar(
        histogrammed_uncertainties[1][0:12],
        histogrammed_values,
        width=(max_x - min_x) / 12,
        align="edge",
        color="lightsteelblue",
        edgecolor="royalblue",
        linewidth="2",
    )

    plt.xlim([min_x, max_x])
    plt.ylim([min_y, max_y])

    plt.xticks(ticks)

    ax = plt.gca()

    if column_name == "quasar_activity_duration":

        def numfmt(x, pos):
            s = "{}".format(int(x / 1000.0))
            return s

        import matplotlib.ticker as tkr

        yfmt = tkr.FuncFormatter(numfmt)

        ax.xaxis.set_major_formatter(yfmt)
        ax.yaxis.set_major_formatter(yfmt)
    ax.get_xaxis().set_tick_params(
        which="both", direction="in", bottom=True, top=True, left=True, right=True
    )
    ax.get_yaxis().set_tick_params(
        which="both", direction="in", bottom=True, top=True, left=True, right=True
    )

fig = plt.gcf()
fig.set_size_inches(9.0, 8.0)
plt.subplots_adjust(wspace=0.4, hspace=0.3)

plt.savefig(f"./figures/error_predictions_histogram_randomized.png", dpi=300)
