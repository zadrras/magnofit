import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import tools.utils as utils


outflow_properties = utils.load_simulated_outflows()

X = utils.to_numpy(outflow_properties[utils.input_params])
y = utils.to_numpy(outflow_properties[utils.output_params])

train_mask, test_mask = utils.split_sets_masks(outflow_properties)

X_test, y_test = X[test_mask], y[test_mask]


X_mean, X_stddev, y_mean, y_stddev = utils.load_normalization()

X_test = utils.normalize(X_test, X_mean, X_stddev)
y_test = utils.normalize(y_test, y_mean, y_stddev)

model = tf.keras.models.load_model("./outputs/model.h5")

y_test_predictions = model.predict(X_test, verbose=1)

y_test_predictions_denorm = utils.denormalize(y_test_predictions, y_mean, y_stddev)
prediction_table = utils.from_numpy(
    y_test_predictions_denorm, column_names=utils.output_params
)

groundtruth_table = outflow_properties[test_mask]

prediction_table["id"] = outflow_properties[test_mask]["id"]

utils.output_params.remove("bulge_mass")

plt.rcParams.update({'font.size': 14})
for idx, column_name in enumerate(utils.output_params):
    plt.subplot(2, int(np.ceil(len(utils.output_params) / 2)), idx + 1)
    bins = 12
    if column_name == "duty_cycle":  # 1
        plt.xlabel(f"True duty cycle")
        plt.ylabel(f"Predicted duty cycle")
        ticks = [0.2, 0.4, 0.6, 0.8, 1.0]
    elif column_name == "quasar_activity_duration":
        plt.xlabel(f"True activity duration [kyr]")
        plt.ylabel(f"Predicted activity duration [kyr]")
        bins = np.linspace(10000, 320000, 12)
        ticks = [100000, 200000, 300000]
    elif column_name == "outflow_solid_angle_fraction":  # 3
        plt.xlabel(f"True solid angle fraction")
        plt.ylabel(f"Predicted solid angle fraction")
        ticks = [0.2, 0.4, 0.6, 0.8, 1.0]
    elif column_name == "bulge_gas_fraction":
        plt.xlabel(f"True bulge gas fraction")
        plt.ylabel(f"Predicted bulge gas fraction")
        ticks = [0.1, 0.2, 0.3]

    clipped_predictions = prediction_table[column_name]
    min_val = np.min(groundtruth_table[column_name])
    max_val = np.max(groundtruth_table[column_name])
    clipped_predictions[clipped_predictions < min_val] = min_val
    clipped_predictions[clipped_predictions > max_val] = max_val
    histogram, *_ = np.histogram2d(
        groundtruth_table[column_name], clipped_predictions, bins=bins, normed=False
    )

    histogram = histogram.T[::-1]
    histogram = histogram / np.sum(histogram, axis=0)[np.newaxis]
    plt.imshow(
        histogram,
        cmap="Greys",
        extent=[min_val, max_val, min_val, max_val],
        vmin=0.0,
        vmax=0.5,
        aspect="auto",
    )

    plt.xlim([min_val, max_val])
    plt.ylim([min_val, max_val])

    plt.xticks(ticks)
    plt.yticks(ticks)

    ax = plt.gca()
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c="r")

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
plt.subplots_adjust(wspace=0.4, hspace=0.25)

plt.savefig("./figures/diagonal_error_heatmap.png", dpi=300)
