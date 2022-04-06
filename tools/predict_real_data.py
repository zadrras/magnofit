import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import tools.utils as utils


real_outflows = utils.load_real_outflows()
real_subset_bright = real_outflows["log_f_Edd"] > -1.65
real_subset_medium = (real_outflows["log_f_Edd"] > -2.5) & (
    real_outflows["log_f_Edd"] < -1.65
)
real_subset_dim = real_outflows["log_f_Edd"] < -2.5

X_real = real_outflows[utils.input_params]

X_mean, X_stddev, y_mean, y_stddev = utils.load_normalization()
X_real = utils.normalize(X_real, X_mean, X_stddev)

model = tf.keras.models.load_model("./outputs/model.h5")
y_real_predictions = model.predict(X_real)

y_real_predictions = utils.denormalize(y_real_predictions, y_mean, y_stddev)

predictions_df = pd.DataFrame(
    y_real_predictions,
    columns=utils.output_params,
    index=real_outflows.index,
)

outflows_and_predictions_df = pd.merge(
    real_outflows, predictions_df, left_index=True, right_index=True
)
outflows_and_predictions_df.to_csv("./outputs/real_predictions.csv")

utils.output_params.remove("bulge_mass")


plt.rcParams.update({"font.size": 13})
for idx, column_name in enumerate(utils.output_params):
    plt.subplot(2, int(np.ceil(len(utils.output_params) / 2)), idx + 1)

    if column_name == "duty_cycle":
        plt.xlabel(f"Duty cycle")
        plt.ylabel(f"N")
        ticks = [0.2, 0.4, 0.6, 0.8, 1.0]
        minval = 0
        maxval = 1
    elif column_name == "quasar_activity_duration":
        plt.xlabel(f"Quasar activity duration [kyr]")
        plt.ylabel(f"N")
        ticks = [100000, 200000, 300000]
        minval = 0
        maxval = 3e5
    elif column_name == "outflow_solid_angle_fraction":
        plt.xlabel(f"Outflow solid angle fraction")
        plt.ylabel(f"N")
        ticks = [0.2, 0.4, 0.6, 0.8, 1.0]
        minval = 0
        maxval = 1
    elif column_name == "bulge_gas_fraction":
        plt.xlabel(f"Bulge gas fraction")
        plt.ylabel(f"N")
        ticks = [0.05, 0.1, 0.15]
        minval = 0
        maxval = 0.15

    plt.hist(
        [
            predictions_df[real_subset_bright][column_name],
            predictions_df[real_subset_medium][column_name],
            predictions_df[real_subset_dim][column_name],
        ],
        bins=12,
        range=(minval, maxval),
        histtype="barstacked",
        stacked=True,
    )

    plt.xlim(minval, maxval)
    plt.ylim(0, 20)
    plt.xticks(ticks)
    plt.yticks([0, 5, 10, 15, 20])

    ax = plt.gca()

    if column_name == "outflow_solid_angle_fraction":
        plt.legend(
            [
                "log $f_{Edd}$ > -1.7",
                "-1.7 > log $f_{Edd}$ > -2.5",
                "log $f_{Edd}$ < -2.5",
            ]
        )
    elif column_name == "quasar_activity_duration":

        def numfmt(x, pos):
            s = "{}".format(int(x / 1000.0))
            return s

        import matplotlib.ticker as tkr

        yfmt = tkr.FuncFormatter(numfmt)

        ax.xaxis.set_major_formatter(yfmt)

    ax.get_xaxis().set_tick_params(
        which="both", direction="in", bottom=True, top=True, left=True, right=True
    )
    ax.get_yaxis().set_tick_params(
        which="both", direction="in", bottom=True, top=True, left=True, right=True
    )

fig = plt.gcf()
fig.set_size_inches(9.0, 8.0)
plt.subplots_adjust(wspace=0.4, hspace=0.25)

plt.savefig(f"./figures/real_predictions.png", dpi=300)
