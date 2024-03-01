import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import tools.utils as utils


real_outflows = utils.load_real_outflows()

X_real = real_outflows[utils.input_params]

X_mean, X_stddev, y_mean, y_stddev = utils.load_normalization()
X_real = utils.normalize(X_real, X_mean, X_stddev)

model = tf.keras.models.load_model("./outputs/model.keras")
y_real_predictions = model.predict(X_real)

y_real_predictions = utils.denormalize(y_real_predictions, y_mean, y_stddev)

predictions_df = pd.DataFrame(
    y_real_predictions,
    columns=utils.output_params,
    index=real_outflows.index,
)

outflow_pred_df = pd.merge(
    real_outflows, predictions_df, left_index=True, right_index=True
)

utils.output_params.remove("bulge_mass")


plt.rcParams.update({"font.size": 14})
for idx, column_name in enumerate(utils.output_params):
    plt.subplot(2, int(np.ceil(len(utils.output_params) / 2)), idx + 1)

    if column_name == "duty_cycle":
        plt.xlabel(f"Duty cycle")
        plt.ylabel(f"N")
        ticks = [0.2, 0.4, 0.6, 0.8, 1.0]
        minval = 0
        maxval = 1
    elif column_name == "quasar_activity_duration":
        plt.xlabel(f"Activity duration [kyr]")
        plt.ylabel(f"N")
        ticks = [100000, 200000, 300000]
        minval = 0
        maxval = 3e5
    elif column_name == "outflow_solid_angle_fraction":
        plt.xlabel(f"Solid angle fraction")
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

    colors = [plt.cm.Set1(i) for i in range(4)]
    plt.hist(
        [
            outflow_pred_df[outflow_pred_df["agn_type"] == "Type 1"][column_name],
            outflow_pred_df[outflow_pred_df["agn_type"] == "Type 2"][column_name],
            outflow_pred_df[outflow_pred_df["agn_type"] == "LINER"][column_name],
            outflow_pred_df[outflow_pred_df["agn_type"] == "HII"][column_name],
        ],
        bins=12,
        range=(minval, maxval),
        histtype="barstacked",
        stacked=True,
        color=colors
    )

    plt.xlim(minval, maxval)
    plt.ylim(0, 20)
    plt.xticks(ticks)
    plt.yticks([0, 5, 10, 15, 20])

    ax = plt.gca()

    if column_name == "outflow_solid_angle_fraction":
        plt.legend(
            [
                "Type 1",
                "Type 2",
                "LINER",
                "HII",
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

plt.savefig(f"./figures/real_predictions_by_type.png", dpi=300)
