import matplotlib.pyplot as plt
import os
import astropy.table


prediction_data = astropy.table.Table.read("./outputs/real_predictions.csv")

bout = prediction_data["outflow_solid_angle_fraction"]
lagn = 10 ** prediction_data["luminosity_AGN_log"]
ledd = 1.26e38 * 10 ** prediction_data["smbh_mass_log"]
edot = (
    10 ** prediction_data["mdot_log_g_s"]
    * (prediction_data["dot_radius"] * 1e5) ** 2
    / 2
)


plt.rcParams.update({"font.size": 12})
# Let's make two scatter plots in a column
fig = plt.figure(figsize=(6.4, 6))
gs = fig.add_gridspec(nrows=2, left=0.13, right=0.91, top=0.95, bottom=0.1, hspace=0)
ax = gs.subplots(sharex=True, sharey=False)
for z in range(len(ax)):
    ax[z].tick_params(
        axis="both",
        which="both",
        direction="in",
        top=True,
        right=True,
        width=1.2,
        labelsize="medium",
    )
    ax[z].tick_params(length=6, width=1.0)
    ax[z].tick_params(which="minor", length=4, width=0.8)

ax[1].set_xscale("log")
ax[1].set_xlim(3e-2, 6)
ax[1].set_xlabel("Inferred solid angle fraction")

ax[0].set_ylabel("$\dot{E}_\mathrm{out}/L_\mathrm{AGN}$")
ax[0].set_ylim(2e-6, 2)
ax[0].set_yscale("log")
ax[0].plot(bout, edot / lagn, ".b", markersize=1.5, alpha=1)

ax[1].set_ylabel("$\dot{E}_\mathrm{out}/L_\mathrm{Edd}$")
ax[1].set_ylim(2e-7, 1e-1)
ax[1].set_yscale("log")
ax[1].plot([3e-2, 6], [6e-4 * 3e-2 ** 2, 6e-4 * 6 ** 2], "--r", alpha=0.7)
ax[1].plot(bout, edot / ledd, ".b", markersize=1.5, alpha=1)


fname = "./figures/bout_vs_lkin.png"
fig.savefig(fname, format="png", dpi=300)
