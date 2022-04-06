import matplotlib.pyplot as plt
import astropy.table
import random

# Read in a file with outflow data
outflow_data = astropy.table.Table.read("./outputs/outflows.hdf5")

agn_shining = outflow_data[outflow_data["luminosity_AGN"] > 0]
subsample = random.sample(range(len(agn_shining)), k=10000)
shining = agn_shining[subsample]

radius = shining["radius"]
velocity = shining["dot_radius"]
mdot = shining["dot_mass"]
lagn = shining["luminosity_AGN"]
pdot = mdot * 1.989e33 / 3.15e7 * velocity * 1e5
edot = mdot * 1.989e33 / 3.15e7 * velocity * velocity / 2.0 * 1e10
pload = pdot * 3.0e10 / lagn  # momentum loading factor, dot(p) / (L_AGN/c)
eload = edot / lagn

plt.rcParams.update({"font.size": 12})
# Let's make three scatter plots in a column
fig = plt.figure(figsize=(6.4, 9))
gs = fig.add_gridspec(nrows=3, left=0.12, right=0.89, top=0.98, bottom=0.06, hspace=0)
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
    ax[z].tick_params(length=7, width=1.5)
    ax[z].tick_params(which="minor", length=4.5, width=1.2)

ax[0].set_xscale("log")
ax[0].set_xlim(1e42, 1e48)
ax[2].set_xlabel("AGN luminosity, erg s$^{-1}$")

ax[0].set_ylabel("Outflow velocity (km s$^{-1}$)")
ax[0].set_ylim(1.01e1, 3e3)
ax[0].set_yscale("log")
ax[0].plot(lagn, velocity, ".b", markersize=1, alpha=0.5)

ax[1].set_ylabel("Mass outflow rate ($M_\odot$ yr$^{-1}$)")
ax[1].set_ylim(1.01e0, 1e4)
ax[1].set_yscale("log")
ax[1].plot(lagn, mdot, ".b", markersize=1, alpha=0.5)

ax[2].set_ylabel("Kinetic power (erg s$^{-1}$)")
ax[2].set_ylim(1e38, 1e45)
ax[2].set_yscale("log")
ax[2].plot(lagn, edot, ".b", markersize=1, alpha=0.5)
ax[2].plot([1e42, 1e48], [1e42, 1e48], "-r")
ax[2].text(
    0.18e43, 0.25e43, r"$L_{AGN}$", fontsize=10, rotation=26, rotation_mode="anchor"
)
ax[2].plot([1e42, 1e48], [0.05 * 1e42, 0.05 * 1e48], "--r")
ax[2].text(
    0.18e43,
    0.05 * 0.25e43,
    r"$0.05 \times L_{AGN}$",
    fontsize=10,
    rotation=26,
    rotation_mode="anchor",
)
ax[2].plot([1e42, 1e48], [0.01 * 1e42, 0.01 * 1e48], ":r")
ax[2].text(
    0.18e43,
    0.01 * 0.25e43,
    r"$0.01 \times L_{AGN}$",
    fontsize=10,
    rotation=26,
    rotation_mode="anchor",
)

fig.savefig("./figures/outflow_scatter_plot.png", dpi=300)
