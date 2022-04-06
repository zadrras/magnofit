import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


plt.rcParams.update({'font.size': 12})
def setup_plot(axes):
    for ax2d in axes:
        for ax in ax2d:
            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.tick_params(axis='both', which='both', direction='in', top=True, right=True, width=1.2, labelsize='medium')
            ax.tick_params(length=7, width=1.5)
            ax.tick_params(which='minor', length=4.5, width=1.2)

full_real_outflows = pd.read_csv("./outputs/generated_outflows_from_real_predictions.csv")
initial_real_outflows = pd.read_csv("./observed_outflows.csv")

os.makedirs("./figures/real_outflows", exist_ok=True)

for name in initial_real_outflows.name:
    calculated_outflow = full_real_outflows[(full_real_outflows.name == name) & (full_real_outflows.luminosity_AGN > 0)]
    if len(calculated_outflow) == 0:
        continue
    sorted_by_time_outflow = calculated_outflow.sort_values(by=['time'])
    filtered_initial_real = initial_real_outflows[initial_real_outflows.name==name]
    filtered_initial_real.reset_index(drop=True, inplace=True)
    fig, axs = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(10,6))
    fig.subplots_adjust(hspace=0, wspace=0)

    setup_plot(axs)

    axis_range = 3 #axis range in terms of observed value uncertainty

    lummin = np.min([10**filtered_initial_real.lum_min_log[0]/axis_range,10**filtered_initial_real.luminosity_AGN_log[0]/10])
    lummax = np.max([10**filtered_initial_real.lum_max_log[0]*axis_range,10**filtered_initial_real.luminosity_AGN_log[0]*10])
    vmin = filtered_initial_real.dot_radius[0]/10**0.15/axis_range
    vmax = filtered_initial_real.dot_radius[0]*10**0.15*axis_range
    rmin = filtered_initial_real.radius[0]/10**0.15/axis_range
    rmax = filtered_initial_real.radius[0]*10**0.15*axis_range
    mmin = np.min([filtered_initial_real.mdot_min[0]/axis_range,filtered_initial_real.derived_dot_mass[0]/10])
    mmax = np.max([filtered_initial_real.mdot_max[0]*axis_range,filtered_initial_real.derived_dot_mass[0]*10])

    axs[0, 0].set_xlim(lummin, lummax)
    axs[0, 0].set_ylim(vmin, vmax)
    axs[0, 0].set_ylabel('Outflow velocity (km s$^{-1}$)')
    axs[0, 0].plot(sorted_by_time_outflow.luminosity_AGN, sorted_by_time_outflow.dot_radius)
    axs[0, 0].plot([10**filtered_initial_real.lum_min_log, 10**filtered_initial_real.lum_max_log], [filtered_initial_real.dot_radius, filtered_initial_real.dot_radius], c='r')
    axs[0, 0].plot([10**filtered_initial_real.luminosity_AGN_log, 10**filtered_initial_real.luminosity_AGN_log], [filtered_initial_real.dot_radius*10**0.15, filtered_initial_real.dot_radius/10**0.15], c='y')
    axs[0, 0].scatter(10**filtered_initial_real.luminosity_AGN_log, filtered_initial_real.dot_radius, c='r', zorder=1000)

    axs[0, 1].set_xlim(rmin, rmax)
    axs[0, 1].plot(sorted_by_time_outflow.radius, sorted_by_time_outflow.dot_radius)
    axs[0, 1].plot([filtered_initial_real.radius/10**0.15, filtered_initial_real.radius*10**0.15], [filtered_initial_real.dot_radius, filtered_initial_real.dot_radius], c='y')
    axs[0, 1].plot([filtered_initial_real.radius, filtered_initial_real.radius], [filtered_initial_real.dot_radius*10**0.15, filtered_initial_real.dot_radius/10**0.15], c='y')
    axs[0, 1].scatter(filtered_initial_real.radius, filtered_initial_real.dot_radius, c='r', zorder=1000)

    axs[1, 0].set_ylim(mmin, mmax)
    axs[1, 0].set_ylabel('Mass outflow rate ($M_\odot$ yr$^{-1}$)')
    axs[1, 0].set_xlabel('AGN luminosity (erg s$^{-1}$)')
    axs[1, 0].plot(sorted_by_time_outflow.luminosity_AGN, sorted_by_time_outflow.dot_mass)
    axs[1, 0].plot([10**filtered_initial_real.lum_min_log, 10**filtered_initial_real.lum_max_log], [filtered_initial_real.derived_dot_mass, filtered_initial_real.derived_dot_mass], c='r')
    axs[1, 0].plot([10**filtered_initial_real.luminosity_AGN_log, 10**filtered_initial_real.luminosity_AGN_log], [filtered_initial_real.mdot_min, filtered_initial_real.mdot_max], c='r')
    axs[1, 0].scatter(10**filtered_initial_real.luminosity_AGN_log, filtered_initial_real.derived_dot_mass, c='r', zorder=1000)

    axs[1, 1].set_xlabel('Radius (kpc)')
    axs[1, 1].tick_params(left=False, right=True, direction='in', grid_linewidth=40, grid_color='green')
    axs[1, 1].plot(sorted_by_time_outflow.radius, sorted_by_time_outflow.dot_mass)
    axs[1, 1].plot([filtered_initial_real.radius/10**0.15, filtered_initial_real.radius*10**0.15], [filtered_initial_real.derived_dot_mass, filtered_initial_real.derived_dot_mass], c='y')
    axs[1, 1].plot([filtered_initial_real.radius, filtered_initial_real.radius], [filtered_initial_real.mdot_min, filtered_initial_real.mdot_max], c='r')
    axs[1, 1].scatter(filtered_initial_real.radius, filtered_initial_real.derived_dot_mass, c='r', zorder=1000)

    fig.suptitle(str(name).split("_")[0])

    fig = plt.gcf()
    fig.set_size_inches(9.0, 6.2)
    plt.savefig(f"./figures/real_outflows/{name}.png", dpi=300)
    plt.close()
