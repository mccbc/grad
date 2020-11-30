import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import pdb

# Set up figure
fig, ax = plt.subplots(1, 1)

# Collect data from external files
cluster_data = np.loadtxt('GalWCls19.txt', skiprows=41)
galaxy_data = np.loadtxt('GalWGal.txt', skiprows=9)

# Bin the clusters by mass
n_bins = 20
cluster_masses = cluster_data[:, 18]
mass_bins = np.linspace(min(cluster_masses), max(cluster_masses), n_bins)
inds = np.digitize(cluster_masses, mass_bins)

colors = pl.cm.jet(np.linspace(0, 1, n_bins))

# For each mass bin in the sample...
for i in range(1, n_bins):
    # Narrow the sample to clusters in this mass range
    clusters = cluster_data[inds == i]
    center_distances = np.array([])
    elliptical_flags = np.array([])

    # Loop over all the clusters in this group
    for j in range(len(clusters)):
        # Collect galaxies in this cluster by matching cluster ID
        galaxies = galaxy_data[galaxy_data[:, 10] == clusters[j, 0]]

        ### Calculate the distance of each galaxy to the cluster center
        r_gal = galaxies[:, 3]
        r_c = clusters[j, 5]
        d_los = np.abs(r_gal - r_c) # Line-of-sight distance

        # Get sky-projected distance
        dRA = galaxies[:, 0] - clusters[j, 1]
        dDEC = galaxies[:, 1] - clusters[j, 2]
        c = np.arccos(np.cos(dRA)*np.cos(dDEC)) # Distance on sky in degrees
        d_sky = r_gal * c # Distance on sky in same units as r_gal

        # Distance from galaxies in this sample to this cluster's center
        d = np.sqrt(d_los**2 + d_sky**2)
        is_elliptical = galaxies[:,9] == 2

        # Add data from this cluster to the mass group's data
        center_distances = np.concatenate((center_distances, d))
        elliptical_flags = np.concatenate((elliptical_flags, is_elliptical))

    # Bin by distance to create x and y data
    n_dist_bins = 50

    dist_bins = np.linspace(min(center_distances), max(center_distances), n_dist_bins)
    dist_inds = np.digitize(center_distances, dist_bins)

    # Find fraction of ellipticals in each distance bin    
    all_bincenters = (dist_bins[:-1] + dist_bins[1:])/2
    frac_ellip = []
    bincenters = []
    for k in range(1, len(dist_bins)):
        if len(elliptical_flags[dist_inds == k]) != 0:
            frac_ellip.append(sum(elliptical_flags[dist_inds == k])/len(elliptical_flags[dist_inds == k]))
            bincenters.append(all_bincenters[k-1])

    # Plot results
    ax.plot(bincenters, frac_ellip, c=colors[i], alpha=0.75)



sm = plt.cm.ScalarMappable(cmap=pl.cm.jet, norm=plt.Normalize(vmin=min(cluster_masses), vmax=max(cluster_masses)))
cbar = fig.colorbar(sm)
cbar.ax.set_ylabel('Cluster Mass', rotation=90)
