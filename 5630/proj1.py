from astropy.io import ascii
import astropy.constants as c
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import numpy as np

def rms(x):
    return np.sqrt(np.sum(x**2)/len(x))

# Load the tables
table1 = ascii.read('lvg_table1.dat')
table2 = ascii.read('lvg_table2.dat')

# Pick out data we actually need
v_gal = table2['VLG']
distance = table1['Dis']
mags = table2['BMag']

# Get rid of the masked values in the Column object
mask = ~(v_gal.mask ^ mags.mask)
v_gal = v_gal[mask]
distance = distance[mask]
mags = mags[mask]

# Calculate the deviation from the Hubble flow
H = 70. * u.Unit('km/(s * Mpc)')
dV = v_gal - H * distance

# Bin by distance or magnitude and calculate the RMS in each bin
def bin_data(indep_var):
    bins = np.linspace(min(indep_var), max(indep_var), 20)
    inds = np.digitize(indep_var, bins)
    rms_array = [rms(dV[inds == i]) for i in range(1, len(bins))]
    bincenters = (bins[:-1] + bins[1:])/2
    return bincenters, rms_array

dist_bins, dist_rms = bin_data(distance)
mag_bins, mag_rms = bin_data(mags)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6), sharey=True)
ax1.scatter(distance, np.abs(dV), s=2, c='k', alpha=0.25, label='Galaxy Data')
ax1.plot(dist_bins, dist_rms, 'm--', label='Binned RMS')
ax1.set_xlabel('Distance (Mpc)')
ax1.grid(alpha=0.25)
ax1.legend()
ax1.set_ylabel('$|dV|$ = $|V_{gal} - HR|$ (km/s)')
ax2.plot(mag_bins, mag_rms, 'm--', label='Binned RMS')
ax2.scatter(mags, np.abs(dV), s=2, c='k', alpha=0.25, label='Galaxy Data')
ax2.set_xlabel('Absolute B Magnitude')
ax2.grid(alpha=0.25)
plt.suptitle('Hubble Flow Deviation')
plt.legend()
plt.tight_layout()
plt.subplots_adjust(top=0.925,
bottom=0.119,
left=0.094,
right=0.981,
hspace=0.,
wspace=0.)
plt.show()

