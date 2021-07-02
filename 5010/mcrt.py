import numpy as np
import random as r
import time
from astropy.utils.console import ProgressBar


# ------------------------------ PARAMETERS ---------------------------------
n_phot = 10000  # number of photons
n_bins = 250  # number of scattering angle bins
chi = 1.  # extinction opacity, cm^-1
tau_f = 6.  # maximum optical depth
outfile = 'run4'
# ---------------------------------------------------------------------------


class Photon(object):
    def __init__(self, mu=0., phi=0., tau=0., x=0., y=0., z=0.):
        self.mu = mu
        self.phi = phi
        self.tau = tau
        self.x = x
        self.y = y
        self.z = z
        self.nsteps = 0

    def step(self):
        '''
        Take one step in the random walk by generating a scattering angle and
        travel distance, in terms of tau. Also track the total number of steps
        taken and the photon's current position in xyz coordinates.
        '''
        # Generate new scattering angle and optical depth
        self.mu = 2.*r.random() - 1.
        self.theta = np.arccos(self.mu)
        self.phi = 2.*np.pi*r.random()
        self.tau += self.mu * np.negative(np.log(1. - r.random()))
        self.nsteps += 1

        # Only update positions if photon hasn't exited yet
        if self.tau <= tau_f:
            self.x += (self.tau / chi) * np.sin(self.theta) * np.cos(self.phi)
            self.y += (self.tau / chi) * np.sin(self.theta) * np.sin(self.phi)
            self.z += (self.tau / chi) * np.cos(self.theta)

    def terminate(self, params, dataframe):
        '''
        Use the photon's exit angle to assign bins, calculate image position,
        and export all quantities to the dataframe.
        '''
        # Bin the exit angle
        i = np.digitize(self.mu, params['mu_bins'])
        j = np.digitize(self.phi, params['phi_bins'])

        # Find image position
        xi = self.z * np.sin(self.theta) \
             - self.y * np.cos(self.theta) * np.sin(self.phi) \
             - self.x * np.cos(self.theta) * np.cos(self.phi)
        yi = self.y * np.cos(self.phi) - self.x * np.sin(self.phi)

        # Update the dataframe
        dataframe['angle'] = i, j
        dataframe['position'] = xi, yi
        dataframe['nsteps'] = self.nsteps


data = np.zeros(n_phot, dtype=[('angle',    float, (2,)),
                               ('position', float, (2,)),
                               ('nsteps',   float, (1,))])

param_tup = (n_phot, n_bins, chi, tau_f, np.linspace(0, 1, n_bins), 
             np.linspace(0, 2*np.pi, n_bins))

params = np.array(param_tup, dtype=[('n_phot',     int,      (1,)),
                                    ('n_bins',     int,      (1,)),
                                    ('chi',      float,      (1,)),
                                    ('tau_f',    float,      (1,)),
                                    ('mu_bins',  float, (n_bins,)),
                                    ('phi_bins', float, (n_bins,))])

# PHOTON LOOP
pb = ProgressBar(n_phot)
for k in range(n_phot):
    p = Photon()
    while p.tau <= params['tau_f']:
        p.step()
    else:
        p.terminate(params, data[k])
    pb.update()

# Save data
np.save(outfile+'.npy', data)
np.save(outfile+'_params.npy', params)

print('\nOutput saved to '+outfile+'.npy')
