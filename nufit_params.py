
import numpy as np

# from NuFit 6.0, http://www.nu-fit.org/?q=node/294
# with IceCube and Super-Kamiokande atmospheric data

# degrees
theta12 = 33.68
theta12_err = 0.5*(0.73+0.70)
theta23 = 43.3
theta23_err = 0.5*(1.0+0.8)
theta13 = 8.56
theta13_err = 0.5*(0.11+0.11)
deltaCP = 212
deltaCP_err = 0.5*(26+41)
# eV^2
delta2_m21 = 7.49e-5
delta2_m21_err = 0.5*(0.19+0.19)*1e-5
delta2_m3l = 2.513e-3
delta2_m3l_err = 0.5*(0.021+0.019)*1e-3

# inverted ordering

# degrees
theta12_inv = 33.68
theta12_inv_err = 0.5*(0.73+0.70)
theta23_inv = 47.9
theta23_inv_err = 0.5*(0.7+0.9)
theta13_inv = 8.59
theta13_inv_err = 0.5*(0.11+0.11)
deltaCP_inv = 274
deltaCP_inv_err = 0.5*(22+25)
# eV^2
delta2_m21_inv = 7.49e-5
delta2_m21_inv_err = 0.5*(0.19+0.19)*1e-5
delta2_m3l_inv = -2.484e-3
delta2_m3l_inv_err = 0.5*(0.020+0.020)*1e-3

theta12 = np.radians(theta12)
theta12_err = np.radians(theta12_err)
theta23 = np.radians(theta23)
theta23_err = np.radians(theta23_err)
theta13 = np.radians(theta13)
theta13_err = np.radians(theta13_err)
deltaCP = np.radians(deltaCP)
deltaCP_err = np.radians(deltaCP_err)

theta12_inv = np.radians(theta12_inv)
theta12_inv_err = np.radians(theta12_inv_err)
theta23_inv = np.radians(theta23_inv)
theta23_inv_err = np.radians(theta23_inv_err)
theta13_inv = np.radians(theta13_inv)
theta13_inv_err = np.radians(theta13_inv_err)
deltaCP_inv = np.radians(deltaCP_inv)
deltaCP_inv_err = np.radians(deltaCP_inv_err)

def get_PMNS(theta12, theta23, theta13, deltaCP, alpha1, alpha2):

    rot_23 = np.array([[
        [1, 0, 0],
        [0, np.cos(theta23), np.sin(theta23)],
        [0, -np.sin(theta23), np.cos(theta23)]
        ]])

    rot_13 = np.array([[
        [np.cos(theta13), 0, np.sin(theta13)*np.exp(-1j*deltaCP)],
        [0, 1, 0],
        [-np.sin(theta13)*np.exp(1j*deltaCP), 0, np.cos(theta13)]
        ]])

    rot_12 = np.array([[
        [np.cos(theta12), np.sin(theta12), 0],
        [-np.sin(theta12), np.cos(theta12), 0],
        [0, 0, 1]
        ]])
    
    majorana = np.diag([np.exp(1j*alpha1), np.exp(1j*alpha2), 1])

    return (rot_23 @ rot_13 @ rot_12 @ majorana)[0]


