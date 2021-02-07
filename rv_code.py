import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.signal as sig
from scipy.optimize import curve_fit, fsolve
from copy import copy
import scipy.special as sps
import scipy.integrate as integrate
plt.rc('font', family='serif')
plt.rc('mathtext', fontset='cm')
import pdb

G = 4*np.pi**2 /(356.25)**2 # G in AU^3/(day^2 Solar Mass)

##### Load data #######
name = 'HD 10442' # Name of the Star
mass = 1.56 # Stellar mass 
mass_err = 0.09 # Stellar mass error
jitter = 4.7 # Jitter value (instrument error)
data = np.loadtxt('HD 10442.dat')
data = np.transpose(data)
t = np.array(data[0], dtype='float64')
vr = np.array(data[1], dtype='float64')
vr_err = np.array(data[2], dtype='float64')
vr_err = np.sqrt(vr_err**2 + jitter**2)

######## Plot data ##########
fig, ax = plt.subplots(figsize = [10, 6])
plt.subplots_adjust(left=0.17,bottom=0.15,right=0.97,top=0.97)
ax.errorbar(t, vr, yerr=vr_err, fmt='o', capsize= 2, color='black', markersize=5, linestyle='none', mfc='black', mec='black')
ax.tick_params(which='both', axis='both', direction='in', top=True, right=True, labelsize=14)
ax.tick_params(which='major', length=5)
ax.set_xlabel(r'Time  (Julian Days)', fontsize=14)
ax.set_ylabel(r'Radial Velocity (m/s)', fontsize=14)
ax.minorticks_on()
fig.savefig(name + '_rv.png')

###### Get best n ############
def periodogram():
	periods = 10**np.linspace(-3, 4, 1000)
	omega = 2*np.pi/periods
	power = sig.lombscargle(t - t[0], vr, omega)
	max_pers = np.argsort(power)
	return omega[max_pers[-1]]

def periodogram_plot():	
	#Plot periodogram
	periods = 10**np.linspace(-3, 4, 1000)
	omega = 2*np.pi/periods
	power = sig.lombscargle(t - t[0], vr, omega)
	fig = plt.figure()
	ax = fig.add_subplot()
	ax.semilogx(periods, power, color = 'k')
	ax.set_xlabel('Period (days)', fontsize=12)
	ax.set_ylabel('Power', fontsize=12)
	ax.minorticks_on()
	fig.savefig(name + '_periodogram.png')
	
def solve_kep_eqn(l, e):
    """ Solve Keplers equation x - e*sin(x) = l for x"""
    try:
    	l[0]
    	res = np.zeros(l.shape)
    	for i, li in enumerate(l):
    		tmp,= fsolve(lambda x: x-e*np.sin(x) - li, li)
    		res[i] = tmp
    except IndexError:
    	res, = fsolve(lambda x: x - e*np.sin(x)-l, l)
    return res
	
def model(t, n, tau, k, w, e):
    """Obtain the radial velocity due to a single planet """
    e_anom = solve_kep_eqn(n*(t-tau), e)
    f = 2*np.arctan2(np.sqrt(1+e)*np.sin(e_anom*.5), np.sqrt(1-e)*np.cos(e_anom*.5))
    RV = k*(np.cos(f + w) + e*np.cos(w))
    return RV

def fit_data():
    """ Fit the RV data with the model """
    # Initial guesses to the parameters
    n0 = periodogram()
    k0 = 40  # Semi-amplitude equal to the maximum stellar velocity
    tau0 = t[ vr == k0]
    if len(tau0) == 0:
        tau0 = t[0]
    w0 = 0; e0 = 0.5
    initial_guess = (n0, tau0, k0, w0, e0)

    # Fit the data
    popt,pcov = curve_fit(model, t, vr, sigma = vr_err, absolute_sigma = True, p0 = initial_guess)
    return popt, pcov

def planet_params():
	n, tau, k, w, e = fit_data()[0]
	if e<0:
		w += np.pi
		e *= -1
	if k<0:
		k*=-1
		w += np.pi
	k1 = (G*n)**(1./3) * (1 - e**2)**(-1./2)
	mass_f = k/k1

	# Assuming most of the mass is in the star
	mp = mass_f * mass**(2./3) # Planet mass in solar masses
	mp *= 9.543e-4 # Planet mass in Jupiter masses
	p = 2*np.pi/n
	a = (G*mass/n**2)**(1./3)
	return mp, e, p, w, a

def errors():
	n, tau, k, w, e = fit_data()[0]
	mp, e, p, w, a = planet_params()
	pcov = fit_data()[1]
	ms = mass; ms_err = mass_err
	n_err = np.sqrt(pcov[0,0])
	k_err = np.sqrt(pcov[2,2])
	w_err = np.sqrt(pcov[3,3])
	p_err = p*np.abs(n_err/n)
	e_err = np.sqrt(pcov[4,4])
	a_err = a*np.sqrt( (2*n_err/(3*n))**2 + (ms_err/(3*ms))**2 )
	mp_err = mp*np.sqrt((2*ms_err/(3*ms))**2 + (k_err/k)**2 + (n_err/(3*n))**2 + (e*e/(1-e*e))**2 *(e_err/e)**2)
	return mp_err, e_err, p_err, w_err, a_err

def results():
	mp, e, p, w, a = planet_params()
	mp_err, e_err, p_err, w_err, a_err = errors()
	outstr = 'Mp > {:.3f} +\- {:.4f} Mj\n'.format(mp, mp_err)
	outstr += 'e = {:.3f} +\- {:.4f}\n'.format(e, e_err)
	outstr += 'P =  {:.3f} +\- {:.4f} days\n'.format(p, p_err)
	outstr += 'a = {:.3f} +\- {:.4f} AU\n'.format(a, a_err)
	print(outstr)

########## Define the parameters
n, tau, k, w, e = fit_data()[0]
mp, e, p, w, a = planet_params()
mp_err, e_err, p_err, w_err, a_err = errors()
t_fit = np.linspace(t[0], t[-1], 1000)
rv = np.array(model(t_fit, n, tau, k, w, e))
residuals = vr - np.array([model(t, n, tau, k, w, e) for x in t])
residuals = residuals[0]

######## Plot periodgram and Print results
periodogram_plot()
print(results())

############# PLOT DATA AND FIT RESULT
fig = plt.figure(figsize=[10,6])
plt.subplots_adjust(left=0.15,bottom=0.1,right=0.97,top=0.99)
gs = gridspec.GridSpec(2,1,height_ratios=[1,0.25])
gs.update(hspace=0.00)

ax1 = plt.subplot(gs[0,0])
ax2 = plt.subplot(gs[1,0],sharex=ax1)
plt.setp(ax1.get_xticklabels(), visible=False)

# Set tick attributes (direction inwards, tick width and length for both axes)
for ax in [ax1, ax2]:
    ax.minorticks_on()
    ax.tick_params(which='both',axis='both',direction='in',top='True',right='True',labelsize=14)
    ax.tick_params(which='major', length=10)
    ax.tick_params(which='minor', length=5)

# Plot Data
ax1.errorbar(t, vr, yerr=vr_err, fmt='o', capsize= 2, color='black', markersize=5, linestyle='none', mfc='black', mec='black')
legstr = 'Mp > {:.2f} $\pm$ {:.2f} Mj \n e = {:.3f} $\pm$ {:.2f} \n P = {:.1f} $\pm$ {:.2f} days \n a = {:.2f} $\pm$ {:.2f} AU'\
		.format(mp, mp_err, e, e_err, p, p_err, a, a_err)
ax1.plot(t_fit, rv, color = 'b', linewidth = 1.5, label = legstr)
ax1.legend(frameon=False,fontsize=10,loc='lower left')
ax1.set_xlabel(r'Time  (Julian Days)', fontsize=12)
ax1.set_ylabel(r'Radial Velocity (m/s)', fontsize=12)
ax1.minorticks_on()

# Plot Residuals
ax2.errorbar(t, residuals, yerr=vr_err, fmt='o', capsize= 2, color='black', markersize=5, linestyle='none', mfc='black', mec='black')
ax2.axhline(0, color='b', linewidth = 1.5)
ax2.get_yticklabels()[-1].set_visible(False)
ax2.minorticks_on()
ax2.set_ylabel('Residuals', fontsize = 12)
ax2.set_xlabel('Time (Julian Days)', fontsize = 12)
fig.savefig(name + '_results.png')

#plt.show()
pdb.set_trace()  ### STOP