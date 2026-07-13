import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import jax
import jax.numpy as jnp
from jax import  jit
jax.config.update('jax_platform_name', 'cpu')
import scipy
from scipy import special as sp
from scipy.signal import convolve
from jax.scipy.signal import convolve
from scipy.ndimage import gaussian_filter1d
import os
import pickle
# import dill
from os.path import dirname, realpath
from cycler import cycler
import seaborn as sns


def init_vars(params):
    """
    initiates t, end, x for easy initial plotting and functions
    """
    # build time and space constraints
    x = jnp.linspace(params['xmin'], params['xmax'], params['Nx'])
    end = params['dt']*params['total_n']*params['compiled_steps'] # final time step
    t = jnp.linspace(params['start'],end,params['total_n'])  # time array
    return x,t,end


def sawtooth(t, width=1):
    """
    jaxified from scipy.signal sawtooth
    """
    t, w = jnp.asarray(t), jnp.asarray(width)
    w = jnp.asarray(w + (t - t))
    t = jnp.asarray(t + (w - w))
    y = jnp.zeros(t.shape)

    # width must be between 0 and 1 inclusive
    mask1 = (w > 1) | (w < 0)
    y = jnp.where(mask1,jnp.nan,y)
    tmod = jnp.mod(t, 2 * jnp.pi)

    # on the interval 0 to width*2*pi function is
    #  tmod / (pi*w) - 1
    mask2 = (1 - mask1) & (tmod < w * 2 * jnp.pi)
    tsub = jnp.where(mask2,tmod,y)
    wsub = jnp.where(mask2,w,y)
    y = jnp.where(mask2,tsub/(jnp.pi*wsub)-1,y)

    return y 


def sin_exp(sinusoid, x, rate=3.0):
    """
    plots sinusoid with exponential from peak to trough
    curve = sin_exp(sl_fun(t),t,rate=6)
    test = jnp.interp(t,jnp.linspace(0,end,curve.size),curve)
    plt.plot(t,test)
    """
    # Function to scale and shift the exponential decay to match the modified_sinusoid's amplitude at the peak and trough
    def scaled_exp_decay(x, peak_x, trough_x, peak_y, trough_y):
        scale = peak_y - trough_y  # Scale factor
        shift = peak_x  # Shift to start from the peak
        decay_rate = rate / (trough_x - peak_x)  # Adjusted decay rate
        return trough_y + scale * jnp.exp(-decay_rate * (x - shift))

    # Copy of the original modified_sinusoid to apply modifications
    modified_sinusoid = jnp.array(sinusoid)

    # Iterate through the modified_sinusoid to find peaks and apply scaled exponential decay to each peak-to-trough segment
    for i in range(1, len(x) - 1):
        if modified_sinusoid[i] > modified_sinusoid[i - 1] and modified_sinusoid[i] > modified_sinusoid[i + 1]:
            # Peak found
            # Find the next trough
            for j in range(i + 1, len(x) - 1):
                if modified_sinusoid[j] < modified_sinusoid[j - 1] and modified_sinusoid[j] < modified_sinusoid[j + 1]:
                    trough_index = j
                    break
            else:
                trough_index = len(x) - 1  # Use the end if no trough is found

            # Apply scaled exponential decay from the peak to the trough
            for k in range(i, trough_index + 1):
                modified_sinusoid = modified_sinusoid.at[k].set(
                    scaled_exp_decay(x[k], x[i], x[trough_index], modified_sinusoid[i], modified_sinusoid[trough_index]))
    
    return modified_sinusoid


def save_object(var, path):
    """
    Save variable/object as a pickle object.
    Parameters
    ----------
    var:
        Object to be saved.
    path: str or path to file
    """
    with open(path+'.pkl', "wb") as buff:
        pickle.dump(var, buff)


def load_object(path):
    """
    Custom load command for pickle object.
    Parameters
    ----------
    path: str or path to file
    Returns
    -------
    var:
       Variable saved in ``path``.
    """
    with open(path + '.pkl', "rb") as input_file:
        return pickle.load(input_file)


def dh_values(sto,means=False,out=False):
    """
    calculate dh between timesteps when eroding and depositing
    """
    dhs_er = []
    dhs_dep = []
    for i in range(len(sto.T)):
        bedf = sto[::2][:,i]
        bedi = sto[::2][:,i-1]
        proxy = sto[1::2][:,i]
        dh = bedf-bedi
        dhe = np.where(dh<0,dh,np.nan) # only take eroded values
        dhd = np.where(dh>0,dh,np.nan) # only take deposited values # not taking >= as obviously no dh
        dhs_dep.append(dhd)
        dhs_er.append(dhe)
    dhs_er = np.array(dhs_er) 
    dhs_dep= np.array(dhs_dep) 
    
    dhs_er = dhs_er[~np.isnan(dhs_er)] # get rid of nans (depositing)
    dhs_dep = dhs_dep[~np.isnan(dhs_dep)] # get rid of nans (eroding)

    if means:
        dhs_er = np.mean(np.abs(dhs_er))
        dhs_dep = np.mean(dhs_dep)
    if out:
        print(f'mean eroded distance = {np.mean(np.abs(dhs_er)):.2e}; mean deposited distance = {np.mean(dhs_dep):.2e}')

    return dhs_er, dhs_dep


def round_2dec(x,decimal,way='up'):
    """
    better way to round for calculating ymin and ymax
    """
    factor = 10**decimal
    if way=='up':
        r = np.ceil(x*factor)/factor
    elif way=='down':
        r = np.floor(x*factor)/factor
    return r

def normalize_y(y, array_length, y_min, y_max):
    """
    normalize y values between new range for plotting pixel coordinates as correct topography
    used in strat col funciton
    """
    norm_y = y_min+(y_max-y_min)*y/(array_length-1)
    return norm_y 


def normalize_topo(value, old_min, old_max, new_min, new_max):
    """
    normalize topographic range to be between 0 and the change in height
    used in strat col function
    """
    norm_topo = ((value-old_min)/(old_max-old_min))*(new_max-new_min)+new_min
    return norm_topo


def center_ticks(data):
    """
    centers ticks for categorical colorbar
    """
    ticks = (np.arange(len(data))+0.5)*(len(data)-1)/len(data)
    return ticks 


@jit
def add_lump(condition, array, value):
    """
    lets me use if else statements in jit compiled functions
    this is for setting lumps to values at certain times (condition is time (i) == some time)
    - note the grid xycoordinates depend on the EXACT ymin and ymax I use, so will need to change that depending on how I do the ymin ymax, does not effect anything
    other than this lump placement
    """
    def true_fun(array, value): # set values
        #x_start, x_end = 47, 65  # works better for flat topo
        #y_start, y_end = 200, 250 
        x_start, x_end = 42, 62
        y_start, y_end = 110, 210 
        #x_start, x_end = 46, 60 # for depth growth
        #y_start, y_end = 120, 140 
        return array.at[y_start:y_end, x_start:x_end].set(value)
    
    def false_fun(array, value): # do nothing
        return array
    
    result = jax.lax.cond(condition, true_fun, false_fun, array, value)
    return result


def bosscher_G(depth=np.linspace(0,100,100),Gm=0.0125,k=0.1,Io=2000,Ik=450,Dt=0,out='both',Iz_base=0,G_base=0):
    """
    ## Bosscher and Schlager, 1992; Monastrea Annularis
    z = np.linspace(0,100,100) # depth (m) , positive downwards
    Gm = 0.0125                 # max growth rate (10-15 mm/yr) converted to m
    k = 0.1                     # extinction coefficient (0.04 to 0.16 1/m)
    Io = 2000                   # surface light intensity (2000-2250 uE/m2*s) (basically max light intensity)
    Ik = 450                    # saturation light intensity (50-450 uE/m2*s)
    Iz = Io*np.exp(-k*z)        # light intensity (at depth z) -> Beer-Lambert Law
    #Pm = 1 # max photosynthetic rate
    # P = Pm*np.tanh(I/Ik) # photosynthetic rate, but as  photosynthetic rate, calcification rate and skeletal growth rate areproportional (Chalker et at., 1988), P can be replaced by skeletal growth rate (G)
    # G = Gm*np.tanh(I/Ik) # skeletal growth; but can replace I from Iz from Beer-Lambert Law
    G = Gm*np.tanh(Io*np.exp(-k*z)/Ik) # final skeletal growth rate
    G_base = offset for minimum
    Iz_base = offset for minimum
    """
    z = depth
    Iz = Io*jnp.exp(-k*z)+Iz_base        # light intensity (at depth z) -> Beer-Lambert Law
    G = Gm*jnp.tanh(Io*jnp.exp(-k*z)/Ik)+G_base # final skeletal growth rate

    # no growth or light above sea surface
    Iz = jnp.where(depth<0,0,Iz)
    G = jnp.where(depth<0,0,G)

    # tidal depth
    rounded_depth = depth #jnp.round(depth, decimals=0) # prevents the side by side floating point error when spatially adjacent samples are constant; decimals depends on the run
    G = jnp.where(rounded_depth<Dt,0,G)
    Iz = jnp.where(rounded_depth<Dt,0,Iz)

    # single return options for jax functions (easier to avoid tuple in current framework)
    if out=='both':
        return Iz,G
    elif out=='light':
        return Iz
    elif out=='growth':
        return G
    
def erf_G(z,Gmax,peak_width,G_depth):
    """
    uses the error function to plot pulses of growth with depth
    uses jax version
    """
    R = Gmax*(1-jax.scipy.special.erf((z-G_depth)/peak_width)**2)

    return R

def erf_G_asym(z, Gmax, G_depth, width_shallow=30, width_deep=80):
    """
    Asymmetric erf growth function with different steepness on each side of the peak.
    - width_shallow: controls slope on the shallow side (z < G_depth)
    - width_deep: controls slope on the deep side (z > G_depth)
    """
    shallow_side = z < G_depth
    deep_side = ~shallow_side

    shallow_term = 1 - jax.scipy.special.erf((z - G_depth) / width_shallow)**2
    deep_term = 1 - jax.scipy.special.erf((z - G_depth) / width_deep)**2

    return Gmax * (shallow_term * shallow_side + deep_term * deep_side)

    
def seq_id(topo,threshold=1e-6,fill_val=1):
    """
    finds significant changes in topography over time, related to hiatuses or erosional periods associated with sea level falls
    -> sequence boundaries
    - takes the gradient over time and finds where this gradient is decreasing
    - doesn't work well with the downsampled grids from make_image so switched to an approach using im_t where we look for large jumps (unconformities in time)

    example usage:
    seq = seq_id(topo=mod['beds'].T,threshold=1e-3)
    plt.imshow(seq,aspect='auto',cmap='Greys',interpolation='nearest');
    plt.colorbar()

    tst = np.where(seq==1,1000,beds.T).T # changed to beds_eroded for now as that is what im_t and such are based upon
    #plt.plot(tst.T,'.',color='tab:red',markersize=1,alpha=0.5);
    print(tst.shape)
    plt.imshow(tst.T,aspect='auto',cmap='coolwarm',interpolation='nearest');
    plt.colorbar()
    """
    # take gradient of topography over time
    grad = np.gradient(topo, axis=1)

    # sign of the gradient
    grad_sign = np.sign(grad)
    
    # find sign changes
    significant_sign_changes = np.zeros_like(grad_sign)
    # compare element with next to flag sign changes given a threshold excluding last element to avoid index out of bounds error
    significant_sign_changes[:-1] = (grad_sign[:-1]!=grad_sign[1:])&(np.abs(grad[:-1])>threshold) # can modify threshold for sensitivity to gradient sign changes
        
    change_mask = np.zeros_like(topo)

    # flag sequence boundaries with fill_val
    change_mask = np.where(significant_sign_changes,fill_val,change_mask)

    return change_mask

def imshow(img,ax=None,out=True,cbar=True,cmap='coolwarm',**kwargs):
    """
    pre made version of plt.imshow for my sed transport arrays to avoid typing the same kwargs all the time
    """
    ax0 = ax if ax else plt.gca()
    im = ax0.imshow(img,aspect='auto',origin='lower',interpolation='nearest',cmap=cmap,**kwargs)
    if cbar:
        cb = plt.colorbar(im)
        if out:
            return im,cb
    if out:
        return im
    
def gaussian_kernel_F(size, sigma):
    """Generate a Gaussian kernel using JAX.
    
    Args:
        size (int): The size of the kernel.
        sigma (float): The standard deviation of the kernel.
        
    Returns:
        jnp.ndarray: A 1D array containing the Gaussian kernel.
    """
    x = jnp.arange(size) - size // 2
    kernel = jnp.exp(-(x**2)/(2*sigma**2))
    #kernel /= kernel.sum()  # Normalize the kernel; ignoring this as not the behaviour as scipy.signal.gaussian; want the kernel to peak at 1
    return kernel
    

def prep_convolve(array,val,sigma,mode='same'):
    """
    jax version of convolution with padding to remove effect on boundaries
    """
    kernel_size = int(sigma*3)*2+1  # 3 sigma rule, adjust as necessary
    gaussian_kernel = gaussian_kernel_F(kernel_size,sigma)
    
    # Pad the array to handle boundaries
    pad_width = kernel_size//2
    array = jnp.pad(array, pad_width, mode='constant', constant_values=(val, val))
    array = convolve(array, gaussian_kernel, mode=mode) / gaussian_kernel.sum() # normalize the convolution to sum; need this to prevent signal explosion (keeps convolution max the same as capacity)
    
    # Remove the padding
    array = array[pad_width:-pad_width]

    return array

def create_excursions(array, excursions, locs, rise_widths, fall_widths, rise_sigmas, fall_sigmas):
    """
    Create non-symmetrical excursions in the array with smooth transitions.

    Parameters:
    array (numpy.ndarray): Initial array of values.
    excursions (list): List of excursion magnitudes.
    locs (list): List of time points where excursions occur.
    rise_widths (list): List of rise widths (number of points for the excursion to reach its peak).
    fall_widths (list): List of fall widths (number of points for the excursion to return to baseline).
    rise_sigmas (list): List of standard deviations for Gaussian kernel during rise.
    fall_sigmas (list): List of standard deviations for Gaussian kernel during fall.

    Returns:
    numpy.ndarray: Array with excursions applied.
    """
    result = np.copy(array)
    temp_arrays = []

    for excursion, loc, rise_width, fall_width, rise_sigma, fall_sigma in zip(excursions, locs, rise_widths, fall_widths, rise_sigmas, fall_sigmas):
        temp_array = np.zeros_like(array)
        
        start_rise = max(0, loc - rise_width)
        peak = loc
        end_fall = min(len(array), loc + fall_width)
        
        # Create rise and fall transitions
        rise_transition = np.linspace(0, excursion, peak - start_rise)
        fall_transition = np.linspace(excursion, 0, end_fall - peak)
        
        # Apply Gaussian smoothing
        smoothed_rise = gaussian_filter1d(rise_transition, sigma=rise_sigma)
        smoothed_fall = gaussian_filter1d(fall_transition, sigma=fall_sigma)
        
        # Combine rise and fall transitions
        smoothed_transition = np.concatenate([smoothed_rise, smoothed_fall])
        
        # Apply the excursion to the temporary array
        transition_start = start_rise
        transition_end = transition_start + len(smoothed_transition)
        temp_array[transition_start:transition_end] += smoothed_transition[:transition_end - transition_start]
        
        temp_arrays.append(temp_array)
    
    for temp_array in temp_arrays:
        result += temp_array
    
    # Smooth the entire result to ensure continuity between excursions
    final_result = gaussian_filter1d(result, sigma=min(rise_sigmas + fall_sigmas))
    
    return final_result

def sign_jumps(arr,before=True):
    """
    find sign jumps in shoreline movement to track transgressive and regressive surfaces
    """
    positive_changes = []
    negative_changes = []
    i = 0
    while i < len(arr) - 2:
        if arr[i] == 1:
            j = i + 1
            while j < len(arr) and arr[j] == 0:
                j += 1
            if j < len(arr) and arr[j] == -1:
                positive_changes.append(i if before else j)
                i = j  # move to the next position after the detected pattern
            else:
                i += 1
        elif arr[i] == -1:
            j = i + 1
            while j < len(arr) and arr[j] == 0:
                j += 1
            if j < len(arr) and arr[j] == 1:
                negative_changes.append(i if before else j)
                i = j  # move to the next position after the detected pattern
            else:
                i += 1
        else:
            i += 1
    
    # convert lists to arrays            
    positive_changes = np.array(positive_changes)
    negative_changes = np.array(negative_changes)

    return positive_changes, negative_changes

def norm01(arr):
    """
    normalize values from 0 to 1
    """
    norm_arr = (arr - np.nanmin(arr)) / (np.nanmax(arr) - np.nanmin(arr))
    return norm_arr

def sbar(x_value,ymin,ymax,capsize=5,lw=1,ax=None,**kwargs):
    ax0 = plt.gca() if not ax else ax
    # midpoint and error values
    y_mid = (ymin + ymax) / 2
    yerr = [[y_mid - ymin], [ymax - y_mid]]  # Absolute distances
    ax0.errorbar(x_value, y_mid, yerr=yerr, fmt='None', capsize=capsize,lw=lw,capthick=lw,**kwargs)

def set_cycler(ax,n_colors,cmap='viridis'):
    """
    set color cycle for plotting
    short form for
    # ax.set_prop_cycle(cycler('color',sns.color_palette('coolwarm',n_colors=len(locs))))

    """
    
    ax.set_prop_cycle(cycler('color',sns.color_palette(cmap,n_colors=n_colors)))

def rem_ol(arr,low=1,high=99):
    """
    remove outliers lower than the lowest 1% and higher than the highest 99% percentiles
    - these outliers are usually a few pixels in each image and make colourbars unreadable
    """
    pc1 = jnp.nanpercentile(arr, low)
    pc99 = jnp.nanpercentile(arr, high)
    arr1 = jnp.clip(arr, pc1, pc99)
    return arr1

def img(w,o,total=1999,ylen=1000):
    
    a = complex(str(o.beds[0].size)+'j')
    b = complex(str(ylen)+'j') #500j
    xi, yi = np.mgrid[o.xmin:o.xmax:a, o.ymin:o.ymax:b]
    
    X = np.tile(np.linspace(o.xmin, o.xmax, o.beds[0].size), total)
    
    H = np.hstack(o.beds_eroded[:total])
    W = np.hstack(w[:total])
    
    mask = ~np.isnan(W)
    X = X[mask]
    H = H[mask]
    W = W[mask]
    
    rbf = scipy.interpolate.NearestNDInterpolator((X, H), W)
    
    ai = rbf(xi, yi)
    ai = ai.T
    
    ai[yi.T > o.beds_raw[total]] = np.nan
    ai[yi.T < o.beds_raw[0]] = np.nan
    return ai


def calc_mol(carb_frac=None,toc_percent=None, topo=None, mode='organic sediment', params=None,
            rho_sed=1.8, mm_C=12.011, C_DIC=0.0024, frac_reactive = 1.0, DIC_mult = 1.0,
            rho_sw=1.025, dx=None, dy=1.0, xmax=None, Nx=None, dz=None, phi=None,sed_scale=False):
    '''
    Converts TOC loss to moles of carbon, either in sediment or seawater pore space.

    Parameters
    ----------
    carb_frac : array-like
        Fraction (0,1) of carbonate-derived carbon, or porosity (fraction) if mode='seawater'.
    toc_percent : array-like
        Percent TOC loss per timestep (%), or porosity (fraction) if mode='seawater'.
    topo : array-like
        Topography array, shape (Nt, Nx) or (Nx, Nt).
    mode : str
        'sediment' or 'seawater' to determine which carbon reservoir to compute.
    params : dict, optional
        Should include 'dx', 'xmax', and 'Nx' if not passing dx, xmax, Nx directly.
    rho_sed : float
        Sediment density in g/cm3.
    mm_C : float
        Molar mass of carbon (g/mol).
    width : float
        Width of a model cell in meters (default is 1 m).
    C_DIC : float
        DIC concentration in mol/kg (converted internally to mol/cm3).
    rho_sw : float
        Seawater density in g/cm3. see Ahm 2018.
    dx, xmax, Nx : float, optional
        Grid spacing and horizontal domain parameters if not using 'params'.
    dz : array-like, optional
        Custom vertical thickness per cell (m). If not provided, derived from topo.

    Returns
    -------
    mol_C : array
        Moles of carbon (same shape as toc_percent).
    v_cm3 : array
        Volume per cell (cm³).
    '''

    cm3_per_m3 = 1e6  # conversion factor

    if params:
        dx = params['dx']
        xmax = params['xmax']
        Nx = params['Nx']
    assert dx is not None and xmax is not None and Nx is not None, "Need dx, xmax, Nx (either via params or directly)"

    #dx_m = dx * (xmax / Nx)

    # compute dz from topo if not provided
    if dz is None:
        dz = jnp.diff(topo, axis=1, prepend=jnp.nan)# prepend=0)
        dz = jnp.clip(dz, 0, None)

    # volume of new sediment per cell in cm3
    dx = 1.0 # m
    dy = 1.0 # m 
    dz = dz if sed_scale else 1.0 # m
    v_cm3 = dx * dy * dz * cm3_per_m3
    
    if mode == 'carbonate sediment':
        m_sed = v_cm3 * rho_sed
        carb_frac = jnp.clip(carb_frac, 0.0, None)
        m_C_carb = m_sed * carb_frac
        mol_C = jnp.clip(m_C_carb / mm_C, 0, None) # avoid negative vals if exist
        return mol_C, v_cm3
    
    if mode == 'organic sediment':
        # porosity (fraction) used to compute porewater volume
        #phi = jnp.clip(phi, 0.0, 1.0)
        # sediment mass (g) and moles of TOC
        m_sed = (v_cm3) * rho_sed # ass organics are in the porespaces, add phi
        toc_frac = jnp.clip(toc_percent, 0, None) / 100.0
        m_C = m_sed * toc_frac * frac_reactive
        mol_C = jnp.clip(m_C / mm_C, 0, None) # avoid negative vals if exist
        return mol_C, v_cm3

    elif mode == 'seawater':
        # porosity (fraction) used to compute porewater volume
        phi = jnp.clip(phi, 0.0, 1.0)
        v_pore_cm3 = phi * v_cm3
        # convert DIC (mol/kg) to mol/cm3 via: (mol/kg) × (g/cm3) × (1 kg / 1000 g)
        C_DIC_cm3 = ((C_DIC * DIC_mult) / 1000) * rho_sw # /1000 to convert /kg to /g
        mol_C = jnp.clip(v_pore_cm3 * C_DIC_cm3, 0, None) # avoid negative vals if exist
        return mol_C, v_cm3

    else:
        raise ValueError("mode must be either 'sediment' or 'seawater'")
    
    
def mol_bal(n_sw, d_sw, n_org, d_org, pad=1e-12):
    """
    Computes the isotopic composition (d13C) of a mixed carbon pool from seawater DIC and respired organic carbon.

    Parameters
    ----------
    n_sw : array-like
        Moles of carbon contributed by seawater (DIC).
    d_sw : array-like
        d13C value (‰) of seawater DIC.
    n_org : array-like
        Moles of carbon contributed by respired organic matter.
    d_org : array-like
        d13C value (‰) of organic matter.
    pad : float, optional
        Small constant added to denominator to avoid division by zero (default is 1e-12).

    Returns
    -------
    d_raw : array-like
        d13C value (‰) of the combined carbon pool.
    """
    sw_value = n_sw * d_sw # mol-weighted isotopic composition of seawater
    org_value = n_org * d_org # mol-weighted isotopic composition of respired organic matter (carbon)
    total_mol = n_org + n_sw + pad  # avoid division by zero
    d_raw = (org_value + sw_value) / total_mol
    return d_raw

def mol_bal_mass(n_sw, d_sw, n_org, d_org, n_carb, d_carb, f_react, pad=1e-12,return_org=False):
    """
    
    """
    # mol balance for d13c_pf
    sw_value = n_sw * d_sw                        # mol-weighted isotopic composition of seawater
    org_value = n_org * d_org                     # mol-weighted isotopic composition of respired organic matter (carbon)
    n_mol_pf = n_org + n_sw                 
    d_pf_raw = (org_value + sw_value) / (n_mol_pf + pad) # avoid division by zero
    
    # porefluid
    n_pf = n_mol_pf
    d13c_pf = d_pf_raw
    # n_pf = n_org
    # d13c_pf = d_org
    
    # existing carbonate
    n_carb_add = f_react * n_carb
    d13c_carb = d_carb
    
    # mol balance
    n_add = jnp.where(n_carb_add<=n_pf,n_carb_add,n_pf) # fraction of existing carbonate reacting as input from porefluid limit to porefluid size
    pf_val = n_add * d13c_pf
    carb_val = n_carb * d13c_carb # want raw moles not with f_react for existing

    # total moles
    tot_mol = n_add + n_carb + pad # avoid division by zero
    
    d_raw = (pf_val + carb_val) / tot_mol
    n_org_pool_new = n_org
    
    if return_org:
        return d_raw, n_org_pool_new
    else:
        return d_raw
    

def compute_f_react(target_percent_per_Myr, dt_kyr=0.1):
    '''
    Computes per-step f_react needed to match a target percent respiration per Myr.

    Parameters
    ----------
    target_percent_per_Myr : float
        Desired total percent of TOC respired per Myr (e.g., 10 for 10%)
    dt_kyr : float
        Model timestep duration in kyr (default is 0.1 kyr)

    Returns
    -------
    f_react : float
        Fraction respired per timestep to match the desired Myr-scale target
    '''
    target_frac = target_percent_per_Myr / 100
    n_steps_per_myr = 1000 / dt_kyr
    f_react = 1 - (1 - target_frac) ** (1 / n_steps_per_myr)
    return f_react

