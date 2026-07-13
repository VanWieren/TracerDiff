### Imports ###
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import jax
import jax.numpy as jnp
from jax import  jit, random, lax
from .utils import *
from .utils import calc_mol, mol_bal, mol_bal_mass
from jax.scipy.signal import convolve
from functools import partial
jax.config.update('jax_platform_name', 'cpu')
jax.config.update('jax_enable_x64', True)  # must be before any jax.numpy arrays are created; needed for reaction diagenesis code

### Model ###
def run(params,
    model_desc,
    hi,
    wi,
    no_res_bounds=False,
    normalize_balance=True,
    sl_fun=lambda x: x,
    rsl=True,
    sec_w_fun=lambda x: x,
    depth_w_fun=None,
    growth_time_fun=lambda x: x,
    growth_time_fun1=lambda x: x,
    org_growth=False,
    carb_growth=True,
    growth_fun_org=None,
    growth_fun_pel=None,
    growth_fun_alg=None,
    growth_fun_coral=None,
    wi_sec=False,
    track_toc=False,
    scale_organics=True,
    swi_dist_calc=False,
    swi_fun=None,
    porosity_fun=None,
    track_react='accumulate',
    plot_skip=10,
    plot_out=True,
    w_transport=True,
    w_set_erode=True,
    ymin_ymax=None,
    full_storage=False,
    storage_level='compiled',
    grid_level='dt',
    const_K=False,
    cmap='rainbow',
    figsize=(9, 3),
    molorg_prev=None,
    molcarb_prev=None
    ):
    """
    Run the sediment transport and growth model.

    This function simulates the evolution of topography and proxy values (e.g., isotopes) through time,
    including carbonate and organic growth, sediment transport, erosion, and diagenetic processes.

    Parameters
    ----------
    params : dict
        Dictionary of input variables and model parameters.
    model_desc : str
        Description or name of the model run.
    hi : array-like
        Initial topography array.
    wi : array-like
        Initial proxy array.
    no_res_bounds : bool, optional
        If True, solve for all boundaries (default False).
    normalize_balance : bool, optional
        If True, normalize mass balance calculations (default True).
    sl_fun : callable, optional
        Function defining temporal changes in sea level (default is identity function).
    rsl : bool, optional
        If True, relative sea level is used (default True).
    sec_w_fun : callable, optional
        Function defining temporal changes in proxy values (default is identity function).
    depth_w_fun : callable, optional
        Function defining depth-related changes to proxy values.
    growth_time_fun : callable, optional
        Function defining temporal variations in growth maxima.
    growth_time_fun1 : callable, optional
        Additional function for temporal growth variations.
    org_growth : bool, optional
        Toggle for enabling organic growth (default False).
    carb_growth : bool, optional
        Toggle for enabling carbonate growth (default True).
    growth_fun_org : callable, optional
        Function for organic growth.
    growth_fun_pel : callable, optional
        Function for pelagic carbonate growth.
    growth_fun_alg : callable, optional
        Function for algal carbonate growth.
    growth_fun_coral : callable, optional
        Function for coral carbonate growth.
    wi_sec : bool, optional
        Toggle for secular changes on the left boundary condition (default False).
    track_toc : bool, optional
        If True, track total organic carbon (default False).
    scale_organics : bool, optional
        If True, organic carbon production is scaled to match the ambient sedimentation rate. Defaults to True.
    swi_dist_calc : bool, optional
        If True, calculate sediment-water interface proximity and diagenesis (default False).
    swi_fun : callable, optional
        Decay function for diagenesis calculations.
    porosity_fun : callable, optional
        Function defining porosity as a function of depth or distance.
    track_react : str, optional
        Mode for diagenetic tracking (e.g., 'accumulate', 'reaction', 'molar', 'respiration').
    plot_skip : int, optional
        Number of time steps to skip when plotting (default 10).
    plot_out : bool, optional
        If True, display the final plot (default True).
    w_transport : bool, optional
        If True, include proxy transport (default True).
    w_set_erode : bool, optional
        If True, update proxy values during erosion (default True).
    ymin_ymax : tuple or None, optional
        If provided, sets the y-axis min/max for grid generation.
    full_storage : bool, optional
        If True, store all time steps in a storage matrix (default False).
    storage_level : str, optional
        Storage mode: 'compiled' or 'dt' (default 'compiled').
    grid_level : str, optional
        Grid update mode: 'dt' or 'compiled' (default 'dt').
    const_K : bool, optional
        If True, use constant diffusion coefficient (default False).
    cmap : str, optional
        Colormap for plotting (default 'coolwarm').
    figsize : tuple, optional
        Figure size for output plot (default (9, 3)).
    molorg_prev : array or None, optional
        Previous storage matrix of moles of respired carbon for diagenesis calculations.

    Returns
    -------
    results : dict
        Dictionary of output arrays and model results.
    """

    # diagenesis
    f_react = params.get('f_react', 0.001)
    scale_f = jit(lambda x:  1.0 - (1.0 - x) ** compiled_steps) # scaling factor to account for 1% of material lost when using only compiled loops
    tau = params.get('tau', 0.1)
    base_depth = params['base_depth'] # bottom n meters to average for lower bound (base_value) of reaction curves
    fuzz = params.get('fuzz',5.0) # for smoothing shoreline boundary
    toc_t_cutoff = params.get('toc_t_cutoff',0) # dont grow TOC before certain tidx (real units)
    sw_DIC_mult = params.get('sw_DIC_mult',1) # multiplier coefficient for to modify concentration of DIC in modern seawater (e.g. C_DIC = sw_DIC_mult * C_DIC)
    
    # build time and space constraints
    grid_ylen = params['grid_ylen'] # number of grid boxes for gridded topography and isotopes for erosion functionality
    org_epsilon = params['org_epsilon']
    alg_epsilon = params['alg_epsilon']
    pel_epsilon = params['pel_epsilon']
    coral_epsilon = params['coral_epsilon']
    conv_sig = params.get('conv_sig', 4)
    org_coef = params.get('org_coef', 1)
    pel_coef = params.get('pel_coef', 1)
    alg_coef = params.get('alg_coef', 1)
    coral_coef = params.get('coral_coef', 1)
    ep = params.get('ep',-25)
    xmin = params['xmin']
    xmax = params['xmax']
    Nx = params['Nx']
    x = jnp.linspace(xmin, xmax, Nx)
    dx = params['dx'] if params['dx'] != 'none' else (xmax-xmin)/(Nx)
    dt = params['dt']
    # for plotting
    end = params['dt']*params['total_n']*params['compiled_steps'] # final time step
    t = jnp.linspace(params['start'],end,params['total_n'])       # time array
    # set up run
    start = params['start']
    total_n = params['total_n']
    compiled_steps = params['compiled_steps']
    marine_K = params['marine_K']
    land_K = params['land_K']
    gamma = params['smooth_K']
    A = params['A'] # mixed layer depth; must scale with run.
    
    # define initial topography and proxy
    h = hi # topo
    w = wi # proxy

    # collect topography and proxy as alternating indices
    c = jnp.zeros(h.size+w.size)  # current timestep
    c = c.at[::2].set(h)          # c[::2] = h
    c = c.at[1::2].set(w)         # c[1::2] = w
    d = jnp.array(c)              # first guess (zero)

    # finite differences
    L =  lambda x: jit(jnp.concatenate)([jnp.array([x[0]]), x[:-1]])    # left shift with constant boundary
    R = lambda x: jit(jnp.concatenate)([x[1:], jnp.array([x[-1]])])     # right shift with constant boundary
    Ux_centered = lambda x: (R(x)-L(x))/(2*dx)                     # first ordered centered difference
    Ux_left = lambda x: (x-L(x))/dx                                # first order upwinding left
    Ux_right = lambda x: (R(x)-x)/dx                               # first order upwinding right


    # Crank Nicolson version
    @jit
    def residual(current, sea_level, dt, dx, p):
        """
        Compute the finite element residual vector for topography and proxy equations.

        Parameters
        ----------
        current : array-like
            Current solution vector (alternating h, w).
        sea_level : float
            Sea level at current time.
        dt : float
            Time step size.
        dx : float
            Spatial step size.
        p : array-like
            Candidate solution vector for the next time step.

        Returns
        -------
        RES : array-like
            Residual vector for use in Newton-Raphson solver.
        """
        h = current[::2]   # topo current time
        w = current[1::2]  # proxy current time
        hf = p[::2]        # topo future time, what we are solving for
        wf = p[1::2]       # proxy future time, what we are solving forf
        h_half = (h+hf)/2  # topo future and current in time /2
        w_half = (w+wf)/2  # proxy future and current in time /2

        # Compute diffusion coefficient (K)
        K = get_K(h_half, sea_level, p)

        # step-offset diffusivities
        if const_K:
            K_L = K
            K_R = K
        else:
            K_L = (L(K)+K)/2   # half-step diffusivity to the left
            K_R = (R(K)+K)/2   # half-step diffusivity to the right

        # total residual initialization
        RES = jnp.zeros_like(current) # current residual

        ### Equation 1 - Normal Diffusion
        RES1 = jnp.zeros_like(hf) # residual in the future after we solve for it
        LHS = (hf-h)/dt
        # discretization from Wei (2013) 3.2.1 treatment of diffusion without the tracer
        RHS = (K_R*(R(h_half)-h_half)-K_L*(h_half-L(h_half)))/(dx**2)
        RES1_temp = RHS - LHS

        if no_res_bounds:
            # solving everything including boundaries
            RES1 = RES1.at[:].set(RES1_temp[:])
        else:
            # leaves boundaries at 0 residual (constant boundary conditions based on my hi and wi)
            RES1 = RES1.at[1:-1].set(RES1_temp[1:-1])

        ### Equation 2 - Tracer Diffusion
        RES2 = jnp.zeros_like(wf)
        if w_transport:
            # Compute left-hand side (LHS) and right-hand side (RHS) for proxy equation
            # the 0* here is correct. the following terms cancel with RHS
            LHS = A*(wf-w)/dt + 0*(w_half)*(hf-h)/dt # 0*(w_half)*(hf-h)/dt

            # Upwinding for proxy transport
            RHS_l = K_L*Ux_left(w_half)*Ux_centered(h_half)  # left upwind
            RHS_r = K_R*Ux_right(w_half)*Ux_centered(h_half)  # right upwind
            RHS = jnp.where(L(h_half) > R(h_half), RHS_l, RHS_r) # upwinding mask to combine

            RES2_temp = RHS - LHS

            if no_res_bounds:
                # solving everything including boundaries
                RES2 = RES2.at[:].set(RES2_temp[:])
            else:
                # leaves boundaries at 0 residual (constant boundary conditions based on my hi and wi)
                RES2 = RES2.at[1:-1].set(RES2_temp[1:-1])

        # Combine residuals for topography and proxy
        RES = RES.at[::2].set(RES1)  # set h to residual eq1 (RES1)
        RES = RES.at[1::2].set(RES2) # set w to residual eq2 (RES2)

        return RES


    @jit
    def get_K(h_half, sea_level, p):
        """
        Compute the diffusion coefficient K as a function of topography and sea level.

        Parameters
        ----------
        h_half : array-like
            Midpoint topography (between time steps).
        sea_level : float
            Current sea level.
        p : array-like
            Solution vector (unused here but kept for interface).

        Returns
        -------
        K : array-like
            Diffusion coefficient array.
        """
        if const_K:
            K = marine_K
        else:
            # Sea level variable K, smoothed by gamma
            K = ((-1 * jnp.tanh((h_half - sea_level) * 1 / gamma) * (marine_K - land_K) / 2) + (land_K + marine_K) / 2)
        return K


    @jit
    def boundary_fluxes(current, sea_level, dt, dx, p):
        """
        Compute boundary fluxes for topography and proxy at the domain edges.

        Parameters
        ----------
        current : array-like
            Current solution vector (alternating h, w).
        sea_level : float
            Sea level at current time.
        dt : float
            Time step size.
        dx : float
            Spatial step size.
        p : array-like
            Candidate solution vector for the next time step.

        Returns
        -------
        L1_bflux : float
            Left boundary topography flux.
        R1_bflux : float
            Right boundary topography flux.
        L2_bflux : float
            Left boundary proxy flux.
        R2_bflux : float
            Right boundary proxy flux.
        """

        h = current[::2]   # topo current time
        w = current[1::2]  # proxy current time
        hf = p[::2]        # topo future time, what we are solving for
        wf = p[1::2]       # proxy future time, what we are solving for
        h_half = (h+hf)/2  # topo future and current in time /2
        w_half = (w+wf)/2  # proxy future and current in time /2

        # diffusion coefficient
        K = get_K(h_half,sea_level,p)

        # Topography (Eqn. 1)
        L1_bflux = 1 * ((K * (h_half - R(h_half)) / dx**2) * dt) 
        R1_bflux = -1 * ((K * (L(h_half) - h_half) / dx**2) * dt) 
        # Proxy (Eqn. 2)
        L2_bflux = w_half
        R2_bflux = w_half

        if no_res_bounds:
            # for testing with no res bounds
            RES_tmp = residual(current, sea_level, dt, dx, p)
            h_tmp = RES_tmp[::2]
            w_tmp = RES_tmp[1::2]

            # indexing is done below
            L1_bflux = h_tmp
            R1_bflux = h_tmp
            L2_bflux = w_tmp
            R2_bflux = w_tmp

        return L1_bflux[0],R1_bflux[-1],L2_bflux[0],R2_bflux[-1]


    @jit
    def loss(c, sea_level, dt, dx, p):
        """
        Compute the loss function (L2 norm of the residual vector).

        Parameters
        ----------
        c : array-like
            Current solution vector.
        sea_level : float
            Sea level at current time.
        dt : float
            Time step size.
        dx : float
            Spatial step size.
        p : array-like
            Candidate solution vector.

        Returns
        -------
        float
            Discrete 2-norm of residual vector.
        """
        return jnp.linalg.norm(residual(c, sea_level, dt, dx, p))


    @jit
    def newton_step_nojac(current, sea_level, dt, dx, p):
        """
        Perform one Newton-Raphson iteration step using JAX to compute the Jacobian.

        Parameters
        ----------
        current : array-like
            Current solution vector.
        sea_level : float
            Sea level at current time.
        dt : float
            Time step size.
        dx : float
            Spatial step size.
        p : array-like
            Candidate solution vector.

        Returns
        -------
        p : array-like
            Updated solution vector after one Newton-Raphson step.
        """
        f = lambda guess: -1 * residual(current, sea_level, dt, dx, guess)
        f = jit(f)
        jac_x_prod = lambda x: jax.jvp(f, [p], [x])[1]
        jac_x_prod = jit(jac_x_prod)
        # Perform conjugate gradient solve for Newton step
        dp = jax.scipy.sparse.linalg.cg(jac_x_prod, -f(p), x0=jnp.zeros_like(p), tol=1e-2, maxiter=3, M=-M)[0]
        p += dp
        return p


    @jit
    def sediment_prod(t, p, dic, sea_level, growth_fun_t, wavg_exp, wavg_obs, mavg_exp, mavg_obs, mtot_exp_i, mtot_obs_i, ml_tot, mr_tot, ml_m_tot, mr_m_tot, R1L_tot, R1R_tot, R2L_tot, R2R_tot, bias_tot, mass_dt):
        """
        Compute carbonate and organic sediment production and update topography/proxy values.

        Parameters
        ----------
        p : array-like
            Current solution vector.
        dic : array-like
            Secular and/or depth-dependent DIC values.
        sea_level : float
            Sea level at current time.
        growth_fun_t : callable
            Temporal growth function.
        wavg_exp, wavg_obs, mavg_exp, mavg_obs, mtot_exp_i, mtot_obs_i, ml_tot, mr_tot, ml_m_tot, mr_m_tot, R1L_tot, R1R_tot, R2L_tot, R2R_tot, bias_tot, mass_dt : various
            Mass balance and tracking variables.

        Returns
        -------
        tuple
            Updated solution vector and diagnostic values.
        """
        # define depth, growth function, and arrays
        depth = sea_level - p[::2] # positive downwards for equations
        ocean_depth = params['ocean_depth']  # depth at which the fore-reef ends, where ocean parameters match those of the open ocean rather than the platform
      
        q = jnp.zeros_like(p)
        qh = q[::2]
        qw = q[1::2]
        ones = jnp.ones_like(qh)

        # Find shoreline position by detecting sign change in topography relative to sea level
        h_tmp = p[::2] - sea_level # this is for zero_crossing_mask
        zero_crossing_mask = (jnp.roll(h_tmp > 0, 1) & (h_tmp < 0)) # finds sign change
        shoreline = jnp.array(jnp.where(zero_crossing_mask!=0,size=1))[0][0]
        new_shore = x[shoreline] # shoreline in x coordinates

        # Calculate growth for each carbonate group, scaled by coefficients if provided
        if growth_fun_alg:
            G_alg = alg_coef*growth_fun_alg(depth) # grow algal carbonate on substrate with depth function for algal sediment, scaled with coefficient
        else:
            G_alg = jnp.zeros_like(depth)
            
        if growth_fun_coral:
            # Coral growth: apply convolution to simulate spatial spreading of growth
            production = growth_fun_coral(depth)      # grab from coral light curve
            growth_potential = jnp.zeros_like(p[::2]) # start with all zero growth
            capacity = 1
            growth_potential_unconvolved = jnp.where(depth>=ocean_depth,capacity,growth_potential)      # everywhere greater than ocean depth gets growth potential of 1
            growth_convolved = prep_convolve(growth_potential_unconvolved ,val=capacity,sigma=conv_sig) # apply guassian filter (convolution) to the data; sigma will have to be played with
            growth_convolved = jnp.where(depth<0,0,growth_convolved)                                    # ensure zeros for subaerial
            coral_prod = production*growth_convolved
            G_coral = coral_coef*coral_prod
        else:
            G_coral = jnp.zeros_like(depth)

        if growth_fun_pel:
            # Pelagic growth: integrate over depth, restrict to deep ocean
            G_pel = jnp.trapezoid(growth_fun_pel(depth)) # for pelagic sediment, integral over depth down to sediment, using jax.scipy.integrate.trapezoid
            G_pel = jnp.where(depth<ocean_depth,0,G_pel) # limit pelagic growth to not be in the shallow banktop; may need to make this shallower
            G_pel = pel_coef*G_pel # scaled with coefficient
        else:
            G_pel = jnp.zeros_like(depth)

        # Mask: only allow growth below sea level
        mask = (depth > 0)

        ## fractions and isotopes of all sources

        # Isotope values for each growth source
        h_grow_coral = jnp.abs(ones*G_coral) # these are abs just in case I accidentally generate negative growth in the pre-made functions
        h_grow_alg = jnp.abs(ones*G_alg)
        h_grow_pel = jnp.abs(ones*G_pel)

        w_grow_org = dic + org_epsilon
        w_grow_pel = dic + pel_epsilon  # pelagic material contains secular values
        w_grow_alg = w_grow_pel + alg_epsilon  #  ‰ offset between algal and pelagic (algal is heavier); enriched shallow material -> see Geyman diurnal paper for idea of epsilon
        w_grow_coral = w_grow_pel + coral_epsilon #  ‰ offset between coral  and pelagic (algal is heavier); enriched shallow material

        # Add new topography from all carbonate sources
        qh_coral = jnp.where(mask,h_grow_coral,qh)
        qh_alg = jnp.where(mask,h_grow_alg,qh)
        qh_pel = jnp.where(mask,h_grow_pel,qh)
        qh_carb  = qh_pel + qh_alg + qh_coral  if carb_growth else jnp.zeros_like(p[::2]) # total new carbonate added

        # Use carbonate sedimentation rate as proxy for burial rate
        sed_rate = qh_carb
        
        if scale_organics:
            sed_rate_norm = sed_rate / jnp.max(sed_rate + 1e-12)  # prevent division by zero
        else:
            sed_rate_norm = jnp.ones_like(sed_rate)
        
        # Organics scaled by sedimentation rate (preservation increases with sed_rate)
        if growth_fun_org:
            G_org_depth = growth_fun_org(depth)
            # want to have some mix of a depth controlled, and a sedimentation controlled bias here. 0.5 is 50:50
            G_org = org_coef * G_org_depth * sed_rate_norm # adding a way to grow more organics where there is more carbonate sedimentation
        else:
            G_org = jnp.zeros_like(depth)

        h_grow_org = jnp.abs(ones*G_org)

        # Add new topography from organic growth
        qh_org = jnp.where(mask,h_grow_org,qh) if org_growth else jnp.zeros_like(p[::2])
        o_sed_rate = qh_org

        if org_growth:
            # Calculate total organic carbon (TOC) percent
            tot_sed = qh_org + qh_carb
            toc = jnp.where(tot_sed > 0, qh_org / tot_sed, 0.0)
            toc_p = toc * 100 # convert to percent TOC
           # toc_p = toc_p * sed_rate_norm
            toc_p = jnp.where(t>=toc_t_cutoff,toc_p,jnp.zeros_like(toc_p))

        # Add new topography from all sediment sources (carbonate + organic)
        qh_sed = qh_carb + qh_org
        p = p.at[::2].add(qh_sed)

        # Calculate weighted average proxy value for new sediment

        # total topography added
        tot_dh = (qh_org + qh_pel + qh_alg + qh_coral)

        # weighted isotopes
        org = (qh_org * w_grow_org)
        pelagic = (qh_pel * w_grow_pel)
        algal = (qh_alg * w_grow_alg)
        coral = (qh_coral * w_grow_coral)

        # Sum weighted isotopes
        weighted_w = (org + pelagic + algal + coral)

        # Calculate average proxy value for new sediment
        qw_avg = toc_p if track_toc else weighted_w / tot_dh

        # Handle hiatus: if no new sediment, use previous proxy value; This is the keystone of the whole model. Allows for advection when not setting
        qw_avg = jnp.where(tot_dh == 0, p[1::2], qw_avg)

        # Update proxy values in solution vector
        p = p.at[1::2].set(qw_avg)

        return p, new_shore, depth, mtot_exp_i, mtot_obs_i, ml_tot, mr_tot, ml_m_tot, mr_m_tot, wavg_exp, wavg_obs, mavg_exp, mavg_obs, R1L_tot, R1R_tot, R2L_tot, R2R_tot, bias_tot, mass_dt, qh_sed, qw_avg, sed_rate, o_sed_rate

    
    @jit
    def solve(i, dt, dx, carry, tolerance=1.0e-2):
        """
        Perform Newton-Raphson solve for a single time step.

        Parameters
        ----------
        i : int
            Current time step index.
        dt : float
            Time step size.
        dx : float
            Spatial step size.
        carry : tuple
            Model state variables.
        tolerance : float, optional
            Convergence tolerance for residual norm (default 1e-2).

        Returns
        -------
        tuple
            Updated model state variables.
        """
        R1L_tot,R1R_tot,R2L_tot,R2R_tot,bias_tot,mass_dt,mtot_exp_i,mtot_obs_i,ml_tot,mr_tot,ml_m_tot,mr_m_tot,wavg_exp,wavg_obs,mavg_exp,mavg_obs,depth,new_shore,sto,grid,gridw,grid_org_prev,grid_carb_prev,grid_dz,gridh,c,ya,base_value_store,base_value_grid,molorg_prev,molcarb_prev,csed_sto,sed_rate,osed_sto,o_sed_rate,dsed_sto = carry

        # pre updated guess array
        # Compute current time and state-dependent parameters
        t = dt * i                # Current model time
        sea_level = sl_fun(t)     # Sea level at current time
        sed_w_t = sec_w_fun(t)    # Secular changes to proxy
        growth_fun_t = growth_time_fun(t) # Time-dependent growth function
        growth_fun_t1 = growth_time_fun1(t)
        depth = sea_level - c[::2]  # Depth below sea level
        if depth_w_fun:
            sed_w_depth = depth_w_fun(depth)    # Proxy gradient by depth

        # Compute seawater isotopes (DIC)
        ones = jnp.ones_like(c[::2])
        dic_s = ones * sed_w_t if sec_w_fun else 0
        dic_d = ones * sed_w_depth if depth_w_fun else 0
        dic = dic_s + dic_d

        c0 = jnp.array(c)

        # Add new sediment before transport/diffusion
        c, new_shore, depth, mtot_exp_i, mtot_obs_i, ml_tot, mr_tot, ml_m_tot, mr_m_tot, wavg_exp, wavg_obs, mavg_exp, mavg_obs, R1L_tot, R1R_tot, R2L_tot, R2R_tot, bias_tot, mass_dt, qh_sed, qw_avg, sed_rate, o_sed_rate = sediment_prod(
            t, c, dic, sea_level, growth_fun_t, wavg_exp, wavg_obs, mavg_exp, mavg_obs, mtot_exp_i, mtot_obs_i, ml_tot, mr_tot, ml_m_tot, mr_m_tot, R1L_tot, R1R_tot, R2L_tot, R2R_tot, bias_tot, mass_dt
        )

        p = jnp.array(c)

        # Setup loss and Newton step functions for this time step
        L2 = lambda x: loss(c, sea_level, dt, dx, x)
        N_S = lambda x: newton_step_nojac(c, sea_level, dt, dx, x)

        # Newton-Raphson iteration for time step
        d = lax.while_loop(lambda x: L2(x) > tolerance, N_S, p)

        # --- Erosion step: remove eroded beds from grids ---
        w_erode_new, dh, er, gridw, gridh = set_erode(gridw, gridh, initial_h=c0[::2], final_h=d[::2], last_w=c0[1::2])
        erode_mask = (dh < 0)
        h_erode = jnp.where(erode_mask, dh, jnp.zeros_like(d[::2]))
        w_erode = jnp.where(erode_mask, w_erode_new, d[1::2])

        if w_set_erode:
            d = d.at[1::2].set(w_erode)

        # --- Update grid to match new solution ---
        gridw, gridh = set_grid(d[::2], gridw, gridh, w_set=d[1::2], ya=ya)

        # Save old grid for reaction calculations
        gridw_old = jnp.copy(gridw)

        # --- SWI distance accumulation and diagenesis after erosion ---
        if swi_dist_calc:
            if grid_level == 'dt':
                if track_react == 'accumulate':
                    gridw, base_value_grid, decay_value, combined_mask, delta = update_grid(
                        gridw, gridh, d[::2], i, sea_level, decay_fun=swi_fun, mode='accumulate'
                    )
                elif track_react == 'reaction':
                    gridw, base_value_grid, decay_value, combined_mask, delta = update_grid(
                        gridw, gridh, d[::2], i, sea_level, decay_fun=swi_fun, mode='reaction',
                        f_react=f_react, d13c_seawater=dic, tau=tau
                    )
                elif track_react == 'molar':
                    gridw, base_value_grid, decay_value, combined_mask, delta, grid_org_prev = update_grid(
                        gridw, gridh, d[::2], i, sea_level, decay_fun=swi_fun, mode='molar',
                        f_react=f_react, d13c_seawater=dic, tau=tau, grid_org_prev=grid_org_prev, grid_carb_prev=grid_carb_prev, grid_dz=grid_dz
                    )
                elif track_react == 'respiration':
                    gridw, base_value_grid, decay_value, combined_mask, delta = update_grid(
                        gridw, gridh, d[::2], i, sea_level, decay_fun=swi_fun, mode='respiration',
                        f_react=f_react, tau=tau
                    )
            else:
                decay_value = jnp.zeros_like(depth)
                combined_mask = jnp.zeros_like(depth)
                delta = jnp.zeros_like(depth)
        else:
            decay_value = jnp.zeros_like(depth)
            combined_mask = jnp.zeros_like(depth)
            delta = jnp.zeros_like(depth)

        # --- Calculate boundary fluxes on converged solution ---
        L1_bflux, R1_bflux, L2_bflux, R2_bflux = boundary_fluxes(c0, sea_level, dt, dx, d)

        # --- Mass balance calculations ---
        mtot_exp_i, mtot_obs_i, wavg_exp, wavg_obs, mavg_exp, mavg_obs, ml_tot, mr_tot, ml_m_tot, mr_m_tot, R1L_tot, R1R_tot, R2L_tot, R2R_tot, bias_tot, mass_dt = mass_balance(
            sea_level, mtot_exp_i, mtot_obs_i, L1_bflux, R1_bflux, L2_bflux, R2_bflux, R1L_tot, R1R_tot, R2L_tot, R2R_tot, bias_tot, mass_dt,
            ml_tot=ml_tot, mr_tot=mr_tot, ml_m_tot=ml_m_tot, mr_m_tot=mr_m_tot, htf=d[::2], hti=c0[::2], wtf=d[1::2], wti=c0[1::2],
            h_erode=h_erode, w_erode=w_erode, dh=dh, qh_sed=qh_sed, qw_avg=qw_avg, gridh=gridh, gridw=gridw, gridw_old=gridw_old,
            decay_value=decay_value, f_react=f_react, ya=ya, combined_mask=combined_mask, delta_grid=delta
        )

        # --- Storage matrix update ---
        if full_storage:
            if storage_level == 'dt':
                # Store current solution
                sto = sto.at[:, i].set(d)
                # rates
                rdiff = sto[::2,i] - sto[::2,i-1]
                csed_sto = csed_sto.at[:,i].set(sed_rate/dt) # store carbonate sedimentation rate
                osed_sto = osed_sto.at[:,i].set(o_sed_rate/dt) # store organic sedimentation rate
                dsed_sto = dsed_sto.at[:,i].set(rdiff/dt) # net topographic change
                if swi_dist_calc:
                    if track_react == 'accumulate':
                        # Accumulate exposure time
                        sto, base_value = update_sto(
                            sto, d, i, sea_level,
                            decay_fun=swi_fun,
                            mode='accumulate', base_value_passed=base_value_grid
                        )
                        base_value_store = base_value_store.at[:, i].set(base_value[:, 0])
                    elif track_react == 'reaction':
                        # Apply reactive diagenesis
                        sto, base_value = update_sto(
                            sto, d, i, sea_level,
                            decay_fun=swi_fun,
                            mode='reaction',
                            f_react=f_react,
                            d13c_seawater=dic[:, None], base_value_passed=base_value_grid, tau=tau
                        )
                        base_value_store = base_value_store.at[:, i].set(base_value[:, 0])
                    elif track_react == 'molar':
                        sto, base_value, molorg_prev = update_sto(
                            sto, d, i, sea_level,
                            decay_fun=swi_fun,
                            mode='molar',
                            f_react=f_react,
                            d13c_seawater=dic[:, None], base_value_passed=base_value_grid, tau=tau, molorg_prev=molorg_prev, molcarb_prev=molcarb_prev,
                        )
                        base_value_store = base_value_store.at[:, i].set(base_value[:, 0])
                    elif track_react == 'respiration':
                        sto, base_value = update_sto(
                            sto, d, i, sea_level,
                            decay_fun=swi_fun,
                            mode='respiration',
                            f_react=f_react,
                            base_value_passed=base_value_grid, tau=tau
                        )
                        base_value_store = base_value_store.at[:, i].set(base_value[:, 0])

        return (R1L_tot, R1R_tot, R2L_tot, R2R_tot, bias_tot, mass_dt, mtot_exp_i, mtot_obs_i, ml_tot, mr_tot, ml_m_tot, mr_m_tot, wavg_exp, wavg_obs, mavg_exp, mavg_obs, depth, new_shore, sto, grid, gridw, grid_org_prev, grid_carb_prev, grid_dz, gridh, d, ya, base_value_store, base_value_grid, molorg_prev, molcarb_prev, csed_sto, sed_rate, osed_sto, o_sed_rate, dsed_sto)


    @jit
    def run_steps(start, end, dt, dx,
                  R1L_tot, R1R_tot, R2L_tot, R2R_tot, bias_tot, mass_dt, mtot_exp_i, mtot_obs_i,
                  ml_tot, mr_tot, ml_m_tot, mr_m_tot, wavg_exp, wavg_obs, mavg_exp, mavg_obs,
                  depth, new_shore, sto, grid, gridw, grid_org_prev, grid_carb_prev, grid_dz, gridh, c, ya, base_value_store, base_value_grid, molorg_prev, molcarb_prev, csed_sto, sed_rate, osed_sto, o_sed_rate, dsed_sto):
        """
        Run multiple time steps using Newton-Raphson solver.

        Parameters
        ----------
        start : int
            Starting time step index.
        end : int
            Ending time step index.
        dt, dx : float
            Time and spatial step sizes.
        ... : various
            Model state variables.

        Returns
        -------
        tuple
            Updated model state variables after running steps.
        """
        S = lambda i, x: solve(i, dt, dx, x)
        S = jit(S)
        return lax.fori_loop(
            start, end, S,
            (R1L_tot, R1R_tot, R2L_tot, R2R_tot, bias_tot, mass_dt, mtot_exp_i, mtot_obs_i, ml_tot, mr_tot, ml_m_tot, mr_m_tot,
             wavg_exp, wavg_obs, mavg_exp, mavg_obs, depth, new_shore, sto, grid, gridw, grid_org_prev, grid_carb_prev, grid_dz, gridh, c, ya, base_value_store, base_value_grid, molorg_prev, molcarb_prev, csed_sto, sed_rate, osed_sto, o_sed_rate, dsed_sto)
        )


    @jit
    def make_grid():
        """
        Set up user-defined grid arrays for vertical (ya) and horizontal (x) dimensions.

        Returns
        -------
        grid : array
            Base grid of vertical layers (ya) for each x.
        gridw : array
            Proxy grid (initialized to NaN).
        gridh : array
            Topography grid (initialized to NaN).
        grid_org_prev : array
            Previous grid for diagenesis (initialized to NaN).
        grid_dz : array
            Grid for vertical increments (initialized to NaN).
        """
        grid = jnp.tile(ya.reshape(-1, 1), xlen)
        gridw = jnp.full(grid.shape, jnp.nan)
        gridh = jnp.array(gridw)
        grid_org_prev = jnp.array(gridw)
        grid_carb_prev = jnp.array(gridw)
        grid_dz = jnp.array(gridw)
        return grid, gridw, gridh, grid_org_prev, grid_carb_prev, grid_dz


    @jit
    def set_grid(bed, gridw, gridh, w_set, ya):
        """
        Update gridded arrays with current bed elevations and proxy values.

        Parameters
        ----------
        bed : array
            Current bed elevations.
        gridw : array
            Proxy grid to update.
        gridh : array
            Topography grid to update.
        w_set : array
            Proxy values to set.
        ya : array
            Vertical grid levels.

        Returns
        -------
        gridw, gridh : arrays
            Updated proxy and topography grids.
        """
        idx_y = jnp.searchsorted(ya[::-1], bed, side='left')
        idx_y = ya.size - 1 - idx_y  # because we reversed ya
        x_idx = jnp.arange(gridw.shape[1])
        gridw = gridw.at[idx_y, x_idx].set(w_set)
        gridh = gridh.at[idx_y, x_idx].set(bed)
        return gridw, gridh
    
    
    @jit
    def set_grid_prev(bed, in_grid, set_val, ya):
        """
        Update a grid with new values at the current bed surface.

        Parameters
        ----------
        bed : array
            Current bed elevations.
        in_grid : array
            Grid to update.
        set_val : array
            Values to set at the surface.
        ya : array
            Vertical grid levels.

        Returns
        -------
        in_grid : array
            Updated grid.
        """
        idx_y = jnp.searchsorted(ya[::-1], bed, side='left')
        idx_y = ya.size - 1 - idx_y  # because we reversed ya
        x_idx = jnp.arange(in_grid.shape[1])
        in_grid = in_grid.at[idx_y, x_idx].set(set_val)
        return in_grid


    @jit
    def grab_last(data):
        """
        Find the last non-NaN value along each column in a grid.

        Parameters
        ----------
        data : array
            Input grid.

        Returns
        -------
        last_vals : array
            Array of last non-NaN values per column.
        """
        last_valid_indices = jnp.argmax(~jnp.isnan(data), axis=0)
        last_vals = jnp.take_along_axis(data, jnp.expand_dims(last_valid_indices, axis=0), axis=0)[0]
        return last_vals


    @jit
    def set_erode(gridw, gridh, initial_h, final_h, last_w):
        """
        Apply erosion: update grid arrays and proxy values at the new eroded surface.

        Parameters
        ----------
        gridw : array
            Proxy grid.
        gridh : array
            Topography grid.
        initial_h : array
            Initial topography before erosion.
        final_h : array
            Topography after erosion.
        last_w : array
            Previous proxy values.

        Returns
        -------
        w_new : array
            Proxy values at new eroded surface.
        dh : array
            Change in topography (dz).
        er : array
            New eroded surface elevations.
        gridw_er : array
            Proxy grid after erosion.
        gridh_er : array
            Topography grid after erosion.
        """
        h_now = initial_h
        h_next = final_h
        dh = h_next - h_now
        dhi = jnp.where(dh < 0, -1 * dh, jnp.zeros_like(dh))
        er = h_now - dhi
        v = (gridh <= er)
        grid_new_surf = jnp.where(v, gridw, jnp.nan)
        w_new = grab_last(grid_new_surf)
        w_new = jnp.where(jnp.isnan(w_new), last_w, w_new)
        gridw_er = jnp.where(~v, jnp.nan, gridw)
        gridh_er = jnp.where(~v, jnp.nan, gridh)
        return w_new, dh, er, gridw_er, gridh_er


    @jit
    def default_decay_fun(vertical_distance, d13c_seawater=1.0, base_value=0.0, tau=0.033):
        """
        Default decay function for diagenesis (e.g., SWI accumulation or porewater d13C).

        Parameters
        ----------
        vertical_distance : array
            Vertical distance below SWI (negative values).
        d13c_seawater : float, optional
            Seawater d13C value (default 1.0).
        base_value : float, optional
            Base value for mixing (default 0.0).
        tau : float, optional
            Decay rate constant (default 0.033).

        Returns
        -------
        d13c_porewater : array
            Decayed d13C values.
        """
        d13c_porewater = (d13c_seawater * jnp.exp(tau * vertical_distance) + base_value * (1.0 - jnp.exp(tau * vertical_distance)))
        return d13c_porewater


    ### Reaction steps on discretized gridded matrices
    @partial(jit, static_argnames=['decay_fun', 'mode'])
    def update_sto(sto, d, i, sea_level, decay_fun=None,
                f_react=f_react, fuzz=fuzz, d13c_seawater=0.0,
                mode='accumulate',base_value_passed=None,tau=tau,molorg_prev=None,molcarb_prev=None):
        """
        Unified storage (sto) updater for sediment-water interface (SWI) proximity accumulation and diagenetic processes.

        This function updates the storage matrix for each time step, applying different diagenetic or accumulation modes.

        Parameters
        ----------
        sto : array
            Full storage matrix (2*Nx, N_time).
        d : array
            Current solution vector (2*Nx,).
        i : int
            Current time step.
        sea_level : float
            Sea level at current time.
        decay_fun : callable, optional
            Decay function of vertical distance. If None, a default exponential decay is used.
        f_react : float, optional
            Reaction fraction per timestep (used in 'reaction', 'molar', and 'respiration' modes for both organic carbon and carbonate reaction scaling).
        fuzz : float, optional
            Smoothing factor for shoreline masking.
        tau : float, optional
            Exponential decay constant for the decay function.
        d13c_seawater : float or array, optional
            Seawater d13C value for diagenetic calculations.
        mode : str
            Mode for updating storage. Supported modes:
                - 'accumulate': Accumulates SWI proximity or decay values for all buried material. No reaction; simply sums up the decay function output for each burial event.
                - 'reaction': Applies a fixed fraction of reaction (f_react) each step toward the decay function value (e.g., simulating porewater diagenesis or isotopic resetting).
                - 'molar': Computes mole-based mixing between respired organic carbon and seawater DIC, updating storage by the effective reaction fraction.
                - 'respiration': Applies organic carbon loss by a fixed fraction (f_react) per step, simulating remineralization or TOC loss.
                - (Other modes may be supported if implemented.)
        base_value_passed : array, optional
            Precomputed base_value from grid; required for decay mixing.
        molorg_prev : array, optional
            Previous storage matrix, required for 'molar' mode.

        Behavior of each mode:
            - 'accumulate': Adds the decay function value (e.g., time or d13C) to storage at each step for all buried material.
            - 'reaction': Moves storage values toward the decay function value by a fraction f_react each step (simulates ongoing reaction).
            - 'molar': Updates storage based on mole-weighted mixing of respired organic C and seawater DIC, using f_react as the mixing fraction.
            - 'respiration': Removes a fraction f_react of stored TOC per step, scaled by the decay function (simulates organic matter loss).

        Returns
        -------
        sto : array
            Updated storage matrix.
        base_value_sto : array
            The base value used for decay mixing, reshaped.
        """

        # we use the base value from update_grid
        base_value_sto = base_value_passed[:, None]  # reshape

        # New handling for vertical_distance and burial_mask
        past_topo = sto[::2, :]      # (Nx, Nt)
        swi = d[::2]                 # (Nx,)
        
        # mask for below the swi and away from shoreline
        vertical_distance = past_topo - swi[:, None]  # (Nx, Nt)

        burial_mask = (past_topo < swi[:, None])      # (Nx, Nt)
        time_mask = (jnp.arange(sto.shape[1]) < i)    # (Nt,)

        shoreline_fuzz = fuzz
        topo_submerged_weight = jax.nn.sigmoid((sea_level - swi)[None, :] / shoreline_fuzz)
        sub_sea_mask = (past_topo <= sea_level) * topo_submerged_weight.T

        valid_mask = ~jnp.isnan(past_topo)
        combined_mask = burial_mask * sub_sea_mask * valid_mask * time_mask[None, :]
        
        # centralize effective_f_react calculation
        effective_f_react = scale_f(f_react) if grid_level == 'compiled' else f_react
                
        if mode == 'molar':
            # carbon moles from respiration calculations
            if molorg_prev is not None:
                # contribution from respired organic carbon; here assuming TOC is 100% carbon (C)
                ## make jax friendly
                Nt = molorg_prev.shape[1]  # static at tracing time
                full_slice = lax.dynamic_slice(molorg_prev, (0, 0), (Nx, Nt))  # static shape
                time_mask = jnp.arange(Nt) < i  # build time_mask dynamically: shape (Nt,), True up to time i
                org_mol_profile = full_slice * time_mask[None, :]  # broadcasted mask
                
                # contribution from moles existing carbonate-derived carbon
                ## some more jax 
                Nt = molcarb_prev.shape[1]  # static at tracing time
                full_slice = lax.dynamic_slice(molcarb_prev, (0, 0), (Nx, Nt))  # static shape
                time_mask = jnp.arange(Nt) < i  # build time_mask dynamically: shape (Nt,), True up to time i
                carb_mol_profile = full_slice * time_mask[None, :]  # broadcasted mask
                
                # calculate moles of seawater DIC in pore space
                phi = jnp.clip(porosity_fun(vertical_distance), 0.0, 1.0) # clip in case weird overshoots
                n_sw,_ = calc_mol(topo=past_topo,phi=phi,mode='seawater',params=params,DIC_mult=sw_DIC_mult) # incorporates phi
                n_sw = rem_ol(n_sw) # remove outliers
                d13c_sw = d13c_seawater # isotopic composition of seawater; relates to dic i define outside of this
               
                # organics and porefluid d13C mass balance
                n_org = phi * org_mol_profile
                d13c_org = d13c_sw + ep # -25‰;  photosynthetic fractionation

                # moles of existing carbonate (passed from molcarb_prev)
                n_carb = carb_mol_profile
                n_carb = (1 - phi) * n_carb # scale by remaining non-pore space
                d13c_carb = sto[1::2, :]
                
                # mol balance (new version taking mass of existing carbonate into account)
                d13c_calc_raw, molorg_prev = mol_bal_mass(n_sw, d13c_sw, n_org, d13c_org, n_carb, d13c_carb, effective_f_react, pad=1e-12,return_org=True)
                d13c_calc = rem_ol(d13c_calc_raw) # remove outliers

                # new update: blend with previous values using combined_mask weights
                d13c_new = sto[1::2, :] * (1 - combined_mask) + d13c_calc * combined_mask
                sto = sto.at[1::2, :].set(d13c_new)
                
        else:
                 
            if decay_fun is None:
                # match the same behavior: simple default for accumulation
                decay_value = default_decay_fun(vertical_distance)
            else:
                decay_value = decay_fun(vertical_distance, d13c_seawater=d13c_seawater if mode != 'respiration' else 1.0, base_value=base_value_sto if mode != 'respiration' else 0.0, tau=tau)

            # storage level scaling
            if storage_level == 'dt':
                scale = 1.0
            elif storage_level == 'compiled':
                scale = compiled_steps

            # Centralize effective_f_react calculation for all applicable modes
            effective_f_react = scale_f(f_react) if grid_level == 'compiled' else f_react

            if mode == 'accumulate':
                contribution = decay_value * combined_mask * scale
                sto = sto.at[1::2, :].add(contribution)

            elif mode == 'reaction':
                # Directly apply the update over full columns (all times, all x-grid)
                delta = effective_f_react * (decay_value - sto[1::2, :]) * combined_mask
                sto = sto.at[1::2, :].add(delta)

            elif mode == 'respiration':
                #phi = jnp.clip(porosity_fun(vertical_distance), 0.0, 1.0) # clip in case weird overshoots
                phi = 1
                # Use f_react for organic carbon respiration as well
                effective_loss = effective_f_react * (phi * decay_value) * combined_mask
                lost_TOC = sto[1::2, :] * effective_loss
                sto = sto.at[1::2, :].add(-lost_TOC)
    
        if mode == 'molar':
            return sto, base_value_sto, molorg_prev
        else:
            return sto, base_value_sto


    def smooth_base_value(var, sigma=1.0, kernel_size=20):
        """
        Smooth the base_value_store along x (columns) using a Gaussian kernel.

        Args:
            var (array): Array to smooth (1D).
            sigma (float): Approximate width of Gaussian (default 3).
            kernel_size (int): Size of the kernel.

        Returns:
            smoothed array (same shape).
        """
        x = jnp.linspace(-3 * sigma, 3 * sigma, kernel_size)
        gauss_kernel = jnp.exp(-0.5 * (x / sigma)**2)
        gauss_kernel /= gauss_kernel.sum()

        # no [:, None] here
        smoothed = convolve(var, gauss_kernel, mode='same')
        return smoothed


    ### Reaction steps on discretized gridded matrices
    @partial(jit, static_argnames=['decay_fun', 'mode'])
    def update_grid(gridw, gridh, current_topo, i, sea_level, decay_fun=None, mode='accumulate',
                    f_react=f_react, fuzz=fuzz, d13c_seawater=0.0, tau=tau, grid_org_prev=None, grid_carb_prev=None, grid_dz=None):
        """
        Unified grid updater for SWI proximity accumulation and diagenetic reaction on gridded data.

        This function updates a 2D grid of proxy values (e.g., d13C, TOC) for each time step and burial depth,
        according to the selected diagenetic or accumulation mode.

        Parameters
        ----------
        gridw : array
            Grid proxy values (e.g., time at SWI, d13C, TOC).
        gridh : array
            Grid elevations (vertical positions).
        current_topo : array
            Current topography (SWI elevation at each x).
        i : int
            Current time step.
        sea_level : float
            Sea level at current time.
        decay_fun : callable, optional
            User-supplied decay function of vertical distance. If None, a default exponential decay is used.
        mode : str, optional
            Mode for grid updating. Supported modes:
                - 'accumulate': Accumulate decay function output (e.g., time or d13C) for all buried grid cells.
                - 'reaction': Move grid proxy values toward decay function value by a fixed fraction (f_react) per step.
                - 'molar': Update grid by mole-weighted mixing of respired organic C and seawater DIC, using f_react as mixing fraction.
                - 'respiration': Remove a fraction (f_react) of grid proxy (e.g., TOC) per step, simulating remineralization.
                - (Other modes may be supported as implemented.)
        f_react : float, optional
            Reaction rate factor for 'reaction', 'molar', and 'respiration' modes (used for both organic carbon respiration and carbonate reaction scaling).
        fuzz : float, optional
            Smoothing factor for shoreline transition.
        tau : float, optional
            Tau value for decay function if none provided.
        d13c_seawater : float or array, optional
            Seawater d13C value for decay function or mixing.
        grid_org_prev : array, optional
            Previous grid of respired organic C (required for 'molar' mode).
        grid_dz : array, optional
            Grid of vertical increments (required for 'molar' mode).

        Behavior of each mode:
            - 'accumulate': Adds decay function value to all valid buried cells (no reaction, just accumulation).
            - 'reaction': Moves each grid cell toward the decay function value by a fraction f_react per step.
            - 'molar': Updates grid by mole-weighted mixing of respired organic carbon and seawater DIC, with f_react as the mixing fraction.
            - 'respiration': Removes a fraction f_react of grid proxy (e.g., TOC) per step, scaled by the decay function.

        Returns
        -------
        gridw : array
            Updated proxy grid.
        base_value_grid : array
            The base value used for decay mixing, smoothed.
        decay_value : array
            The decay function value for each grid cell.
        combined_mask : array
            Mask applied to determine where updates occur.
        delta : array
            The change applied to gridw (for mass balance tracking).
        """

        current_topo_broadcast = current_topo[None, :]
        vertical_distance = gridh - current_topo_broadcast

        # Match update_sto: average proxy values up to base_depth above deepest burial
        deepest_burial = jnp.nanmin(gridh, axis=0)  # (Nx,)
        mask_bottom = (gridh - deepest_burial[None, :]) <= base_depth
        masked_proxy = jnp.where(mask_bottom, gridw, jnp.nan)
        base_value_grid = jnp.nanmean(masked_proxy, axis=0)  # (Nx,); jnp.ones(200)*1.0

        # add smoothing here (there is some vertical striping from the per x-grid averaging)
        base_value_grid = smooth_base_value(base_value_grid)

        # Masks
        burial_mask = gridh <= current_topo_broadcast
        valid_mask = ~jnp.isnan(gridh)
        sub_sea_mask = gridh <= sea_level
        shoreline_weight = jax.nn.sigmoid((sea_level - current_topo)[None, :] / fuzz)
        combined_mask = burial_mask * sub_sea_mask * valid_mask * shoreline_weight

        # centralize effective_f_react calculation for all applicable modes
        effective_f_react = scale_f(f_react) if grid_level == 'compiled' else f_react

        if mode == 'molar':
            # Molar mode logic, similar to update_sto for molar
            if grid_org_prev is not None:
                       
                # calculate moles seawater DIC in pore space
                phi = jnp.clip(porosity_fun(vertical_distance), 0.0, 1.0) # porosity
                n_sw,_ = calc_mol(topo=None,phi=phi,dz=grid_dz,mode='seawater',params=params,DIC_mult=sw_DIC_mult) # accounts for porosity (phi) internally
                n_sw = rem_ol(n_sw) # remove outliers
                d13c_sw = d13c_seawater # isotopic composition of seawater; relates to dic i define outside of this
                        
                # organics and porefluid d13C mass balance; grid_org_prev contains the respired organic molar data, same shape as grid
                n_org = phi*grid_org_prev
                d13c_org = d13c_sw + ep  # -25 per mil, from params # make sure this is seawater minus ep!!!

                # calculate moles of existing carbonate (passed from grid_carb_prev)
                n_carb = grid_carb_prev
                n_carb = (1 - phi) * n_carb
                d13c_carb = gridw # d13c composition of existing carbonate
                
                # mol balance (new version taking mass of existing carbonate into account)
                d13c_calc_raw,grid_org_prev = mol_bal_mass(n_sw, d13c_sw, n_org, d13c_org, n_carb, d13c_carb, effective_f_react, pad=1e-12,return_org=True)
                d13c_calc = rem_ol(d13c_calc_raw) # remove outlier
                
                # blend with previous values using combined_mask weights
                gridw = gridw * (1 - combined_mask) + d13c_calc * combined_mask
                
                # return these. not needed anymore, but more work to get rid of redundant returns for now
                delta = jnp.zeros_like(gridw)
                decay_value = jnp.zeros_like(gridw)
                
            else:
                delta = jnp.zeros_like(gridw)
                decay_value = jnp.zeros_like(gridw)

        else:
            # setup decay curve
            if decay_fun is None:
                # match the same behavior: simple default for accumulation
                decay_value = default_decay_fun(vertical_distance)
            else:
                decay_value = decay_fun(vertical_distance, d13c_seawater=d13c_seawater if mode != 'respiration' else 1.0, base_value=base_value_grid if mode != 'respiration' else 0.0, tau=tau)

            if mode == 'accumulate':
                update = decay_value * combined_mask
                delta = 0.0
                gridw = gridw + update

            elif mode == 'reaction':
                delta = effective_f_react * (decay_value - gridw) * combined_mask
                gridw = gridw + delta

            elif mode == 'respiration':
                #phi = jnp.clip(porosity_fun(vertical_distance), 0.0, 1.0) # porosity
                phi = 1
                # Use f_react for organic carbon respiration as well
                effective_loss = effective_f_react * (phi * decay_value) * combined_mask
                lost_TOC = gridw * effective_loss
                gridw = gridw - lost_TOC
                delta = jnp.zeros_like(gridw) # needs a return

        if mode == 'molar':
            return gridw, base_value_grid, decay_value, combined_mask, delta, grid_org_prev
        else:
            return gridw, base_value_grid, decay_value, combined_mask, delta


    @jit
    def mass_balance(sea_level, mtot_exp_i, mtot_obs_i, L1_bflux, R1_bflux, L2_bflux, R2_bflux, R1L_tot, R1R_tot, R2L_tot, R2R_tot, bias_tot, mass_dt, ml_tot, mr_tot, ml_m_tot, mr_m_tot, htf, hti, wtf, wti, h_erode, w_erode, dh, qh_sed, qw_avg, gridh, gridw, gridw_old, decay_value, f_react, ya, combined_mask,delta_grid):
        """
        Compute mass balance and account for boundary loss.

        Args:
            mtot_exp_i (float): Cumulative total expected mass.
            mtot_obs_i (float): Cumulative total observed mass.
            L1_bflux (float): Left boundary topography flux.
            R1_bflux (float): Right boundary topography flux.
            L2_bflux (float): Left boundary proxy flux.
            R2_bflux (float): Right boundary proxy flux.
            ml_tot (float): Cumulative observed proxy mass.
            mr_tot (float): Cumulative expected proxy mass.
            h_trans (array-like): Transported topography mass.
            qh_sed (array-like): Sediment input mass.
            wtf (array-like): Proxy values.

        Returns:
            tuple: Updated mass totals and balances:
                - mtot_exp_i: Total expected mass.
                - mtot_obs_i: Total observed mass.
                - ml_tot: Observed proxy mass.
                - mr_tot: Expected proxy mass.
        """

        # define transported mass
        h_trans = htf - hti

        # define eroded mass and isotopes
        eroded_mass = -jnp.nansum(h_erode) if w_set_erode else 0 # erosion depth (h_erode is negative for erosion)
        eroded_isotopes = jnp.nansum(h_erode * w_erode) if w_set_erode else 0

        # boundary isotopes without diff_dir as simplified; left side will always increase when on a right facing ramp, and right side will always decrease
        w_L = wtf[1]
        w_R = wtf[-2]

        ### cumulative boundary returns ### maybe a problem because new boundary topo should be calculated before the diff_dir() funs, but I need the signs to add them?
        # save boundary values for return
        R1L_tot += L1_bflux
        R1R_tot += R1_bflux

        ##### cumulative; mass-weighted proxy #####

        # boundary isotopes, dynamic for if using no res bounds or not
        if no_res_bounds:
            L1B_iso = L2_bflux
            R1B_iso = R2_bflux
        else:
            L1B_iso = w_L # isotopic value at left boundary
            R1B_iso = w_R # isotopic value at right boundary

        L1B_w = L1_bflux * L1B_iso # single values so don't need to sum and dot
        R1B_w = R1_bflux * R1B_iso

        # save proxy values for return
        R2L_tot = L1B_iso
        R2R_tot = R1B_iso

        ## reaction balance; from grids
        # add reaction expected contribution
        dy = abs(jnp.diff(ya)[0])  # vertical grid spacing
        delta_expected = delta_grid # f_react * (decay_value - gridw) * combined_mask (passed from initial funciton to avoid redundancy)
        expected_reaction_mass = jnp.nansum(delta_expected) * dy

        # add reaction observed contribution
        delta_observed = gridw - gridw_old
        observed_reaction_mass = jnp.nansum(delta_observed) * dy

        # Observed
        ml_tot += jnp.nansum(jnp.dot(h_trans,wtf)) + eroded_isotopes + observed_reaction_mass

        # Expected; # adding boundary condition inputs and outputs; need erosion here to balance the mass
        mr_tot += jnp.nansum(jnp.dot(qh_sed,qw_avg)) + L1B_w + R1B_w + eroded_isotopes + expected_reaction_mass

        # sum total added mass from all growth functions
        mtot_obs_i += jnp.nansum(h_trans) + eroded_mass # again need to add eroded mass here to balance the mass
        mtot_exp_i += jnp.nansum(qh_sed) + L1_bflux + R1_bflux + eroded_mass

        # normalize to convert units to w only - need to use consistent denominator to avoid error propogation
        unified_denom = mtot_obs_i + L1_bflux + R1_bflux # or mtot_exp_i if preferred
        den = unified_denom if normalize_balance else 1 # if normalize_balance is False, then den = 1 (no normalization)
        ml = ml_tot / den # observed
        mr = mr_tot / den # expected

        ##### cumulative; mass only #####
        # observed
        ml_m_tot += jnp.nansum(h_trans) + eroded_mass

        # expected - adding the boundaries here makes up for the loss
        m_bounds = jnp.nansum(qh_sed) + L1_bflux + R1_bflux + eroded_mass
        mr_m_tot += m_bounds
        mass_dt = m_bounds - eroded_mass # add non cumulative total mass for bias calculations; without erosion

        # vars
        ml_m = ml_m_tot
        mr_m = mr_m_tot

        ### cumulative mass balance returns ###
        wavg_obs = ml
        wavg_exp = mr
        mavg_obs = ml_m
        mavg_exp = mr_m

        ### Bias; not including erosion because later we want to account for this and leave erosion imbalance as that is real imbalance
        # quantify bias; [residual between obs and exp at the end]/[total number of dt]/[number of dx]/[average mass added per dt]
        # terms without erosion
        ml_tot_no_erosion = ml_tot - eroded_isotopes # no erosion version
        mr_tot_no_erosion = mr_tot - eroded_isotopes # no erosion version
        mtot_obs_i_no_erosion = mtot_obs_i - eroded_mass # no erosion version
        #mtot_exp_i_no_erosion = mtot_exp_i - eroded_mass # no erosion version; commented out as not used anywhere
        unified_denom_no_erosion = mtot_obs_i_no_erosion + L1_bflux + R1_bflux # or mtot_exp_i if preferred
        den_no_erosion = unified_denom_no_erosion if normalize_balance else 1 # no erosion version

        ### Separate versions excluding erosion ###
        # Observed and expected without erosion
        ml_no_erosion = ml_tot_no_erosion / den_no_erosion
        mr_no_erosion = mr_tot_no_erosion / den_no_erosion

        steps = params['total_n']*params['compiled_steps']*params['dt'] # I think dt should be here; same as end at top
        mass_mean = jnp.nanmean(mass_dt) # non-cumulative mean mass added through growth functions per dt
        # excluding erosion
        bias_tot = (mr_no_erosion - ml_no_erosion) / steps / params['Nx'] / mass_mean  # bias excludes erosion (non absolute version)

        return mtot_exp_i, mtot_obs_i, wavg_exp, wavg_obs, mavg_exp, mavg_obs, ml_tot, mr_tot, ml_m_tot, mr_m_tot, R1L_tot, R1R_tot, R2L_tot, R2R_tot, bias_tot, mass_dt


    def get_effective_fraction(f_react=None, compiled_steps=params['compiled_steps'], mode='reaction'):
        """
        Calculate effective reaction/loss per dt and per compiled chunk based on the mode.
        For 'reaction' and 'respiration' modes, return calculated percent reacted/loss respectively.
        For 'accumulate' or other modes, return 0.0 for both values.
        Returns decimal fractions, multiply by 100 for percent
        """
        if mode == 'reaction' or mode =='respiration':
            per_dt = f_react
            per_compiled = scale_f(f_react)
            return per_dt, per_compiled
        else:
            # for accumulate or unsupported modes
            return 0.0, 0.0
    
    
    # calculate percent change due to reaction curves
    f_dt, f_frac = get_effective_fraction(f_react=f_react, compiled_steps=compiled_steps,mode=track_react)
    
    ## Initiate variables and run parameters
    step = 0
    hs = []     # topo
    ws = []     # proxy
    ts = []     # time
    ds = []     # depth
    sh = []     # shoreline
    wobs = []   # observed average w
    wexp = []   # expected average w
    mobs = []   # observed average mass
    mexp = []   # expected average mass
    mtot_exp = []   # total accumulated expected mass
    mtot_obs = []   # total accumulated observed mass
    mltot = []  # mass balance cumsum term for right side
    mrtot = []  # mass balance cumsum term for right side
    mlmtot = []  # mass balance cumsum term for right side; only mass
    mrmtot = []  # mass balance cumsum term for right side; only mass
    R1Ltot = [] # boundary residual 1 left
    R1Rtot = [] # boundary residual 1 right
    R2Ltot = [] # boundary residual 2 left
    R2Rtot = [] # boundary residual 2 right
    biastot = [] # bias term for mass balance
    massdt = [] # mass added per dt
    compiled_steps = params['compiled_steps']
    total_n = params['total_n']
    dt = params['dt']
    sed_rate = jnp.zeros_like(c[::2]) # initialize carbonate sed_rate
    o_sed_rate = jnp.zeros_like(c[::2]) # initialize organic sed_rate

    dic_arr = []

    # initialize base_value from gridded approach
    base_value_grid = jnp.full((Nx,), jnp.nan)  # NOT called base_value

    if full_storage:
        if storage_level == 'dt':
            # preload final, full-sized matrix into the model run
            sto = jnp.full((d.size,total_n*compiled_steps), jnp.nan) # version with all nans instead of zeros
            base_value_store = jnp.full((Nx, total_n*compiled_steps), jnp.nan)
             # track carbonate growth rate
            csed_sto = jnp.full((d[::2].size,total_n*compiled_steps), jnp.nan) # store sedimentation rate for carbonate only
            osed_sto = jnp.full((d[::2].size,total_n*compiled_steps), jnp.nan) # store sedimentation rate for organics only
            dsed_sto = jnp.full((d[::2].size,total_n*compiled_steps), jnp.nan) # store sedimentation rate for topographic difference (hf - hi)

        elif storage_level =='compiled':
            sto = jnp.full((d.size,total_n), 0.) # version with zeros and 2000 steps only
            base_value_store = jnp.full((Nx, total_n), jnp.nan)
            csed_sto = jnp.full((d[::2].size,total_n), jnp.nan) # store sedimentation rate for carbonate only
            osed_sto = jnp.full((d[::2].size,total_n), jnp.nan) # store sedimentation rate for organics only
            dsed_sto = jnp.full((d[::2].size,total_n), jnp.nan) # store sedimentation rate for topographic difference (hf - hi)

    else:
        sto = jnp.full((1,1), jnp.nan) # place holder for low memory useage when don't need the storage matrix
        base_value_store = jnp.full((1,1), jnp.nan)
        csed_sto = jnp.full((1,1), jnp.nan) # store sedimentation rate for carbonate only
        osed_sto = jnp.full((1,1), jnp.nan) # store sedimentation rate for organics only
        dsed_sto = jnp.full((1,1), jnp.nan) # store sedimentation rate for topographic difference (hf - hi)

    # make previous model storage available if not passed
    # molorg_prev must be of same shape as current sto in current model run
    if molorg_prev is None:
        molorg_prev = jnp.zeros_like(sto[1::2])
    
    if molcarb_prev is None:
        molcarb_prev = jnp.zeros_like(sto[1::2])
        
    ## preload gridded space array (x,h) not making it optional as should be part of core model
    # calculate max and min values
    if not ymin_ymax:
        if rsl:
            ymin,ymax = np.min(hi)-np.max(abs(sl_fun(t))), np.max(hi)+np.max(abs(sl_fun(t)))
        else:
            ymin,ymax = 0,400
    else:
        ymin,ymax = ymin_ymax[0],ymin_ymax[1]

    print(f'calculated ymin,ymax = {ymin:.2f}, {ymax:.2f}')
     # define vertical grid and layers for dynamic base_value
    ya = jnp.linspace(ymax, ymin, grid_ylen)

    # initiate grids and values
    xlen = x.size # using full range of x values (Nx = 100 here)
    grid,gridw,gridh,grid_org_prev,grid_carb_prev,grid_dz = make_grid()
    print(f'mean grid dy = {np.abs(np.mean(np.diff(ya))):.3e}') # print grid dy

    wavg_exp = wi[0] # jnp.sum(hi * wi) / (jnp.sum(hi) + 1e-8)     # pre-allocate avg mass balance expected w; using average initial proxy value; avoiding div 0; as no initial topography, just wi[0]
    wavg_obs = wi[0] # wavg_exp # pre-allocate mass balance observed w; assume matches expected initially
    mavg_exp = 0     # pre-allocate avg mass balance expected mass
    mavg_obs = 0     # pre-allocate mass balance observed mass
    mtot_exp_i = 0 # jnp.nansum(hi) # normalize by number of elements
    mtot_obs_i = 0 # mtot_exp_i     # assume observed equals expected initially
    ml_tot = 0
    mr_tot = 0
    ml_m_tot = 0
    mr_m_tot = 0
    rsl = jnp.max(hi)  # assuming sea level initially floods the highest point of topography
    depth = rsl - hi                # initial depth relative to sea level (was np.zeros_like(h))
    new_shore = jnp.zeros_like(h[0])  # for shoreline return
    R1L_tot = jnp.zeros_like(hi[0]) # boundaries total topo left
    R1R_tot = jnp.zeros_like(hi[0])  # boundaries total topo right
    R2L_tot = hi[0]*wi[0]  # boundaries proxy left (start with initial value)
    R2R_tot = hi[-1]*wi[-1]  # boundaries proxy right (start with initial value)
    bias_tot = 0 # initially assume no bias
    mass_dt = 0 # initia mass added through growth functions per dt

    # make the inverse jacobian once so you can feed it to the cg algorithm
    f = jit(lambda guess: residual(c, 0, dt, dx, guess)) # isolate the guess as the variable in residual function, sea level = 0 here as placeholder
    J = jax.jacfwd(f)(d)  # calculate jacobian of residual
    key = random.PRNGKey(0)
    jitter = jnp.eye(c.size)*random.normal(key, shape=(d.size,))*1e-3
    M = jnp.linalg.inv(J+jitter)

    ##  Model name and description
    todays_date = str(datetime.now())[:16] # limit to minutes
    name = f'{model_desc}_A{A}_ylen{grid_ylen}_{todays_date}' # [:-6]

    ## run model
    prog =  lambda x: tqdm(range(x),desc=f'running model: {name}',unit=' compiled steps')

    # time loop
    for i in prog(total_n):
        t_idx = i*compiled_steps*dt
        sea_level = sl_fun(t_idx)
        sed_w_t = sec_w_fun(t_idx)
        
        if wi_sec:
            # secular change to initial left proxy boundary value (c[1::2], so c[1] should be first value of proxy?) it seems to work!
            w_init_sec = sec_w_fun(t_idx)
            c = c.at[1].set(w_init_sec) # c[1::2] first
            c = c.at[-1].set(w_init_sec) # c[1::2] last

        hs.append(c[::2])
        ws.append(c[1::2])
        ts.append(step*dt)
        ds.append(depth)
        sh.append(new_shore)
        wexp.append(wavg_exp)
        wobs.append(wavg_obs)
        mexp.append(mavg_exp)
        mobs.append(mavg_obs)
        mtot_exp.append(mtot_exp_i)
        mtot_obs.append(mtot_obs_i)
        mltot.append(ml_tot)
        mrtot.append(mr_tot)
        mlmtot.append(ml_m_tot)
        mrmtot.append(mr_m_tot)
        R1Ltot.append(R1L_tot)
        R1Rtot.append(R1R_tot)
        R2Ltot.append(R2L_tot)
        R2Rtot.append(R2R_tot)
        biastot.append(bias_tot)
        massdt.append(mass_dt)

        # newton iterations - into jax
        R1L_tot,R1R_tot,R2L_tot,R2R_tot,bias_tot,mass_dt,mtot_exp_i,mtot_obs_i,ml_tot,mr_tot,ml_m_tot,mr_m_tot,wavg_exp,wavg_obs,mavg_exp,mavg_obs,depth,new_shore,sto,grid,gridw,grid_org_prev,grid_carb_prev,grid_dz,gridh,d,ya,base_value_store,base_value_grid,molorg_prev,molcarb_prev,csed_sto,sed_rate,osed_sto,o_sed_rate,dsed_sto = run_steps(
            step,step+compiled_steps,dt,dx,
            R1L_tot,R1R_tot,R2L_tot,R2R_tot,bias_tot,mass_dt,mtot_exp_i,mtot_obs_i,ml_tot,mr_tot,ml_m_tot,mr_m_tot,
            wavg_exp,wavg_obs,mavg_exp,mavg_obs,depth,new_shore,sto,grid,gridw,grid_org_prev,grid_carb_prev,grid_dz,gridh,c,ya,base_value_store,base_value_grid,molorg_prev,molcarb_prev,csed_sto,sed_rate,osed_sto,o_sed_rate,dsed_sto
        )
        
        depth = sea_level - d[::2] # positive downwards for equations
        if depth_w_fun:
            sed_w_depth =  depth_w_fun(depth)    # changes to w as function of depth for simplified lateral gradients

        # seawater isotopes (DIC)
        ones = jnp.ones_like(d[::2])
        dic_s = ones*sed_w_t if sec_w_fun else 0 # secular isotopic value of seawater dic
        dic_d = ones*sed_w_depth if depth_w_fun else 0 # isotopic value as function of depth
        dic = dic_s + dic_d  # secular + gradient. if depth gradient does not exist, it is just secular
        
        if full_storage and storage_level == 'compiled':
            # current solution
            sto = sto.at[:, i].set(d)
            # rates
            rdiff = sto[::2,i] - sto[::2,i-1]
            csed_sto = csed_sto.at[:,i].set(sed_rate/dt) # store carbonate sedimentation rate
            osed_sto = osed_sto.at[:,i].set(o_sed_rate/dt) # store organic sedimentation rate
            dsed_sto = dsed_sto.at[:,i].set(rdiff/dt) # net topographic change

            # update grid_org_prev and grid_dz (sw mole calcs)
            if track_react == 'molar':
                # get beds
                bed_now = sto[::2, i]
                bed_prev = sto[::2, i - 1]

                # change in height and dz; and handle erosion (remove it)
                dh = bed_now - bed_prev
                dz_col = jnp.clip(dh, 0.0, None)
                dz_col = jnp.where(dh < 0, jnp.nan, dz_col)
                
                # get previous moles at current step
                molC_now = molorg_prev[:, i]
                molcarb_now = molcarb_prev[:, i]
                
                # set grids
                grid_org_prev = set_grid_prev(d[::2], grid_org_prev, molC_now, ya)
                grid_carb_prev = set_grid_prev(d[::2], grid_carb_prev, molcarb_now, ya)
                grid_dz = set_grid_prev(bed_now, grid_dz, dz_col, ya)
            
            # grid reaction step
            if swi_dist_calc:
                
                if grid_level=='compiled':
                    if track_react == 'accumulate':
                        gridw, base_value_grid, decay_value, combined_mask, delta = update_grid(
                            gridw, gridh, d[::2], i, sea_level, decay_fun=swi_fun, mode='accumulate'
                        )
                    elif track_react == 'reaction':
                        gridw, base_value_grid, decay_value, combined_mask, delta = update_grid(
                            gridw, gridh, d[::2], i, sea_level, decay_fun=swi_fun, mode='reaction',
                            f_react=f_react, d13c_seawater=dic, tau=tau
                        )
                    elif track_react == 'molar':
                        gridw, base_value_grid, decay_value, combined_mask, delta, _ = update_grid(
                            gridw, gridh, d[::2], i, sea_level, decay_fun=swi_fun, mode='molar',
                            f_react=f_react, d13c_seawater=dic, tau=tau, grid_org_prev=grid_org_prev, grid_carb_prev=grid_carb_prev, grid_dz=grid_dz
                        )
                    elif track_react == 'respiration':
                        gridw, base_value_grid, decay_value, combined_mask, delta = update_grid(
                            gridw, gridh, d[::2], i, sea_level, decay_fun=swi_fun, mode='respiration',
                            f_react=f_react, tau=tau
                        )

            if swi_dist_calc:
            
                if track_react == 'accumulate':
                    sto, base_value = update_sto(
                        sto, d, i, sea_level,
                        decay_fun=swi_fun,
                        mode='accumulate', base_value_passed=base_value_grid
                    )
                    base_value_store = base_value_store.at[:, i].set(base_value[:, 0])
                elif track_react == 'reaction':
                    sto, base_value = update_sto(
                        sto, d, i, sea_level,
                        decay_fun=swi_fun,
                        mode='reaction',
                        f_react=f_react,
                        d13c_seawater=dic[:, None], base_value_passed=base_value_grid,tau=tau
                    )
                    base_value_store = base_value_store.at[:, i].set(base_value[:, 0])
                elif track_react == 'molar':
                    # apply reactive diagenesis
                    sto, base_value, _ = update_sto(
                        sto, d, i, sea_level,
                        decay_fun=swi_fun,
                        mode='molar',
                        f_react=f_react,
                        d13c_seawater=dic[:, None], base_value_passed=base_value_grid, tau=tau, molorg_prev=molorg_prev, molcarb_prev=molcarb_prev,
                    )
                    base_value_store = base_value_store.at[:, i].set(base_value[:, 0])
                elif track_react == 'respiration':
                    sto, base_value = update_sto(
                        sto, d, i, sea_level,
                        decay_fun=swi_fun,
                        mode='respiration',
                        f_react=f_react,
                        base_value_passed=base_value_grid,tau=tau
                    )
                    base_value_store = base_value_store.at[:, i].set(base_value[:, 0])

        # step out of jax into python
        c = d # re update guess

        step += compiled_steps # next batch of compiled steps

    # make into numpy arrays
    base_value_store = np.array(base_value_store)
    sto = np.array(sto)
    csed_sto = np.array(csed_sto)
    osed_sto = np.array(osed_sto)
    dsed_sto = np.array(dsed_sto)
    gridw = np.array(gridw)
    grid_org_prev = np.array(grid_org_prev)
    grid_carb_prev = np.array(grid_carb_prev)
    grid_dz = np.array(grid_dz)
    gridh = np.array(gridh)
    dic_arr.append(dic)
    ts = np.array(ts)
    ds = np.array(ds)   # depth (or whatever I return and place here); Inside model calculations (incl. growth funs), (+) depth means deeper (changed this so everywhere depth is positive, consistency)
    sh = np.array(sh)
    beds = np.array(hs)
    proxy = np.array(ws)
    SL_fun = jit(sl_fun)
    rsl_strat = np.array(SL_fun(ts))
    w_strat_t = np.array(sec_w_fun(ts))
    wexp = np.array(wexp)
    wobs = np.array(wobs)
    mexp = np.array(mexp)
    mobs = np.array(mobs)
    mtot_exp = np.array(mtot_exp_i)
    mtot_obs = np.array(mtot_obs_i)
    mltot = np.array(mltot)
    mrtot = np.array(mrtot)
    mlmtot = np.array(mlmtot)
    mrmtot = np.array(mrmtot)
    R1Ltot = np.array(R1Ltot)
    R1Rtot = np.array(R1Rtot)
    R2Ltot = np.array(R2Ltot)
    R2Rtot = np.array(R2Rtot)
    biastot = np.array(biastot)
    time_array = np.ones_like(beds)*np.arange(start,total_n)[:,np.newaxis] # for colouring basin by time

    ## remove eroded beds (vectorized, faster) ##
    beds_eroded = np.minimum.accumulate(beds[::-1,:], axis=0)[::-1,:]
    mask = (beds > beds_eroded) # mask where erosion needs to happen
    proxy_eroded = np.where(mask, np.nan, proxy)

    # storage version
    if swi_dist_calc:
        beds_sto = sto[::2].T
        proxy_sto = sto[1::2].T
        beds_sto_eroded = np.minimum.accumulate(beds_sto[::-1,:], axis=0)[::-1,:]
        mask_sto = (beds_sto > beds_sto_eroded)
        proxy_sto_eroded = np.where(mask_sto, np.nan, proxy_sto)
    else:
        beds_sto = beds
        proxy_sto = proxy
        beds_sto_eroded = beds_eroded
        proxy_sto_eroded = proxy

    # remove eroded beds from arrays
    sh_eroded = np.where(mask, np.nan, sh[:,np.newaxis]) # as shoreline is a single value per timestep need to manually add an axis
    ds_eroded = np.where(mask, np.nan, ds)
    ts_eroded = np.where(mask, np.nan, time_array)

    ### simple plot of model results
    if plot_out:
        if full_storage:
            vmin=np.nanmin(sto[1::2])
            vmax=np.nanmax(sto[1::2])
        else:
            vmin=np.nanmin(proxy_eroded)
            vmax=np.nanmax(proxy_eroded)

        _,ax = plt.subplots(1,2,figsize=figsize,layout='constrained')
        for i in tqdm(range(len(ts)),desc = 'plotting results',unit=' beds'):
            if i%plot_skip==0:
                sc = ax[0].scatter(x,beds[i],c=proxy[i],marker='o',alpha=1,zorder=100,cmap=cmap,vmin=vmin, vmax=vmax)
                ax[1].plot(x,proxy[i],zorder=100,color='grey',alpha=0.1)
        ax[0].scatter(x,beds[-1],c=proxy[-1],marker='o',alpha=1,edgecolor='.3',lw=0.5,zorder=100,cmap=cmap,vmin=vmin, vmax=vmax)
        ax[0].plot(x,beds[-1],color='.3',ls='-',zorder=1000)
        ax[0].plot(x,beds[0],color='.3',ls='-',zorder=1000)
        ax[0].set_ylabel('topography')
        ax[0].set_xlabel('x')
        ax[1].set_xlabel('x')
        ax[1].set_ylabel('proxy')

        mod_A = params['A']
        ax[0].set_title(f'A = {mod_A}')
        plt.colorbar(sc,ax=ax[0],label='w')
        print(f'actual ymin,ymax = {np.nanmin(beds):.2f}, {np.nanmax(beds):.2f}')

        if full_storage: # these functions rely on storage_matrix
            # Look at average erodion/deposition distance for each dt
            _,_ = dh_values(sto,means=True,out=True)

    # grab all local vars so I can make a dict with the following names and not type them out twice
    local_vars = locals()

    # return output as dictionary
    var_names = ['name',
                 'params',
                 'wi',
                 'dic',
                 'gridw',
                 'grid_org_prev',
                 'grid_carb_prev',
                 'grid_dz',
                 'gridh',
                 'csed_sto',
                 'osed_sto',
                 'dsed_sto',
                 'sto',
                 'start',
                 'end',
                 'x',
                 'dx',
                 'xmin',
                 'xmax',
                 'Nx',
                 'ymin',
                 'ymax',
                 'total_n',
                 'beds',
                 'beds_eroded',
                 'proxy',
                 'proxy_eroded',
                 'ts',
                 'ts_eroded',
                 'rsl_strat',
                 'w_strat_t',
                 'ds',
                 'ds_eroded',
                 'sh',
                 'sh_eroded',
                 'wexp',
                 'wobs',
                 'mexp',
                 'mobs',
                 'mtot_exp',
                 'mtot_obs',
                 'mltot',
                 'mrtot',
                 'mlmtot',
                 'mrmtot',
                 'R1Ltot',
                 'R1Rtot',
                 'R2Ltot',
                 'R2Rtot',
                 'biastot',
                 'massdt',
                 'normalize_balance',
                 'proxy_sto',
                 'proxy_sto_eroded',
                 'beds_sto',
                 'beds_sto_eroded',
                 'base_value_store',
                 'f_dt',
                 'f_frac']

    # avoiding retyping all the names
    results = {vname: local_vars[vname] for vname in var_names}

    return results