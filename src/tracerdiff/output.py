import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from IPython.display import clear_output
import time
import seaborn as sns
import os
import scipy
from ipywidgets import interact
from matplotlib.colors import ListedColormap
import jax.numpy as jnp
from scipy.stats import norm, uniform, gaussian_kde
from .utils import *
from cycler import cycler

class Model_output:
    """

    Args:

    Returns:

    """
    def __init__(self,facies_data,facies_colours,im_ylen=None,tol=2,shore=True,images=True,use_gridw=False,swi_dist_calc=False,**model_output_vars):

        # auto define all params from **model_output_vars (model outputs from run() in model_dev.py)
        for key,val in model_output_vars.items():
            setattr(self, key, val)

        # if im_ylen not defined match grid_ylen
        if not im_ylen:
            im_ylen = self.params['grid_ylen']
            print(f'im_ylen = {im_ylen} matching grid_ylen')

        # if calculating swi; use sto[::2] for topography and sto[1::2] for proxy as these are made with compiled step loops matching shape of beds and proxy, but with modfied stratigraphy for time since surface
        # full version of arrays copied here. doing like this as avoids changing most functions
        self.proxy_raw = self.proxy
        self.beds_raw = self.beds
        self.proxy_eroded_raw = self.proxy_eroded
        self.beds_eroded_raw = self.beds_eroded

        if swi_dist_calc:
            self.proxy = self.proxy_sto
            self.beds = self.beds_sto
            self.proxy_eroded = self.proxy_sto_eroded
            self.beds_eroded = self.beds_sto_eroded

        # facies attributes
        self.facies_colours = facies_colours
        self.facies_data, self.labels, self.cmap = self.init_facies(facies_data,facies_colours)

        # images
        self.images = images
        self.im_ylen = im_ylen
        if self.images:
            if use_gridw:
                self.im_w = self.gridw
            else:
                self.im_w, _ = self.im_plot(show='proxy', ylen=im_ylen)
            self.im_d, self.extent = self.im_plot(show='depth',ylen=im_ylen)
            self.im_h, _ = self.im_plot(show='topo',ylen=im_ylen)
            self.im_t, _ = self.im_plot(show='time',ylen=im_ylen)

            # facies map
            self.facies_pred = self.map_facies(ter_thresh=-0.01,plot=False)

        # Track failed shore indices
        self.failed_shore_idxs = []

        # shoreline
        self.shore = shore
        if self.shore:
            self.sl, self.diff = self.grab_shore(targ='beds', tol=tol)
            if self.diff is not None:
                self.tran_idx, self.reg_idx = sign_jumps(self.diff)
            else:
                self.tran_idx, self.reg_idx = [], []

    ## these should be moved to plotting eventually
    def init_facies(self,facies_data,facies_colours):
        """
        facies_data: list of dicts

        facies_data:
        # example data -> all made up
        # facies_data = [{'name':'Terrestrial','type': 'land'}, # terrestrial is always everyting <0 (above water)
        #                {'name':'Shelly Grainstone','type': 'gaussian', 'mean': 5, 'std': 5},
        #                {'name':'Oolitic Grainstone','type': 'gaussian', 'mean': 10, 'std': 10},
        #                {'name':'Packstone','type': 'gaussian', 'mean': 20, 'std': 10},
        #                {'name':'Wackestone','type': 'gaussian', 'mean': 30, 'std': 10},
        #                {'name':'Mudstone','type': 'gaussian', 'mean': 100, 'std': 10}]

        # define facies with options for gaussian, uniform, or real data; distribution options as follows:
        # 'type': 'gaussian', 'mean': 50, 'std': 10
        # 'type': 'uniform', 'min': 70, 'max': 90
        # 'type': 'data', 'data': np.random.uniform(120, 140, size=1000)
        # 'type': 'data', 'data': your_data_array

        """
        import copy
        facies_data = copy.deepcopy(facies_data)
        # initiate widths for terrestrial and upper bound for next one; assumes terrestrial is first
        # terrestrial defaults
        # define new terrestrial facies
        terrestrial = {
            'name': 'Terrestrial',
            'type': 'land',
            'min': 1,
            'max': 2,
            'width': 1
            }
        
        # insert at the beginning of the list
        facies_data.insert(0, terrestrial)
        # next facies
        facies_data[1]['min'] = -0.01 # nicely bounds with terrestrial
        # last facies
        facies_data[-1]['max'] = np.nanmax(self.ds)

        # add labels and colours
        labels = [facies_data[i]['name'] for i in range(len(facies_data))]
        # set colormap from facies colours
        cmap = ListedColormap(facies_colours)

        # add colours to dict
        for facies, color in zip(facies_data, facies_colours):
            facies['colour'] = color
        for i in range(len(facies_data)):
            if 'colour' not in facies_data[i]:
                if i < len(facies_colours):
                    facies_data[i]['colour'] = facies_colours[i]
                else:
                 facies_data[i]['colour'] = 'grey'  # default fallback

        return facies_data, labels, cmap


    def make_image(self,show='proxy',ylen=500,total=None,btarg='eroded'):
        """

        Args:

        Returns:

        """
        if total is None:
            total = self.total_n-1

        a = complex(str(self.beds[0].size)+'j')
        b = complex(str(ylen)+'j') #500j
        xi, yi = np.mgrid[self.xmin:self.xmax:a, self.ymin:self.ymax:b]

        X = np.tile(np.linspace(self.xmin, self.xmax, self.beds[0].size), total)

        # this needs to be fixed. when making animations where limiting image to certain end time, it extrapolates using erode-removed version giving misleading data
        # but when you use the full beds arrays, the eroded material obscures the data. and so needs to be dynamic somehow.
        if btarg=='eroded':
            H = np.hstack(self.beds_eroded[:total])
            W = np.hstack(self.proxy_eroded[:total])
            D = np.hstack(self.ds_eroded[:total])
            T = np.hstack(self.ts_eroded[:total])
        elif btarg=='full':
            # without erosion
            H = np.hstack(self.beds[:total])
            W = np.hstack(self.proxy[:total])
            D = np.hstack(self.ds[:total])
            T = np.hstack(self.ts[:total])

        #remove nans
        # X = X[~np.isnan(W)] # distance
        # H = H[~np.isnan(W)] # topography
        # D = D[~np.isnan(W)] # depth
        # W = W[~np.isnan(W)] # proxy
        # T = T[~np.isnan(T)] # time

        # modified version to handle the possibility of swi calc version from sto
        mask = ~np.isnan(W)
        X = X[mask]
        H = H[mask]
        W = W[mask]
        D = D[mask]
        T = T[mask]

        if show == 'proxy':
            rbf = scipy.interpolate.NearestNDInterpolator((X, H), W)
        elif show=='topo':
            rbf = scipy.interpolate.NearestNDInterpolator((X, H), H)
        elif show =='depth':
            rbf = scipy.interpolate.NearestNDInterpolator((X, H), D)
        elif show =='time':
            rbf = scipy.interpolate.NearestNDInterpolator((X, H), T)

        ai = rbf(xi, yi)

        ai = ai.T

        # modified with raw arrays
        ai[yi.T > self.beds_raw[total]] = np.nan
        ai[yi.T < self.beds_raw[0]] = np.nan
        return ai


    def im_plot(self,show='proxy',ylen=500,total=None,btarg='eroded'):
        """
        Args:

        Returns:

        """
        extent=(self.xmin, self.xmax, self.ymin, self.ymax)

        image = self.make_image(show=show,ylen=ylen,total=total,btarg=btarg)
        #image = np.flipud(image)

        return image,extent


    def secular_dic(self,loc,time_image):
        """
        finds the 'expected' secular dic value when running with secular change
        loc = basin gridpoint index
        time_image = image (im_t) coloured by time from get_image
        dic = secular_dic expected
        y = y values of image
        """
        time_im = time_image[:,loc][~np.isnan(time_image[:,loc])]
        sec_w = self.w_strat_t
        gt = lambda t: sec_w[t]
        dic = []
        for time in time_im:
            time = int(time)
            sec_dic = gt(time)
            dic.append(sec_dic)
        dic = np.array(dic)

        return dic


    def plot_grids(self,out=False,sto=False,diff=False,cmap='coolwarm'):
        """

        """
        fig,ax = plt.subplots(1,3,figsize=(12,3),constrained_layout=True)

        if sto:
            array_out = self.sto[1::2]
            array_out_topo = self.sto[::2]
            vmin = np.nanmin(array_out)
            vmax = np.nanmax(array_out)
        else:
            array_out = self.proxy.T
            array_out_topo = self.beds.T
            vmin = np.nanmin(array_out)
            vmax = np.nanmax(array_out)

        #if extent:
        img = ax[0].imshow(self.gridw,aspect='auto',cmap=cmap,interpolation='none',extent=self.extent)#,vmin=vmin, vmax=vmax)
        ax[0].plot(self.x,self.beds_eroded_raw[-1],color='.3',lw=1)
        ax[0].plot(self.x,self.beds_eroded_raw[0],color='.3',lw=1)
        #else:
            #img = ax[0].imshow(self.gridw,aspect='auto',cmap=cmap,interpolation='none',vmin=vmin, vmax=vmax)
        plt.colorbar(img,ax=ax[0],label='w')
        img = ax[1].imshow(array_out,aspect='auto',cmap=cmap,interpolation='none',vmin=vmin, vmax=vmax)
        plt.colorbar(img,ax=ax[1],label='w')
        if diff:
            # targ = np.diff(self.sto[::2].T,axis=0)
            # targ1 = np.gradient(self.sto[::2].T,axis=0) # maintains shape and has
            # these ones line up with other axes
            targ = np.diff(array_out_topo[::-1],axis=1)
            targ1 = np.gradient(array_out_topo[::2][::-1],axis=1) # maintains shape and has
            lim = 1.5*np.max(abs(targ)) # 1.25e-3
            a = ax[2].contourf(targ1,cmap='seismic',vmin=-lim,vmax=lim)
            plt.colorbar(a,ax=ax[2],format='%.1e',label='erosion / deposition')
            ax[2].set_title('diff h')
        else:
            img = ax[2].imshow(array_out_topo,aspect='auto',cmap=cmap,interpolation='none',vmin=np.nanmin(array_out_topo), vmax=np.nanmax(array_out_topo))
            plt.colorbar(img,ax=ax[2],label='h')
            ax[2].set_title('h')

        # label
        ax[0].set_title('gridded w')
        ax[1].set_title('w')
        ax[0].set_ylabel('h')
        ax[0].set_xlabel('x')
        [ax[i].set_xlabel('t') for i in [1,2]];
        [ax[i].set_ylabel('x') for i in [1,2]];
        if out:
            return fig

    def plot_components(self,xaxis='time',out=False):
        """

        """
        # histograms of growth fractions
        fig,ax = plt.subplots(2,4,figsize=(12,4),layout='constrained',sharex=True)
        axs = ax.ravel()
        labs = ['transported','eroded','algal','pelagic']*2
        targs = [self.th,self.eh,self.ah,self.ph,
                self.tw,self.ew,self.aw,self.pw]
        #targs = [i.ravel() for i in targs]
        for i,j in enumerate(targs):
            if xaxis=='time':
                axs[i].plot(j)
            elif xaxis=='distance':
                axs[i].plot(j.T)
            if i<=3:
                axs[i].set_ylabel(f'h$_{{{labs[i][:1]}}}$')
                axs[i].set_title(labs[i])
            else:
                axs[i].set_ylabel(f'w$_{{{labs[i][:1]}}}$')
                if xaxis=='time':
                    axs[i].set_xlabel('t')
                elif xaxis=='distance':
                    axs[i].set_xlabel('x')
            if out:
                return fig,ax

    ## Animation function - at some point should be integrated into the class framework
    def animate_w(self, type='proxy', ylen=250, scatter=True, mp4=True, model_name=None,norm=None,show='proxy',lab='$w$',alpha=1,
                   frames_dir=None, ani_dir=None, output_format='mp4', fps=15, ax=None, skip=20,surfaces=True,
                   wait_time=0.001, figsize=(8,3.5), pal='coolwarm'):
        """
        new simple animation function that uses the storage matrix (targ) to effectively animate the beds, proxy and sea level history of the basin
        anim_type can be eroded (using only 2000 total_n and grids) to show current layer and all underlying beds without eroded bits, here targ must be model output variable
        (mod in this notebook)
        if anim_type is full, use storage matrix variable (sto) and will use all time steps
        # note: there is a gap between the grid and the scatter as the grid does not build and erode down, it just goes to where it was eroded to, possible future fix
        - also assumes functions have names as in this one (growth, secular) could be generalized eventually
        - this doesn't work well on super small scale runs (the buf gets messed up becauase it is not using all of the timesteps and only the total_n)
        """
        if mp4:
            for i in [':',' ']: # ffmpeg breaks from the colon and space in the time
                model_name = model_name.replace(i,'_')
            # clean up old jpgs
            [os.remove(frames_dir+file) for file in os.listdir(frames_dir) if file.endswith('.jpg')]
        end = self.params['total_n']

        for idx in range(1,end):
            if idx%skip==0:
                # don't want eroded for these
                bed = self.beds[idx]
                proxy = self.proxy[idx]

                fig, ax = plt.subplots(1,figsize=figsize)#,layout='constrained')
                if not mp4:
                    clear_output(wait=True)

                vmin = np.nanmin(self.proxy)
                vmax = np.nanmax(self.proxy)
                #vmin = np.nanmin(self.sto[1::2])
                #vmax = np.nanmax(self.sto[1::2])

                cmap = sns.color_palette(pal,as_cmap=True)

                fin = self.sl.size
                array = self.beds[idx]
                shore = self.sl[idx]

                if type=='proxy':
                    # figure goes here
                    if scatter:
                        ax.scatter(self.x,bed,c=proxy,marker='s',alpha=1,zorder=100,cmap=cmap,s=10,lw=0.15,edgecolor='k',vmin=vmin,vmax=vmax)
                    tim,_ = self.im_plot(show=show,ylen=ylen,total=idx)
                    sm = imshow(tim,extent=self.extent,ax=ax,cmap=pal,norm=norm,vmin=vmin,vmax=vmax)
                    plt.colorbar(sm,label=lab)

                elif type=='beds':
                    ax.set_prop_cycle(cycler('color',sns.color_palette(pal,n_colors=int(self.total_n/skip))))
                    ax.plot(self.x,bed,alpha=alpha)
                    for loc  in range(idx):
                        if loc%skip==0:
                            ax.plot(self.x,self.beds_eroded[loc,:],alpha=alpha)

                # mark shore and rsl
                ax.hlines(y=self.rsl_strat[idx],xmin=shore,xmax=self.x[-1],label='rsl')
                ax.plot(self.x,array,color='grey')
                ax.scatter(x=shore,y=self.rsl_strat[idx],zorder=1000,marker='o',facecolor='w',edgecolor='tab:red',label='shoreline')
                # add T-R surfaces
                if surfaces:
                    ax.plot(self.x,self.beds_eroded[self.tran_idx,:].T,color='.2',alpha=1,ls=':',lw=1,label='transgressive surface')
                    ax.plot(self.x,self.beds_eroded[self.reg_idx,:].T,color='.2',alpha=1,ls='--',lw=1,label='regressive surface')

                axi = ax.inset_axes([0.1,0.1,0.3,0.3])
                axi.plot(self.ts[:fin],self.rsl_strat[:fin])
                axi.plot(self.ts[idx],self.rsl_strat[idx],'r.')
                axii = axi.twinx()
                axii.plot(self.ts[:fin],self.rsl_strat,color='tab:red')
                axii.plot(self.ts[idx],self.rsl_strat[idx],'k.')
                axi.set_yticks([])
                axii.set_yticks([])
                axii.set_ylabel(r'shore $\rightarrow$ deep')

                div = self.params['dt']
                ax.set_title(f't = {idx}')#; w$_{{secular}}$ = {sec_w_fun(idx*div):.0f}')

                if mp4:
                    fig.savefig(f'{frames_dir}frame{str(idx).zfill(6)}.jpg',dpi=300)
                    print(f'\rprocessing frame {idx} / {end}',end='')
                    plt.close('all')
                else:
                    time.sleep(wait_time)
                    plt.show()
        if mp4:
            command = (
            f'ffmpeg -y -framerate {fps} -pattern_type glob '
            f'-i "{frames_dir}/*.jpg" -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" ' # the space at the end is important
            f'-c:v libx264 -r 15 -y -vb 20M {ani_dir+model_name}.{output_format} -loglevel quiet ')
            #!{command} # works in jupyter
            os.system(command) # works from py file
            # clean up old jpgs for space saving
            [os.remove(frames_dir+file) for file in os.listdir(frames_dir) if file.endswith('.jpg')]
            print('\nprocessing completed')

    def strat_col_im(self,loc,lw=0.5,seq_color='r',seq_lw=1.25,edgecolor='k',targ=None,htarg=None,ftarg=None,scale=True,plot=True,ax=None,x_axis=True,out=True,seq_bounds=False,total=None,seq_ax=None):
        """
        returns: targ_val: basically just a column of proxies (w,depth, whatever from im_w, im_d), and it's scaled y values
        returns per column topographic y min and max from im_h

        # to calculate the scale factor and shift for y-coordinate normalization to match topography we use functions in utils.py
        # normalize_y() and normalize_topo()

        """
        if total is None:
            total = self.total_n

        if targ is None:
            targ = self.im_w

        if htarg is None:
            htarg = self.im_h

        if ftarg is None:
            ftarg = self.facies_pred

        #if plot:
        ax0 = ax if ax else plt.gca()

        # as units for grids are pixel coordinates, extract the actual topographic range from this location
        y_min,y_max = np.nanmin(htarg[:,loc]),np.nanmax(htarg[:,loc])

        # vertical section through loc in the predicted facies map, remove nans so only stratigraphy
        column = ftarg[:, loc][~np.isnan(ftarg[:, loc])]

        # grab isotopes from same column if targ
        if np.any(targ):
            targ_val = targ[:,loc][~np.isnan(targ[:,loc])]
            #new_y_val = [normalize_y(y, targ_val.shape[0], y_min, y_max) for y in range(targ_val.shape[0])] # pixel to topo
            #new_y_val = [normalize_y(y, targ_val.shape[0], 0, abs(y_max-y_min)) for y in range(targ_val.shape[0])] # topo to height from zero
        new_y_val = [normalize_y(y, column.shape[0], 0, abs(y_max-y_min)) for y in range(column.shape[0])] # topo to height from zero
        new_y_val = np.array(new_y_val) # convert to array

        # add made-up diff change to end so last block is plotted (after we grab shapes so lines up with proxy)
        column = np.append(column, -1)
        if plot:
            # find changes in facies (they are all numeric values 1,2,3, etc..)
            diff = np.diff(column, prepend=column[0])
            where_diff = np.where(diff!=0)[0]

            start = 0
            for y in where_diff:
                f_idx = column[start]
                facies = self.facies_data[int(f_idx)] # facies from dict

                # normalize from pixels to topography
                start_norm = normalize_y(start,len(column),y_min,y_max)
                end_norm = normalize_y(y,len(column),y_min,y_max)

                # second normalization step to the already normalized heights scaled between 0 and net change
                start_topo_norm = normalize_topo(start_norm, y_min, y_max, 0, y_max-y_min)
                end_topo_norm = normalize_topo(end_norm,  y_min, y_max, 0, y_max-y_min)

                # find heights and plot topography
                height = end_topo_norm - start_topo_norm
                rect = patches.Rectangle((0, start_topo_norm), facies['width'], height, facecolor=facies['colour'], edgecolor=edgecolor,lw=lw)
                ax0.add_patch(rect)
                start = y

            if scale:
                scaled_y_max = max(new_y_val)
                ax0.set_xlim(0,max([a['width'] for a in self.facies_data]))  # 1.05) # # Set x-axis limits to accommodate the rectangles
                ax0.set_ylim(0, scaled_y_max)  # Set y-axis limits to the normalized range
                #ticks = np.arange(0,scaled_y_max,10)
                #ax0.set_yticks([i for i in ticks])
                ax0.spines[['right','top']].set_visible(False)
                #ax0.set_ylabel('stratigraphic height (m)')

        # sequence boundaries
        if seq_bounds:
            tr,re = self.tr_bounds(loc)
            seq_y = np.concatenate([tr,re])
        else:
            seq_y = 0.

        seq_ax = ax0 if not seq_ax else seq_ax

        # if seq_bounds:
        #     for i in seq_y:
        #         seq_ax.axhline(i,color=seq_color,lw=seq_lw)

        if plot:
            if x_axis:
                ax0.set_xlabel('grain size')
            else:
                ax0.spines[['bottom']].set_visible(False)
                ax0.set_xticks([])
                ax0.set_xticklabels([])

        if out:
            #if targ:
                #return targ_val,new_y_val,y_min,y_max,seq_y
            #else:
            return targ_val,new_y_val,y_min,y_max,seq_y


    def facies_legend(self,ax=None,scale=1,x_axis=True,sort=False,aspect=0.5,offset=0.2,label_right=False,lab_size=12,label='facies',inf_out=True,**kwargs):
        """
        scale:  # for scaling bar widths

        """
        ax0 = ax if ax else plt.gca()

        # sort dictionary if not in order (redundant here)
        # made optional so can keep order of things with same width if needed
        if sort:
            sort_fdata = sorted(self.facies_data, key=lambda x: x['width'])
        else:
            sort_fdata = self.facies_data[::-1] # reversed so coarse at top
        widths = [scale*a['width'] for a in sort_fdata]
        colors = [a['colour'] for a in sort_fdata]
        flab = [a['name'] for a in sort_fdata]
        dlab = [f"{abs(a['min']):.0f} - {abs(a['max']):.0f} m" for a in sort_fdata]
        clab = [f"{a['name']} ({abs(a['min']):.0f} - {abs(a['max']):.0f} m)" for a in sort_fdata]
        if 'terrestrial' in flab:
            dlab[flab.index('terrestrial')] = ' > 0 m'
            clab[flab.index('terrestrial')] = 'Terrestrial ( > 0 m)'
        elif 'Terrestrial' in flab:
            dlab[flab.index('Terrestrial')] = ' > 0 m'
            clab[flab.index('Terrestrial')] = 'Terrestrial ( > 0 m)'

        if label=='facies':
            labels = flab
        elif label=='depth':
            labels = dlab
        elif label=='both':
            labels = clab

        # plot bars
        bars = ax0.barh(list(range(len(self.facies_data))),widths,height=[1]*len(self.facies_data),left=0,color=colors,align='edge',lw=0.5,edgecolor='k',**kwargs)
        ax0.set_xlim(0,max(widths)+0.05)
        if x_axis:
            ax0.set_xticks(np.arange(0,max(widths)+0.5,0.5))
        else:
            ax0.spines[['bottom']].set_visible(False)
            ax0.set_xticks([])
            ax0.set_xticklabels([])
        ax0.set_yticks([0.5 + n for n in range(len(self.facies_data))])
        ax0.set_yticklabels(labels)
        if aspect:
            ax0.set_aspect(aspect)
        ax0.spines[['top','right']].set_visible(False)
        ax0.set_ylim(-0.05,len(self.facies_data))
        if label_right:
            ax0.axis('off')
            # Add text labels with offset
            for bar,label in zip(bars,labels):
                width = bar.get_width()
                ax0.text(width + offset,    # x position (width of the bar + offset)
                        bar.get_y() + bar.get_height()/2,   # y position (center of the bar)
                        label,       # the text to display
                        va='center',      # vertical alignment
                        ha='left',fontsize=lab_size)        # horizontal alignment
        if inf_out:
            return flab,dlab


    def map_facies(self,plot=False,ter_thresh=0,ax=None):
        """
        depth array with positive values downwards (im_d)
        - currently if some facies is undefined by the dict, it reverts to land (i guess because prob is zero?)
        - ter_thresh = the cutoff for what is classified as terrestrial. can make this different to clean up facies
        # example data -> all made up
        # facies_data = [{'name':'Terrestrial','type': 'land'}, # terrestrial is always everyting <0 (above water)
        #                {'name':'Shelly Grainstone','type': 'gaussian', 'mean': 5, 'std': 5},
        #                {'name':'Oolitic Grainstone','type': 'gaussian', 'mean': 10, 'std': 10},
        #                {'name':'Packstone','type': 'gaussian', 'mean': 20, 'std': 10},
        #                {'name':'Wackestone','type': 'gaussian', 'mean': 30, 'std': 10},
        #                {'name':'Mudstone','type': 'gaussian', 'mean': 100, 'std': 10}]
        """

        # mask nans
        depth_array = self.im_d
        nan_mask = np.isnan(depth_array)

        # probabilities array, now including land
        probabilities = np.zeros((depth_array.shape[0],depth_array.shape[1],len(self.facies_data)))

        # set distribution for each facies definition, ensuring nans in depth data are accounted for
        for i, facies in enumerate(self.facies_data):
            prob = np.full(depth_array.shape, np.nan)  # array of nans

            if facies['type'] == 'land':
                prob[~nan_mask] = np.where(depth_array[~nan_mask] < ter_thresh, 1, 0) # depths < 0 are considered terrestrial
            else:
                if facies['type'] == 'gaussian':
                    dist = norm(facies['mean'], facies['std']) # gaussian distribution (scipy.stats.norm.pdf)
                elif facies['type'] == 'uniform':
                    dist = uniform(facies['min'],facies['max']-facies['min']) # uniform distribution (scipy.stats.uniform)
                elif facies['type'] == 'data':
                    dist = gaussian_kde(facies['data']) # pdf from data

                prob[~nan_mask] = dist.pdf(depth_array[~nan_mask])  # apply pdf only to non-nans
                prob[(depth_array < ter_thresh) & (~nan_mask)] = 0  # zero probability for depths < 0 (subaerial)

            # set probabilities
            probabilities[:, :, i] = prob

        # normalize probabilities, ensuring to exclude nan masked areas
        probabilities_sum = np.nansum(probabilities,axis=2,keepdims=True)
        with np.errstate(divide='ignore',invalid='ignore'):
            normalized_probabilities = np.where(probabilities_sum!=0,probabilities/probabilities_sum,np.nan)

        # np.argmax to find the indices of the maximum probabilities along the facies axis
        most_probable_facies = np.argmax(normalized_probabilities,axis=2)

        # convert most_probable_facies to float to allow setting nan values again
        most_probable_facies = most_probable_facies.astype(float)
        most_probable_facies[nan_mask] = np.nan

        if plot:
            s = ax.imshow(most_probable_facies,aspect='auto',origin='lower',cmap=self.cmap,interpolation='nearest',extent=self.extent)
            #tick_locs = (np.arange(len(facies_colours))+0.5)*(len(facies_colours)-1)/len(facies_colours) # set ticks to center of each colourbar box
            tick_locs = center_ticks(self.facies_colours) # set ticks to center of each colourbar box, going to last one because colour bar has length with land, need one less
            cbar = plt.colorbar(s,ax=ax,ticks=tick_locs)
            cbar.ax.set_yticklabels(self.labels)
            ax.plot(self.x,self.beds[0],color='.3')
            ax.plot(self.x,self.beds[-1],color='.3')

        return most_probable_facies


    def plot_facies_val(self,facies_val,val,plot_type='kde',label_loc='right',bin_width=0.05,skip_ter=True,xlim=None,
                        out=True,**kwargs):
        """
        default skips first dict item which is land/ terrestrial
        """
        if skip_ter:
            n = len(self.labels)-1
            idx = 1
        else:
            n = len(self.labels)
            idx = 0

        # find number of bins from provided bin width
        num_bins = int(np.ceil((np.nanmax(val)-np.nanmin(val))/bin_width))

        # calculate the bin edges
        bins = np.linspace(np.nanmin(val), np.nanmax(val), num_bins + 1)

        fig,ax = plt.subplots(n,1,sharex=True,sharey=False,**kwargs)
        ax[-1].set_xlabel('proxy (w)')
        for i,(f,c) in enumerate(zip(self.labels[idx:],self.facies_colours[idx:])):
            val_facies = facies_val[f]
            n = len(val_facies)

            if plot_type=='kde':
                sns.kdeplot(val_facies,fill=True,ax=ax[i],label=f,color=c,edgecolor='.3')

            elif plot_type=='hist':
                sns.histplot(val_facies,ax=ax[i],stat='density',label=f,color=c,edgecolor='.3',bins=bins)

            ax[i].set_yticks([])
            ax[i].spines[['right','left','top']].set_visible(False)
            ax[i].set_ylabel(None)
            if label_loc=='right':
                ax[i].text(1,0.8,f'{f}; n = {n}',transform=ax[i].transAxes,ha='right')
            elif label_loc=='left':
                ax[i].text(0,0.8,f'{f}; n = {n}',transform=ax[i].transAxes,ha='left')
            if xlim:
                ax[i].set_xlim(xlim)
        if out:
            return fig


    def get_facies_val(self,facies_map,data,as_df=False):
        """
        for extracting values binned by facies from another image array
        e.g.: w_ms = facies_w['Mudstone']
        # data is the image we are extracting from (i.e. im_w)
        """
        facies_values = {}

        for i, facies in enumerate(self.facies_data):

            # extract values from w array at these indices
            values = data[facies_map==i]

            # store the extracted values in the dictionary using facies names as keys
            facies_values[facies['name']] = values

        if as_df:
            facies_values = pd.DataFrame.from_dict(facies_values,orient='index').T
            facies_values = facies_values.melt(var_name='facies', value_name='w')

        return facies_values


    def show_facies_dists(self,x_end=None, where_x_end=None, out=True):
        """

        """
        depth_array = self.im_d
        # range of depth values for plotting
        depth_range = np.linspace(np.nanmin(depth_array),np.nanmax(depth_array),2000)
        num = len(self.facies_data)-1 # ignore land

        # where to apply x_end
        if not where_x_end:
            where_x_end = range(num)

        # plotting everything but land
        fig,ax = plt.subplots(1,num,figsize=(9,1.25),layout='constrained')

        for i,facies in enumerate(self.facies_data[1:]): # first is always land
            if facies['type'] == 'gaussian':
                dist = norm(facies['mean'], facies['std'])  # gaussian distribution
                f = dist.pdf(depth_range)
                ax[i].plot(depth_range,f,label=facies['name'],color='.3',lw=1.25)
                ax[i].fill_between(depth_range,f,alpha=0.75,color=facies['colour'])
            elif facies['type'] == 'uniform':
                dist = uniform(loc=facies['min'], scale=facies['max']-facies['min']) # Uniform distribution
                f = dist.pdf(depth_range)
                ax[i].plot(depth_range, f, label=facies['name'],color='.3',lw=1.25)
                ax[i].fill_between(depth_range,f,alpha=0.75,color=facies['colour'])
            elif facies['type'] == 'data':
                # Plot distribution from real data using KDE
                kde = gaussian_kde(facies['data'])
                f = kde.pdf(depth_range)
                ax[i].plot(depth_range, f, label=facies['name'],color='.3',lw=1.25)
                ax[i].fill_between(depth_range,f,alpha=0.75,color=facies['colour'])

            # formatting
            ax[i].set_xlabel('Depth')
            ax[i].spines[['right','left','top']].set_visible(False)
            ax[i].set_yticks([])
            ax[i].set_xlabel('depth')
            # if x_end:
            #     for j in where_x_end:
            #         ticks = np.round(np.linspace(0,x_end,5)) # includes positives
            #         ax[j].set_xticks(ticks)
            #         ax[j].set_xlim(0,ticks[-1])
            ticks = np.round(np.linspace(0,np.nanmax(depth_array),5)) # includes positives
            ax[i].set_xticks(ticks)
            ax[i].set_xlim(0,ticks[-1])
            ax[i].set_ylim(bottom=0)
            ax[i].set_title(facies['name'],fontsize=10)
        if out:
            return fig

    def explore_strat(self,xlim=False,secular=False,surfaces=False,heights=False,im_targ=None,cmap='coolwarm',pad=0,seq_bounds=False,interpolation='none',surf_color='k',figsize=(14, 4)):
        """
        pad: allows to erase epsilon offset to better visualize sequence boundary changes
        """

        cmap = self.cmap if not cmap else cmap
        lims = (1,self.x.size-1,1)
        extent = self.extent
        if im_targ is None:
            im_targ = self.im_w

        @interact(loc=lims)
        def f(loc):
            obs = self.im_w[:,loc][~np.isnan(self.im_w[:,loc])];
            if len(obs)>0:
                _, ax = plt.subplots(1,5,figsize=figsize,sharey=False,width_ratios=[0.2,0.2,1,0.05,.2],layout='constrained')
                # strat_col
                targ_val,new_y_val,y_min,y_max,seq_y =  self.strat_col_im(loc,targ=self.im_w,out=True,ax=ax[0],x_axis=True,seq_bounds=seq_bounds)
                # isotopes
                ax[1].tick_params(axis='y', labelleft=False)
                ax[1].sharey(ax[0])
                ax[1].plot(targ_val,new_y_val,marker='.',ls='-',color='.3',mfc='tab:red',mec='none',ms=0) # ms=6,lw=0.5
                #ax[1].axvline(0,ls='--',color='.3',lw=0.75)
                ax[1].set_xlabel('w')
                if secular:
                    dic = self.secular_dic(loc,self.im_t)
                    ax[1].plot(dic+pad,new_y_val,ls='-',color='dodgerblue',zorder=0) # lw=0.5

                if surfaces:
                    _,diff = self.grab_shore(targ='beds')
                    ax[2].plot(self.x,self.beds_eroded[self.tran_idx,:].T,color=surf_color,alpha=1,ls='--',lw=1.25,label='transgressive surface');
                    ax[2].plot(self.x,self.beds_eroded[self.reg_idx,:].T,color=surf_color,alpha=1,ls='-',lw=1.25,label='reg dressive surface');
                    tr,re = self.tr_bounds(loc) # grab scaled y-value locations for TR cycle changes
                    # ax[1].axhline(tr,color='tab:green',ls='-',lw=1,label='TS')
                    # ax[1].axhline(re,color='k',ls='-',lw=1,label='RS')
                    # ax[1].legend(frameon=False,handlelength=0.75)

                #ax[1].set_xlim(np.nanmin(sto[1::2])-2,np.nanmax(sto[1::2])+2)
                if xlim:
                    ax[1].set_xlim(xlim)
                else:
                    ax[1].set_xlim(np.nanmin(self.proxy)-2,np.nanmax(self.proxy)+2)
                ax[2].imshow(im_targ,aspect='auto',origin='lower',cmap=cmap,interpolation=interpolation,extent=extent)
                cola = loc*self.params['xmax']/self.params['Nx'] # convert Nx location fo xgrid
                ax[2].axvline(cola,ls='--',zorder=1000)
                ax[2].plot(self.x,self.beds[0],color='.3')
                ax[2].plot(self.x,self.beds[-1],color='.3')

                if heights:
                    ax[2].axhline(y_min,ls='--',zorder=1000)
                    ax[2].axhline(y_max,ls='--',zorder=1000)
                    ax[2].text(182,y_max,f'{y_max:.2f}',va='center')
                    ax[2].text(182,y_min,f'{y_min:.2f}',va='center')
                    ax[2].text(0,40,f'diff = {y_max-y_min:.2f}')
                ax[2].text(cola,ax[2].get_ylim()[1]+5,f'{cola}',ha='center')
                self.facies_legend(ax=ax[4])
                ax[3].axis('off')
                widths = [a['width'] for a in self.facies_data]
                for width in widths:
                    for i in [0,4]:
                        ax[i].axvline(width,color='grey',alpha=0.5,ls='--',zorder=0,lw=0.75)
                axi = ax[2].inset_axes([0.125,0.2,0.3,0.2])
                axi.hist(self.ds[self.ds>=0].ravel(),color='grey',bins=25)
                max_d = np.nanmax(self.ds[self.ds>=0])
                min_d = np.nanmin(self.ds[self.ds>=0])
                axi.spines[['left','right','top']].set_visible(False)
                axi.set_yticks([])
                axi.set_xlabel('water depths (m)')
                axi.set_xlim(min_d,max_d)
                axi.set_xticks(np.arange(min_d,max_d,40))
            else:
                print('All NaNs')
                plt.clf()
            plt.show() # for interactive seem to need this now


    def Nx_toX(self, locs, out='grid'):
        """

        """
        b = self.params['xmax'] / self.params['Nx']
        b0 = b if out == 'grid' else 1 / b

        # ensure locs is iterable if it's a list or numpy array
        if isinstance(locs, (list, np.ndarray)):
            cols = [int(loc * b0) for loc in locs]  # iterate over elements
        else:
            cols = int(locs * b0)  # single value case

        return cols if isinstance(locs, (list, np.ndarray)) else cols  # return same type

    def shore_x(self, t_idx, tol=0.25, targ='beds_eroded'):
        """
        simple shoreline calculation to find where rsl~=topo within a tolerence
        """
        if targ == 'beds_eroded':
            array = self.beds_eroded_raw[t_idx]
        elif targ == 'beds':
            array = self.beds_raw[t_idx]
        target_value = self.rsl_strat[t_idx]
        # Exclude NaNs from the array before comparison
        valid = ~np.isnan(array)
        # Check if all values are invalid (all NaN)
        if not np.any(valid):
            print(f'shore_x: all values are NaN at timestep {t_idx}')
            return None
        crossings = np.where(np.abs(array[valid] - target_value) <= tol)[0]

        if len(crossings) > 0:
            valid_indices = np.where(valid)[0]
            last_idx = valid_indices[crossings[-1]]
            return np.floor(self.x[last_idx])
        else:
            return None


    def grab_shore(self, end=None, tol=0.95, plot=False, targ='beds_eroded'):
        """

        """
        sl = []
        if not end:
            end = self.ts.size
        printed_warning = False
        for i in range(end):
            idx = self.shore_x(i, tol, targ)
            if idx is not None:
                sl.append(idx)
            else:
                self.failed_shore_idxs.append(i)
                sl.append(np.nan)
                if not printed_warning:
                    print('Warning: some shoreline points could not be computed (e.g., NaNs or no crossings)')
                    printed_warning = True
        sl = np.array(sl)
        # Enhanced check for None, type, and dtype validity
        if sl is None or not isinstance(sl, np.ndarray) or not np.issubdtype(sl.dtype, np.number):
            print('Warning: sea level array (sl) is None or not a valid numeric array')
            return None, None
        if np.any(np.isnan(sl)):
            print('Warning: sea level array (sl) contains NaNs in grab_shore')
            return None, None
        if plot:
            plt.plot(self.ts[:end], sl, label='shoreline')
            plt.title(sl.shape)
            plt.legend()
            axi = plt.gca().twinx()
            axi.plot(self.ts[:end], self.rsl_stra[:end], color='tab:red', label='sea level')
            axi.legend()

        try:
            diff = np.sign(np.gradient(sl))
        except Exception as e:
            print(f'Gradient computation failed: {e}')
            return sl, None

        return sl, diff

    def check_shoreline(self,out=None,img=False,targ='beds',tran_idx=None,reg_idx=None,surf=False,surf_color='.2',legend=True,figsize=(5,3.5),ylen=100,show='proxy',norm=None):
        """

        """
        sl = self.sl
        end=sl.size
        lims = (1,end-1)
        @interact(i=lims)
        def f(i):
            plt.figure(figsize=figsize)
            if targ=='beds_eroded':
                array = self.beds_eroded_raw[i]
            elif targ=='beds':
                array = self.beds_raw[i]
            shore = sl[i]
            if img:
                tim,extent = out.im_plot(show=show,ylen=ylen,total=i)
                sm = imshow(tim,extent=extent,cmap='coolwarm',norm=norm)
                plt.colorbar(sm)
            plt.gca().hlines(y=self.rsl_strat[i],xmin=shore,xmax=self.x[-1],label='rsl')
            plt.plot(self.x,array,color='grey')
            plt.scatter(x=shore,y=self.rsl_strat[i],zorder=1000,marker='o',facecolor='w',edgecolor='tab:red',label='shoreline')
            if legend:
                plt.legend()
            if surf:
                plt.plot(self.x,self.beds_eroded_raw[tran_idx,:].T,color=surf_color,alpha=1,ls=':',lw=1,label='transgressive surface');
                plt.plot(self.x,self.beds_eroded_raw[reg_idx,:].T,color=surf_color,alpha=1,ls='--',lw=1,label='regressive surface');

            axi = plt.gca().inset_axes([0.1,0.1,0.3,0.3])
            axi.plot(self.ts[:end],self.rsl_strat[:end])
            axi.plot(self.ts[i],self.rsl_strat[i],'r.')
            axii = axi.twinx()
            axii.plot(self.ts[:end],sl,color='tab:red')
            axii.plot(self.ts[i],sl[i],'k.')
            axi.set_yticks([])
            axii.set_yticks([])
            axii.set_ylabel(r'shore $\rightarrow$ deep')
            plt.show() # for interactive seem to need this now



    def tr_bounds(self,loc):
        """
        grab normalized yvalue spots for transgressive-regressive cycle boundaries in stratigraphic context
        """
        # transgressive-regressive cycles
        base = self.beds_eroded[0,loc].T
        top = max(self.beds_eroded[:,loc].T)

        th = self.beds_eroded[self.tran_idx,:].T
        rh = self.beds_eroded[self.reg_idx,:].T

        tr = th[loc,:] - base
        re = rh[loc,:] - base

        return tr,re


    def tr_bars(self,loc,xval=None,yval=None,line=False,ax=None,colors=['tab:blue', 'tab:red'],lines=['-','--'],
                legend=True,base=0,alpha=0.5,edgecolor='.2',xticks=False,zorder=10,lw=1,mask_out=False,
                n_interp=10000,yscale=1,yshift=0,clip=True):
        """
        color order is opposite to tr,re as up until each boundary it is the opposite sequence
        hence, the order is regressive color, transgressive color
        """
        ax0 = plt.gca() if not ax else ax
        tr,re = self.tr_bounds(loc)
        if line:
            # apply scale as needed to y values (default is unscaled)
            yval = yval*yscale+yshift
            tr = tr*yscale+yshift
            re = re*yscale+yshift

        # min_val = min(min(tr), min(re))
        # max_val = max(max(tr), max(re))
        combined = np.sort(np.concatenate((tr, re)))

        if line:
            # ensure xval and yval are the same length
            if len(xval) != len(yval):
                min_len = min(len(xval), len(yval))
                xval = xval[:min_len]
                yval = yval[:min_len]
            # interpolate for denser masking (less gaps)
            interp_func = scipy.interpolate.interp1d(yval, xval)
            y_dense = np.linspace(min(yval), max(yval), n_interp)
            x_dense = interp_func(y_dense)
        if not line:
            for t in tr:
                ax0.axhline(t,color=edgecolor,ls='-',lw=1,zorder=zorder)
            for r in re:
                ax0.axhline(r,color=edgecolor,ls='-',lw=1,zorder=zorder)
        if not xticks:
            ax0.set_xticks([])

        # Initialize the previous boundary
        prev_boundary = base

        # if len(tr)==0:
        #     ax0.axhspan(facecolor=colors[1], alpha=alpha, ymin=ax0.get_ylim()[0], ymax=ax0.get_ylim()[-1])

        # elif len(re)==0:
        #     ax0.axhspan(facecolor=colors[0], alpha=alpha, ymin=ax0.get_ylim()[0], ymax=ax0.get_ylim()[-1])

        # else:
        # Plot the hspans
        for i,val in enumerate(combined):

            # Determine the color based on the index
            color = colors[i%2]
            linestyle = lines[i%2]

            if line:
                mask = (y_dense >= prev_boundary) & (y_dense <= val)
                ax0.plot(x_dense[mask], y_dense[mask], color=color,zorder=zorder,lw=lw,ls=linestyle,alpha=alpha,clip_on=clip)

            else:
                # Plot the hspan
                ax0.axhspan(prev_boundary, val, facecolor=color, alpha=alpha,edgecolor=edgecolor,zorder=zorder)

            # Update the previous boundary
            prev_boundary = val

        if line:
            mask = (y_dense >= prev_boundary)
            ax0.plot(x_dense[mask], y_dense[mask], color=colors[len(combined) % 2],zorder=zorder,lw=lw,ls=lines[len(combined) % 2],alpha=alpha)
        else:
            # Plot the final span from the last max to the ylimit
            ax0.axhspan(prev_boundary, ax0.get_ylim()[-1], facecolor=colors[len(combined)%2], alpha=alpha,edgecolor=edgecolor,zorder=zorder)

        if legend:
            el = [patches.Patch(facecolor=colors[0], edgecolor='none',label='Regressive'),
                  patches.Patch(facecolor=colors[-1], edgecolor='none',label='Transgressive')]
            ax0.legend(handles=el,loc=3,bbox_to_anchor=(0,1.025),ncols=len(el),frameon=True,framealpha=1,fontsize=9,handletextpad=0.4,markerfirst=True,handlelength=0.75)
        if mask_out:
            return mask

    def image_diff(self,proxy_obs=None,im_l_lim=10,im_r_lim=-10):
        """

        """
        # calculate layer-weighted average w per x gridpoint and compare to final imasges
        proxy_obs = self.im_w if proxy_obs is None else proxy_obs

        # use eroded versions, as the images won't contain all the un-eroded beds
        proxy_exp = self.proxy_eroded
        topo_exp = self.beds_eroded

        # calculate the topographic mass (layer thickness) for weighting; the change in height in between timesteps (0 to 2000) in the y direction
        mass_diff = np.abs(np.diff(topo_exp, axis=0, prepend=np.nan))  # compute layer thickness
        weighted_proxy_exp = proxy_exp * mass_diff  # weight expected values by layer thickness

        # compute weighted mean for expected values
        weighted_sum_exp = np.nansum(weighted_proxy_exp, axis=0)  # sum weighted values
        mass_sum = np.nansum(mass_diff, axis=0)  # sum of masses for normalization
        weighted_mean_exp = weighted_sum_exp / (mass_sum + 1e-8)  # weighted mean for expected; added a small val to avoid div by 0

        # filter out columns with all NaN values in proxy_obs
        valid_columns = ~np.all(np.isnan(proxy_obs), axis=0)  # boolean mask for valid columns
        filtered_proxy_obs = proxy_obs[:, valid_columns]  # keep only valid columns
        filtered_weighted_mean_exp = weighted_mean_exp[valid_columns]  # align expected with valid columns

        # compute mean for observed values using filtered columns
        mean_obs = np.nanmean(filtered_proxy_obs, axis=0)

        # calculate absolute and relative differences
        raw_diff = mean_obs - filtered_weighted_mean_exp
        abs_diff = np.abs(mean_obs - filtered_weighted_mean_exp)
        rel_diff = (abs_diff / np.abs(mean_obs)) * 100  # relative difference in percentage
        rel_diff_noabs = (raw_diff / np.abs(mean_obs)) * 100  # relative difference in percentage

        return mean_obs[im_l_lim:im_r_lim], weighted_mean_exp[im_l_lim:im_r_lim], rel_diff[im_l_lim:im_r_lim], rel_diff_noabs[im_l_lim:im_r_lim], abs_diff[im_l_lim:im_r_lim], raw_diff[im_l_lim:im_r_lim]

    def mass_balance(self,figsize=(5.25,8.5),figsize_no=(5.15,6.25),show='proxy',rsl=True,grid=False,exp_m='--',obs_m='-',diff_m='-',percent=True,start=1,ylim=None,xlim=None,out=False,lloc='best',
                     plot_bias=True,im_l_lim=10,im_r_lim=-10):
        """
        plot mass balance
        - maybe should plot exp vs obs and the diff on lower panel, and then the mass per dt with the bias on upper?
        """
        if self.normalize_balance:
            yunits = 'w'
        else:
            yunits = 'h*w'

        if show=='proxy':
            obs = self.wobs[start:] # skip first index as start is 0
            exp = self.wexp[start:]
            ylaba = f'cumulative\nmass-normalized proxy ({yunits})'

        elif show=='mass':
            obs = self.mobs[start:] # skip first index as start is 0
            exp = self.mexp[start:]
            ylaba = 'cumulative mass (h)'
        else:
            print('show="proxy" or show="mass"')

        # limit bias to start time
        bias = self.biastot[start:]
        mass = self.massdt[start:]
        bias = np.abs(bias) # make absolute
        f = bias / mass # fraction bias relative to input mass per dt
        f_log = np.log10(f) # log bias

        ### Image comparison ###
        if self.images:
            mean_obs, weighted_mean_exp, rel_diff, rel_diff_noabs, abs_diff, raw_diff = self.image_diff(im_l_lim=im_l_lim,im_r_lim=im_r_lim)

        if not self.images:
            fig,ax = plt.subplot_mosaic(
                """
                bb
                aa
                aa
                """,figsize=figsize_no,width_ratios=(1,1),layout='constrained')
        else:
            fig,ax = plt.subplot_mosaic(
                """
                bb
                aa
                aa
                cc
                """,figsize=figsize,width_ratios=(1,1),layout='constrained')#,sharex=True)

        if percent:
            tol = 1e-6 if exp.all()==0 else 0
            diff = ((obs - exp) / np.abs(exp+tol)) * 100 # with absolute
            ylab = '% difference\n( obs - exp ) / exp'
        else:
            diff = (obs - exp) # positive difference is observed > expected and negative is observed < expected
            ylab = 'difference\n( obs - exp )'

        # plot expected vs observed - main panel
        ax['a'].plot(obs,obs_m,color='tab:orange',label='observed')
        ax['a'].plot(exp,exp_m,color='tab:blue',label='expected')
        ax['a'].legend(frameon=False,ncol=2,loc=lloc,handlelength=1,columnspacing=0.5,handletextpad=0.5,bbox_to_anchor=(0.6,1.1),fontsize=10.5)
        if grid:
            ax['a'].grid(ls=':')
        ax['a'].set_xlabel(r'time $\rightarrow$')
        ax['a'].set_ylabel(ylaba)
        if ylim:
            ax['a'].set_ylim(ylim[0],ylim[1])
        if xlim:
            ax['a'].set_xlim(xlim[0],xlim[1])

        # difference - plot along expected vs observed
        dcol = 'tab:green'
        axa = ax['a'].twinx()
        axa.axhline(0,ls='--',color=dcol,zorder=100,alpha=0.5)
        axa.plot(np.arange(len(diff)),diff,diff_m,label='diff',color='tab:green',alpha=0.5)
        axa.set_ylabel(ylab,color=dcol)
        axa.tick_params(axis='y', colors=dcol)
        axa.set_zorder(0)  # Lower the z-order of the twinx axis
        ax['a'].set_zorder(1)   # Raise the z-order of the main axis
        ax['a'].patch.set_visible(False)  # Hide the background of the twinx axis
        ax['a'].spines['right'].set_color(dcol)

        if rsl:
            axi = ax['b'].twinx()
            axi.plot(self.ts*self.params['dt'],self.rsl_strat,color='.6',alpha=0.6,label='sea level')
            axi.set_yticks([])
            axi.legend(frameon=False,handlelength=0.85,bbox_to_anchor=(0.35,1.25))
            axi.spines['left'].set_visible(False)
            axi.set_zorder(0)  # Lower the z-order of the twinx axis
            ax['b'].set_zorder(1)   # Raise the z-order of the main axis
            ax['b'].patch.set_visible(False)  # Hide the background of the twinx axis

        if plot_bias:
            bcol = 'tab:purple'
            # twinx for bias plotting
            if grid:
                ax['b'].grid(ls=':')
            ax['b'].plot(f_log,ls='-',label='bias', color=bcol,alpha=0.75)#,linewidth=0.75)
            ax['b'].set_ylabel('log bias / mass flux')#, color=bcol)

        if self.images:
            ## image comparison
            #ax['c'].set_title(f'gridded image comparison; expected\n(model arrays weighted by layer thickness)\nvs observed (images) [{im_l_lim}:{im_r_lim}]',fontsize=11)
            #ax['c'].set_title(f'expected\n(model arrays weighted by layer thickness)\nvs observed (images) [{im_l_lim}:{im_r_lim}]',fontsize=11)
            ax['c'].set_title(f'raster interpolation [{im_l_lim}:{im_r_lim}]',fontsize=11)
            ax['c'].plot(mean_obs, label='obs (mean)',color='tab:orange')
            ax['c'].plot(weighted_mean_exp, label='exp (weighted mean)', linestyle='--',color='tab:blue')
            ax['c'].set_xlabel('x index')
            ax['c'].set_ylabel('mean $w$ )')
            #ax['c'].legend(frameon=False)
            if grid:
                ax['c'].grid(ls=':')

            # differences
            axc = ax['c'].twinx()
            axc.plot(rel_diff,alpha=0.2,color='tab:green')
            axc.set_ylabel('% difference',color='tab:green')
            axc.set_xlabel('x index')
            axc.set_zorder(0)  # Lower the z-order of the twinx axis
            ax['c'].set_zorder(1)   # Raise the z-order of the main axis
            ax['c'].patch.set_visible(False)  # Hide the background of the twinx axis
            axc.tick_params(axis='y', colors='tab:green')
            ax['c'].spines['right'].set_color('tab:green')
            ax['c'].text(0.05,0.85,f'ylen = {self.im_ylen}',transform=ax['c'].transAxes,fontsize=12)

        if out:
            return fig,ax,diff


    def im_res_compare(self,ylens=[250,500,1000,5000],mean_diff_out=False,leg=True,figsize=(4,3.5),pal='RdBu_r',exp_col='tab:red',inset=False,lloc='best',ncols='single',im_l_lim=10,im_r_lim=-10):
        """

        """
        fig,ax = plt.subplots(2,1,figsize = figsize,layout='constrained',sharex=True,height_ratios=(0.5,1))

        # set colours
        #pal='RdBu_r'
        ax[0].set_prop_cycle(cycler('color',sns.color_palette(pal,n_colors=len(ylens))))
        ax[1].set_prop_cycle(cycler('color',sns.color_palette(pal,n_colors=len(ylens))))
        if inset:
            axi = ax[1].inset_axes([0.35,0.5,0.3,0.3])
            axi.set_prop_cycle(cycler('color',sns.color_palette(pal,n_colors=len(ylens))))

        mb = [] # observations
        md = [] # mean differences
        rd = [] # relative difference not absolute
        ims = [] # images

        # loop over ylens
        for idx, ylen in enumerate(ylens):
            im_w_tst = self.make_image(show='proxy',ylen=ylen)
            mobs, weight_mexp, rel_diff, rel_diff_noabs, abs_diff, raw_diff = self.image_diff(proxy_obs=im_w_tst,im_l_lim=im_l_lim,im_r_lim=im_r_lim)

            # plot observed
            ax[1].plot(mobs, alpha=0.85,label=ylen)
            ax[1].set_xlabel('xgrid location')
            ax[1].set_ylabel('mean w value')

            # plot relative differences
            ax[0].plot(rel_diff, alpha=0.85)
            #ax[0].hist(np.log10(rel_diff),alpha=0.85)
            ax[0].set_ylabel('% difference')

            mean_diff = np.mean(rel_diff)
            md.append(mean_diff)
            ims.append(im_w_tst)
            mb.append(mobs)
            rd.append(rel_diff_noabs)

            if inset:
                # plot decrease in mean difference
                axi.scatter(x=ylen,y=mean_diff,marker='o',edgecolor='w',clip_on=False,zorder=1000)

            # # calculate mean and standard deviation with a rolling window
            # window_size = 20  # adjust as needed
            # rolling_mean = np.convolve(mobs, np.ones(window_size)/window_size, mode='valid')
            # rolling_std = np.array([np.std(mobs[i:i+window_size]) for i in range(len(mobs)-window_size+1)])

            # # 2-sigma envelope
            # lower_bound = rolling_mean - 2 * rolling_std
            # upper_bound = rolling_mean + 2 * rolling_std
            # ax[1].fill_between(range(window_size-1, len(mobs)), lower_bound, upper_bound, alpha=0.85)

        # plot expected (should be same for all of them)
        ax[1].plot(weight_mexp,'--',color=exp_col,label='exp',lw=0.75)

        if ncols=='multi':
            l = len(ylens)
            f = l if l%2==0 else l+1
            ncols = f/2
        elif ncols=='single':
            ncols = 1

        if leg:
            ax[1].legend(frameon=False,fontsize=9,loc=lloc,handlelength=0.85,ncols=ncols)

        # clean up
        ax[0].grid(ls=':')
        ax[1].grid(ls=':')
        ax[0].set_title(f'current: {self.im_ylen} ; limits: [{im_l_lim}:{im_r_lim}]')
        if inset:
            axi.set_ylim(bottom=0)
            axi.set_xlim(0,max(ylens))
            axi.set_xticks([])
            axi.axhline(1,ls='--',color='w')
            axi.set_title('mean % diff vs ylen',fontsize=8)

        md = np.array(md)

        if mean_diff_out:
            return fig, md, mb, rd, ims

        else:
            return fig
