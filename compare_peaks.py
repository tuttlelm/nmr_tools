import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import matplotlib.lines as mlines
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
import re
import scipy.spatial
from scipy.optimize import linear_sum_assignment


from datetime import datetime
now = datetime.now()
date = now.strftime("%d%b%Y")

pd.set_option('display.max_columns',None) 
pd.set_option('display.max_colwidth', None)


mpl_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
mpl_colors_dark = ['#005794', '#df5f0e', '#0c800c', '#b60708', '#74479d', '#6c362b', '#b357a2', '#5f5f5f', '#9c9d02', '#079eaf']
mpl_colors_light = ['#2f97e4', '#ff9f2e', '#4cc04c', '#f64748', '#b487ed', '#ac766b', '#f397e2', '#9f9f9f', '#dcdd42', '#37deef']
mpl_colors_light2 = ['#4fb7f4', '#ffbf4e', '#6ce06c', '#f66768', '#d4a7fd', '#cc968b', '#f3b7f2', '#bfbfbf', '#fcfd62', '#57feff']


###### Functions
## Read in xpk and unify labels

def makelist(thing):
    '''
    utility function to make any variable into a list
    '''
    thing = [thing] if not isinstance(thing, list) else thing
    return thing

def read_xpk(file, header_lines=5,strip=True,sort=True):
    # nmrview .xpk files are nonstandard formatting with some columns in brackets
    # otherwise space deliminated 
    # bracketed columns: *.J *.U *.L comment
    
    raw=pd.DataFrame()
    with open(file) as fp:
        for i, line in enumerate(fp):
            #Assuming header_lines is correct, convert that line to the column values
            if i == header_lines:
                xpk_columns = ['pk']+line.rstrip().replace('\t',' ').split(' ')
                raw = pd.DataFrame(columns=xpk_columns)
            #otherwise, get the data after the column names
            #break lines up to capture any curly bracketed content as a single cell entry
            #this retains the curly brackets so output matches what an xpk text file looks like
            elif (i > header_lines):                
                breaks = [0]
                for j,v in enumerate(line):
                    if v == "{":
                        breaks += [j]
                    if v == "}":
                        breaks += [j+1]
                breaks += [len(line)]

                items = []
                for k in range(len(breaks)-1):
                    item=line[breaks[k]:breaks[k+1]]
                    if item[0] == "{":
                        items.append(item)
                    else: 
                        for j,v in enumerate(item.split()):
                            items.append(v)
                #print (items)             
                raw.loc[len(raw)]=items
    if strip:
        raw = raw.replace('{|}','',regex=True)
    if sort:
        label_columns = [l for l in raw.columns if ".L" in l]
        raw['res'] = raw[label_columns[0]].str.extract('(\d+)').astype('Int64') #get res from first .L
        raw = raw.sort_values(by = ['res'])

    return raw


def rename_HN(row):

    if '.' in row['HN.L']:
        resi, suffix = row['HN.L'].split('.')
    else:
        resi = row['HN.L']
        if len(resi)>0: suffix = 'unk'
        else: suffix = ''

    if any(echar in suffix for echar in ['E','e']):
        suffix = 'HE1'
    elif any(hchar in suffix for hchar in ['hn','H','HN','hn','h']):
        suffix = 'HN'
    else: suffix = suffix
    if len(resi)>0:
        new_label = resi+'.'+suffix
    else: new_label = ''

    return new_label

def rename_N(row):

    if '.' in row['15N.L']:
        resi, suffix = row['15N.L'].split('.')
    else:
        resi = row['15N.L']
        if len(resi)>0: suffix = 'unk'
        else: suffix = ''

    if any(hchar in suffix for hchar in ['N','N15','n']):
        suffix = 'N'
    else: suffix = suffix
    if len(resi)>0:
        new_label = resi+'.'+suffix
    else: new_label = ''

    return new_label


## Calculate CSD and CSP scores

def get_ddhnavg(data,scale_n = 5.0,samples=None,reference=None,uselabel='HN'):
    if samples is None:
        samples = list(data['sample'].unique())
    if reference is None:
        reference = samples[0]
    samples = [reference] + [x for x in samples if x is not reference]
    label = 'HN.L' if uselabel in ['HN','hn','h'] else '15N.L'
    ref_residues = list(data[label][data['sample']==reference].unique())
    if '' in ref_residues: ref_residues.remove('')
    #ref_residues.sort()
    csps = pd.DataFrame()
    csps['residue']=ref_residues
    csps['resid']=[x.split('.')[0] for x in csps['residue']]
    csps['resid']=csps['resid'].astype(int)
    for i,sample in enumerate(samples):
        df = data.copy()[data['sample']==sample]
        hdict = dict([(i,x) for i,x in zip(df[label], df['HN.P'])])
        ndict = dict([(i,x) for i,x in zip(df[label], df['15N.P'])])
        csps['HN_'+str(i)]=csps.get('HN_'+str(i),csps['residue'].map(hdict)).astype(float)
        csps['N_'+str(i)]=csps.get('N_'+str(i),csps['residue'].map(ndict)).astype(float)
    for i in range(1,len(samples)):
        csps['ddHN_'+str(i)] = csps['HN_'+str(i)]-csps['HN_0']
        csps['ddN_'+str(i)] = csps['N_'+str(i)]-csps['N_0']
        csps['ddHNavg_'+str(i)]=np.sqrt(csps['ddHN_'+str(i)]**2 + (csps['ddN_'+str(i)]/scale_n)**2)
    csps = csps.sort_values(by=['resid','residue'],ascending=[True,False],ignore_index=True)
    return samples, csps


def get_csd(XYcoord,samples,penalty=0):
    '''
    XYcoord is a dictionary of the H,NH peaks (can be scaled) with keys matching 'samples'
    penalty is the ppm value to assign to any missing peaks between lists, e.g. 0.5
    '''

    csd = {}   
    Ddists = {}
    avg_score = {}    
    num_missing = {}
    row_idx, col_idx = {}, {}

    avg_score_penalized = {}
    penalty_2d = []

    missing2d = []

    for ref in samples: 
        missing_row = []
        penalty_row = []
        for sample in samples:
            k = sample+'_vs_'+ref
            num_missing[k] = len(XYcoord[ref]) - len(XYcoord[sample])

            Ddists[k] = scipy.spatial.distance.cdist(XYcoord[ref],XYcoord[sample],metric='sqeuclidean')
            row_idx[k], col_idx[k] = linear_sum_assignment(Ddists[k])
            avg_score[k] = np.sqrt(Ddists[k][row_idx[k], col_idx[k]]).mean()
            
            if penalty != 0:
                min_peaks = min(len(XYcoord[ref]),len(XYcoord[sample]))
                max_peaks = max(len(XYcoord[ref]),len(XYcoord[sample]))
                avg_score_penalized[k] = avg_score[k]*min_peaks/max_peaks + penalty * (max_peaks - min_peaks)/max_peaks
                # sum is score * num_peaks ... avg * min_peaks + 0.5ppm * #missing/max_peaks
            else: 
                avg_score_penalized[k] = avg_score[k]           

            missing_row += [num_missing[k]]
            penalty_row += [avg_score_penalized[k]]

        missing2d += [missing_row]
        penalty_2d += [penalty_row]

    csd['samples'] = samples
    csd['missing2d'] = np.array(missing2d)
    csd['score2d'] = np.array(penalty_2d)
    csd['row_idx'] = row_idx
    csd['col_idx'] = col_idx
    csd['Ddists'] = Ddists
    
    return csd


def get_csp(apks,samples,penalty=0):
    #get csps from assignments 

    csp = {}
    all_csps={}
    samp={}
    sum_csp={}
    avg_csp={}
    avg_csp_penalty={}
    csp_penalty = 0.5 #ppm
    for ref in samples:
        all_csps[ref] = pd.DataFrame()
        samp[ref],all_csps[ref] = get_ddhnavg(apks,samples=samples,reference=ref)
        npks_ref = len(apks[(apks['sample']==ref) & (apks['HN.L'].str.contains('.H'))]['HN.L'].unique())
        #print(ref, samp)
        for n,s in enumerate(samp[ref]):
            npks_comp = len(apks[(apks['sample']==s) & (apks['HN.L'].str.contains('.H'))]['HN.L'].unique())
            if len(all_csps[ref]) == 0:
                sumcsp = np.nan
                avgcsp = np.nan
                avgcsp_penalty = np.nan        
            else:
                try:
                    sumcsp = all_csps[ref]['ddHNavg_'+str(n)].sum()
                    avgcsp = all_csps[ref]['ddHNavg_'+str(n)].mean() 
                    if penalty != 0: 
                        avgcsp_penalty = sumcsp + csp_penalty * abs(npks_ref - npks_comp)
                        avgcsp_penalty = avgcsp_penalty/max(npks_comp,npks_ref)
                    else: 
                        avgcsp_penalty = avgcsp
                    if len(all_csps[ref]['ddHNavg_'+str(n)]) == all_csps[ref]['ddHNavg_'+str(n)].isna().sum():
                        sumcsp = np.nan
                        avgcsp = np.nan
                        avgcsp_penalty = np.nan
                except: 
                    sumcsp=0
                    avgcsp=0
                    avgcsp_penalty=0
            sum_csp[s+'_vs_'+ref] = sumcsp
            avg_csp[s+'_vs_'+ref] = avgcsp
            avg_csp_penalty[s+'_vs_'+ref] = avgcsp_penalty

    csp_avg_penalty_2d =[]
    for ref in samples: #compare_xpks:
        csp_pen_row = []
        for s in samples:       
            k = s+'_vs_'+ref        
            csp_pen_row += [avg_csp_penalty[k]]
        csp_avg_penalty_2d += [csp_pen_row]
    csp_avg_penalty_2d = np.array(csp_avg_penalty_2d)


    csp['score2d'] = np.array(csp_avg_penalty_2d)
    csp['samp'] = samp
    csp['all_csps'] = all_csps

    return csp


def get_allxpks(xpk_paths,samplenames=None,Header_Lines=5):
    # this will join all xpk file data in xpk_paths into one dataframe 
    # uses file name for sample name
    # works even if mixed dimensions (e.g. HSQCs + HNCACB)
    if samplenames is not None: samplenames = makelist(samplenames)
    xpk_paths = makelist(xpk_paths)
    
    all_xpks = pd.DataFrame()
    for i,file in enumerate(xpk_paths):
        test_xpk = read_xpk(os.path.join(file),header_lines=Header_Lines)
        if samplenames is None:
            test_xpk['sample'] = os.path.basename(file)[:-4]
        else: 
            try: 
                test_xpk['sample'] = samplenames[i] 
            except: 
                print("Error with samplenames list, using peaklist name")
                test_xpk['sample'] = os.path.basename(file)[:-4]
        all_xpks = pd.concat([all_xpks,test_xpk], ignore_index=True,axis=0)

    #to remove curly brackets from entries 
    all_xpks = all_xpks.replace('{|}','',regex=True)
    all_xpks['HN.L'] = all_xpks.apply(rename_HN,axis=1) #unify label names for matching 
    if 'N.L' in all_xpks.columns:
        all_xpks = all_xpks.rename({'N.L':'15N.L','N.P':'15N.P','N.W':'15N.W','N.B':'15N.B','N.E':'15N.E','N.J':'15N.J','N.U':'15N.U'},axis='columns')
    all_xpks['15N.L'] = all_xpks.apply(rename_N,axis=1) #unify label names for matching 

    return all_xpks

def get_coords(all_xpks,ntohn_scale = 5.0):
    samples = all_xpks['sample'].unique()
    
    hn = {}
    n, n_scaled = {}, {}
    X, X_scaled = {}, {}
    all_xpks['15N.P_scaled'] = all_xpks['15N.P'].astype(float)/ntohn_scale
    for sample in samples:
        X[sample] = all_xpks[all_xpks['sample']==sample][['HN.P','15N.P']].astype(float).to_numpy(copy=True)
        X_scaled[sample] =  all_xpks[all_xpks['sample']==sample][['HN.P','15N.P_scaled']].astype(float).to_numpy(copy=True)
        hn[sample] = all_xpks[all_xpks['sample']==sample]['HN.P'].astype(float).to_numpy(copy=True)
        n[sample] = all_xpks[all_xpks['sample']==sample]['15N.P'].astype(float).to_numpy(copy=True)
        n_scaled[sample] = n[sample]/ntohn_scale

    if ntohn_scale != 1:
        return X, X_scaled
    else: return X

##Plot functions

def plot_score_missing(score2d,missing2d,samples,scoretype = 'CSD',title_text=None,savepath=None):
        
    corr1 = missing2d
    corr2 = score2d
    mask1 = np.tril(np.ones_like(corr1, dtype=bool))
    mask2 = np.triu(np.ones_like(corr2, dtype=bool))

    fig, ax = plt.subplots(figsize=(6, 6),)#ncols=2,gridspec_kw={'width_ratios':[20,1],},)
    cax = fig.add_axes([.93, 0.11, .05, 0.62])
    #im1 = ax.imshow(uptri_score,cmap="coolwarm")
    #im2 = ax.imshow(lowtri_missing,cmap="Greens")

    rwr = [(0.78, 0.20, 0.15), (1,1,1), (0.78, 0.20, 0.15)]  # Red, White, Red

    rwr_cmap = LinearSegmentedColormap.from_list('red_white_red', rwr)

    vmax = round(corr2.max(),2)+.01
    vmax1 = round(np.abs(corr1).max(),2)

    sns.heatmap(corr2, mask=mask2, cmap='coolwarm', annot=True, fmt='.2f',linewidths=0.5, linecolor='k',
                square=True,   ax=ax,cbar_ax=cax,cbar_kws={"pad":-10.0}, vmin=0,vmax=vmax)
    #sns.heatmap(corr1, mask=mask1, cmap='binary', annot=True, vmin=0,vmax=0, linewidths=0.5, linecolor='k',
    #            square=True,   ax=ax,cbar=None); #cbar_kws={'label': 'Difference in peaks picked',})
    sns.heatmap(corr1, mask=mask1, cmap=rwr_cmap, annot=True, vmin=-vmax1,vmax=vmax1, linewidths=0.5, linecolor='k',
                square=True,   ax=ax,cbar=None); #cbar_kws={'label': 'Difference in peaks picked',})


    # Rotate the tick labels and set their alignment.

    plt.setp(ax.get_yticklabels(), rotation=0, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    if scoretype == 'CSD':
        ylabel = "Average paired CSD, ppm"
    else: ylabel = "CSP score, ppm"
    cax.set_ylabel(ylabel,rotation=-90,labelpad=18) #CSD = Chemical Shift Distance
    cax.axvline(x=0, color='k',linewidth=3)
    cax.axvline(x=vmax-0.0, color='k',linewidth=3)
    cax.axhline(y=0,color='k',linewidth=3)
    cax.axhline(y=vmax,color='k',linewidth=3)

    ax.set_xticklabels(samples)
    ax.set_yticklabels(samples)

    ax.patch.set_facecolor('grey')
    ax.patch.set_edgecolor('black')
    ax.patch.set_hatch('xxx')

    ax.axhline(y=0, color='k',linewidth=3)
    ax.axhline(y=corr1.shape[1], color='k',linewidth=3)
    ax.axvline(x=0, color='k',linewidth=3)
    ax.axvline(x=corr1.shape[0], color='k',linewidth=3)

    if title_text is None: title_text = "Similarity of Spectra"
    ax.set_title(title_text)

    ax_legend_elements = []
    ax_legend_elements = [ mlines.Line2D([0],[0], color='w',markerfacecolor = 'w',markeredgecolor='k',
                                        marker='s',markersize=10, markeredgewidth=1.5,) ]
    fig.legend(handles=ax_legend_elements,labels=['Difference\nin #peaks\n (row-col)'],handletextpad=0.1,
            loc='upper left',bbox_to_anchor=(0.88,0.9),frameon=False,framealpha=1,edgecolor='k')

    #fig.tight_layout()

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    plt.show()

    if savepath is not None:
        print("saving plot to ",savepath)
        type = savepath[:-3].lower()
        if type == 'svg':
            fig.savefig(savepath,format='svg',bbox_inches='tight')
        else: 
            fig.savefig(savepath,format='pdf',dpi=600,bbox_inches='tight')


def plot_score_score(lowerscore,upperscore,samples,scoretypes=['CSD','CSP'],title_text=None,savepath=None):

    corr1 = upperscore
    corr2 = lowerscore
    mask1 = np.tril(np.ones_like(corr1, dtype=bool))
    mask2 = np.triu(np.ones_like(corr2, dtype=bool))

    fig, ax = plt.subplots(figsize=(6, 6),)#ncols=2,gridspec_kw={'width_ratios':[20,1],},)
    cax = fig.add_axes([1.05, 0.11, .05, 0.77])
    cax1 = fig.add_axes([1.05, 0.11, .05, 0.77])

    vmax = round(np.nanmax(corr2,),2)+.01
    vmax1 = round(np.nanmax(corr1),2)+.01
    #vmax = max(vmax,vmax1)

    sns.heatmap(corr2, mask=mask2, cmap='coolwarm', annot=True, fmt='.2f',linewidths=0.5, linecolor='k',
                square=True,   ax=ax,cbar_ax=cax,cbar_kws={"pad":-10.0}, vmin=0,vmax=vmax)
    sns.heatmap(corr1, mask=mask1, cmap='coolwarm', annot=True, fmt='.2f',linewidths=0.5, linecolor='k',
                square=True,   ax=ax,vmin=0,vmax=vmax1,cbar_ax=cax1,cbar_kws={"pad":-10.0},)  


    # Rotate the tick labels and set their alignment.

    plt.setp(ax.get_yticklabels(), rotation=0, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    #cax.set_ylabel("Assigned ←     Average CSP, ppm     → Paired",rotation=-90,labelpad=15)
    cax.yaxis.tick_left()
    cax.yaxis.set_label_position("left")
    ylabel={}
    ylabel['CSD'] = "Average paired CSD, ppm"
    ylabel['CSP'] = "Average assigned CSP, ppm"
    other = list(set(scoretypes).difference(['CSD','CSP']))
    for k in other:
        ylabel[k] = k

    cax.set_ylabel("   "+ylabel[scoretypes[0]],rotation=90,labelpad=5,loc='bottom')
    cax.axvline(x=0, color='k',linewidth=2)
    cax.axvline(x=vmax-0.00, color='k',linewidth=2)
    cax.axhline(y=0,color='k',linewidth=3)
    cax.axhline(y=vmax,color='k',linewidth=3)
    cax1.set_ylabel(ylabel[scoretypes[1]]+"                                ",rotation=-90,labelpad=15,loc='center')
    cax1.axvline(x=0, color='k',linewidth=3)
    cax1.axvline(x=vmax1-0.00, color='k',linewidth=3)
    cax1.axhline(y=0,color='k',linewidth=3)
    cax1.axhline(y=vmax1,color='k',linewidth=3)

    ax.set_xticklabels(samples)
    ax.set_yticklabels(samples)

    ax.patch.set_facecolor('grey')  
    ax.patch.set_edgecolor('black')
    ax.patch.set_hatch('xxx')

    ax.axhline(y=0, color='k',linewidth=3)
    ax.axhline(y=corr1.shape[1], color='k',linewidth=3)
    ax.axvline(x=0, color='k',linewidth=3)
    ax.axvline(x=corr1.shape[0], color='k',linewidth=3)

    if title_text is None: title_text = "Similarity of Spectra"
    ax.set_title(title_text)

    #find na values in original data and white them out, add 'nd' text
    #currently just applied to the assigned CSPs since paired shouldn't be missing
    for y,x in np.argwhere(np.isnan(np.ma.masked_where(mask1,corr1))):
        ax.add_patch(
            patches.Rectangle(
                (x, y),1.0,1.0,
                edgecolor='k',facecolor='white', lw=0.5 ) );
        ax.text(x+.5,y+.5,'nd',ha="center",va="center")    


    #plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    plt.show()

    if savepath is not None:
        print("saving plot to ",savepath)
        type = savepath[:-3].lower()
        if type == 'svg':
            fig.savefig(savepath,format='svg',bbox_inches='tight')
        else: 
            fig.savefig(savepath,format='pdf',dpi=600,bbox_inches='tight')


def plot_csp_csd_spectra(XYcoord,csp,csd,ref,samples,savedir=None,SVG=False):
    samples = makelist(samples)
    label_max = len(max(samples, key=len))
    ref_padding = ' '*(label_max-len(ref))
    for compare in [s for s in samples if s is not ref]:
        comp_index = csp['samp'][ref].index(compare)
        comp_padding = ' '*(label_max-len(compare))
        Xrefa = csp['all_csps'].copy()[ref]['HN_0'].to_numpy(copy=True)
        Yrefa = csp['all_csps'].copy()[ref]['N_0'].to_numpy(copy=True)
        Xcompa = csp['all_csps'].copy()[ref]['HN_'+str(comp_index)].to_numpy(copy=True)
        Ycompa = csp['all_csps'].copy()[ref]['N_'+str(comp_index)].to_numpy(copy=True)

        Xrefa[np.argwhere(np.isnan(Xcompa))] = np.nan #needs to be in both
        Xcompa[np.argwhere(np.isnan(Xrefa))] = np.nan #

        Xrefall = XYcoord[ref].T[0]
        Xref = XYcoord[ref][csd['row_idx'][compare+'_vs_'+ref]].T[0]
        Xcompall = XYcoord[compare].T[0]
        Xcomp = XYcoord[compare][csd['col_idx'][compare+'_vs_'+ref]].T[0]
        Yrefall = XYcoord[ref].T[1]
        Yref = XYcoord[ref][csd['row_idx'][compare+'_vs_'+ref]].T[1]
        Ycompall = XYcoord[compare].T[1]
        Ycomp = XYcoord[compare][csd['col_idx'][compare+'_vs_'+ref]].T[1]

        fig, ax = plt.subplots(figsize=(15, 5), ncols=2, nrows = 1, squeeze=True,num=1,clear=True)

        ax[0].plot(Xrefa,Yrefa,'ro',label=ref)
        ax[0].plot(Xrefall,Yrefall,'o',color='none',markerfacecolor=None,markeredgecolor='r')
        ax[0].plot(Xcompa,Ycompa,'bo',label=compare)
        ax[0].plot(Xcompall,Ycompall,'o',color='none',markerfacecolor=None,markeredgecolor='b')
        for i in range(min(len(Xrefa),len(Xcompa))):
            x_ref,y_ref = Xrefa[i],Yrefa[i]
            x_comp,y_comp = Xcompa[i],Ycompa[i]
            ax[0].plot([x_ref,x_comp],[y_ref,y_comp],'k-')
        ax[0].invert_xaxis()
        ax[0].invert_yaxis()
        ax[0].set_xlabel("$^1$H, ppm")
        ax[0].set_ylabel("$^{15}$N, ppm")
        ax[0].set_title(ref+" vs. "+compare+", Assigned")
        #ax[0].legend(loc='lower right');

        ax[1].plot(Xref,Yref,'ro',label=ref+ref_padding)
        #ax[1].plot(Xrefall,Yrefall,'ro',)#label=ref) #just for presentation plot

        ax[1].plot(Xrefall,Yrefall,'o',color='none',markerfacecolor=None,markeredgecolor='r',label=ref+" unpaired"+ref_padding)
        ax[1].plot(Xcomp,Ycomp,'bo',label=compare+comp_padding)
        #ax[1].plot(Xcompall,Ycompall,'bo',)#label=compare) #just for presentation plot
        ax[1].plot(Xcompall,Ycompall,'o',color='none',markerfacecolor=None,markeredgecolor='b',label=compare+" unpaired"+comp_padding)
        for i in range(min(len(Xref),len(Xcomp))):
            x_ref,y_ref = Xref[i],Yref[i]
            x_comp,y_comp = Xcomp[i],Ycomp[i]
            ax[1].plot([x_ref,x_comp],[y_ref,y_comp],'k-')
        ax[1].invert_xaxis()
        ax[1].invert_yaxis()
        ax[1].set_xlabel("$^1$H, ppm")
        ax[1].set_ylabel("$^{15}$N, ppm")
        ax[1].set_title(ref+" vs. "+compare+", Unassigned Paired")
        ax[1].legend(loc='upper left',bbox_to_anchor=(1,0.8),frameon=False);#loc='lower right',ncol=2);
        fig.tight_layout()
        plt.show()

        if savedir is not None:
            filename = 'csp_csd_spectra_'+ref+'_vs_'+compare+'_'+date
            print("saving plot to ",savedir,filename)
            if SVG:
                fig.savefig(os.path.join(savedir,filename+'.svg'),format='svg',bbox_inches='tight')
            else: 
                fig.savefig(os.path.join(savedir,filename+'.pdf'),format='pdf',dpi=600,bbox_inches='tight')


######## end Functions
