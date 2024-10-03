import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import re
pd.set_option('display.max_columns',None) 
pd.set_option('display.max_colwidth', None)
# from io import StringIO
# import csv

# nmrview .xpk files are nonstandard formatting with some columns in brackets
# otherwise space deliminated 
# bracketed columns: *.J *.U *.L comment

###### Functions
def read_xpk(file, header_lines=5,strip=True,sort=True):
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

def get_ddhnavg(data,scale_n = 5.0,samples=None,reference=None,uselabel='HN'):
    if samples == None:
        samples = list(data['sample'].unique())
    if reference == None:
        reference = samples[0]
    samples = [reference] + [x for x in samples if x is not reference]
    label = 'HN.L' if uselabel in ['HN','hn','h'] else '15N.L'
    ref_residues = list(data[label][data['sample']==reference].unique())
    if '' in ref_residues: ref_residues.remove('')
    #print("ref_residues",ref_residues)
    #ref_residues.sort()
    csps = pd.DataFrame()
    csps['residue']=ref_residues
    csps['resid']=[x.split('.')[0] for x in csps['residue']]
    csps['resid']=csps['resid'].astype(int)
    for i,sample in enumerate(samples):
        df = data.copy()[data['sample']==sample]
        hdict = dict([(i,x) for i,x in zip(df[label], df['HN.P'])])
        ndict = dict([(i,x) for i,x in zip(df[label], df['15N.P'])])
        csps['HN_'+str(i)]=csps.get('HN_'+str(i),csps['residue'].map(hdict)).astype(np.float16)
        csps['N_'+str(i)]=csps.get('N_'+str(i),csps['residue'].map(ndict)).astype(np.float16)
    for i in range(1,len(samples)):
        csps['ddHN_'+str(i)] = csps['HN_'+str(i)]-csps['HN_0']
        csps['ddN_'+str(i)] = csps['N_'+str(i)]-csps['N_0']
        csps['ddHNavg_'+str(i)]=np.sqrt(csps['ddHN_'+str(i)]**2 + (csps['ddN_'+str(i)]/scale_n)**2)
    csps = csps.sort_values(by=['resid','residue'],ascending=[True,False],ignore_index=True)
    return samples, csps


def rename_HN(row):

    if '.' in row['HN.L']:
        resi, suffix = row['HN.L'].split('.')
        try:
            if not resi[0].isdigit():
                resi = resi[1:]
        except: pass
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


######## end Functions


#Get the data
Header_Lines = 5
#project_dir = '/home/tuttle/data/nmrdata/gal4'
#project_dir = '/home/tuttle/data/nmrdata/FimH/peaklists'
project_dir = '/home/tuttle/data/nmrdata/Karen/Ubc6-180x/peaklists'

xpk_files = [ f for f in os.listdir(project_dir) if f[-4:]=='.xpk'  ]
samples = [ff[:-4] for ff in xpk_files] 

# this will join all xpk file data in the project_dir into one dataframe 
# works even if mixed dimensions (e.g. HSQCs + HNCACB)
all_xpks = pd.DataFrame()
for file in xpk_files:
    test_xpk = read_xpk(os.path.join(project_dir,file),header_lines=Header_Lines)
    test_xpk['sample']=file[:-4]
    all_xpks = pd.concat([all_xpks,test_xpk], ignore_index=True,axis=0)

#to remove curly brackets from entries 
all_xpks = all_xpks.replace('{|}','',regex=True)
all_xpks['HN.L'] = all_xpks.apply(rename_HN,axis=1) #unify label names for matching 

#scale the 15N shifts
samples = all_xpks['sample'].unique()
ntohn_scale = 5.0
hn = {}
n, n_scaled = {}, {}
X, X_scaled = {}, {}
all_xpks['15N.P_scaled'] = all_xpks['15N.P'].astype(float)/ntohn_scale
for sample in samples:
    X[sample] = all_xpks[all_xpks['sample']==sample][['HN.P','15N.P']].astype(float).to_numpy()
    X_scaled[sample] =  all_xpks[all_xpks['sample']==sample][['HN.P','15N.P_scaled']].astype(float).to_numpy()
    hn[sample] = all_xpks[all_xpks['sample']==sample]['HN.P'].astype(float).to_numpy()
    n[sample] = all_xpks[all_xpks['sample']==sample]['15N.P'].astype(float).to_numpy()
    n_scaled[sample] = n[sample]/ntohn_scale

# Plot spectra overlay
mpl_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
mpl_colors_dark = ['#005794', '#df5f0e', '#0c800c', '#b60708', '#74479d', '#6c362b', '#b357a2', '#5f5f5f', '#9c9d02', '#079eaf']
mpl_colors_light = ['#2f97e4', '#ff9f2e', '#4cc04c', '#f64748', '#b487ed', '#ac766b', '#f397e2', '#9f9f9f', '#dcdd42', '#37deef']
mpl_colors_light2 = ['#4fb7f4', '#ffbf4e', '#6ce06c', '#f66768', '#d4a7fd', '#cc968b', '#f3b7f2', '#bfbfbf', '#fcfd62', '#57feff']

# samples = ['WT','G16P','G31P','G65P','G116P','G117P','L34K', 'R60P', 'V27CL34C',
#             'WT_Mannose','G16P_Mannose','G31P_Mannose','G65P_Mannose', 'L34K_Mannose','R60P_Mannose','V27CL34C_Mannose']
fig = plt.figure(figsize=(9,6))
for i,sample in enumerate(samples):
    plt.scatter(hn[sample],n[sample],label=sample,marker='o',alpha=0.5,facecolors=mpl_colors[i%10],edgecolors=mpl_colors_dark[i%10])
plt.legend(bbox_to_anchor=(1,0.5),loc='center left',fancybox=True,shadow=True)
ax = plt.gca()
ax.invert_xaxis()
ax.invert_yaxis()
ax.set_xlabel("$^1$H, ppm")
ax.set_ylabel("$^{15}$N, ppm");
plt.tight_layout()
#plt.savefig(os.path.join(project_dir,'FimH_GtoP_HSQC_all_picked_peaks_26june2024.pdf'),format='pdf',dpi=600)

# Manually set the sample order
#all_xpks['sample'].unique()
samples = ['180X_WT', '180X_H38Q','180X_H55Q','180X_C87N',
           '180X_H94A','180X_H94D', '180X_H94E','180X_H94K', '180X_H94N','180X_H94Q', '180X_H94S',  '180X_H94Y', 
           '180X_G123A','180X_GS123-4AA', '180X_S124A']

# compute the unassigned average paired peak distance

#from collections import defaultdict # = defaultdict(dict)
import scipy.spatial
from scipy.optimize import linear_sum_assignment

#samples = all_xpks['sample'].unique()

#ref = 'WT'
# compare_xpks = list(samples)
# compare_xpks.remove(ref)

D,D_scaled = {}, {}
score = {}
avg_score = {}
score_scaled = {}
avg_score_scaled = {}
num_missing = {}
row_ind, col_ind = {}, {}
score2d = []
avg_score2d = []

missing2d = []

for ref in samples: #compare_xpks:
    score_row = []
    avg_score_row = []
    missing_row = []
    for sample in samples:
        k = sample+'_vs_'+ref
        num_missing[k] = len(X[ref]) - len(X[sample])

        D[k] = scipy.spatial.distance.cdist(X[ref],X[sample],metric='sqeuclidean')
        row_ind[k], col_ind[k] = linear_sum_assignment(D[k])
        score[k] = D[k][row_ind[k], col_ind[k]].sum()
        avg_score[k] = np.sqrt(D[k][row_ind[k], col_ind[k]]).mean()

        D_scaled[k] = scipy.spatial.distance.cdist(X_scaled[ref],X_scaled[sample],metric='sqeuclidean')
        row_ind[k], col_ind[k] = linear_sum_assignment(D_scaled[k])
        score_scaled[k] = D_scaled[k][row_ind[k], col_ind[k]].sum()
        avg_score_scaled[k] = np.sqrt(D_scaled[k][row_ind[k], col_ind[k]]).mean()

        score_row += [score_scaled[k]]
        avg_score_row += [avg_score_scaled[k]]
        missing_row += [num_missing[k]]
    score2d += [score_row]
    avg_score2d += [avg_score_row]
    missing2d += [missing_row]
    
score2d = np.array(score2d)
avg_score2d = np.array(avg_score2d)
missing2d = np.array(missing2d)

# compute the assigned average chemical shift perturbation
#samples = ['WT','L34A','L34K', 'R60P', 'V27CL34C', 'WT_Mannose', 'L34A_Mannose', 'L34K_Mannose','R60P_Mannose','V27CL34C_Mannose']
all_csps={}
samp={}
sum_csp={}
avg_csp={}
for ref in samples:
    all_csps[ref] = pd.DataFrame()
    samp[ref],all_csps[ref] = get_ddhnavg(all_xpks,samples=samples,reference=ref)
    #print(ref)
    for n,s in enumerate(samp[ref]):
        if len(all_csps[ref]) == 0:
            sumcsp = np.nan
            avgcsp = np.nan        
        else: 
            try: 
                sumcsp = all_csps[ref]['ddHNavg_'+str(n)].sum()
                avgcsp = all_csps[ref]['ddHNavg_'+str(n)].mean()           
                if len(all_csps[ref]['ddHNavg_'+str(n)]) == all_csps[ref]['ddHNavg_'+str(n)].isna().sum():
                    sumcsp = np.nan
                    avgcsp = np.nan
            except: 
                sumcsp=0
                avgcsp=0
        sum_csp[s+'_vs_'+ref] = sumcsp
        avg_csp[s+'_vs_'+ref] = avgcsp

csp_2d = []
csp_avg_2d = []
for ref in samples: #compare_xpks:
    csp_row = []
    csp_avg_row = []
    for s in samples:       
        k = s+'_vs_'+ref        
        csp_row += [sum_csp[k]]
        csp_avg_row += [avg_csp[k]]
        #print(k)
    #print(csp_row)
    csp_2d += [csp_row]  
    csp_avg_2d += [csp_avg_row]
csp_2d = np.array(csp_2d)
csp_avg_2d = np.array(csp_avg_2d)


#plot the heatmap of the paired CSP and difference in number of picked peaks 

import seaborn as sns
import matplotlib.lines as mlines

#sns.set_theme(style="white")

corr1 = missing2d
corr2 = avg_score2d
mask1 = np.tril(np.ones_like(corr1, dtype=bool))
mask2 = np.triu(np.ones_like(corr2, dtype=bool))

fig, ax = plt.subplots(figsize=(7.5, 7.5),)#ncols=2,gridspec_kw={'width_ratios':[20,1],},)
cax = fig.add_axes([.93, 0.11, .05, 0.62])
#im1 = ax.imshow(uptri_score,cmap="coolwarm")
#im2 = ax.imshow(lowtri_missing,cmap="Greens")

vmax = round(corr2.max(),2)+.01

sns.heatmap(corr2, mask=mask2, cmap='coolwarm', annot=True, fmt='.2f',linewidths=0.5, linecolor='k',
            square=True,   ax=ax,cbar_ax=cax,cbar_kws={"pad":-10.0}, vmin=0,vmax=vmax)
sns.heatmap(corr1, mask=mask1, cmap='binary', annot=True, vmin=0,vmax=0, linewidths=0.5, linecolor='k',
            square=True,   ax=ax,cbar=None); #cbar_kws={'label': 'Difference in peaks picked',})


# Rotate the tick labels and set their alignment.

plt.setp(ax.get_yticklabels(), rotation=0, ha="right", rotation_mode="anchor")
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
cax.set_ylabel("Average paired CSP, ppm",rotation=-90,labelpad=18)
cax.axvline(x=0, color='k',linewidth=2)
cax.axvline(x=vmax-0.004, color='k',linewidth=2)
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

ax.set_title("Similarity of Ubc6-180X spectra")

ax_legend_elements = []
ax_legend_elements = [ mlines.Line2D([0],[0], color='w',markerfacecolor = 'w',markeredgecolor='k',
                                    marker='s',markersize=10, markeredgewidth=1.5,) ]
fig.legend(handles=ax_legend_elements,labels=['Difference\nin #peaks\n (row-col)'],handletextpad=0.1,
           loc='upper left',bbox_to_anchor=(0.89,0.9),frameon=False,framealpha=1,edgecolor='k')

#fig.tight_layout()

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
plt.show()

filename = 'Ubc6-180X_Similarity_Heatmap_23sept2024'
fig.savefig(os.path.join(project_dir,filename+'.svg'),format='svg',bbox_inches='tight')
fig.savefig(os.path.join(project_dir,filename+'.pdf'),format='pdf',dpi=600,bbox_inches='tight')

# plot heatmap with both unassigned paired CSP and assigned CSP

import seaborn as sns
import matplotlib.lines as mlines
import matplotlib.patches as patches

#sns.set_theme(style="white")

corr1 = csp_avg_2d #np.nan_to_num(csp_avg_2d,copy=True,nan=0)
corr2 = avg_score2d
mask1 = np.tril(np.ones_like(corr1, dtype=bool))
mask2 = np.triu(np.ones_like(corr2, dtype=bool))

fig, ax = plt.subplots(figsize=(7.5, 7.5),)#ncols=2,gridspec_kw={'width_ratios':[20,1],},)
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
cax.set_ylabel("   Average paired CSP, ppm",rotation=90,labelpad=5,loc='bottom')
cax.axvline(x=0, color='k',linewidth=2)
cax.axvline(x=vmax-0.00, color='k',linewidth=2)
cax.axhline(y=0,color='k',linewidth=3)
cax.axhline(y=vmax,color='k',linewidth=3)
cax1.set_ylabel("Average assigned CSP, ppm                                ",rotation=-90,labelpad=15,loc='center')
cax1.axvline(x=0, color='k',linewidth=2)
cax1.axvline(x=vmax1-0.00, color='k',linewidth=2)
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

ax.set_title("Similarity of Ubc6-180X spectra")


#find na values in original data and white them out, add 'nd' text
for y,x in np.argwhere(np.isnan(np.ma.masked_where(mask1,corr1))):
    ax.add_patch(
        patches.Rectangle(
            (x, y),1.0,1.0,
            edgecolor='k',facecolor='white', lw=0.5 ) );
    ax.text(x+.5,y+.5,'nd',ha="center",va="center")    


#plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
plt.show()


filename = 'Ubc6-180X_Similarity_Heatmap_paired_vs_assigned_23sept2024'
#fig.savefig(os.path.join(project_dir,filename+'.svg'),format='svg',bbox_inches='tight')
#fig.savefig(os.path.join(project_dir,filename+'.pdf'),format='pdf',dpi=600,bbox_inches='tight')


# Plot the spectra comparisons between two samples for assigned CSP and unassigned paired CSP

ref = '180X_WT'
#compare = '180X_H38Q'
#compare = samples[10]

for compare in samples[1:]:
    comp_index = samp[ref].index(compare)
    Xrefa = all_csps[ref]['HN_0'].to_numpy()
    Yrefa = all_csps[ref]['N_0'].to_numpy()
    Xcompa = all_csps[ref]['HN_'+str(comp_index)].to_numpy()
    Ycompa = all_csps[ref]['N_'+str(comp_index)].to_numpy()

    Xrefall = X[ref].T[0]
    Xref = X[ref][row_ind[compare+'_vs_'+ref]].T[0]
    Xcompall = X[compare].T[0]
    Xcomp = X[compare][col_ind[compare+'_vs_'+ref]].T[0]
    Yrefall = X[ref].T[1]
    Yref = X[ref][row_ind[compare+'_vs_'+ref]].T[1]
    Ycompall = X[compare].T[1]
    Ycomp = X[compare][col_ind[compare+'_vs_'+ref]].T[1]

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

    ax[1].plot(Xref,Yref,'ro',label=ref)
    #ax[1].plot(Xrefall,Yrefall,'ro',)#label=ref) #just for presentation plot

    ax[1].plot(Xrefall,Yrefall,'o',color='none',markerfacecolor=None,markeredgecolor='r',label=ref+" unpaired")
    ax[1].plot(Xcomp,Ycomp,'bo',label=compare)
    #ax[1].plot(Xcompall,Ycompall,'bo',)#label=compare) #just for presentation plot
    ax[1].plot(Xcompall,Ycompall,'o',color='none',markerfacecolor=None,markeredgecolor='b',label=compare+" unpaired")
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
    fig.savefig(os.path.join(project_dir,'Ubc6-180X_pkmatch'+compare+'_vs_'+ref+'_23sept2024.pdf'),format='pdf',dpi=600)
