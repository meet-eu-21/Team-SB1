#!/usr/bin/env python
# coding: utf-8

# ### Meet - U  2021
# #### Team SB1 Sorbonne University  
# #### Challenge CPT /  Detection of chromatine compartements.  
# 

# In[1]:


import numpy as np
import pandas as pd

import HiCtoolbox  # based on Léopold Carron tool

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

from scipy import sparse
from scipy.spatial import ConvexHull
from scipy.stats import ttest_ind

from itertools import combinations

from sklearn.manifold import MDS 
from sklearn.cluster import KMeans,AgglomerativeClustering,SpectralClustering
from sklearn import metrics
from sklearn.metrics import precision_recall_curve,PrecisionRecallDisplay,classification_report

from hmmlearn import hmm

import h5py
import sys


# ### 1. Data pre-processing pipeline
# 

# In[2]:


# Necessary functions

#BINNING ( based on L. Carron code)
def binning_intra(filename,R):                 # takes file name , and resolution          
    contacts=pd.read_csv(filename, sep="\t",header=None)
    contacts=np.int_(np.array(contacts))
    contacts=np.concatenate((contacts,np.transpose(np.array([contacts[:,1],contacts[:,0],contacts[:,2]]))), axis=0)
    contacts= sparse.coo_matrix( (contacts[:,2], (contacts[:,0],contacts[:,1]))) 
    binned_map=HiCtoolbox.bin2d(contacts,R,R) #sparse array, for mememory saving 
    lenght=np.shape(contacts)[0]
    del contacts
    
  
    return binned_map,lenght    # gives the binned contacts map and lenght of concatenated matrix

#filtering
def filter(binned_map,factor):      # takes binned contact map
    filtered_map,bin_sel=HiCtoolbox.filteramat(binned_map,factor) # filtering parameters to adjust in HiCtoolbox
    
    return filtered_map,bin_sel   # filtered binned contact map and list of saved bins

# Sequential Componenet Normalisation
def SCN(filtered_map):    #takes filtered contact map 
    scn_map=HiCtoolbox.SCN(filtered_map.copy()) 
    return scn_map    # return normalised map

# Observed/Expected Normalisation
def diagonals(scn_map):   # takes scn normalised contact matrix
    mean=[]
    n=scn_map.shape[0]
    contact_map=np.zeros((n,n))

    mean=[np.mean(scn_map.diagonal(i)) for i in range(-n+1,n)]
    for i in range(n):
        for j in range(n):
            contact_map[i, j] = scn_map[i,j]/mean[j-i+n-1]
    return contact_map  # return O/E normalised contact matrix

# Correlation matrix excluding filtered bins

def correlations(contact_map):  # takes normalise contact map 
    corr_mat=np.corrcoef(np.array(contact_map)) # calcul of Pearson correlation
    corr_mat=np.nan_to_num(corr_mat, nan=0.0)
    return corr_mat  # gives correlation matrix 


# In[3]:


filename="Chr22_100kb.txt"  # chromosome contact map 
R=100000                    # resolution


# In[4]:


binned_map,lenght=binning_intra(filename,R)


# In[5]:


filtered_map,bin_sel=filter(binned_map,factor=1.5)


# In[6]:


scn_map=SCN(filtered_map)


# In[7]:


contact_map=diagonals(scn_map)


# In[8]:


# visualisation of contact map
plt.figure(figsize=(5, 5)) 
ax=sns.heatmap(contact_map, norm=LogNorm())
plt.xlabel('Chromosome regions in 100 kb resolution')
ax.set_title("Chromosomal contact map")


# In[9]:


corr_mat=correlations(contact_map)  # correlations matrix 


# In[10]:


#visualisation filtered correlation matrix
plt.figure(figsize=(5, 5)) 
v=np.min([abs(np.min(np.min(corr_mat))),abs(np.max(np.max(corr_mat)))]) 
ax=sns.heatmap(corr_mat,cmap="RdBu",center=0,square=True,vmin=-1*v,vmax=v) # croping and harmonizing the log scale
plt.xlabel('Chromosome regions in kb resolution')
ax.set_title("Correlation matrix")


# In[11]:


#Correlations matrix -including filtered bins with centromer
corr_mat_full=pd.DataFrame(np.zeros(binned_map.shape))
corr_mat_full.iloc[bin_sel,bin_sel]=corr_mat


# In[12]:


# visualisation matrix of correlation with all bins and centromer

plt.figure(figsize=(5, 5)) 
border=np.where(corr_mat_full>0)[0][0]  # elimation of zero zone for some chromosomes !
nr_of_sticks=11
yticklabels = np.ceil(np.linspace(border,corr_mat_full.shape[0],nr_of_sticks))
xticklabels = np.ceil(np.linspace(border,corr_mat_full.shape[0],nr_of_sticks))
yticks = np.linspace(0,corr_mat_full.shape[0]-border,nr_of_sticks)
xticks = np.linspace(0,corr_mat_full.shape[0]-border,nr_of_sticks)
v=np.min([abs(np.min(np.min(corr_mat_full))),abs(np.max(np.max(corr_mat_full)))]) 

ax=sns.heatmap(corr_mat_full.iloc[border:,border:],cmap="RdBu",center=0,square=True,vmin=-1*v,vmax=v) # # log scale bar  cutt of 
ax.set_yticks(yticks)
ax.set_xticks(xticks)
ax.set_yticklabels(yticklabels)
ax.set_xticklabels(xticklabels)
plt.xlabel('Chromosome regions en 100 kb res')
ax.set_title("Chr22 - Correlation matrix")
plt.show()


# ### 2. Compartements Detection

# #### 2.1 Spectral analysis of correlation matrix
# 

# In[13]:


# Gold standard eigenvector
gold_ev=pd.read_csv("chr22_VP.txt", sep="\t",header=None) # GOLD STANDARD COMPARTEMENT
gold_ev.loc[gold_ev[0]==-1]=0



# In[14]:


#EIGENVECTORS, EIGENVALUES  of  correlations matrix not filtered

"""ATTENTION !!! 
Sometimes linalg gives eigenvectors with flipped signs comparing to gold standard / mirror reflexion.
It might also happen that  a part of  the vector corresponding to short arm of chromosome has correct
signs and the second part of vector corresponding to longer arm is inversed.( or oppositely)
It's about 15-20% of cases. 
 
To deal with that numerically  we can calculate the correlation coefficient beetwen
eigenvector/flipped eigenvector and gene density in correct resolution. 
Higher correlation coefficient indicates the correct values of eigenvector. In reality  it 
comes down to cheking the sign of correlation coefficient. Strongly positive indicates the correct 
sign of ev. Sometimes correlation coefficient is close to 0 . It brings the suspicion that only part of 
eigenvector is flipped (before/after centromere)

The visual comparison with gold standard is the easiest method however it's obviously a cheated technique.

"""

eigenvalues_full, eigenvectors_full=np.linalg.eig(corr_mat_full)  # numpy linalg tool
eigenvectors_full= eigenvectors_full.T
first_ev_full=eigenvectors_full[0]


# In[15]:


#gold standard  vs obtained eigenvector
plt.plot(np.arange(len(first_ev_full)),first_ev_full) 
plt.title("Chr22 25kb eigenvector TeamSB1  and gold standard results")
plt.xlabel("Chromosomal positions in 25kb resolution")
plt.plot(np.arange(gold_ev.shape[0]),gold_ev,color="orange")
plt.show()


# In[16]:


# saving to file in expected format  
columns=["startPOS","endPOS","cptTYPE"]
startPOS=np.arange(0,first_ev_full.shape[0]*R,R).reshape(1,-1)
endPOS=np.arange(R,(first_ev_full.shape[0]+1)*R,R).reshape(1,-1)
cptTYPE=-1*np.sign(first_ev_full).reshape(1,-1)
compartments=pd.DataFrame(np.concatenate([startPOS,endPOS,cptTYPE]).T)
compartments.columns=columns
compartments["chr i"]="chr 2" # HERE CHANGE TO ADEQUATE COMPARTEMENT
cols = list(compartments.columns)
cols = [cols[-1]] + cols[:-1]
compartments= compartments[cols]
#compartments     #  1 =active , -1=non active, 0 - filtered not considered region


# In[17]:


filename="TeamSB1_chr22_cpts_100" #  SAVE THE FILE
np.savetxt(filename,compartments,delimiter='\t', fmt="%s")


# In[18]:


# eigenvector of filterd matrix
eigenvalue_filtered, eigenvector_filtered=np.linalg.eig(corr_mat)
eigenvector_filtered= eigenvector_filtered.T
first_ev_filtered=eigenvector_filtered[0]


# In[19]:


#bar plot visualisation 
colors_AB=[]
for i in range(eigenvector_filtered.shape[0]):
    if eigenvector_filtered[0][i] >0:
        colors_AB.append("red")
    if  eigenvector_filtered[0][i] <=0:
        colors_AB.append("blue")
plt.bar(np.arange(len(first_ev_filtered)),first_ev_filtered,color=colors_AB)
plt.xlabel("Chr regions in 100 kb")  #  adapt the resolution
plt.title("Chr22 compartements-TeamSB1 result") 
plt.show() 


# In[20]:


# barplot visualisation
def barcode(data, title):
    sns.set_theme(style="ticks")
    fig = plt.figure(figsize=(5,2))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(data, cmap='bwr', aspect='auto', interpolation='nearest')
    ax.axes.get_yaxis().set_visible(False)
    plt.xlabel("Bins in 100kb ")
    plt.title(title)


# In[21]:


barcode(np.sign(first_ev_filtered).reshape(1,-1),"1st eigenvector")


# #### 2.2 Gene density analysis 
# 

# In[22]:


#GENE DENSITY  # h5py tool necessary

gen_den_filename = "chr22.hdf5"  # gene annotations file reading 

def gene_density(gen_den_filename):
    with h5py.File(gen_den_filename, "r") as f:  
        # List all groups
        print("Keys: %s" % f.keys())
        a_group_key = list(f.keys())[0]

    # Get the data
        data = list(f[a_group_key])
    
    gene_annotations=[]
    for i in range(len(data)):
        gene_annotations.append(data[i][0])
    return gene_annotations


# In[23]:


gene_annotations=gene_density(gen_den_filename)


# In[24]:


# Eigenevctor sign test !
#correlation coeficient beetwen gene density and 1st eigenevector,necessary to define if eigenvector is swaped

r=np.corrcoef(gene_annotations[:len(first_ev_full)],first_ev_full)
print(r)


# In[25]:


# eigenvector adjustment
if r[0,1]<0:
    first_ev_full,first_ev_filtered=-1*first_ev_full,-1*first_ev_filtered


# In[26]:


#visualisation of gene density
plt.plot(gene_annotations)
plt.title("Gene annotations")
plt.xlabel('Chr regions in 100 kb ') #here adapt resolution
plt.show()


# In[27]:


# gene density base pair coverage for each gene 
gene_den=np.array(gene_annotations)/R
gene_den=gene_den[bin_sel]


# #### 2.3. Confusion table comparisons

# ##### Gold  standard vs. obtained compartements

# In[28]:


# precision, recall, f-score 

y_true =np.sign(first_ev_full) # 
y_pred = np.sign(gold_ev[:len(first_ev_full)]) # lengh issue but in this context doesn't have an impact on the results
target_names = ['Active', 'Inactive', 'No cmt']
print(classification_report(y_true, y_pred, target_names=target_names))


# ##### Gene_density compartements vs obtained compartements

# In[29]:



#compartement criterium -> mean
# precision, recall, f-score
gd_compartements=gene_den.copy()
mean=np.mean(gene_den)
for i in range(len(gene_den)):
    if gene_den[i]>=mean:
        gd_compartements[i]=1
    else:
        gd_compartements[i]=-1
mean


# In[30]:


#confusion table
if r[0,1]>0:
    y_true =np.sign(eigenvector_filtered[0])
else: 
    y_true =-1*np.sign(eigenvector_filtered[0]) # for flipped eigenvector
    
y_pred=gd_compartements

target_names = ['Active', 'Inactive']
print(classification_report(y_true, y_pred, target_names=target_names))


# #### 2.3 Epigenetic marks analysis

# In[31]:


alpha=0.227
EpiGfilename="E116_15.bed"


# In[32]:


epigen=pd.read_csv(EpiGfilename, sep="\t",header=None)
epigen=epigen[:-1] 
epigen.iloc[:,3]=epigen.iloc[:,3].astype(int)
nr_of_marks=np.max(epigen.iloc[:,3])


# In[33]:


def mark_coverage(selectedmark,selectedchr):  # tales selected mark and chromosome
    chr_epigen=epigen.loc[epigen[0] == selectedchr]  # chromosome choice
    chr_mark=chr_epigen.loc[chr_epigen[3] == selectedmark] # epigenetic mark choice
    n=int(np.ceil(int(chr_epigen.iloc[-1,2])/R))
    mark_bins=np.zeros(n)
    for i in range(chr_mark.shape[0]):
        start,end=int(np.floor(int(chr_mark.iloc[i,1])/R)),int(np.floor(int(chr_mark.iloc[i,2])/R))
        mark_bins[start:(end+1)]=int(chr_mark.iloc[i,3])
    return chr_mark, mark_bins # epigenetic table with selected mark, bin attribution to that mark


# ##### 2.3.1  3D visualistaion of chromosome with epigenetic markes annotated

# In[34]:


# 3D distance matrix based on Léopold Carron code
    
contact_map_3D=HiCtoolbox.SCN(filtered_map.copy()) 
contact_map_3D=np.asarray(contact_map_3D)**alpha #now we are not sparse at all
dist_matrix = HiCtoolbox.fastFloyd(1/contact_map_3D) #shortest path on the matrix
dist_matrix=dist_matrix-np.diag(np.diag(dist_matrix))#remove the diagonal
dist_matrix=(dist_matrix+np.transpose(dist_matrix))/2
XYZ,E=HiCtoolbox.sammon(dist_matrix, 3)


# In[35]:


mark=[]


# In[36]:


# epigenetic mark base pair coverage per bin 

# Adapted Leopold Carron  code for  colors of marks

def colors(selectedmark,selectedchr):  # takes selected mark (int) and chromosome(str)
    
    
    color=pd.read_csv(EpiGfilename,delimiter='\t',header=None,names=[1,2,3,4])
    color=color[color[1]==selectedchr]   #take only chr of interest
    color[4]=color[4].astype("int")
    number=color[4].max() #number of color in the file
    print(number)
    color_vec = sparse.csr_matrix((lenght, number+1), dtype=int) # passing via spares matrix for memory reasons 
    color_vec=color_vec.toarray()
    print(color_vec.shape)
    i=0

    while i<np.shape(color)[0]:
    
        color_vec[int(color[2].iloc[i]):int(color[3].iloc[i]),int(color[4].iloc[i])]=1
        i+=1
    color_v=sparse.csr_matrix(color_vec)

    color_bins=HiCtoolbox.bin2d(color_v,R,1)
    color_bins=color_bins/np.amax(color_bins)

    print('Bp cover by this mark, has to be >0 :',np.sum(color_bins[:,selectedmark]) )

    color_chr=color_bins[bin_sel] #filter the epi by removed bin in HiC
    color_chr=color_chr[:,selectedmark] #now color2 is 1D
    color_chr=np.float64(color_chr.todense()) #type issue
    
    return color_chr # returns  mark coverage by bin


# In[37]:


color_chr=colors(1,"chr22")
color_chr=np.array(color_chr)
color_chr=color_chr.reshape(color_chr.shape[0],)


# In[38]:


# import to 3D pdb file 
print("Output shape : ",np.shape(XYZ),np.shape(color_chr))
#point rescale
hull=ConvexHull(XYZ)
scale=100/hull.area**(1/3)
XYZ=XYZ*scale
HiCtoolbox.writePDB('3Dcolors_'+str(alpha)+'.pdb',XYZ,color_chr) #  colors of epigemic mark or compartemenst


# In[39]:


# table of all bin coverage of all marks  # takes cuple of min to run
marks=[]
for i in range(1,nr_of_marks+1):
    color_chr=np.array(colors(i,"chr22"))
    marks.append(color_chr)
marks=np.array(marks)
marks_list=marks.reshape(marks.shape[:2]) 


# In[40]:


# subplot visualisation of all epigenetic marks

fig, axs = plt.subplots(nr_of_marks+1,figsize=(15,15))
sub_colors=["red","orange","lightgreen","green","darkgreen","magenta","yellow","blue","violet","pink","purple","brown","grey","lightgrey","black"]

fig.suptitle('Distribution of 15 epigenetic marks ')
axs[0].bar(np.arange(len(first_ev_filtered)),first_ev_filtered,color=colors_AB)
axs[0].set_ylabel("ev cpts")
for i in range(1,nr_of_marks+1):
    axs[i].bar(np.arange(marks_list.shape[1]),marks_list[i-1,:],color=sub_colors[i-1])
    axs[i].set_ylabel(str(i))


# ### 3. Detecting subcompartments ( >2 cpts)

# #### 3.1.Hiden Markov Model

# In[41]:


# necessary fonctions

def bic(likelihood_fn, k, X):
    """likelihood_fn: Function. Should take as input X and give out   the log likelihood
                  of the data under the fitted model.
           k - int. Number of parameters in the model. The parameter that we are trying to optimize.
                   
           X - array. Data that been fitted upon.
    """
    BIC = np.log(X.shape[0])*k - 2*likelihood_fn(X)
    return BIC

def aic(likelihood_fn, k, X):
    """likelihood_fn: Function. Should take as input X and give out   the log likelihood
                  of the data under the fitted model.
           k - int. Number of parameters in the model. The parameter that we are trying to optimize.
           X - array. Data that been fitted upon.
    """
    AIC = 2*k - 2*likelihood_fn(X)
    return AIC

def HMM(X,n_components): # takes fitted array and nr of compartments of the choice
    """ n_components=nr of clusters
         X - array. Data that been fitted upon."""
    
    hmm_curr = hmm.GaussianHMM(n_components=n_components, covariance_type='diag')
    hmm_curr.fit(X)
    hmm_cpts=hmm_curr.predict(X)
    n_features = hmm_curr.n_features
    free_parameters = 2*(n_components*n_features) + n_components*(n_components-1) + (n_components-1)
    bic_curr = bic(hmm_curr.score, free_parameters, X)
    aic_curr= aic(hmm_curr.score, free_parameters, X)
    sil_score=metrics.silhouette_score(X,hmm_cpts) # silhouette score 
    return bic_curr,aic_curr,sil_score,hmm_cpts #  returns BIC, AIC, silhouette and prediction of states sequence

def bic_hmmlearn(X,max_states): # takes array to bi fitted and max nr of states/compartements 
    """ iterates after all possible nr of compartements from min=2 compartements until max"""
    
    bic_l,aic_l = [],[]
    n_states_range = range(2,max_states)

    
    k_cpts=[]
    bayes=[]
    sil_scores=[]
    for n_components in n_states_range:
        bic_curr,aic_curr,sil_score,hmm_cpts=HMM(X,n_components)
        
        bic_l.append(bic_curr)
        aic_l.append(aic_curr)
        sil_scores.append(sil_score)
        k_cpts.append(hmm_cpts)
        
    return bic_l,aic_l,sil_scores,k_cpts # return list of AIC, BIC, silhoutte, predictions for each initial nr of states


# In[42]:



max_states=10 # max nr of compartements
bic_list,aic_list,sil_scores,k_cpts =bic_hmmlearn(corr_mat,max_states)


# In[43]:


plt.plot(np.arange(2,max_states),bic_list,label="BIC")
plt.plot(np.arange(2,max_states),aic_list,label="AIC")
plt.title( "AIC/BIC -chr 22" )
plt.xlabel("nr of clusters HMM")
plt.legend(loc="upper left")
plt.show()


# In[44]:


plt.plot(np.arange(2,max_states),sil_scores)
plt.title("HMM model score")
plt.xlabel("nr of compartements")
plt.ylabel('score')
plt.show()


# In[45]:


for i in range(len(k_cpts)):
    barcode(k_cpts[i].reshape(1,-1),"Clusters")


# #### 3.2 Clustering methods

# #### K-MEANS

# In[46]:


# finding optimal nr of clusters based on silouhette  metrics

def K_Means(array,n_clusters):
    kmeans = KMeans(n_clusters = n_clusters)
    kmeans.fit(corr_mat)
    kmeans_pred=kmeans.labels_
    sil_score=metrics.silhouette_score(corr_mat, kmeans_pred)
    return kmeans_pred,sil_score
def K_Means_optim(array,max_nr_clusters):
    sil_scores,kmeans_preds=[],[]
    for i in range(2,max_nr_clusters):
        kmeans_pred,sil_score= K_Means(array,i)
        sil_scores.append(sil_score)
        kmeans_preds.append(kmeans_pred)
        return kmeans_preds,sil_scores
    


# In[47]:


kmeans_preds,sil_scores_kmeans=K_Means_optim(corr_mat,10)
opt_nr_cpts_km=np.argmax(sil_scores_kmeans)
#plt.plot(np.arange(2,25),sil_scores_kmeans)


# #### HIERARCHICAL CLUSTERING

# In[48]:


def Hierarchical(array,n_clusters):
    hierar = AgglomerativeClustering(n_clusters = n_clusters)
    hierar.fit(corr_mat)
    hierar_pred=hierar.labels_
    sil_score=metrics.silhouette_score(corr_mat, hierar_pred)
    return hierar_pred,sil_score
def Hierarchical_optim(array,max_nr_clusters):
    sil_scores,hierar_preds=[],[]
    for i in range(2,max_nr_clusters):
        hierar_pred,sil_score=Hierarchical(array,i)
        sil_scores.append(sil_score)
        hierar_preds.append(hierar_pred)
    return hierar_preds,sil_scores
    


# In[49]:


hier_preds,sil_scores_h=Hierarchical_optim(corr_mat,10)
opt_nr_cpts_h=np.argmax(sil_scores_h)
plt.plot(np.arange(2,10),sil_scores_h)


# #### SPECTRAL CLUSTERING

# In[50]:


def Spectral(array,n_clusters):
    spec = SpectralClustering(n_clusters = n_clusters)
    spec.fit(corr_mat)
    spec_pred=spec.labels_
    sil_score=metrics.silhouette_score(corr_mat, spec_pred)
    return spec_pred,sil_score
def Spectral_optim(array,max_nr_clusters):
    sil_scores,spectral_preds=[],[]
    for i in range(2,max_nr_clusters):
        spectral_pred,sil_score=Spectral(array,i)
        sil_scores.append(sil_score)
        spectral_preds.append(spectral_pred)
    return spectral_preds,sil_scores
    


# In[51]:


spectral_preds,sil_scores_s=Spectral_optim(corr_mat,10)
opt_nr_cpts_s=np.argmax(sil_scores_s)
plt.plot(np.arange(2,10),sil_scores_s)


# In[52]:


# Barcodes visualisation
k_clusters=3
for i in range(k_clusters):
    barcode(spectral_preds[i].reshape(1,-1),"Clusters")


# #### 3.3. p-value approach , analysis of significant differences of epigenetic marks in  subcompartements
# 

# In[53]:


# division of correlation matrics into two matrice A active et B non active 
pos_A=np.where(np.sign(eigenvector_filtered[0])==1) #  "active" bins indexes
pos_B=np.where(np.sign(eigenvector_filtered[0])==-1) # inactive bins indexes

# Option 1 taking into consideration all contacts
A_map_full=corr_mat[list(pos_A[0]),:]
B_map_full=corr_mat[list(pos_B[0]),:]

# Option matrix contacts in between  regions active and non active regions excluded
A_map=corr_mat[list(pos_A[0]),:]
A_map=A_map[:,list(pos_A[0])]
sns.heatmap(A_map)
plt.show()

B_map=corr_mat[list(pos_B[0]),:]
B_map=B_map[:,list(pos_B[0])]
sns.heatmap(B_map)
plt.show()


# In[54]:


# WELCH test 



def ttest_subs(sub_sequence,nr_of_subcpts):
    
    """ Perform Welch test with not equal samlpe size between 2 or more subcompartements based on 
    epigenetic marks. For > 2 subcompartements  takes all possible combinations without remplacement
    sub_sequence= predicted sequence of compartements in alghorithm of the choice
    nr_of_subcpts=nr of clusters/subcompartements"""
    
    marks_sub=[]
    combin=list(combinations(np.arange(nr_of_subcpts),2)) # for more than 2 subcompartmeents we need to compare  all possible combnations 
 
    for i in range(nr_of_subcpts):
        index_sub=np.where(sub_sequence==i)
        marks_sub.append(marks_list[:,index_sub[0]])
    epi_test=[]
    for i in range(15):
        for pair in combin:
            tt=ttest_ind(marks_sub[pair[0]][i,:],marks_sub[pair[1]][i,:])
            epi_test.append(tt)   
    return epi_test,combin

    


# In[55]:


nr_of_subcpts=2
bic_sb,aic_sb,sil_sb,subcpts_A=HMM(A_map,nr_of_subcpts) # HMM subscompartements in active comp.
bic_sb,aic_sb,sil_sb,subcpts_B=HMM(B_map,nr_of_subcpts) # HMM subscompartements in inactive comp.

epi_test_A,combin_A=ttest_subs(subcpts_A,nr_of_subcpts) # welch -tests
epi_test_B,combin_B=ttest_subs(subcpts_B,nr_of_subcpts)


# In[56]:


epi_test_A=np.array(epi_test_A)[:,1].reshape(15,len(combin_A))
epi_test_B=np.array(epi_test_B)[:,1].reshape(15,len(combin_B))


# In[57]:


# mergin rsults for comparements A, B , for visualisation purposes
epi_test_A=pd.DataFrame(epi_test_A)
epi_test_B=pd.DataFrame(epi_test_B)
columns_A=[]
columns_B=[]
for i in range(len(combin_A)):  # lenght of combinations list are equal as we suppose the equal nr of compartements
    columns_A.append("A "+str(combin_A[i]))
    columns_B.append("B "+str(combin_B[i]))
epi_test_A.columns=columns_A
epi_test_B.columns=columns_B


# In[58]:


epi_test=epi_test_A+epi_test_B
epi_test=pd.concat([epi_test_A,epi_test_B],axis=1)
epi_test.index=np.arange(1,16)


# In[59]:


plt.figure(figsize=(10,10))
ax=sns.heatmap(epi_test,annot=True,cmap="magma",vmin=0.0,vmax=0.05) # croping and harmonizing the log scale
#ax=sns.heatmap(corr_mat_full.iloc[border:,border:],cmap="RdBu",center=0,square=True,vmin=-1*v,vmax=v)
ax.set_title("P-values Epigenetic marks - 2 HMM subclusters")


# #### Correlations

# In[60]:




def correlations(clusters_pred,nr_of_subcpts):
    """
    Calculates de correlation between compartements and epigenetic marks
    clusters_pred=cluster/states prediction sequence 
    nr_of_subcpts=nr of clusters"""
    
    epigenic_correlations=np.zeros((nr_of_subcpts,15))
    for i in range(nr_of_subcpts):
        index=np.where(clusters_pred==i)[0]
        
        e=first_ev_filtered[index]
        mark_l=marks_list.reshape(15,len(first_ev_filtered))
        mark_cpt=mark_l[:,index]
        for j in range(15):
            r=np.corrcoef(mark_cpt[j,:],e)[0,1]
            epigenic_correlations[i,j]=r
    

        
       
    return epigenic_correlations


# In[61]:


nr_of_subcpts=5
clusters_pred=spectral_preds[3] # sequence of compartements of the choice , nr of subcompartements has to equal with nr_of_subcpts
epigenetic_correlations=correlations(clusters_pred,nr_of_subcpts)


# In[62]:


epigenic_correlations=pd.DataFrame(epigenetic_correlations)
columns=["Active TSS","Flanking active TSS","Transcr at gene 5' and 3'","Strong transcription","Weak Transcription","Genetic Enhancers","Enhancers","ZNF genes and repeats","Heterochromatin","Bivalent/ poisoned TSS","Flanking bivalent TSS","Bivalent Enhancer","Repressed PolyComb","Weak repressed PolyComb","Quiescent/low"]


# In[63]:


epigenic_correlations.columns=columns


# In[64]:


plt.figure(figsize=(16,9))
ax=sns.heatmap(epigenic_correlations,annot=True,cmap="coolwarm") # croping and harmonizing the log scale

ax.set_title("Epigenetic marks - HMM clusters Correlation matrix")


# In[ ]:




