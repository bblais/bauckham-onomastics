#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('config', 'InlineBackend.print_figure_kwargs={\'facecolor\' : "w"}')


# In[2]:


from defs import *


# In[3]:


def simulate_texts(source_counts,rest_counts,Nsims=50000,
              distribution='dirichlet'):
    
    from scipy.stats import dirichlet,multinomial,uniform
    from numpy import percentile
    
    rest=np.array(rest_counts)
    source=np.array(source_counts)
    
    rv_all=dirichlet(1+rest)
    f=array(rest).ravel()
    f=f/f.sum()
    
    N=source.sum()
    K=len(source)

    n_all=[]
    for i in tqdm(range(Nsims)):
        if distribution=='dirichlet':
            p=rv_all.rvs().ravel()
        elif distribution=='frequency':
            p=f
        elif distribution=='uniform':
            p=np.ones(K)/K
        else:
            raise ValueError(f"Distribution '{distribution}'' not implemented")

        nn=multinomial(n=N,p=p).rvs().ravel()
        n_all.append(nn)
    n_all=array(n_all)    
    
    
    pl,pm,pu=np.percentile(n_all,[2.5,50,97.5],axis=0)
        
    return pl,pm,pu,n_all


# In[4]:


def find_names(use_names,names):
    idx=[]
    for name in use_names:
        equivalence={'Lazarus':'Eleazar',
                     'Judas':'Judah',
                     'John':'Yohanan',
                     'Ananias':'Hananiah',
                     'Jesus':'Joshua',
                     'Matthew':'Mattathias',
                     'Annas':'Hanan',
                     'James':'Jacob',
                    'Manaen':'Menahem',
                     'Zechariah':'Zachariah',
                     'Barabbas':'Abba', 
                    }

        try:
            idx.append(names.index(equivalence[name]))
        except KeyError:
            idx.append(names.index(name))

    return idx


# In[5]:


use_names=['Simon','Joseph','Lazarus','Judas','John','Ananias',
          'Jesus','Jonathan','Matthew','Annas','Ishmael','James',
          'Manaen','Saul','Yoezer','Levi','Honi','Dositheus','Zechariah','Samuel',
          'Barabbas','Alexander','Hezekiah','Phineas','Herod','Benaiah']


# In[16]:


counts_data={'gospels_acts_unattested':pd.read_csv('counts_gospels_acts_unattested.csv'),
      'josephus':pd.read_csv('counts_josephus.csv'),
     }
counts_data


# In[17]:


col='gospels_acts_unattested'
source=array(counts_data[col][col])
rest=array(counts_data[col]['rest'])
names=list(counts_data[col]['name'])

pl,pm,pu,n_all=simulate_texts(source,rest,Nsims=100000,);
pl_uniform,pm_uniform,pu_uniform,n_all_uniform=simulate_texts(source,rest,Nsims=100000,distribution='uniform');

idx=find_names(use_names,names)


# # Chart 1

# ![image.png](attachment:4ca94b16-79fa-4ba2-8e07-83aabfb719d9.png)

# In[22]:


N=len(use_names)
x=np.arange(N)
y=source[idx]

yerr_mid=(pl[idx]+pu[idx])/2
yerr=np.vstack((yerr_mid-pl[idx],pu[idx]-yerr_mid))
plt.errorbar(x[idx],yerr_mid,yerr=yerr,fmt='k-',
             linewidth=0,
             markersize=0,elinewidth=1,capsize=6,clip_on=False)
plt.plot(x[y>pu_uniform[idx]],y[y>pu_uniform[idx]],'kD',
         markersize=10)
plt.plot(x[y<=pu_uniform[idx]],y[y<=pu_uniform[idx]],'wD',markeredgecolor= "black",
         markersize=10)

fill_between(x, pu_uniform[idx],color='lightgray')

plt.grid(False)
xticks(range(len(use_names)));
gca().set_xticklabels(use_names,rotation=45,size=14,ha="right",rotation_mode='anchor');
plt.ylim(([0,9]))
plt.box('off')
gca().spines['top'].set_visible(False)
gca().spines['right'].set_visible(False)
gca().spines['left'].set_visible(False)
gca().spines['bottom'].set_color('gray')
ylabel('Number of occurences')


# # Chart 2

# In[23]:


col='josephus'
source=array(counts_data[col][col])
rest=array(counts_data[col]['rest'])
names=list(counts_data[col]['name'])

pl,pm,pu,n_all=simulate_texts(source,rest,Nsims=50000,);
pl_uniform,pm_uniform,pu_uniform,n_all_uniform=simulate_texts(source,rest,Nsims=50000,distribution='uniform');

idx=find_names(use_names,names)


# ![image.png](attachment:b4bd3404-130f-4395-a8e0-3b4e5f4aeafd.png)

# In[25]:


N=len(use_names)
x=np.arange(N)
y=source[idx]

yerr_mid=(pl[idx]+pu[idx])/2
yerr=np.vstack((yerr_mid-pl[idx],pu[idx]-yerr_mid))
plt.errorbar(x,yerr_mid,yerr=yerr,fmt='k-',
             linewidth=0,
             markersize=0,elinewidth=1,capsize=6,clip_on=False)
plt.plot(x[y>pu_uniform[idx]],y[y>pu_uniform[idx]],'kD',
         markersize=10)
plt.plot(x[y<=pu_uniform[idx]],y[y<=pu_uniform[idx]],'wD',markeredgecolor= "black",
         markersize=10)


fill_between(x, pu_uniform[idx],color='lightgray')

plt.grid(False)
xticks(range(len(use_names)));
gca().set_xticklabels(use_names,rotation=45,size=14,ha="right",rotation_mode='anchor');
plt.ylim(([0,30]))
plt.box('off')
gca().spines['top'].set_visible(False)
gca().spines['right'].set_visible(False)
gca().spines['left'].set_visible(False)
gca().spines['bottom'].set_color('gray')
ylabel('Number of occurences')


# # Chart 3

# ![image.png](attachment:34878064-8476-44a3-999d-16f21001c3bd.png)

# In[26]:


col='gospels_acts_unattested'
source=array(counts_data[col][col])
rest=array(counts_data[col]['rest'])
names=list(counts_data[col]['name'])
pl,pm,pu,n_all=simulate_texts(source,rest,Nsims=50000,);


# In[27]:


figure(figsize=(10,6))
rare_counts=n_all[:,rest==1].sum(axis=1)
y,_=np.histogram(rare_counts,bins=np.arange(0,23)-0.5)
y=y/sum(y)*100
x=np.arange(len(y))
plt.bar(x,y,color='gray',edgecolor='k',linewidth=.5,width=1)
plt.bar(x[x<=4],y[x<=4],color='black',edgecolor='k',linewidth=.5,width=1)

gca().set_yticks([0,2,4,6,8,10,12,14])
gca().set_yticklabels([f"{_}%"  for _ in [0,2,4,6,8,10,12,14]],fontsize=16)
gca().set_xlim([0,22])
gca().set_xticks(np.arange(23))
gca().set_xticklabels(np.arange(23),fontsize=16)

plt.grid(None)
gca().spines['top'].set_visible(False)
gca().spines['right'].set_visible(False)
gca().spines['left'].set_visible(False)
gca().spines['bottom'].set_color('gray')

xlabel('Number of rare name occurrences drawn')
ylabel('Percentage of draws')


# In[ ]:




