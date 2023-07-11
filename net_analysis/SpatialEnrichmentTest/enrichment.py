import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from typing import List
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
from scipy.stats import norm, binom, binomtest
from tqdm.auto import tqdm
def _attributes(categorical):
    encoder = OneHotEncoder(sparse=True, handle_unknown='ignore')
    encoder.fit(categorical)
    y = encoder.transform(categorical)
    return y, encoder.categories_[0], encoder

def _spatial_enrichment(xy:np.ndarray, categoricals:np.ndarray, method:str='knn', k:int=5, r:float=None, fixed_resample:bool=False, A=None):

    # Check input
    if xy.ndim != 2:
        raise ValueError(f'Input `xy` must be a two-dimensional numpy array.')
    if categoricals.ndim != 1:
        raise ValueError(f'Input `categoricals` must be a two-dimensional numpy array.')
    if method not in {'knn', 'radius'}:
        raise ValueError(f'Valid methods are `knn` or `radius`.')
    if method == 'knn' and k is None:
        raise ValueError(f'Parameter `knn` must be defined if `method` is set to `knn`.')
    if method == 'radius' and r is None:
        raise ValueError(f'Parameter `r` must be defined if `method` is set to `radius`.') 


    # Create graph
    if A is None:
        if method == 'knn':
            from sklearn.neighbors import kneighbors_graph
            A = kneighbors_graph(xy, k, include_self=False)
        else:
            from sklearn.neighbors import radius_neighbors_graph
            A = radius_neighbors_graph(xy, r, include_self=False)
    else:
        A = sp.csr_matrix(A)        
    # Attributes
    categoricals = categoricals.reshape((-1,1))
    y, unique_categorical, _ = _attributes(categoricals)

    # Compute baseline
    mu = (y.T @ A @ y).A  

    # Estimate mean
    num_edges = A.count_nonzero()
    num_vertices = y.count_nonzero()
    counts = y.sum(axis=0).A.ravel()

    # Compute diagonal elements
    if not fixed_resample:
        diagonal = np.diag(counts * (counts ) / (num_vertices * (num_vertices)))
        off_diagonal = np.outer(counts,counts) / (num_vertices * (num_vertices))
        np.fill_diagonal(off_diagonal, 0.0)
        p = diagonal + off_diagonal      

    else:
        # Number of edges with a black root
        n_black_edges = (mu - np.diag(np.diag(mu))).sum(axis=1, keepdims=True)
        # Probability that each edge end-point has a particular color.
        p_end = counts / (counts.sum() - counts.reshape((-1,1)))
        # Probability of an edge
        p = n_black_edges * p_end / num_edges
    binominal_coefs = (p, num_edges)
    mu = np.array(mu, dtype='int')
    return mu, binominal_coefs, unique_categorical, A


def _shuffle_labels_exception(labels, exception):
    labels = np.array(labels)
    mask = labels != exception
    labels[mask] = np.random.choice(labels[mask], size=len(labels[mask]))
    return labels


def _spatial_enrichment_mc(target, other, xy:np.ndarray, categoricals: np.ndarray, method:str='knn', k:int=5, r:float=None, fixed_resample:bool=False, A=None, n_mcs:int=1000):

    # Check input
    if xy.ndim != 2:
        raise ValueError(f'Input `xy` must be a two-dimensional numpy array.')
    if categoricals.ndim != 1:
        raise ValueError(f'Input `categoricals` must be a two-dimensional numpy array.')
    if method not in {'knn', 'radius'}:
        raise ValueError(f'Valid methods are `knn` or `radius`.')
    if method == 'knn' and k is None:
        raise ValueError(f'Parameter `knn` must be defined if `method` is set to `knn`.')
    if method == 'radius' and r is None:
        raise ValueError(f'Parameter `r` must be defined if `method` is set to `radius`.') 

    # Create graph
    if A is None:
        if method == 'knn':
            from sklearn.neighbors import kneighbors_graph
            A = kneighbors_graph(xy, k, include_self=False)
        else:
            from sklearn.neighbors import radius_neighbors_graph
            A = radius_neighbors_graph(xy, r, include_self=False)
    else:
        A = sp.csr_matrix(A)
    n_edges = A.count_nonzero()
    lil_graph = sp.lil_matrix(A).rows
    target_indices = np.where(categoricals == target)[0]

    # Attributes
    categoricals = categoricals.reshape((-1,1))
    y, unique_categorical, encoder = _attributes(categoricals)
    target = encoder.transform([[target]]).argmax(axis=1).A[0,0]
    other = encoder.transform([[other]]).argmax(axis=1).A[0,0]
    categoricals = y.argmax(axis=1).A
    
    mc_samples = []
    shuffled = np.array(categoricals).copy().ravel()
    categoricals = categoricals.ravel()
    for _ in tqdm(range(n_mcs)):
        # Randomly shuffle labels
        if fixed_resample:
            # Re shuffle everything except target label
            shuffled = _shuffle_labels_exception(categoricals, target)
            shuffled_target_indices = target_indices
        else:
            # Reshuffle everything
            shuffled = np.random.choice(categoricals, size=len(shuffled))
            shuffled_target_indices = np.where(shuffled == target)[0]
        # Count neighbors
        count = 0
        if len(shuffled_target_indices) > 0:
            neighboring_labels = shuffled[np.concatenate(lil_graph[shuffled_target_indices])]
            count = np.sum(neighboring_labels==other)
        mc_samples.append(count) 
    mc_samples = np.array(mc_samples)
    mu_true = np.zeros(len(unique_categorical))
    neighboring_labels = categoricals[[np.concatenate(lil_graph[target_indices])]]
    mu_true = np.sum(neighboring_labels==other)
    # Compute normal coef
    mu = np.mean(mc_samples,axis=0)
    std = np.std(mc_samples,axis=0)
    normal_coef = (mu, std)

    # Compute binominal coef.
    n = n_edges
    p =  mu / n
    bin_coef = (p,n)

    # Compute true mean
    return unique_categorical, mu_true, normal_coef, bin_coef, mc_samples


def find_topk_scores(neighborhood_results, k:int):
    zscores, labels = neighborhood_results['z_scores'], neighborhood_results['labels']
    zscores_flat = zscores.ravel()
    ind = np.argpartition(zscores.ravel(), -k)[-k:]
    result = []
    ind = np.flip(ind[np.argsort(zscores_flat[ind])])
    for i in ind:
        zscore = zscores_flat[i]
        a,b = np.where(zscores == zscore)
        result.append((labels[a[0]], labels[b[0]]))
    return result


def find_topk_colocalized(neighborhood_results, k:int):
    zscores, labels = neighborhood_results['z_scores'].copy(), neighborhood_results['labels']
    zscores = np.minimum(zscores, zscores.T)
    zscores_flat = zscores.ravel()
    ind = np.argpartition(zscores.ravel(), -2*k)[-2*k:]
    result = []
    ind = np.flip(ind[np.argsort(zscores_flat[ind])])
    for i in ind:
        zscore = zscores_flat[i]
        a,b = np.where(zscores == zscore)
        result.append((labels[a[0]], labels[b[0]]))
    return list(dict.fromkeys(result))



def debug(xy:np.ndarray, categoricals: np.ndarray, method:str='knn', k:int=5, r:float=None, fixed_resample:bool=True, A=None):
    # Compute analytical scores
    mu, bin_coefs, labels, A= _spatial_enrichment(xy,
        categoricals, 
        method=method, 
        k=k, 
        r=r, 
        fixed_resample=fixed_resample, 
        A=A
    )

    # Shape of subplot
    nrows, ncols = bin_coefs[0].shape
    p, n = bin_coefs

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15,15))
    for i in range(nrows):
        for j in range(ncols):
            # Run with Monte Carlo
            _, mu_true, normal_coef_mc, bin_coef_mc, mc_samps = _spatial_enrichment_mc(
                labels[i], 
                labels[j],
                xy, 
                categoricals, 
                method, 
                k, 
                r, 
                fixed_resample, 
                A
            )
            # Draw the histogram            
            y, x, _ = axs[i,j].hist(mc_samps, bins=30)
            xint = np.arange(int(x.max()+1))
            # Find max of histogram (so we can scale pdfs)
            ymax = np.max(y)

            # Compute pdf/pmfs.
            normal = norm.pdf(x, normal_coef_mc[0], normal_coef_mc[1]); normal = normal / normal.max() * ymax
            binominal = binom.pmf(xint, bin_coef_mc[1], bin_coef_mc[0]); binominal = binominal / binominal.max() * ymax
            bin_anal = binom.pmf(xint, bin_coefs[1], bin_coefs[0][i,j]); bin_anal = bin_anal / bin_anal.max() * ymax
            # Plot Normal distribution fitted using Monte Carolo
            axs[i,j].plot(x, normal, '-o', label='Normal distribution (MC fit).')
            axs[i,j].plot(xint, binominal, '-o', label='Binominal distribution (MC fit).')
            axs[i,j].plot(xint, bin_anal, '-o', label='Analytical binominal fit.')
            

    plt.legend()
    plt.show()




def _check_inputs(xy:np.ndarray=None, categoricals:np.ndarray=None, df:pd.DataFrame=None,  xy_columns:List[str]=None, label_column:str=None, method:str='knn', k:int=5, r:float=None, fixed_resample:bool=True):
    if df is None and xy is None:
        raise ValueError('Either `xy` or `df` must defined.')
    if (xy is not None and categoricals is None) or (categoricals is not None and xy is None):
        raise ValueError('Both `xy` and `categoricals` must defined.')
    if xy is None and categoricals is None:    
        if df is None or xy_columns is None or label_column is None:
            raise ValueError('Arguments `df`, `xy_column` and `label_column` must be all be defined.')
        for xy_column in xy_columns:
            if xy_column not in df:
                raise ValueError(f'Could not find column `{xy_column}` in input DataFrame.')
        if label_column not in df:
            raise ValueError(f'Could not find column `{label_column}` in input DataFrame.')
        if df is not None:
            xy = df[xy_columns].to_numpy()
            categoricals = df[label_column].to_numpy()
    return xy, categoricals


def spatial_enrichment(xy:np.ndarray=None, categoricals:np.ndarray=None, df:pd.DataFrame=None,  xy_columns:List[str]=None, label_column:str=None, method:str='knn', k:int=5, r:float=None,  fixed_resample:bool=True):
    """Compute spatial enrichment score from spatial cateogrical data

    Parameters
    ----------
    xy : np.ndarray, optional
        Spatial coordinates, must be `n times dim` shaped array.
    categoricals : np.ndarray, optional
        Array with categorical labels for each observation.
    df : pd.DataFrame, optional
        DataFrame with data, can be used instead of `xy` and `categoricals`, by default None.
    xy_columns : List[str], optional
        Which columns on `df` contains spatial coordinates, by default None
    label_column : str, optional
        Which column in `df` contains categoricals labels, by default None
    method : str, optional
        Which method to use when constructing spatial connectivitys, can be `radius` or `knn`, by default 'knn'
    k : int, optional
        Number of neighbors, only used when `method` is set to `knn`, by default 5
    r : float, optional
        Radius of ball search, only used when `method` is set to `radius`, by default None
    fixed_resample : bool, optional
        Wether all markers should be resampled when computing the null distribution or kept fixed
        , by default True

    Returns
    -------
    Dict
        Dictionary with keys:

    labels : np.ndarraay
        `n` shaped array with unique categorical labels
    counts : np.ndarray
        `n` times `n` shaped array with neighborhood counts.
        For example, counts[i,j] shows the frequency of observations
        with label `label[j]` that are withing the neighborhood
        of observations with label `label[i]`.
    z_scores : np.ndarray
        `n` times `n` shaped array with z-scores. For example,
        a high (positive) in entry `z_scores[i,j]` indicates that
        label `label[j]` is enriched around label `label[i]`
    p_{two_sided, one_sided} : np.ndarray
        `n` times `n` shaped array with p-values.

    Example
    -------
    Compute enrichment scores using numpy arrays
        
        >>> result = spatial_enrichment(xy, categoricals)
        >>> print(result["z_scores"]) 

    Compute enrichment scores using dataframe

        >>> result = spatial_enrichment(df=df, xy_column=['x',  'y'], label_column='labels')
        >>> print(result["z_scores])

    """    


    xy, categoricals = _check_inputs(xy, categoricals, df,  xy_columns, label_column, method, k, r, fixed_resample)
    counts, binominal_coefs, labels, A = _spatial_enrichment(xy, categoricals, method=method, k=k, r=r, fixed_resample=fixed_resample)

    # Compute z-scores
    p, n = binominal_coefs
    with np.errstate(divide='ignore', invalid='ignore'):
        z_scores = (counts - binom.mean(n, p)) / binom.std(n, p)
    if fixed_resample:
        np.fill_diagonal(z_scores, 0.0)
    # Compute p-values
    p_one_sided = norm.sf(abs(z_scores))
    p_two_sided = p_one_sided*2


    # Make output dictionary
    outputs = dict(
        counts=counts,
        bin_coefs=binominal_coefs,
        labels=labels,
        z_scores=z_scores,
        p_two_sided=p_two_sided,
        p_one_sided=p_one_sided,
        params=dict(
            xy=xy,
            categoricals=categoricals,
            method=method,
            k=k,
            r=r,
            fixed_resample=fixed_resample,
            A=A
        )
    )

    return outputs




def qc_plot_enrichment_histogram(enrichment_result, neighborhood_label, query_label):
    
    label1 = np.where(enrichment_result['labels']==neighborhood_label)[0]
    label2 = np.where(enrichment_result['labels']==query_label)[0]
    p,n = enrichment_result['bin_coefs'][0][label1,label2], enrichment_result['bin_coefs'][1]
    mu_true = enrichment_result['counts'][label1,label2]
    fig,axs = plt.subplots(nrows=1, ncols=1)

    labels_mc, _, _, _, mc_samples = _spatial_enrichment_mc(neighborhood_label, query_label, **enrichment_result['params'])

    # Keep only data with enrichment neighborhood_label -> query_label    
    
    y, x, _ = axs.hist(np.array(mc_samples, dtype='int'), bins=50)
    scale = np.max(y)
    xint = np.arange(int(np.min(x))-1,int(np.max(x))+1)
    yval = binom.pmf(xint, n, p); yval = yval/yval.max() * scale
    #yval = norm.pdf(xint, p*n, np.sqrt(p*n)); yval = yval/yval.max() * scale
    
    axs.plot(xint, yval, label='Fitted null distribution')
    axs.plot([mu_true, mu_true], [0, scale], color='red', label='True mean')
    axs.set_xlabel('Counts')
    plt.legend()


if __name__ == '__main__':
    from enrichment2 import spatial_enrichment
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    df = pd.read_csv('example_data.csv')
    result = spatial_enrichment(df=df, xy_columns=['x','y'], label_column='label')

    qc_plot_enrichment_histogram(result, 'ALBUMIN', 'CPS1')
    print(result['labels'][1])
    print(result['labels'][26])


    # Generate some random example data
    npts, ngenes = int(1e6), 200
    xy = np.random.rand(npts, 2)
    genes = np.random.randint(low=0, high=ngenes, size=npts)

    resut = spatial_enrichment(xy, genes, method='knn', fixed_resample=False)


#######################################################################################
    xy = np.array([[0,0],[0,1],[0,2],[0,3],[0,4],[5,2],[5,4],[6,6],[6,7],[6,8]])
    cat = np.array([0,0,0,0,0,1,1,2,2,2])
    A = np.zeros((10,10))
    A[0,5] = 1.0
    A[1,6] = 1.0
    A[2,9] = 1.0
    A[3,8] = 1.0
    A[4,7] = 1.0
    np.fill_diagonal(A, 0.0)
###################################################
    dataframe = pd.DataFrame({
        'x' : [1,2,3,4,5,6,7,8,3,2],
        'y' : [1,2,3,4,5,2,4,4,1,1],
        'label' : [0,1,1,1,2,2,0,1,1,0]
    }) 


    xy = dataframe[['x','y']].to_numpy()
    cat = dataframe['label'].to_numpy()
    A=None
####################################################

####################################################
    result = spatial_enrichment(xy=xy, categoricals=cat, k=3, fixed_resample=False)
    qc_plot_enrichment_histogram(result, 0, 1)
    debug(xy, cat, method='knn', k=3, fixed_resample=False, A=A)
    plt.show()
