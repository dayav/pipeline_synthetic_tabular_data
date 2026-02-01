import gc
from typing import Dict, List, Optional, Union
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
import hnswlib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import faiss
from tqdm.auto import tqdm

from sklearn.neighbors import NearestNeighbors  
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from privacy_sampling.heom import fit_heom_encoder, heom_to_vec


def compare_rows(i, df_real, df_synth, categorical_cols, numeric_cols, col_length, weights_feature):
            
    comp_results = pd.DataFrame(index=[i for i in range(len(df_real))], columns=df_real.columns)
    
    for col in categorical_cols:
        comp_results[col] = df_real[col] == df_synth.iloc[i][col]
        
    for col in numeric_cols:
        comp_results[col] = abs(df_real[col] - df_synth.iloc[i][col])
    
    
    comp_results = comp_results.dropna()
    comp_results[categorical_cols] = comp_results[categorical_cols].astype(int).replace({1: 0, 0: 1})
    
    for col in numeric_cols:
        column_range = comp_results[col].max() - comp_results[col].min()
        if column_range != 0:
            comp_results[col] = (comp_results[col] - comp_results[col].min()) / column_range
    
    weighted_comp_results = comp_results * weights_feature
    weighted_comp_results['Sum'] = weighted_comp_results.sum(axis=1)
    # weighted_comp_results['Sum norm CTGAN'] = weighted_comp_results['Sum']
    
    min_comp_results = weighted_comp_results['Sum'].min()
    min_comp_results_idx = weighted_comp_results['Sum'].idxmin()
    del comp_results
    del weighted_comp_results
    return min_comp_results, min_comp_results_idx

def dissimilarity(weights_feature: list, read_data: pd.DataFrame, synth_data: pd.DataFrame) -> tuple:
    """
    Computes dissimilarity between real and synthetic data using weighted features and parallel processing.

    This function calculates the dissimilarity between real and synthetic data samples based on categorical 
    and numerical columns. It processes the data in parallel to compare each row in the synthetic data with 
    the real data, applying feature weights for the comparison.

    Args:
        weights_feature (list): A list of weights for each feature, used to weight the contribution of each feature 
            to the dissimilarity measure.
        read_data (pd.DataFrame): The real dataset to compare against.
        synth_data (pd.DataFrame): The synthetic dataset to be compared.

    Returns:
        tuple: A tuple containing:
            - min_diss_gen (list): A list of minimum dissimilarity values for each synthetic data point.
            - min_diss_gen_idx (list): A list of indices of the real data points with minimum dissimilarity to each synthetic data point.

    Example:
        min_diss_gen, min_diss_gen_idx = dissimilarity(weights_feature, real_data, synth_data)

    Notes:
        - The function separates categorical and numerical columns for comparison.
        - It uses parallel processing (`Parallel` from `joblib`) to speed up row-wise comparison of synthetic 
          data against real data.
        - The `compare_rows` function is used to compute the dissimilarity between rows in the real and synthetic datasets.
        - Data is processed in batches of 5000 rows to optimize memory usage.
        - Garbage collection is performed after each batch to free up memory.
    """


    # df_synth = df_real_cardio_ctgan_first
    # df_real = df_real_cardio_train
    categorical_cols = read_data.select_dtypes(include='category').columns
    numeric_cols = read_data.select_dtypes(exclude='category').columns

    col_length = len(synth_data.columns)
    df_length = len(synth_data)

    min_diss_gen_idx = []
    min_diss_gen = []
    min_diss = []
    comp_results_df = []
    last_i = 0
    first_index = 0
    last_length = 5000

    while first_index != df_length :
       
        results = Parallel(n_jobs=-1)(delayed(compare_rows)(i, read_data, synth_data, categorical_cols, numeric_cols, col_length, weights_feature) for i in range(first_index, last_length))
        min_diss_values = []
        min_diss_idx_values = []
        for min_diss, min_diss_idx in results:
            min_diss_values.append(min_diss)
            min_diss_idx_values.append(min_diss_idx)
        min_diss_gen = min_diss_gen + min_diss_values
        min_diss_gen_idx = min_diss_gen_idx + min_diss_idx_values 

        
        del min_diss
        first_index = last_length
        left_length = df_length - last_length
        added_last_length = min(5000, left_length)
        last_length += added_last_length
        gc.collect()
        print(first_index,' ------finished------', last_length)
    return min_diss_gen, min_diss_gen_idx

def normalize_to_sum_1(numbers):
    total = sum(numbers)
    normalized_numbers = [x / total for x in numbers]
    return normalized_numbers

def sampling_reject(model, real_data, min_diss, cat_features, nb_data_sampled ) :
    """
    Generates synthetic data using a rejection sampling approach based on dissimilarity measures.

    This function generates synthetic data from the provided model, then iteratively replaces samples 
    with new candidates if the dissimilarity between the synthetic and real data is below a specified threshold.

    Args:
        model: The model used to generate synthetic data.
        real_data (pd.DataFrame): The real dataset used for comparison.
        min_diss (float): The minimum dissimilarity threshold. Samples with dissimilarity below this threshold 
            will be replaced.
        cat_features (list): A list of categorical feature names in the dataset.
        nb_data_sampled (int): The number of synthetic data samples to generate.

    Returns:
        pd.DataFrame: A DataFrame containing the final synthetic data after rejection sampling.

    Example:
        synth_data = sampling_reject(model, real_data, min_diss=0.2, cat_features=['col1', 'col2'], nb_data_sampled=100)

    Notes:
        - The function iterates through synthetic data samples and replaces them if their dissimilarity 
            with real data is below `min_diss`.
        - Dissimilarity is computed using the `dissimilarity` and `compare_rows` functions.
        - Categorical features are converted to the `category` dtype to handle categorical variables appropriately.
    """
     
    synth_data = model.sample(nb_data_sampled)
    
    for column in cat_features :
        real_data[column] = real_data[column].astype('category')
        synth_data[column] = synth_data[column].astype('category')
    
    dcr_list = []
    weigths = [1] * len(real_data.columns)
    min_results_tests_dfirst_new, _ = dissimilarity(normalize_to_sum_1(weigths), real_data, synth_data)
    min_results_dict = {i : minimum   for i, minimum in zip(range(len(min_results_tests_dfirst_new)), min_results_tests_dfirst_new)}

    min_results_dict_filt = [i for i in min_results_dict if min_results_dict[i] < min_diss ]

    i = 0
    not_finished = True
    while not_finished :
        data_candidate = model.sample(1)
        weigths = [1] * len(real_data.columns)
        categorical_cols = real_data.select_dtypes(include='category').columns
        numeric_cols = real_data.select_dtypes(exclude='category').columns
        min_results_tests_candidate_new, minimum_candidate_real = compare_rows(0,real_data, data_candidate, categorical_cols, numeric_cols,  24, normalize_to_sum_1(weigths))
        if (min_results_tests_candidate_new > min_diss) :
            synth_data = synth_data.drop(min_results_dict_filt[i])
            # Concatenate DataFrames
            synth_data = pd.concat([synth_data, data_candidate])
            i += 1
            if i == len(min_results_dict_filt) :
                not_finished = False
            for column in cat_features :
                synth_data[column] = synth_data[column].astype('category')

    return synth_data.reset_index(drop=True)


def sampling_reject_epsilon(model, real_data: pd.DataFrame, min_epsilon: float, cat_features: list, 
                            num_features: list, nb_data_sampled: int) -> pd.DataFrame:
    """
    Generates synthetic data using a rejection sampling approach based on epsilon-nearest neighbors.

    This function generates synthetic data and compares it to real data using epsilon-based rejection sampling. 
    It replaces synthetic data points if their nearest neighbor in the real data is too close, adjusting the 
    synthetic data until the epsilon criterion is met.

    Args:
        model: The model used to generate synthetic data.
        real_data (pd.DataFrame): The real dataset used for comparison.
        min_epsilon (float): The minimum epsilon value. Synthetic data will be adjusted until the epsilon value is below this threshold.
        cat_features (list): A list of categorical feature names in the dataset.
        num_features (list): A list of numerical feature names in the dataset.
        nb_data_sampled (int): The number of synthetic data samples to generate.

    Returns:
        pd.DataFrame: A DataFrame containing the final synthetic data after epsilon-based rejection sampling.

    Example:
        synth_data = sampling_reject_epsilon(model, real_data, min_epsilon=0.1, cat_features=['col1', 'col2'], num_features=['col3', 'col4'], nb_data_sampled=100)

    Notes:
        - The function computes nearest neighbors between real and synthetic data, adjusting the synthetic data 
          until the epsilon value (proportion of synthetic points that violate the nearest neighbor condition) 
          is below `min_epsilon`.
        - Categorical features are one-hot encoded using `pd.get_dummies` for both real and synthetic datasets.
        - Numerical features are scaled using `MinMaxScaler`.
    """


    synth_data = model.sample(nb_data_sampled)

    real_data_dummies = pd.get_dummies(real_data, columns=cat_features, dtype=int)
    synth_data_dummies = pd.get_dummies(synth_data, columns=cat_features, dtype=int)
    real_data_dummies_scaled = real_data_dummies.copy()
    synth_data_dummies_scaled = synth_data_dummies.copy()

    extra_columns = set(real_data_dummies_scaled.columns) - set(synth_data_dummies_scaled.columns)

    for column in extra_columns:
        synth_data_dummies_scaled[column] = 0

    synth_data_dummies_scaled = synth_data_dummies_scaled[real_data_dummies_scaled.columns]
    if (len(num_features) != 0) :
        scaler = MinMaxScaler() 
        real_data_dummies_scaled[num_features] = scaler.fit_transform(real_data_dummies_scaled[num_features])
        synth_data_dummies_scaled[num_features] = scaler.transform(synth_data_dummies_scaled[num_features])

    nbrs = NearestNeighbors(n_neighbors = 2).fit(real_data_dummies_scaled)
    distance, indice = nbrs.kneighbors(real_data_dummies_scaled)

    nbrs_hat = NearestNeighbors(n_neighbors = 1).fit(real_data_dummies_scaled)
    distance_hat, indice_hat = nbrs_hat.kneighbors(synth_data_dummies_scaled)

    R_Diff= distance_hat[:,0] - distance[:,1]

    epsilon_id = np.sum(R_Diff<0) / nb_data_sampled

    while (epsilon_id >= min_epsilon): #diff vs epsilon and calculated
        data_candidate = model.sample(1)
        data_candidate_dummies = pd.get_dummies(data_candidate, columns=cat_features, dtype=int)
        if (len(num_features) != 0) :
            data_candidate_dummies[num_features] = scaler.transform(data_candidate_dummies[num_features])
        extra_columns = set(real_data_dummies_scaled.columns) - set(data_candidate_dummies.columns)
        for column in extra_columns:
            data_candidate_dummies[column] = 0
        
        data_candidate_dummies = data_candidate_dummies[real_data_dummies_scaled.columns]
        distance_temp, _ = nbrs_hat.kneighbors(data_candidate_dummies)
        should_changhe = distance_hat[:,0] < distance_temp
        if (should_changhe.any()) :

            argmin_distance = distance_hat[:,0].argmin()
            synth_data.iloc[argmin_distance] = data_candidate.iloc[0]
            distance_hat[:,0][argmin_distance] = distance_temp[:,0][0]
            
            R_Diff= distance_hat[:,0] - distance[:,1]
            epsilon_id = np.sum(R_Diff<0) / nb_data_sampled

    return synth_data.reset_index(drop=True)


def get_epsilon(real_data: pd.DataFrame, synth_data: pd.DataFrame, cat_features: list, num_features: list) -> float:
    """
    Computes the epsilon measure of synthetic data quality by comparing real and synthetic datasets.

    This function calculates epsilon, which quantifies how well the synthetic data captures the distribution 
    of the real data. The method uses one-hot encoding for categorical features and scales numerical features 
    using MinMax scaling. Nearest neighbors are then computed to assess the similarity between the two datasets.

    Args:
        real_data (pd.DataFrame): The real dataset containing both categorical and numerical features.
        synth_data (pd.DataFrame): The synthetic dataset to be compared against the real dataset.
        cat_features (list): A list of categorical feature names in the dataset.
        num_features (list): A list of numerical feature names in the dataset.

    Returns:
        float: The epsilon value, representing the proportion of synthetic data points that are closer to real 
        data points than the nearest neighbor distances in the real data.

    Example:
        epsilon = get_epsilon(real_data, synth_data, cat_features=['col1', 'col2'], num_features=['col3', 'col4'])

    Notes:
        - The function first one-hot encodes categorical features using `pd.get_dummies` and scales the numerical 
          features using `MinMaxScaler`.
        - Nearest neighbors are computed twice: once to find the nearest neighbors within the real data, and once 
          to compare synthetic data points to real data.
        - Epsilon is computed as the proportion of synthetic data points whose nearest neighbor in the real data 
          is closer than the second nearest neighbor within the real data itself.
    """

    real_data_dummies = pd.get_dummies(real_data, columns=cat_features, dtype=int)
    synth_data_dummies = pd.get_dummies(synth_data, columns=cat_features, dtype=int)
    real_data_dummies_scaled = real_data_dummies.copy()
    synth_data_dummies_scaled = synth_data_dummies.copy()

    extra_columns = set(real_data_dummies_scaled.columns) - set(synth_data_dummies_scaled.columns)

    for column in extra_columns:
        synth_data_dummies_scaled[column] = 0

    synth_data_dummies_scaled = synth_data_dummies_scaled[real_data_dummies_scaled.columns]

    if (len(num_features) != 0) :
        scaler = MinMaxScaler() 
        real_data_dummies_scaled[num_features] = scaler.fit_transform(real_data_dummies_scaled[num_features])
        synth_data_dummies_scaled[num_features] = scaler.transform(synth_data_dummies_scaled[num_features])

    nbrs = NearestNeighbors(n_neighbors = 2).fit(real_data_dummies_scaled)
    distance, indice = nbrs.kneighbors(real_data_dummies_scaled)

    nbrs_hat = NearestNeighbors(n_neighbors = 1).fit(synth_data_dummies_scaled)
    distance_hat, indice_hat = nbrs_hat.kneighbors(real_data_dummies_scaled)

    R_Diff= distance_hat[:,0] - distance[:,1]
    # R_Diff = distance_hat[:, 0] - distance[indice_hat[:, 0], 1]

    return np.sum(R_Diff<0) / len(synth_data)

def get_epsilon_numerical_only(real_data: pd.DataFrame, synth_data: pd.DataFrame, num_features: list) -> float:
    """
    Computes the epsilon measure of synthetic data quality by comparing real and synthetic datasets
    using only numerical features.

    This function calculates epsilon, which quantifies how well the synthetic data captures the distribution 
    of the real data. Numerical features are scaled using MinMax scaling. Nearest neighbors are then computed 
    to assess the similarity between the two datasets.

    Args:
        real_data (pd.DataFrame): The real dataset containing numerical features.
        synth_data (pd.DataFrame): The synthetic dataset to be compared against the real dataset.
        num_features (list): A list of numerical feature names in the dataset.

    Returns:
        float: The epsilon value, representing the proportion of synthetic data points that are closer to real 
        data points than the nearest neighbor distances in the real data.

    Example:
        epsilon = get_epsilon_numerical_only(real_data, synth_data, num_features=['col3', 'col4'])

    Notes:
        - The function scales the numerical features using `MinMaxScaler`.
        - Nearest neighbors are computed twice: once to find the nearest neighbors within the real data, and once 
        to compare synthetic data points to real data.
        - Epsilon is computed as the proportion of synthetic data points whose nearest neighbor in the real data 
        is closer than the second nearest neighbor within the real data itself.
    """
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.neighbors import NearestNeighbors
    import numpy as np

    # Select only numerical features
    real_data_scaled = real_data[num_features].copy()
    synth_data_scaled = synth_data[num_features].copy()
    
    # Scale numerical features
    scaler = MinMaxScaler()
    real_data_scaled = scaler.fit_transform(real_data_scaled)
    synth_data_scaled = scaler.transform(synth_data_scaled)

    # Compute nearest neighbors within the real data
    nbrs = NearestNeighbors(n_neighbors=2).fit(real_data_scaled)
    distance, _ = nbrs.kneighbors(real_data_scaled)

    # Compute nearest neighbors between synthetic and real data
    nbrs_hat = NearestNeighbors(n_neighbors=1).fit(real_data_scaled)
    distance_hat, _ = nbrs_hat.kneighbors(synth_data_scaled)

    # Compute R_Diff and epsilon
    R_Diff = distance_hat[:, 0] - distance[:, 1]
    epsilon = np.sum(R_Diff < 0) / len(synth_data)

    return epsilon

def sampling_reject_epsilon_numerical_only(model, real_data: pd.DataFrame, min_epsilon: float, 
                            num_features: list, nb_data_sampled: int) -> pd.DataFrame:
    """
    Generates synthetic data using a rejection sampling approach based on epsilon-nearest neighbors,
    considering only numerical features.

    Args:
        model: The model used to generate synthetic data.
        real_data (pd.DataFrame): The real dataset used for comparison.
        min_epsilon (float): The minimum epsilon value. Synthetic data will be adjusted until the epsilon value is below this threshold.
        num_features (list): A list of numerical feature names in the dataset.
        nb_data_sampled (int): The number of synthetic data samples to generate.

    Returns:
        pd.DataFrame: A DataFrame containing the final synthetic data after epsilon-based rejection sampling.

    Example:
        synth_data = sampling_reject_epsilon(model, real_data, min_epsilon=0.1, num_features=['col1', 'col2'], nb_data_sampled=100)

    Notes:
        - Only numerical features are scaled and used for comparison.
        - Nearest neighbors are computed to compare synthetic and real data distributions.
    """

    # Sample synthetic data
    synth_data = model.sample(nb_data_sampled)

    # Scale numerical features
    scaler = MinMaxScaler()
    real_data_scaled = scaler.fit_transform(real_data[num_features])
    synth_data_scaled = scaler.transform(synth_data[num_features])

    # Compute nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=2).fit(real_data_scaled)
    distances, _ = nbrs.kneighbors(real_data_scaled)

    nbrs_hat = NearestNeighbors(n_neighbors=1).fit(real_data_scaled)
    distances_hat, _ = nbrs_hat.kneighbors(synth_data_scaled)

    # Calculate epsilon
    R_Diff = distances_hat[:, 0] - distances[:, 1]
    epsilon_id = np.sum(R_Diff < 0) / nb_data_sampled

    # Rejection sampling loop
    while epsilon_id >= min_epsilon:
        data_candidate = model.sample(1)
        data_candidate_scaled = scaler.transform(data_candidate[num_features])

        # Compute distance for the candidate
        distance_temp, _ = nbrs_hat.kneighbors(data_candidate_scaled)

        # Replace the worst point if necessary
        should_replace = distances_hat[:, 0] < distance_temp
        if should_replace.any():
            argmin_distance = distances_hat[:, 0].argmin()
            synth_data.iloc[argmin_distance] = data_candidate.iloc[0]
            distances_hat[argmin_distance, 0] = distance_temp[:,0][0]

            # Recompute epsilon
            R_Diff = distances_hat[:, 0] - distances[:, 1]
            epsilon_id = np.sum(R_Diff < 0) / nb_data_sampled

    return synth_data.reset_index(drop=True)



# ------------------------------ build index
def _build_hnsw(X, ef=200, M=16):
    idx = hnswlib.Index(space='l2', dim=X.shape[1])
    idx.init_index(max_elements=X.shape[0], ef_construction=ef, M=M)
    idx.add_items(X, np.arange(X.shape[0]))
    idx.set_ef(ef)
    return idx

# ------------------------------ ε score
def get_epsilon_heom_hnsw(real_df, synth_df, num_cols, cat_cols):
    sc, maps, dim = fit_heom_encoder(real_df, num_cols, cat_cols)
    Xr = heom_to_vec(real_df,  sc, maps, num_cols, cat_cols, dim)
    Xs = heom_to_vec(synth_df, sc, maps, num_cols, cat_cols, dim)

    hnsw = _build_hnsw(Xr)
    radii = np.sqrt(hnsw.knn_query(Xr, k=2)[1][:, 1])   # 1‑NN distance

    idx1, dist1 = hnsw.knn_query(Xs, k=1)
    return (np.sqrt(dist1[:, 0]) - radii[idx1[:, 0]] < 0).mean()

# ------------------------------ rejection‑with‑replacement
def sampling_reject_epsilon_hnsw(model, real_df,
                                 min_eps, num_cols, cat_cols,
                                 n_samples):

    sc, maps, dim = fit_heom_encoder(real_df, num_cols, cat_cols)
    Xr = heom_to_vec(real_df, sc, maps, num_cols, cat_cols, dim)
    hnsw = _build_hnsw(Xr)
    radii = np.sqrt(hnsw.knn_query(Xr, k=2)[1][:, 1])

    def surplus(vec):
        idx, d = hnsw.knn_query(vec.reshape(1, -1), k=1)
        return np.sqrt(d[0, 0]) - radii[idx[0, 0]]

    synth = model.sample(n_samples).reset_index(drop=True)
    Xs    = heom_to_vec(synth, sc, maps, num_cols, cat_cols, dim)
    R     = np.array([surplus(v) for v in Xs])
    eps   = (R < 0).mean()

    bar = tqdm(total=float('inf'), unit='swap', leave=False)
    bar.set_description(f"ε = {eps:.4f}  (target < {min_eps})")
    while eps >= min_eps:
        cand = model.sample(1)
        Xc   = heom_to_vec(cand, sc, maps, num_cols, cat_cols, dim)
        rc   = surplus(Xc[0])

        w = R.argmin()
        if ((R[w] < 0 and rc >= 0) or
            (R[w] >= 0 and rc >= 0 and rc > R[w])):
            synth.iloc[w] = cand.iloc[0]
            Xs[w]         = Xc
            R[w]          = rc
            eps           = (R < 0).mean()
            bar.set_description(f"ε = {eps:.4f}  (target < {min_eps})")
    bar.close()
    return synth.reset_index(drop=True)


def build_faiss_cpu(X):
    idx = faiss.IndexFlatL2(X.shape[1])   # exact
    idx.add(X.astype(np.float32))
    return idx

# uncomment next 4 lines if faiss‑gpu available
def build_faiss_gpu(X, gpu_id=0):
    res  = faiss.StandardGpuResources()
    idx  = faiss.index_cpu_to_gpu(res, gpu_id, faiss.IndexFlatL2(X.shape[1]))
    idx.add(X.astype(np.float32));  return idx

def get_epsilon_heom_faiss(real_df, synth_df, num_cols, cat_cols,
                           use_gpu=False):
    sc, maps, dim = fit_heom_encoder(real_df, num_cols, cat_cols)
    Xr = heom_to_vec(real_df,  sc, maps, num_cols, cat_cols, dim)
    Xs = heom_to_vec(synth_df, sc, maps, num_cols, cat_cols, dim)

    index = build_faiss_cpu(Xr)  # replace with build_faiss_gpu if needed
    # index = build_faiss_gpu(Xr)
    radii = np.sqrt(index.search(Xr, 2)[0][:, 1])

    D1, I1 = index.search(Xs, 1)
    return (np.sqrt(D1[:, 0]) - radii[I1[:, 0]] < 0).mean()

def get_epsilon_any(
    real_data: pd.DataFrame,
    synth_data: pd.DataFrame,
    cat_features: list,
    num_features: list
) -> float:
    """
    ANY-rule epsilon in one-hot + MinMax space.
    Unsafe if ∃ real i s.t. ||s - x_i|| < r_i, where r_i = 2-NN radius of x_i among REAL.

    Returns: epsilon_any = fraction of synthetic points that fall inside at least one real ball B_i.
    """

    # --- 1) One-hot and align columns ---
    real_dum = pd.get_dummies(real_data, columns=cat_features, dtype=int)
    synth_dum = pd.get_dummies(synth_data, columns=cat_features, dtype=int)

    # Align columns to REAL
    missing = set(real_dum.columns) - set(synth_dum.columns)
    for c in missing:
        synth_dum[c] = 0
    synth_dum = synth_dum[real_dum.columns]

    # --- 2) Scale numeric columns (fit on REAL) ---
    real_scaled = real_dum.copy()
    synth_scaled = synth_dum.copy()
    if num_features:
        scaler = MinMaxScaler()
        real_scaled[num_features]  = scaler.fit_transform(real_scaled[num_features])
        synth_scaled[num_features] = scaler.transform(synth_scaled[num_features])

    # --- 3) Per-real radii r_i via 2-NN among REAL ---
    nn2 = NearestNeighbors(n_neighbors=2, metric='euclidean').fit(real_scaled)
    D_rr, I_rr = nn2.kneighbors(real_scaled)   # shape (n_r, 2)
    r = D_rr[:, 1]                             # r_i = distance to nearest other REAL
    R_max = float(r.max())

    # --- 4) Radius-neighbors from SYNTH to REAL at radius R_max ---
    rad = NearestNeighbors(radius=R_max, metric='euclidean').fit(real_scaled)
    dlists, ilists = rad.radius_neighbors(synth_scaled, radius=R_max, return_distance=True)

    # --- 5) ANY-rule: unsafe if ANY neighbor has d < r_i ---
    unsafe = 0
    for dj, ij in zip(dlists, ilists):
        if len(ij) == 0:
            continue  # no real within R_max -> safe
        if np.any(dj < r[ij]):
            unsafe += 1

    return unsafe / len(synth_data)

def sampling_reject_epsilon_faiss(model, real_df,
                                  min_eps, num_cols, cat_cols,
                                  n_samples, use_gpu=False):

    sc, maps, dim = fit_heom_encoder(real_df, num_cols, cat_cols)
    Xr = heom_to_vec(real_df, sc, maps, num_cols, cat_cols, dim)

    index = build_faiss_cpu(Xr)
    radii = np.sqrt(index.search(Xr, 2)[0][:, 1])

    def surplus(vec):
        d, i = index.search(vec.reshape(1, -1), 1)
        return np.sqrt(d[0, 0]) - radii[i[0, 0]]

    synth = model.sample(n_samples).reset_index(drop=True)
    Xs    = heom_to_vec(synth, sc, maps, num_cols, cat_cols, dim)
    R     = np.array([surplus(v) for v in Xs])
    eps   = (R < 0).mean()

    bar = tqdm(total=float('inf'), unit='swap', leave=False)
    bar.set_description(f"ε = {eps:.4f}  (target < {min_eps})")

    while eps >= min_eps:
        cand = model.sample(1)
        Xc   = heom_to_vec(cand, sc, maps, num_cols, cat_cols, dim)
        rc   = surplus(Xc[0])

        w = R.argmin()
        if ((R[w] < 0 and rc >= 0) or
            (R[w] >= 0 and rc >= 0 and rc > R[w])):
            synth.iloc[w] = cand.iloc[0]
            Xs[w]         = Xc
            R[w]          = rc
            eps           = (R < 0).mean()
            bar.set_description(f"ε = {eps:.4f}  (target < {min_eps})")
    bar.close()

    return synth.reset_index(drop=True)


def _faiss_knn(X, k):
    index = faiss.IndexFlatL2(X.shape[1])
    index.add(X.astype(np.float32))
    D, I = index.search(X.astype(np.float32), k)
    return D, I

def _faiss_knn_pairwise(Q, X):
    index = faiss.IndexFlatL2(X.shape[1])
    index.add(X.astype(np.float32))
    D, I = index.search(Q.astype(np.float32), 1)
    return D, I

def get_epsilon_tabnet(real_df, synth_df, embed_fn, use_faiss=True):
    Zr = embed_fn(real_df)     # (n_r, d)
    Zs = embed_fn(synth_df)    # (n_s, d)

    if use_faiss:
        D_rr, _  = _faiss_knn(Zr, 2)     # self + nearest other
        D_rs, I1 = _faiss_knn_pairwise(Zs, Zr)   # helper below
    else:       # fallback to sklearn for tiny datasets
        nn2 = NearestNeighbors(2).fit(Zr)
        nn1 = NearestNeighbors(1).fit(Zr)
        D_rr, _  = nn2.kneighbors(Zr)
        D_rs, I1 = nn1.kneighbors(Zs)

    radii = np.sqrt(D_rr[:, 1])          # first other real record
    d_rs  = np.sqrt(D_rs[:, 0])
    return (d_rs - radii[I1[:, 0]] < 0).mean()


def fit_aia_guard(real_df, sens_cols, num_cols, cat_cols,
                  q: float = 0.6, min_count: int = 5, seed: int = 42):
    """
    Train per‑column AIA guards without creating a 'RARE' class.
    - For categorical S: keep only classes with >= min_count for training.
    - For numeric/high‑card S: quantile‑bin; reduce bins until all bins >= min_count.
    - Calibrate probs when possible; fall back gracefully when not.
    Returns dict[s] -> (pipeline, tau, X_cols).
    """
    guards = {}

    for s in sens_cols:
        X_cols = [c for c in real_df.columns if c != s]
        num = [c for c in num_cols if c in X_cols]
        cat = [c for c in cat_cols if c in X_cols]

        pre = ColumnTransformer(
            [('num', MinMaxScaler(), num),
             ('cat', OneHotEncoder(handle_unknown='ignore'), cat)],
            remainder='drop'
        )

        y_raw = real_df[s]

        # ---- build a training label with enough support, WITHOUT modifying df
        if pd.api.types.is_numeric_dtype(y_raw) or y_raw.nunique() > 50:
            # numeric/high-card → quantile bins; shrink until all bins >= min_count
            n_bins = min(10, max(2, y_raw.nunique()))
            while n_bins > 2:
                y_train = pd.qcut(y_raw, q=n_bins, duplicates='drop', labels=False)
                if pd.Series(y_train).value_counts().min() >= min_count:
                    break
                n_bins -= 1
            if n_bins <= 2:
                y_train = (y_raw > y_raw.median()).astype(int)
            train_mask = np.ones(len(y_train), dtype=bool)  # everyone kept
        else:
            # categorical → train only on frequent classes
            y_cat = y_raw.astype('category')
            keep_vals = y_cat.value_counts()[lambda s: s >= min_count].index
            train_mask = y_cat.isin(keep_vals).to_numpy()
            y_train = y_cat[train_mask].astype('category')

        X_train = real_df.loc[train_mask, X_cols]
        # if after filtering we have <2 classes, disable the guard for this S
        if pd.Series(y_train).nunique() < 2:
            # make a trivial pipe that outputs uniform probabilities
            base = LogisticRegression(max_iter=1)  # dummy fit
            pipe = Pipeline([('prep', pre), ('clf', base)]).fit(
                X_train.head(2), pd.Series([0, 1])  # tiny dummy
            )
            tau = 1.0  # never reject on this S
            guards[s] = (pipe, tau, X_cols)
            continue

        # ---- model + calibration (safe CV depending on support)
        base = LogisticRegression(
            max_iter=1000, multi_class='multinomial',
            class_weight='balanced', solver='saga'
        )
        counts = pd.Series(y_train).value_counts()
        minc = int(counts.min())

        if minc >= 3:
            cv = min(5, minc)
            clf = CalibratedClassifierCV(base, method='sigmoid', cv=cv)
            pipe = Pipeline([('prep', pre), ('clf', clf)]).fit(X_train, y_train)
        elif minc == 2:
            Xtr, Xva, ytr, yva = train_test_split(
                X_train, y_train, test_size=0.2,
                stratify=y_train, random_state=seed
            )
            base_pipe = Pipeline([('prep', pre), ('clf', base)]).fit(Xtr, ytr)
            pipe = CalibratedClassifierCV(base_pipe, method='sigmoid', cv='prefit').fit(Xva, yva)
        else:
            pipe = Pipeline([('prep', pre), ('clf', base)]).fit(X_train, y_train)

        # threshold on *all* real rows (pipe can score any X)
        conf = pipe.predict_proba(real_df[X_cols]).max(axis=1)
        tau  = np.quantile(conf, q)
        guards[s] = (pipe, tau, X_cols)

    return guards

def is_high_confidence(batch_df, guards):
    """True if any sensitive column is too predictable from X\\{s}."""
    risky = np.zeros(len(batch_df), dtype=bool)
    for s, (pipe, tau, X_cols) in guards.items():
        conf = pipe.predict_proba(batch_df[X_cols]).max(axis=1)
        risky |= (conf > tau)
    return risky

def sampling_reject_epsilon_tabnet(
        generator_model,          
        real_df   : 'pd.DataFrame',
        min_eps   : float,
        embed_fn,
        n_samples : int,
        use_faiss : bool = True,
        random_state: int = 42, num_cols = None, cat_cols = None,
         sens_cols=None, aia_guard=None
    ):
    """
    Epsilon‑constrained rejection‑with‑replacement in TabPFN/Transformer latent
    space.  Works with any embed_fn that maps a DataFrame ⇒ np.ndarray[N, d].
    """

    rng = np.random.default_rng(random_state)

    # ---------- 0. latent radii for REAL data ------------------------
    Zr = embed_fn(real_df).astype(np.float32)

    if use_faiss:
        D_rr, _ = _faiss_knn(Zr, 2)              # self + nearest other
    else:
        nn2 = NearestNeighbors(2).fit(Zr)
        D_rr, _ = nn2.kneighbors(Zr)

    radii = np.sqrt(D_rr[:, 1])                  # r_i for every real row

    # ---------- 1. initial synthetic batch ---------------------------
    synth_df = generator_model.sample(n_samples).reset_index(drop=True)
    Zs       = embed_fn(synth_df).astype(np.float32)

    if use_faiss:
        D_rs, idx1 = _faiss_knn_pairwise(Zs, Zr)
    else:
        nn1 = NearestNeighbors(1).fit(Zr)
        D_rs, idx1 = nn1.kneighbors(Zs)

    surplus = np.sqrt(D_rs[:, 0]) - radii[idx1[:, 0]]
    eps     = (surplus < 0).mean()
    bar = tqdm(total=float('inf'), unit='swap', leave=False)
    bar.set_description(f"ε = {eps:.4f}  (target < {min_eps})")

    if sens_cols:
        print('with aia gard')
        aia_guard = aia_guard or fit_aia_guard(real_df, sens_cols, num_cols, cat_cols)
    
  
    # ---------- 2. replacement loop ---------------------------------
    while eps >= min_eps:

        cand   = generator_model.sample(1)
        z_cand = embed_fn(cand)[0].astype(np.float32)

        # distance cand → real
        if use_faiss:
            d_cand, idx_cand = _faiss_knn_pairwise(
                                 z_cand.reshape(1, -1), Zr)
        else:
            d_cand, idx_cand = nn1.kneighbors(z_cand.reshape(1, -1))

        r_cand = np.sqrt(d_cand[0, 0]) - radii[idx_cand[0, 0]]
        worst  = surplus.argmin()

        # replacement rule: always accept if worst is unsafe and cand is safe,
        # otherwise accept if cand improves the minimum surplus
        accept = ((surplus[worst] < 0 <= r_cand) or
                  (r_cand > surplus[worst] >= 0))

        if sens_cols and accept:
            if is_high_confidence(cand, aia_guard)[0]:
                accept = False  # fail the AIA guard
        
        if accept:
            synth_df.iloc[worst] = cand.iloc[0]
            Zs[worst]            = z_cand
            surplus[worst]       = r_cand
            eps = (surplus < 0).mean()
            bar.set_description(f"ε = {eps:.4f}  (target < {min_eps})")
    bar.close()

    return synth_df.reset_index(drop=True)


import numpy as np, pandas as pd

def audit_aia_guard(guards, real_df, synthetic=None, q_hint=None):
    """
    Inspect fitted guards. For each sensitive column s:
      - show n_classes, tau, flag-rate on real (and synthetic, if given)
      - confidence quantiles so you see where tau sits
    """
    rows = []
    for s, (pipe, tau, X_cols) in guards.items():
        # predicted probabilities on REAL rows
        proba = pipe.predict_proba(real_df[X_cols])
        conf  = proba.max(axis=1)
        flagged_real = (conf > tau)

        # quantify the distribution to see how aggressive tau is
        qs    = [0.50, 0.75, 0.90, 0.95, 0.99]
        qvals = np.quantile(conf, qs)
        row = dict(
            sensitive=s,
            n_classes=int(proba.shape[1]),
            tau=float(tau),
            flag_rate_real=float(flagged_real.mean()),
            conf_median=float(qvals[0]),
            conf_q75=float(qvals[1]),
            conf_q90=float(qvals[2]),
            conf_q95=float(qvals[3]),
            conf_q99=float(qvals[4]),
            conf_max=float(conf.max()),
            equal_tau=int(np.sum(np.isclose(conf, tau)))
        )

        # optionally compare on current synthetic table
        if synthetic is not None and set(X_cols).issubset(synthetic.columns):
            conf_s = pipe.predict_proba(synthetic[X_cols]).max(axis=1)
            row['flag_rate_synth'] = float((conf_s > tau).mean())
            row['conf_median_synth'] = float(np.median(conf_s))
        if q_hint is not None:
            row['target_flag_rate'] = float(1 - q_hint)

        rows.append(row)

    rep = pd.DataFrame(rows).sort_values('flag_rate_real', ascending=False)
    print(rep.to_string(index=False))
    return rep

def get_epsilon_heom_any(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    num_cols: list,
    cat_cols: list
) -> float:
    """
    ANY-rule epsilon in HEOM space:
      unsafe(s) iff ∃ real i with HEOM(s, x_i) < r_i,
      where r_i = min_{k≠i} HEOM(x_i, x_k) (2-NN radius among REAL).
    """

    # --- 1) HEOM vectorization (your encoder) ---
    sc, maps, dim = fit_heom_encoder(real_df, num_cols, cat_cols)
    Xr = heom_to_vec(real_df,  sc, maps, num_cols, cat_cols, dim)  # (n_r, d)
    Xs = heom_to_vec(synth_df, sc, maps, num_cols, cat_cols, dim)  # (n_s, d)

    # --- 2) Per-real radii r_i via 2-NN among REAL in HEOM space ---
    nn2 = NearestNeighbors(n_neighbors=2, metric='euclidean').fit(Xr)
    D_rr, _ = nn2.kneighbors(Xr)
    r = D_rr[:, 1]                      # r_i
    R_max = float(r.max())

    # --- 3) Radius-neighbors from SYNTH to REAL at radius R_max ---
    rad = NearestNeighbors(radius=R_max, metric='euclidean').fit(Xr)
    dlists, ilists = rad.radius_neighbors(Xs, radius=R_max, return_distance=True)

    # --- 4) ANY-rule: unsafe if ANY neighbor has d < r_i ---
    unsafe = 0
    for dj, ij in zip(dlists, ilists):
        if len(ij) == 0:
            continue
        if np.any(dj < r[ij]):
            unsafe += 1

    return unsafe / len(synth_df)


def _faiss_build_index(X: np.ndarray, use_gpu: bool = False, gpu_id: int = 0):
    d = X.shape[1]
    cpu = faiss.IndexFlatL2(d)
    if use_gpu:
        res  = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, gpu_id, cpu)
    else:
        index = cpu
    index.add(X.astype(np.float32))
    return index

def _faiss_range_search_compat(index, Qb: np.ndarray, Rmax2: float):
    """
    Compatibility wrapper for FAISS range_search:
    - Newer/alt bindings:   index.range_search(Q, radius) -> (lims, D, I)
    - Others (C++-style):   index.range_search(Q, radius, RangeSearchResult)
    Returns (lims, D, I) as numpy arrays.
    """
    # Ensure float32, contiguous
    Qb = np.ascontiguousarray(Qb.astype(np.float32))
    # Try 2-arg form first
    try:
        lims, D, I = index.range_search(Qb, Rmax2)
        lims = np.asarray(lims, dtype=np.int64)
        D    = np.asarray(D,    dtype=np.float32)
        I    = np.asarray(I,    dtype=np.int64)
        return lims, D, I
    except TypeError:
        pass  # Fall back to 3-arg RangeSearchResult form

    # 3-arg form with RangeSearchResult
    rs = faiss.RangeSearchResult(Qb.shape[0])
    index.range_search(Qb, Rmax2, rs)
    lims = faiss.vector_to_array(rs.lims).astype(np.int64)
    D    = faiss.vector_to_array(rs.distances).astype(np.float32)
    I    = faiss.vector_to_array(rs.labels).astype(np.int64)
    # rs.free() may not exist on all builds; guard it.
    if hasattr(rs, "free"):
        rs.free()
    return lims, D, I

def get_epsilon_heom_any_faiss(
    real_df: "pd.DataFrame",
    synth_df: "pd.DataFrame",
    num_cols: list,
    cat_cols: list,
    *,
    use_gpu: bool = False,
    gpu_id: int = 0,
    batch_size: int = 100_000,
) -> float:
    """
    ANY-rule epsilon in HEOM space using FAISS (exact, version-agnostic wrapper).

    Unsafe(s) ⇔ ∃ real i: HEOM(s, x_i) < r_i, where r_i is 2-NN distance of x_i among REAL.
    Implemented via FAISS range search at R_max = max_i r_i, then per-neighbor strict test d^2 < r_i^2.
    """
    # 1) HEOM vectorization
    sc, maps, dim = fit_heom_encoder(real_df, num_cols, cat_cols)
    Xr = heom_to_vec(real_df,  sc, maps, num_cols, cat_cols, dim).astype(np.float32)
    Xs = heom_to_vec(synth_df, sc, maps, num_cols, cat_cols, dim).astype(np.float32)

    n_r = Xr.shape[0]
    if n_r < 2 or len(Xs) == 0:
        return 0.0  # no radii or nothing to score

    # 2) Index & per-real radii (squared)
    index = _faiss_build_index(Xr, use_gpu=use_gpu, gpu_id=gpu_id)
    D_rr2, _ = index.search(Xr, 2)             # squared L2: [self (0), nearest other]
    r2       = D_rr2[:, 1].astype(np.float32)  # r_i^2
    Rmax2    = float(r2.max())

    # 3) Range search by batches, ANY test
    unsafe = 0
    for s in range(0, len(Xs), batch_size):
        e  = min(len(Xs), s + batch_size)
        Qb = Xs[s:e]

        lims, D, I = _faiss_range_search_compat(index, Qb, Rmax2)
        # lims has length (nbatch+1); entries slice D/I per query j
        for j in range(e - s):
            beg, end = lims[j], lims[j + 1]
            if end > beg:
                dj2 = D[beg:end]
                ij  = I[beg:end]
                # ANY rule: exists neighbor with d^2 < r_i^2
                if np.any(dj2 < r2[ij]):
                    unsafe += 1

    return unsafe / len(Xs)


def _build_hnsw(X: np.ndarray, ef: int = 200, M: int = 16) -> hnswlib.Index:
    idx = hnswlib.Index(space='l2', dim=X.shape[1])
    idx.init_index(max_elements=X.shape[0], ef_construction=ef, M=M)
    idx.add_items(X, np.arange(X.shape[0]))
    idx.set_ef(ef)  # ef controls recall at query time
    return idx

def get_epsilon_heom_any_hnsw(
    real_df,
    synth_df,
    num_cols,
    cat_cols,
    *,
    ef: int = 300,
    M: int = 16,
    k0: int = 64,
    k_max: int = 8192
) -> float:
    """
    ANY-rule epsilon in HEOM space using HNSW (approximate).

    Unsafe(s) ⇔ ∃ real i: HEOM(s, x_i) < r_i, where r_i is the 2-NN distance of x_i among REAL.
    We enumerate neighbors with d^2 < Rmax^2 via adaptive K (increase until K-th dist >= Rmax^2).
    """

    # 1) HEOM vectors (float32)
    sc, maps, dim = fit_heom_encoder(real_df, num_cols, cat_cols)
    Xr = heom_to_vec(real_df,  sc, maps, num_cols, cat_cols, dim).astype(np.float32)
    Xs = heom_to_vec(synth_df, sc, maps, num_cols, cat_cols, dim).astype(np.float32)

    n_r = Xr.shape[0]
    if n_r < 2 or len(Xs) == 0:
        return 0.0

    # 2) HNSW index on REAL
    hnsw = _build_hnsw(Xr, ef=ef, M=M)

    # 3) Radii (squared) via 2-NN among REAL (approximate)
    # knn_query returns (labels, distances^2)
    labels_rr, D_rr2 = hnsw.knn_query(Xr, k=2)
    r2    = D_rr2[:, 1].astype(np.float32)   # r_i^2
    Rmax2 = float(r2.max())

    # 4) Adaptive-K ANY test for each synthetic
    unsafe = 0
    for q in Xs:
        K = min(k0, n_r)
        while True:
            I, D2 = hnsw.knn_query(q.reshape(1, -1), k=K)
            I  = I[0]
            D2 = D2[0].astype(np.float32)

            # If we enumerated all real points, we're done.
            if K >= n_r:
                if np.any(D2 < r2[I]):
                    unsafe += 1
                break

            # If the farthest among top-K is >= Rmax2, then all neighbors with d^2 < Rmax2
            # are included within these K (assuming good recall). Check ANY rule now.
            if D2[-1] >= Rmax2:
                mask = (D2 < Rmax2)
                if np.any(mask) and np.any(D2[mask] < r2[I[mask]]):
                    unsafe += 1
                break

            # Otherwise, increase K and try again (there may be neighbors within Rmax2 beyond K)
            K = min(K * 2, n_r)
            if K > k_max:
                # Safety cap: evaluate with the neighbors we have (approximate)
                mask = (D2 < Rmax2)
                if np.any(mask) and np.any(D2[mask] < r2[I[mask]]):
                    unsafe += 1
                break

    return unsafe / len(Xs)


def get_epsilon_heom_any_weighted_k(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    num_cols: List[str],
    cat_cols: List[str],
    *,
    k: int = 2,
    weights_by_col: Optional[Dict[str, float]] = None,
    compute_inverse_entropy: bool = True,
    numeric_bins: Union[str, int] = 'fd',   # Freedman–Diaconis or an int
    normalize_mean_to_one: bool = True,
    clip_max_weight: Optional[float] = None
) -> float:
    """
    ANY-rule epsilon in HEOM space with per-column weights and k-th neighbor radii.
    Implements weighted Minkowski (p=2) by scaling HEOM vectors by sqrt(weight) and
    then using Euclidean distances.

    Assumes:
      fit_heom_encoder(real_df, num_cols, cat_cols) -> (scaler, cat_maps, dim)
        - numeric dims = indices [0 .. len(num_cols)-1]
        - cat_maps[c][value] = (index_in_vector, scale)
      heom_to_vec(df, scaler, cat_maps, num_cols, cat_cols, dim) -> np.ndarray
    """

    # ---- checks ----
    n_s = len(synth_df)
    if n_s == 0:
        return 0.0
    if k < 2:
        raise ValueError("k must be >= 2.")

    # ---- HEOM vectors ----
    scaler, cat_maps, dim = fit_heom_encoder(real_df, num_cols, cat_cols)
    Xr = heom_to_vec(real_df,  scaler, cat_maps, num_cols, cat_cols, dim).astype(np.float32)
    Xs = heom_to_vec(synth_df, scaler, cat_maps, num_cols, cat_cols, dim).astype(np.float32)
    n_r = Xr.shape[0]
    if n_r < 2:
        raise ValueError("Need at least two real rows to define radii.")

    # ---- per-column weights ----
    def _shannon_entropy_from_counts(counts: np.ndarray) -> float:
        total = counts.sum()
        if total <= 0: return 0.0
        p = counts[counts > 0].astype(float) / float(total)
        return float(-(p * np.log(p)).sum())

    def _inverse_entropy_weights(
        df: pd.DataFrame, num_cols_: List[str], cat_cols_: List[str],
        *, numeric_bins_: Union[str, int] = 'fd', eps: float = 1e-12,
        normalize_mean_to_one_: bool = True, clip_max_: Optional[float] = None
    ) -> Dict[str, float]:
        w: Dict[str, float] = {}
        for col in num_cols_:
            s = df[col].dropna().to_numpy()
            if s.size <= 1 or np.all(s == s[:1]):
                H = 0.0
            else:
                try:
                    counts, _ = np.histogram(s, bins=numeric_bins_)
                except Exception:
                    counts, _ = np.histogram(s, bins=32)
                H = _shannon_entropy_from_counts(counts)
            w[col] = 1.0 / (H + eps)
        for col in cat_cols_:
            counts = df[col].value_counts(dropna=False).to_numpy()
            H = _shannon_entropy_from_counts(counts)
            w[col] = 1.0 / (H + eps)
        if normalize_mean_to_one_ and len(w) > 0:
            m = float(np.mean(list(w.values())))
            if m > 0:
                for c in w: w[c] /= m
        if clip_max_ is not None:
            for c in w: w[c] = min(w[c], clip_max_)
        return w

    if weights_by_col is None and compute_inverse_entropy:
        weights_by_col = _inverse_entropy_weights(
            real_df, num_cols, cat_cols,
            numeric_bins_=numeric_bins,
            eps=1e-12,
            normalize_mean_to_one_=normalize_mean_to_one,
            clip_max_=clip_max_weight
        )
    elif weights_by_col is None:
        weights_by_col = {c: 1.0 for c in list(num_cols) + list(cat_cols)}

    # ---- expand column weights to per-dimension weights via cat_maps ----
    def _build_dim_weight_vector(cat_maps_: dict, dim_: int) -> np.ndarray:
        w_dim_ = np.ones(dim_, dtype=np.float32)
        # numeric dims first
        for j, col in enumerate(num_cols):
            w_dim_[j] = float(weights_by_col.get(col, 1.0))
        # categorical dims
        for col in cat_cols:
            w_col = float(weights_by_col.get(col, 1.0))
            for _, (idx, _scale) in cat_maps_[col].items():
                if 0 <= idx < dim_:
                    w_dim_[idx] = w_col
        return w_dim_

    w_dim = _build_dim_weight_vector(cat_maps, dim)
    # emulate weighted Minkowski (p=2) via scaling by sqrt(w)
    w_scale = np.sqrt(np.clip(w_dim, 0.0, None)).astype(np.float32)

    # ---- scale vectors ----
    Xr_w = Xr * w_scale
    Xs_w = Xs * w_scale

    # ---- k-NN radii among REAL (euclidean on scaled vectors) ----
    k_eff = min(max(k, 2), n_r)
    nnk = NearestNeighbors(n_neighbors=k_eff, metric='euclidean', algorithm='auto').fit(Xr_w)
    D_rr, _ = nnk.kneighbors(Xr_w)
    r = D_rr[:, k_eff - 1].astype(np.float32)      # distance to k-th neighbor
    R_max = float(r.max())

    # ---- radius neighbors from SYNTH to REAL (same scaled space) ----
    rad = NearestNeighbors(radius=R_max, metric='euclidean', algorithm='auto').fit(Xr_w)
    dlists, ilists = rad.radius_neighbors(Xs_w, radius=R_max, return_distance=True)

    # ---- ANY rule ----
    unsafe = 0
    for dj, ij in zip(dlists, ilists):
        if len(ij) == 0:
            continue
        if np.any(dj < r[ij]):
            unsafe += 1

    return float(unsafe) / float(n_s)


def sampling_reject_epsilon_heom_knn_any(
    model, real_df: pd.DataFrame,
    min_eps: float, num_cols: list, cat_cols: list,
    n_samples: int, *, verbose: bool = True
    ) -> pd.DataFrame:
    """
    Rejection-with-replacement in HEOM space (ANY rule) using exact kNN from scikit-learn.

    Unsafe(s) <=> ∃ real i: ||s - x_i|| < r_i, where r_i is 2-NN distance among REAL.
    We work with squared margins m2(s) = min_i(||s-x_i||^2 - r_i^2).

    Returns a synthetic DataFrame with epsilon (ANY) < min_eps (if possible).
    """

    # ---------- HEOM vectors for REAL ----------
    sc, maps, dim = fit_heom_encoder(real_df, num_cols, cat_cols)
    Xr = heom_to_vec(real_df, sc, maps, num_cols, cat_cols, dim).astype(np.float32)
    n_r = Xr.shape[0]

    # ---------- per-real radii r_i (2-NN among REAL) ----------
    nn2 = NearestNeighbors(n_neighbors=2, metric='euclidean').fit(Xr)
    D_rr, _ = nn2.kneighbors(Xr)         # Euclidean
    r  = D_rr[:, 1]
    r2 = r * r
    Rmax  = float(r.max())
    Rmax2 = float(r2.max())

    # Two indices: 1-NN (for fallback dmin) and radius-neighbors up to Rmax
    nn1  = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(Xr)
    rad  = NearestNeighbors(radius=Rmax,   metric='euclidean').fit(Xr)

    # ---------- helper: squared ANY margin for a batch ----------
    def margins2_batch(Xq: np.ndarray) -> np.ndarray:
        # radius neighbors within Rmax
        dlists, ilists = rad.radius_neighbors(Xq, radius=Rmax, return_distance=True)
        # nearest neighbor distance for fallback
        Dmin, Imin = nn1.kneighbors(Xq)
        Dmin2 = (Dmin[:, 0] * Dmin[:, 0]).astype(np.float32)

        out = np.empty(len(Xq), dtype=np.float32)
        for j in range(len(Xq)):
            ids = ilists[j]
            if len(ids) == 0:
                # No real within Rmax ⇒ all distances >= Rmax >= r_i ⇒ margin^2 >= Dmin^2 - Rmax^2 >= 0
                out[j] = Dmin2[j] - Rmax2
            else:
                dj  = dlists[j].astype(np.float32)
                dj2 = dj * dj
                out[j] = np.min(dj2 - r2[ids])
        return out

    # ---------- initial synthetic + margins ----------
    synth = model.sample(n_samples).reset_index(drop=True)
    Xs    = heom_to_vec(synth, sc, maps, num_cols, cat_cols, dim).astype(np.float32)
    R2    = margins2_batch(Xs)
    eps   = float((R2 < 0).mean())

    bar = tqdm(total=float('inf'), unit='swap', leave=False) if verbose else None
    if bar is not None:
        bar.set_description(f"ε_any = {eps:.4f}  (target < {min_eps})")

    # ---------- replacement loop ----------
    while eps >= min_eps:
        cand   = model.sample(1)
        Xc     = heom_to_vec(cand, sc, maps, num_cols, cat_cols, dim).astype(np.float32)
        rc2    = margins2_batch(Xc)[0]

        w = int(np.argmin(R2))
        accept = ((R2[w] < 0 <= rc2) or (R2[w] >= 0 and rc2 > R2[w]))
        if accept:
            synth.iloc[w] = cand.iloc[0]
            Xs[w]         = Xc[0]
            R2[w]         = rc2
            eps           = float((R2 < 0).mean())
            if bar is not None: bar.set_description(f"ε_any = {eps:.4f}  (target < {min_eps})")

    if bar is not None: bar.close()
    return synth.reset_index(drop=True)

def _build_hnsw(X: np.ndarray, ef: int = 200, M: int = 16) -> hnswlib.Index:
    idx = hnswlib.Index(space='l2', dim=X.shape[1])
    idx.init_index(max_elements=X.shape[0], ef_construction=ef, M=M)
    idx.add_items(X, np.arange(X.shape[0]))
    idx.set_ef(ef)
    return idx

def sampling_reject_epsilon_heom_hnsw_any(
    model, real_df: pd.DataFrame,
    min_eps: float, num_cols: list, cat_cols: list,
    n_samples: int, *,
    ef: int = 300, M: int = 16,
    k0: int = 32, k_max: int = 8192,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Rejection-with-replacement in HEOM space (ANY rule) using HNSW (approximate).
    Adaptive-K ensures we enumerate all neighbors within Rmax *provided* the ANN
    result contains them among the top-K (recall depends on ef).
    """

    # ---------- HEOM vectors for REAL ----------
    sc, maps, dim = fit_heom_encoder(real_df, num_cols, cat_cols)
    Xr = heom_to_vec(real_df, sc, maps, num_cols, cat_cols, dim).astype(np.float32)
    n_r = Xr.shape[0]
    hnsw = _build_hnsw(Xr, ef=ef, M=M)

    # ---------- per-real radii r_i ----------
    D2_rr = hnsw.knn_query(Xr, k=2)[1]   # squared dists
    r2    = D2_rr[:, 1].astype(np.float32)
    Rmax2 = float(r2.max())

    # ---------- helper: adaptive-K ANY squared margin for one vector ----------
    def margin2_one(vec: np.ndarray) -> float:
        K = min(k0, n_r)
        while True:
            I, D2 = hnsw.knn_query(vec.reshape(1, -1), k=K)
            I  = I[0]
            D2 = D2[0].astype(np.float32)

            # If K == n_r, we enumerated all.
            if K >= n_r:
                mask = (D2 < Rmax2)
                if not np.any(mask):
                    return float(D2[0] - Rmax2)
                return float(np.min(D2[mask] - r2[I[mask]]))

            # If the K-th distance >= Rmax2, then all neighbors with d^2 < Rmax2 are within top-K.
            if D2[-1] >= Rmax2:
                mask = (D2 < Rmax2)
                if not np.any(mask):
                    return float(D2[0] - Rmax2)
                return float(np.min(D2[mask] - r2[I[mask]]))

            # Otherwise, increase K (there may exist neighbors within Rmax2 beyond current K)
            K = min(K * 2, n_r)
            if K > k_max:
                # Safety cap; approximate result
                mask = (D2 < Rmax2)
                if not np.any(mask):
                    return float(D2[0] - Rmax2)
                return float(np.min(D2[mask] - r2[I[mask]]))

    # ---------- initial synthetic + margins ----------
    synth = model.sample(n_samples).reset_index(drop=True)
    Xs    = heom_to_vec(synth, sc, maps, num_cols, cat_cols, dim).astype(np.float32)
    R2    = np.array([margin2_one(v) for v in Xs], dtype=np.float32)
    eps   = float((R2 < 0).mean())

    bar = tqdm(total=float('inf'), unit='swap', leave=False) if verbose else None
    if bar is not None: bar.set_description(f"ε_any ≈ {eps:.4f}  (target < {min_eps})")

    # ---------- replacement loop ----------
    while eps >= min_eps:
        cand = model.sample(1)
        Xc   = heom_to_vec(cand, sc, maps, num_cols, cat_cols, dim).astype(np.float32)
        rc2  = margin2_one(Xc[0])

        w = int(np.argmin(R2))
        accept = ((R2[w] < 0 <= rc2) or (R2[w] >= 0 and rc2 > R2[w]))
        if accept:
            synth.iloc[w] = cand.iloc[0]
            Xs[w]         = Xc[0]
            R2[w]         = rc2
            eps           = float((R2 < 0).mean())
            if bar is not None: bar.set_description(f"ε_any ≈ {eps:.4f}  (target < {min_eps})")

    if bar is not None: bar.close()
    return synth.reset_index(drop=True)

def _faiss_build_index(X: np.ndarray, use_gpu: bool = False, gpu_id: int = 0):
    d = X.shape[1]
    cpu = faiss.IndexFlatL2(d)
    if use_gpu:
        res  = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, gpu_id, cpu)
    else:
        index = cpu
    index.add(X.astype(np.float32))
    return index

def _faiss_margins2_any(index, Q: np.ndarray, r2: np.ndarray, Rmax2: float, batch_size: int = 100_000) -> np.ndarray:
    """Exact ANY-rule squared margins for a batch Q using FAISS range search at Rmax."""
    nq = Q.shape[0]
    out = np.empty(nq, dtype=np.float32)

    # 1-NN for fallback dmin^2
    Dmin2, _ = index.search(Q.astype(np.float32), 1)
    Dmin2 = Dmin2[:, 0].astype(np.float32)

    # Range search per batch
    for s in range(0, nq, batch_size):
        e = min(nq, s + batch_size)
        Qb = Q[s:e].astype(np.float32)

        rs = faiss.RangeSearchResult(e - s)
        index.range_search(Qb, Rmax2, rs)
        lims = faiss.vector_to_array(rs.lims).astype(np.int64)
        D    = faiss.vector_to_array(rs.distances).astype(np.float32)
        I    = faiss.vector_to_array(rs.labels).astype(np.int64)

        for j in range(e - s):
            beg, end = lims[j], lims[j + 1]
            if end <= beg:
                # no neighbors within Rmax -> margin^2 ≥ Dmin^2 - Rmax^2
                out[s + j] = Dmin2[s + j] - Rmax2
            else:
                dj2 = D[beg:end]
                ij  = I[beg:end]
                out[s + j] = np.min(dj2 - r2[ij])

        rs.free()  # no-op on CPU; frees GPU temp if any
    return out

def sampling_reject_epsilon_heom_faiss_any(
    model, real_df: pd.DataFrame,
    min_eps: float, num_cols: list, cat_cols: list,
    n_samples: int, *,
    use_gpu: bool = False, gpu_id: int = 0,
    batch_size: int = 100_000, verbose: bool = True
) -> pd.DataFrame:
    """
    Rejection-with-replacement in HEOM space (ANY rule) using FAISS (exact).
    Works with CPU or single-GPU IndexFlatL2.
    """

    # ---------- HEOM vectors for REAL ----------
    sc, maps, dim = fit_heom_encoder(real_df, num_cols, cat_cols)
    Xr = heom_to_vec(real_df, sc, maps, num_cols, cat_cols, dim).astype(np.float32)
    n_r = Xr.shape[0]

    # ---------- FAISS index & per-real radii ----------
    index = _faiss_build_index(Xr, use_gpu=use_gpu, gpu_id=gpu_id)
    D_rr2, _ = index.search(Xr, 2)         # squared dists: self(0), nearest-other
    r2       = D_rr2[:, 1].astype(np.float32)
    Rmax2    = float(r2.max())

    # ---------- initial synthetic + margins ----------
    synth = model.sample(n_samples).reset_index(drop=True)
    Xs    = heom_to_vec(synth, sc, maps, num_cols, cat_cols, dim).astype(np.float32)
    R2    = _faiss_margins2_any(index, Xs, r2, Rmax2, batch_size=batch_size)
    eps   = float((R2 < 0).mean())

    bar = tqdm(total=float('inf'), unit='swap', leave=False) if verbose else None
    if bar is not None: bar.set_description(f"ε_any = {eps:.4f}  (target < {min_eps})")

    # ---------- replacement loop ----------
    while eps >= min_eps:
        cand = model.sample(1)
        Xc   = heom_to_vec(cand, sc, maps, num_cols, cat_cols, dim).astype(np.float32)
        rc2  = _faiss_margins2_any(index, Xc, r2, Rmax2, batch_size=1)[0]

        w = int(np.argmin(R2))
        accept = ((R2[w] < 0 <= rc2) or (R2[w] >= 0 and rc2 > R2[w]))
        if accept:
            synth.iloc[w] = cand.iloc[0]
            Xs[w]         = Xc[0]
            R2[w]         = rc2
            eps           = float((R2 < 0).mean())
            if bar is not None: bar.set_description(f"ε_any = {eps:.4f}  (target < {min_eps})")

    if bar is not None: bar.close()
    return synth.reset_index(drop=True)


# def sampling_reject_epsilon_heom_any_weighted_k(
#     model,
#     real_df: pd.DataFrame,
#     min_eps: float,
#     num_cols: List[str],
#     cat_cols: List[str],
#     n_samples: int,
#     *,
#     k: int = 2,
#     weights_by_col: Optional[Dict[str, float]] = None,
#     compute_inverse_entropy: bool = True,
#     numeric_bins: Union[str, int] = 'fd',
#     normalize_mean_to_one: bool = True,
#     clip_max_weight: Optional[float] = None,
#     verbose: bool = True
# ) -> pd.DataFrame:
#     """
#     Rejection-with-replacement in HEOM space (ANY rule) with per-column weights and k-th neighbor radii.
#     Weighted Minkowski (p=2) is implemented by scaling the HEOM vectors by sqrt(weight) and using Euclidean NNs.
#     """

#     if k < 2: raise ValueError("k must be >= 2.")
#     if n_samples <= 0: raise ValueError("n_samples must be positive.")
#     if not (0.0 <= min_eps < 1.0): raise ValueError("min_eps must be in [0,1).")

#     # ---- HEOM vectors for REAL ----
#     scaler, cat_maps, dim = fit_heom_encoder(real_df, num_cols, cat_cols)
#     Xr = heom_to_vec(real_df, scaler, cat_maps, num_cols, cat_cols, dim).astype(np.float32)
#     n_r = Xr.shape[0]
#     if n_r < 2:
#         raise ValueError("Need at least two real rows to define radii.")

#     # ---- per-column weights (same helpers as above) ----
#     def _shannon_entropy_from_counts(counts: np.ndarray) -> float:
#         total = counts.sum()
#         if total <= 0: return 0.0
#         p = counts[counts > 0].astype(float) / float(total)
#         return float(-(p * np.log(p)).sum())

#     def _inverse_entropy_weights(
#         df: pd.DataFrame, num_cols_: List[str], cat_cols_: List[str],
#         *, numeric_bins_: Union[str, int] = 'fd', eps: float = 1e-12,
#         normalize_mean_to_one_: bool = True, clip_max_: Optional[float] = None
#     ) -> Dict[str, float]:
#         w: Dict[str, float] = {}
#         for col in num_cols_:
#             s = df[col].dropna().to_numpy()
#             if s.size <= 1 or np.all(s == s[:1]):
#                 H = 0.0
#             else:
#                 try:
#                     counts, _ = np.histogram(s, bins=numeric_bins_)
#                 except Exception:
#                     counts, _ = np.histogram(s, bins=32)
#                 H = _shannon_entropy_from_counts(counts)
#             w[col] = 1.0 / (H + eps)
#         for col in cat_cols_:
#             counts = df[col].value_counts(dropna=False).to_numpy()
#             H = _shannon_entropy_from_counts(counts)
#             w[col] = 1.0 / (H + eps)
#         if normalize_mean_to_one_ and len(w) > 0:
#             m = float(np.mean(list(w.values())))
#             if m > 0:
#                 for c in w: w[c] /= m
#         if clip_max_ is not None:
#             for c in w: w[c] = min(w[c], clip_max_)
#         return w

#     if weights_by_col is None and compute_inverse_entropy:
#         weights_by_col = _inverse_entropy_weights(
#             real_df, num_cols, cat_cols,
#             numeric_bins_=numeric_bins, eps=1e-12,
#             normalize_mean_to_one_=normalize_mean_to_one,
#             clip_max_=clip_max_weight
#         )
#     elif weights_by_col is None:
#         weights_by_col = {c: 1.0 for c in list(num_cols) + list(cat_cols)}

#     # ---- expand per-column weights to per-dimension weights via cat_maps ----
#     def _build_dim_weight_vector(cat_maps_: dict, dim_: int) -> np.ndarray:
#         w_dim_ = np.ones(dim_, dtype=np.float32)
#         for j, col in enumerate(num_cols):
#             w_dim_[j] = float(weights_by_col.get(col, 1.0))
#         for col in cat_cols:
#             w_col = float(weights_by_col.get(col, 1.0))
#             for _, (idx, _scale) in cat_maps_[col].items():
#                 if 0 <= idx < dim_:
#                     w_dim_[idx] = w_col
#         return w_dim_

#     w_dim   = _build_dim_weight_vector(cat_maps, dim)
#     w_scale = np.sqrt(np.clip(w_dim, 0.0, None)).astype(np.float32)

#     # ---- scale REAL vectors once ----
#     Xr_w = Xr * w_scale

#     # ---- per-real radii r_i via k-NN among REAL (euclidean on scaled vectors) ----
#     k_eff = min(max(k, 2), n_r)
#     nnk = NearestNeighbors(n_neighbors=k_eff, metric='euclidean', algorithm='auto').fit(Xr_w)
#     D_rr, _ = nnk.kneighbors(Xr_w)
#     r  = D_rr[:, k_eff - 1].astype(np.float32)
#     r2 = r * r
#     Rmax  = float(r.max())
#     Rmax2 = Rmax * Rmax

#     # ---- indices for queries from synthetic points ----
#     nn1 = NearestNeighbors(n_neighbors=1, metric='euclidean', algorithm='auto').fit(Xr_w)
#     rad = NearestNeighbors(radius=Rmax, metric='euclidean', algorithm='auto').fit(Xr_w)

#     # ---- helper: squared ANY margin for a batch (work in scaled space) ----
#     def _margins2_batch(Xq_raw: np.ndarray) -> np.ndarray:
#         Xq = (Xq_raw * w_scale).astype(np.float32)  # scale query once here
#         dlists, ilists = rad.radius_neighbors(Xq, radius=Rmax, return_distance=True)
#         Dmin, _ = nn1.kneighbors(Xq)
#         Dmin2 = (Dmin[:, 0] * Dmin[:, 0]).astype(np.float32)

#         out = np.empty(len(Xq), dtype=np.float32)
#         for j in range(len(Xq)):
#             ids = ilists[j]
#             if len(ids) == 0:
#                 out[j] = Dmin2[j] - Rmax2
#             else:
#                 dj  = dlists[j].astype(np.float32)
#                 dj2 = dj * dj
#                 out[j] = float(np.min(dj2 - r2[ids]))
#         return out

#     # ---- initial synthetic + margins ----
#     synth = model.sample(n_samples).reset_index(drop=True)
#     Xs    = heom_to_vec(synth, scaler, cat_maps, num_cols, cat_cols, dim).astype(np.float32)
#     R2    = _margins2_batch(Xs)
#     eps   = float((R2 < 0).mean())

#     bar = tqdm(total=float('inf'), unit='swap', leave=False) if verbose else None
#     if bar is not None:
#         bar.set_description(f"ε_any(k={k_eff}, weighted) = {eps:.4f}  (target < {min_eps})")

#     # ---- rejection-with-replacement loop ----
#     while eps >= min_eps:
#         cand = model.sample(1)
#         Xc   = heom_to_vec(cand, scaler, cat_maps, num_cols, cat_cols, dim).astype(np.float32)
#         rc2  = _margins2_batch(Xc)[0]

#         widx   = int(np.argmin(R2))  # current worst (smallest margin)
#         accept = ((R2[widx] < 0 <= rc2) or (R2[widx] >= 0 and rc2 > R2[widx]))
#         if accept:
#             synth.iloc[widx] = cand.iloc[0]
#             R2[widx]         = rc2
#             eps              = float((R2 < 0).mean())
#             if bar is not None:
#                 bar.set_description(f"ε_any(k={k_eff}, weighted) = {eps:.4f}  (target < {min_eps})")

#     if bar is not None:
#         bar.close()
#     return synth.reset_index(drop=True)

def sampling_reject_epsilon_heom_any_weighted_k(
    model,
    real_df: pd.DataFrame,
    min_eps: float,
    num_cols: List[str],
    cat_cols: List[str],
    n_samples: int,
    *,
    k: int = 2,
    weights_by_col: Optional[Dict[str, float]] = None,
    compute_inverse_entropy: bool = True,
    numeric_bins: Union[str, int] = 'fd',
    normalize_mean_to_one: bool = True,
    clip_max_weight: Optional[float] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Rejection-with-replacement in HEOM space (ANY rule) with per-column weights and k-th neighbor radii.
    Weighted Minkowski (p=2) is implemented by scaling the HEOM vectors by sqrt(weight) and using Euclidean NNs.
    This version computes the ANY margin using k-NN only (no radius queries) with an exact early-stop bound.
    """

    # ---- basic checks ----
    if k < 2: raise ValueError("k must be >= 2.")
    if n_samples <= 0: raise ValueError("n_samples must be positive.")
    if not (0.0 <= min_eps < 1.0): raise ValueError("min_eps must be in [0,1).")

    # ---- HEOM vectors for REAL ----
    scaler, cat_maps, dim = fit_heom_encoder(real_df, num_cols, cat_cols)
    Xr = heom_to_vec(real_df, scaler, cat_maps, num_cols, cat_cols, dim).astype(np.float32)
    n_r = Xr.shape[0]
    if n_r < 2:
        raise ValueError("Need at least two real rows to define radii.")

    # ---- per-column weights (same as original) ----
    def _shannon_entropy_from_counts(counts: np.ndarray) -> float:
        total = counts.sum()
        if total <= 0: return 0.0
        p = counts[counts > 0].astype(float) / float(total)
        return float(-(p * np.log(p)).sum())

    def _inverse_entropy_weights(
        df: pd.DataFrame, num_cols_: List[str], cat_cols_: List[str],
        *, numeric_bins_: Union[str, int] = 'fd', eps: float = 1e-12,
        normalize_mean_to_one_: bool = True, clip_max_: Optional[float] = None
    ) -> Dict[str, float]:
        w: Dict[str, float] = {}
        for col in num_cols_:
            s = df[col].dropna().to_numpy()
            if s.size <= 1 or np.all(s == s[:1]):
                H = 0.0
            else:
                try:
                    counts, _ = np.histogram(s, bins=numeric_bins_)
                except Exception:
                    counts, _ = np.histogram(s, bins=32)
                H = _shannon_entropy_from_counts(counts)
            w[col] = 1.0 / (H + eps)
        for col in cat_cols_:
            counts = df[col].value_counts(dropna=False).to_numpy()
            H = _shannon_entropy_from_counts(counts)
            w[col] = 1.0 / (H + eps)
        if normalize_mean_to_one_ and len(w) > 0:
            m = float(np.mean(list(w.values())))
            if m > 0:
                for c in w: w[c] /= m
        if clip_max_ is not None:
            for c in w: w[c] = min(w[c], clip_max_)
        return w

    if weights_by_col is None and compute_inverse_entropy:
        weights_by_col = _inverse_entropy_weights(
            real_df, num_cols, cat_cols,
            numeric_bins_=numeric_bins, eps=1e-12,
            normalize_mean_to_one_=normalize_mean_to_one,
            clip_max_=clip_max_weight
        )
    elif weights_by_col is None:
        weights_by_col = {c: 1.0 for c in list(num_cols) + list(cat_cols)}

    # ---- expand per-column weights to per-dimension weights via cat_maps ----
    def _build_dim_weight_vector(cat_maps_: dict, dim_: int) -> np.ndarray:
        w_dim_ = np.ones(dim_, dtype=np.float32)
        for j, col in enumerate(num_cols):
            w_dim_[j] = float(weights_by_col.get(col, 1.0))
        for col in cat_cols:
            w_col = float(weights_by_col.get(col, 1.0))
            for _, (idx, _scale) in cat_maps_[col].items():
                if 0 <= idx < dim_:
                    w_dim_[idx] = w_col
        return w_dim_

    w_dim   = _build_dim_weight_vector(cat_maps, dim)
    w_scale = np.sqrt(np.clip(w_dim, 0.0, None)).astype(np.float32)

    # ---- scale REAL vectors once (C-contiguous helps sklearn) ----
    Xr_w = np.ascontiguousarray(Xr * w_scale, dtype=np.float32)

    # ---- REAL-to-REAL kNN to get r_i (k-th neighbor radii) ----
    # Use a single kNN index and reuse it everywhere.
    try:
        nn = NearestNeighbors(metric='euclidean', algorithm='auto', n_jobs=-1).fit(Xr_w)
    except TypeError:
        # older sklearn without n_jobs on NearestNeighbors
        nn = NearestNeighbors(metric='euclidean', algorithm='auto').fit(Xr_w)

    k_eff = min(max(k, 2), n_r)
    D_rr, _ = nn.kneighbors(Xr_w, n_neighbors=k_eff)
    r  = D_rr[:, k_eff - 1].astype(np.float32)  # self is at position 0, so k-th includes self at 0
    r2 = r * r
    Rmax  = float(r.max())
    Rmax2 = Rmax * Rmax

    
    def _margins2_batch_knn_only(Xq_raw: np.ndarray, init_neighbors: int = 64, micro_batch: int = 1) -> np.ndarray:
        """
        EXACT ANY-margin with kNN only, but memory-safe:
        - Processes queries one-by-one (or in very small micro-batches)
        - Grows n_neighbors progressively per query until the exact early-stop is satisfied
        - Avoids allocating (num_queries x m) blocks that can trigger OOM

        Returns: array (len(Xq_raw),) of float32 squared margins.
        """
        if Xq_raw.dtype != np.float32:
            Xq_raw = Xq_raw.astype(np.float32)

        # Scale queries once
        Xq = Xq_raw * w_scale
        nq = Xq.shape[0]
        out = np.empty(nq, dtype=np.float32)

        # Process in tiny micro-batches (default: 1) to cap peak memory
        for start in range(0, nq, micro_batch):
            end = min(start + micro_batch, nq)
            for j in range(start, end):
                x = Xq[j : j+1]  # shape (1, d)

                # progressive neighbor growth for THIS query only
                m = min(init_neighbors, n_r)
                best = np.inf
                while True:
                    D, I = nn.kneighbors(x, n_neighbors=m, return_distance=True)
                    # D: (1, m) float64, I: (1, m) int64
                    D = D[0].astype(np.float32)
                    I = I[0]
                    D2 = D * D

                    # candidates only from neighbors within Rmax
                    within = (D <= Rmax)
                    if np.any(within):
                        cand = D2[within] - r2[I[within]]
                        # update best exact value
                        cb = float(cand.min())
                        if cb < best:
                            best = cb
                    else:
                        # if no neighbor within Rmax, exact fallback
                        cb = float(D2[0] - Rmax2)
                        if cb < best:
                            best = cb

                    # EXACT early-stop checks:
                    # 1) last returned neighbor already beyond Rmax -> all radius neighbors seen
                    # 2) best <= d_last^2 - Rmax^2 -> no unseen neighbor can improve min
                    dlast = float(D[-1])
                    dlast2 = float(D2[-1])
                    done = (dlast >= Rmax) or (best <= (dlast2 - Rmax2)) or (m == n_r)

                    if done:
                        out[j] = np.float32(best)
                        break

                    # grow neighbors for THIS query only (keeps allocations small)
                    m = min(n_r, max(m * 2, m + 16))

        return out


    # ---- initial synthetic + margins ----
    synth = model.sample(n_samples).reset_index(drop=True)
    Xs    = heom_to_vec(synth, scaler, cat_maps, num_cols, cat_cols, dim).astype(np.float32)
    R2    = _margins2_batch_knn_only(Xs)
    eps   = float((R2 < 0).mean())

    bar = tqdm(total=float('inf'), unit='swap', leave=False) if verbose else None
    if bar is not None:
        bar.set_description(f"ε_any(k={k_eff}, weighted) = {eps:.4f}  (target < {min_eps})")
    iter_ctr = 0

    # ---- rejection-with-replacement loop (unchanged semantics) ----
    while eps >= min_eps:
        cand = model.sample(1)
        Xc   = heom_to_vec(cand, scaler, cat_maps, num_cols, cat_cols, dim).astype(np.float32)
        rc2  = _margins2_batch_knn_only(Xc)[0]

        widx   = int(np.argmin(R2))  # current worst (smallest margin)
        accept = ((R2[widx] < 0 <= rc2) or (R2[widx] >= 0 and rc2 > R2[widx]))
        if accept:
            synth.iloc[widx] = cand.iloc[0]
            R2[widx]         = rc2
            eps              = float((R2 < 0).mean())
            if bar is not None:
                # Update only occasionally to reduce string/IO overhead
                iter_ctr += 1
                if (iter_ctr & 0xFF) == 0 or eps < min_eps:
                    bar.set_description(f"ε_any(k={k_eff}, weighted) = {eps:.4f}  (target < {min_eps})")

    if bar is not None:
        bar.close()
    return synth.reset_index(drop=True)

    

