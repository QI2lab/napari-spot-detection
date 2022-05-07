import numpy as np
from tysserand import tysserand as ty


def compute_distances(source, target, method='xy_z_orthog', dist_fct='euclidian', tilt_vector=None):
    """
    Parameters
    ----------
    source : ndarray
        Coordinates of the first set of points.
    target : ndarray
        Coordinates of the second set of points.
    method : str
        Method used to compute distances. If 'xyz', standard distances are computed considering all axes
        simultaneously. If 'xy_z_orthog' 2 distances are computed, for the xy pkane and along the z axis 
        respectively. If 'xy_z_tilted' 2 distances are computed for the tilted plane and  its normal axis.
    
    Example
    -------
    >>> source = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    >>> target = np.array([[0, 0, 0], [-3, 0, 2], [0, 0, 10]])
    >>> distance(source, target)
        (array([0, 4, 0]), array([0., 2., 5.]))
    >>> distance(source, target, dist_fct='L1')
        (array([0, 4, 0]), array([0, 2, 7]))
    
    """
    if method == 'xyz':
        if dist_fct == 'euclidian':
            dist = np.sqrt(np.sum((source - target)**2, axis=1))
        elif dist_fct == 'L1':
            dist = np.sum((source - target), axis=1)
        else:
            dist = dist_fct(source, target, axis=1)
        return dist
    elif method == 'xy_z_orthog':
        if dist_fct == 'euclidian':
            dist_xy = np.sqrt(np.sum((source[:, 1:] - target[:, 1:])**2, axis=1))
            dist_z = np.abs(source[:, 0] - target[:, 0])
        elif dist_fct == 'L1':
            dist_xy = np.sum(np.abs((source[:, 1:]  - target[:, 1:])), axis=1)
            dist_z = np.abs(source[:, 0] - target[:, 0])
        else:
            dist_xy = dist_fct(source[:, 1:], target[:, 1:], axis=1)
            dist_z = dist_fct(source[:, 0], target[:, 0])
        return dist_z, dist_xy
    elif method == 'xy_z_tilted':
        raise NotImplementedError("Method 'xy_z_tilted' will be implemented soon")


def cut_graph_bidistance(dist_z, dist_xy, max_z, max_xy, pairs=None):
    """
    Apply 2 thresholds on distances, along the z axis and in the xy plane,
    to cut a graph of closest neighbors, i.e. to trim edges.

    Parameters
    ----------
    dist_z : array
        Distances between nodes along the z axis.
    dist_xy : array
        Distances between nodes in the xy plane.
    max_z : float
        Distance threshold along the z axis.
    max_xy : float
        Distance threshold in the xy plane.
    pairs : ndarray, optionnal
        Array of pairs of nodes' indices defining the network, of shape nb_nodes x 2.
        If not None, this array is filtered and returned in addition to the boolean filter.
    
    Returns
    -------
    select : array
        Boolean filter used to select pairs of nodes considered as close to each other.
    filtered_pairs : ndarray
        Filtered array of pairs of nodes' indices that are close to each other.
    """

    select = np.logical_and(dist_z <= max_z, dist_xy  <= max_xy)
    if pairs is not None:
        filtered_pairs = pairs[select, :]
        return select, pairs
    else:
        return select


def find_neighbors(pairs, n):
    """
    Return the list of neighbors of a node in a network defined 
    by edges between pairs of nodes. 
    
    Parameters
    ----------
    pairs : array_like
        Pairs of nodes' id that define the network's edges.
    n : int
        The node for which we look for the neighbors.
        
    Returns
    -------
    neigh : array_like
        The indices of neighboring nodes.
    """
    
    left_neigh = pairs[pairs[:,1] == n, 0]
    right_neigh = pairs[pairs[:,0] == n, 1]
    neigh = np.hstack( (left_neigh, right_neigh) ).flatten()
    
    return neigh


def neighbors_k_order(pairs, n, order):
    """
    Return the list of up the kth neighbors of a node 
    in a network defined by edges between pairs of nodes
    
    Parameters
    ----------
    pairs : array_like
        Pairs of nodes' id that define the network's edges.
    n : int
        The node for which we look for the neighbors.
    order : int
        Max order of neighbors.
        
    Returns
    -------
    all_neigh : list
        The list of lists of 1D array neighbor and the corresponding order
    
    
    Examples
    --------
    >>> pairs = np.array([[0, 10],
                        [0, 20],
                        [0, 30],
                        [10, 110],
                        [10, 210],
                        [10, 310],
                        [20, 120],
                        [20, 220],
                        [20, 320],
                        [30, 130],
                        [30, 230],
                        [30, 330],
                        [10, 20],
                        [20, 30],
                        [30, 10],
                        [310, 120],
                        [320, 130],
                        [330, 110]])
    >>> neighbors_k_order(pairs, 0, 2)
    [[array([0]), 0],
     [array([10, 20, 30]), 1],
     [array([110, 120, 130, 210, 220, 230, 310, 320, 330]), 2]]
    """
    
    # all_neigh stores all the unique neighbors and their oder
    all_neigh = [[np.array([n]), 0]]
    unique_neigh = np.array([n])
    
    for k in range(order):
        # detected neighbor nodes at the previous order
        last_neigh = all_neigh[k][0]
        k_neigh = []
        for node in last_neigh:
            # aggregate arrays of neighbors for each previous order neighbor
            neigh = np.unique(find_neighbors(pairs, node))
            k_neigh.append(neigh)
        # aggregate all unique kth order neighbors
        if len(k_neigh) > 0:
            k_unique_neigh = np.unique(np.concatenate(k_neigh, axis=0))
            # select the kth order neighbors that have never been detected in previous orders
            keep_neigh = np.in1d(k_unique_neigh, unique_neigh, invert=True)
            k_unique_neigh = k_unique_neigh[keep_neigh]
            # register the kth order unique neighbors along with their order
            all_neigh.append([k_unique_neigh, k+1])
            # update array of unique detected neighbors
            unique_neigh = np.concatenate([unique_neigh, k_unique_neigh], axis=0)
        else:
            break
        
    return all_neigh


def flatten_neighbors(all_neigh):
    """
    Convert the list of neighbors 1D arrays with their order into
    a single 1D array of neighbors.

    Parameters
    ----------
    all_neigh : list
        The list of lists of 1D array neighbor and the corresponding order.

    Returns
    -------
    flat_neigh : array_like
        The indices of neighboring nodes.
        
    Examples
    --------
    >>> all_neigh = [[np.array([0]), 0],
                     [np.array([10, 20, 30]), 1],
                     [np.array([110, 120, 130, 210, 220, 230, 310, 320, 330]), 2]]
    >>> flatten_neighbors(all_neigh)
    array([  0,  10,  20,  30, 110, 120, 130, 210, 220, 230, 310, 320, 330])
        
    Notes
    -----
    Code from the mosna library https://github.com/AlexCoul/mosna
    """
    
    list_neigh = []
    for neigh, order in all_neigh:
        list_neigh.append(neigh)
    flat_neigh = np.concatenate(list_neigh, axis=0)

    return flat_neigh


def merge_nodes(coords, weight):
    """
    Merge nodes coordinates by averaging them.

    Parameters
    ----------
    coords : ndarray
        Coordinates of nodes, array of shape nb_nodes x 3.
    weight : array
        Weight of nodes for coordinates averaging, 
        array fo shape nb_nodes x 1.

    Returns
    -------
    merged_coords : ndarray
        Coordinates of merged nodes.
    
    Examples
    --------
    >>> coords = np.array([[0, 0, 0], [2, -4, 8]])
    >>> weight = np.array([1, 1]).reshape((len(coords), -1))
    >>> merge_nodes(coords, weight)
    array([ 1., -2.,  4.])
    """

    tot_weight = weight.sum()
    merged_coords = np.sum(coords * weight, axis=0) / weight.sum()
    return merged_coords


def merge_cluster_nodes(coords, pairs, weights=None, split_big_clust=False, cluster_size=None):
    """
    Merge nodes that are in the same connected cluster, for all cluster in a graph.

    Parameters
    ----------
    coords : ndarray
        Coordinates of nodes, array of shape nb_nodes x 3.
    pairs : ndarray
        Array of pairs of nodes' indices defining the network, of shape nb_nodes x 2.
    weight : array
        Weight of nodes for coordinates averaging. The image intensity at nodes
        coordinates can be used as weights.

    Returns
    -------
    merged_coords : ndarray
        Coordinates of merged nodes.
    """

    nb_nodes = len(coords)
    if weights is None:
        weights = np.ones(nb_nodes)
    # make list of nodes indices on which we will iterate
    iter_nodes = np.arange(nb_nodes)
    # variable storing new merged coordinates
    merged_coords = []
    # for each node, detect all its connected neighbors, even indirectly
    for i in np.arange(nb_nodes):
        # check if we have processed all nodes
        if i >= len(iter_nodes):
            break
        else:
            node_id = iter_nodes[i]
            detected_neighbors = flatten_neighbors(neighbors_k_order(pairs, node_id, nb_nodes))
            # delete these coordinates nodes indices to avoid reprocessing the same neighbors
            select = np.isin(iter_nodes, detected_neighbors, assume_unique=True, invert=True)
            iter_nodes = iter_nodes[select]
            # merge nodes coordinates
            if len(detected_neighbors) == 1:
                merged_coords.append(coords[node_id])
            else:
                # detect if cluster likely contains multiple spots
                if split_big_clust:
                    if cluster_size is None:
                        raise ValueError("`cluster_size` has to be given to split big clusters")
                    # work on it latter, for now use small distance thresholds
                    # actually merge peaks
                    cluster_coords = merge_nodes(coords[detected_neighbors], 
                                                 weights[detected_neighbors].reshape(-1, 1))
                else:
                    cluster_coords = merge_nodes(coords[detected_neighbors], 
                                                 weights[detected_neighbors].reshape(-1, 1))
                merged_coords.append(cluster_coords)
    merged_coords = np.vstack(merged_coords)
    return merged_coords


def filter_nearby_peaks(coords, max_z, max_xy, weight_img=None,
                        split_big_clust=False, cluster_size=None):
    """
    Merge nearby peaks in an image by building a radial distance graph and cutting it given
    distance thresholds in the xy plane and along the z axis.

    Parameters
    ----------
    coords : ndarray
        Coordinates of nodes, array of shape nb_nodes x 3.
    max_z : float
        Distance threshold along the z axis.
    max_xy : float
        Distance threshold in the xy plane.
    weight_img : ndarray
        Image used to find peaks, now used to weight peaks coordinates during merge.
        If None, equal weight is given to peaks coordinates.
    split_big_clust : bool
        If True, cluster big enough to contain multiple objects of interest (like spots)
        are split into sub-clusters.
    cluster_size : list | array
        The threshold z and x/y size of clusters above which they are split.
    
    Returns
    -------
    merged_coords : ndarray
        The coordinates of merged peaks.
    """

    # build the radial distance network using the bigest radius: max distance along z axis
    pairs = ty.build_rdn(coords=coords, r=max_z)
    source = coords[pairs[:, 0]]
    target = coords[pairs[:, 1]]
    # compute the 2 distances arrays
    dist_z, dist_xy = compute_distances(source, target)
    # perform grph cut from the 2 distance thresholds
    _, pairs = cut_graph_bidistance(dist_z, dist_xy, max_z, max_xy, pairs=pairs)

    if weight_img is not None:
        # need ravel_multi_index to get pixel values of weight_img at several 3D coordinates
        amplitudes_id = np.ravel_multi_index(coords.transpose(), weight_img.shape)
        weights = weight_img.ravel()[amplitudes_id]
    else:
        weights = None  # array of ones will be generated in merge_cluster_nodes
    # merge nearby nodes coordinates
    merged_coords = merge_cluster_nodes(coords, pairs, weights,
                                        split_big_clust=split_big_clust, 
                                        cluster_size=cluster_size)

    return merged_coords
