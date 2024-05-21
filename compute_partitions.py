import numpy as np

BINARY_SEARCH_LOW_DIM = 1
BINARY_SEARCH_HIGH_DIM = 1000


def compute_hidden_dims(total_param_number, global_weights_factor, in_dim, output_dim, hidden_layers, groups):
    """
    Compute the dimensions of the hidden layers for a neural network.

    Parameters:
    total_param_number (int): The requested total number of parameters in the network.
    global_weights_factor (float): The global weights ratio to be approximated in the network.
    in_dim (int): The dimension of the input.
    output_dim (int): The dimension of the output.
    hidden_layers (int): The number of hidden layers in the network.
    groups (int): The number of partitions in the signal.

    Returns:
    tuple: The dimensions of the local and global hidden layers.
    """
    global_hidden_dim, global_weights_number = compute_global_hidden_dim(total_param_number, global_weights_factor,
                                                                         in_dim, hidden_layers)
    target_param_number = (total_param_number - global_weights_number) * 0.95
    print(f"Looking for {target_param_number:.1f} local weights")

    local_hidden_dim, local_weights_number = compute_local_hidden_dim(in_dim, output_dim, hidden_layers, groups, target_param_number)

    current_global_weight_factor = global_weights_number / total_param_number
    global_hidden_dim, global_weights_number = adjust_global_weights(in_dim, hidden_layers, total_param_number, global_weights_factor, global_hidden_dim, global_weights_number, local_weights_number, current_global_weight_factor)

    return local_hidden_dim, global_hidden_dim


def compute_global_hidden_dim(total_param_number, global_weights_factor, in_dim, hidden_layers):
    """
    Compute the dimension of the global hidden layer.

    Returns:
    tuple: The dimension of the global hidden layer and the number of global weights.
    """
    target_param_number = total_param_number * global_weights_factor
    global_hidden_dim = binary_search(in_dim, hidden_layers, target_param_number, compute_global_weights)
    global_weights_count = compute_global_weights(in_dim, global_hidden_dim, hidden_layers)

    return global_hidden_dim, global_weights_count


def compute_local_hidden_dim(in_dim, output_dim, hidden_layers, groups, target_param_number):
    """
    Compute the dimension of the local hidden layer.

    tuple: The dimension of the local hidden layer and the number of local weights.
    """
    local_hidden_dim = binary_search(in_dim, hidden_layers, target_param_number, compute_local_weights, output_dim=output_dim, groups=groups)
    local_weights_number = compute_local_weights(in_dim=in_dim, output_dim=output_dim, hidden_dim=local_hidden_dim, hidden_layers=hidden_layers, groups=groups)

    return local_hidden_dim, local_weights_number


def adjust_global_weights(in_dim, hidden_layers, total_param_number, global_weights_factor, global_hidden_dim, global_weights_number, local_weights_number, current_global_weight_factor):
    """
    Adjust the number of global weights based on the current number of local and global weights.

    Returns:
    tuple: The adjusted dimension of the global hidden layer and the number of global weights.
    """
    while (global_weights_number + local_weights_number > 0.99 * total_param_number) and (np.abs(current_global_weight_factor - global_weights_factor) < 0.25 * global_weights_factor):
        global_hidden_dim -= 1
        global_weights_number = compute_global_weights(in_dim, global_hidden_dim, hidden_layers)
        current_global_weight_factor = global_weights_number / total_param_number

    while (global_weights_number + local_weights_number < 0.99 * total_param_number) and (np.abs(current_global_weight_factor - global_weights_factor) < 0.25 * global_weights_factor):
        global_hidden_dim += 1
        global_weights_number = compute_global_weights(in_dim, global_hidden_dim, hidden_layers)
        current_global_weight_factor = global_weights_number / total_param_number

    return global_hidden_dim, global_weights_number


def binary_search(in_dim, hidden_layers, target_param_number, weight_func, **kwargs):
    """
    Perform binary search to find the optimal dimension for the hidden layer.
    """
    low = BINARY_SEARCH_LOW_DIM
    high = BINARY_SEARCH_HIGH_DIM
    while low < high:
        mid = (low + high) // 2
        if weight_func(in_dim=in_dim, hidden_dim=mid, hidden_layers=hidden_layers, **kwargs) < target_param_number:
            low = mid + 1
        else:
            high = mid

    return low


def compute_global_weights(in_dim, hidden_dim, hidden_layers):
    return (in_dim * hidden_dim) + (hidden_layers * (hidden_dim ** 2)) + (hidden_layers * hidden_dim) + hidden_dim


def compute_local_weights(in_dim, output_dim, hidden_dim, hidden_layers, groups):
    return (groups * in_dim * hidden_dim) + (hidden_layers * groups * (hidden_dim ** 2)) + (
                groups * output_dim * hidden_dim) + (groups * (hidden_layers + 1) * hidden_dim) + (groups * output_dim)
