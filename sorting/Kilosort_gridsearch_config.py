from sklearn.model_selection import ParameterGrid


# Define the parameter grid to be searched over using ops variables from Kilosort_config_3.m
# All parameter combinations are tried, with lower parameters being fully explored first
def get_KS_params_grid():
    grid = dict(
        # Th=[[12, 6], [10, 4], [8, 4], [7, 3], [6, 3], [4, 2], [2, 1], [1, 0.5]],
        # Th=[[10, 4], [5, 2], [2, 1], [1, 0.5]],
        # Th=[[3, 1.5], [2, 1], [1.5, 0.75], [1, 0.5]],
        Th=[[10, 4], [7, 3], [5, 2.5], [4, 2], [3, 1.5], [2, 1], [1.5, 0.75], [1, 0.5]]
        # Th=[[10, 2], [9, 2], [8, 2], [7, 2], [6, 2], [5, 2], [4, 2]],
        # long_range=[[30, 3], [30, 1]],
        # spkTh=[[-3, -6, -9], [-6]],
        # lam=[10, 15],
        # nfilt_factor=[12, 4],  # [1,4,16],
        # AUCsplit=[0.8, 0.9], #,0.95,0.99],
        # momentum=[[20, 400], [60, 600]],
    )
    return ParameterGrid(grid)
