import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scanpy as sc
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
from scipy.stats import ttest_ind, mannwhitneyu, ranksums

def add_spatial_coordinates(adata, csv_path):
    # Read the CSV file containing cell names and coordinates
    
    # Example usage:
    #adata_with_spatial = add_spatial_coordinates(adata, '/fs/cbsuvlaminck2/workdir/in68/Utils/Curio_B/A0018_012_BeadBarcodes.txt')
    spatial_df = pd.read_csv(csv_path, sep='\t', index_col=0, names=['x', 'y'])

    # Merge the spatial information with the original AnnData object
    adata_spatial = pd.merge(adata.obs, spatial_df, left_index=True, right_index=True, how='left')

    # Add NaN for cells without coordinates
    adata_spatial[['x', 'y']] = adata_spatial[['x', 'y']].where(pd.notna(adata_spatial[['x', 'y']]), np.nan)

    # Print a message about the number of cells with added coordinates
    print(f"Added spatial coordinates for {adata_spatial[['x', 'y']].count().min()} cells.")

    # Add the 'spatial' component to the AnnData object
    adata.obsm['spatial'] = adata_spatial[['x', 'y']].values
    nan_indices = np.isnan(adata.obsm['spatial']).any(axis=1)

    # Drop the instances where any element is nan
    adata = adata[~nan_indices]

    return adata

def calculate_celltype_neighbors(adata, k_neighbors, celltype_label,cell_obs):
    """
    Calculate the count of K nearest neighbors of each spot that belong to a specific cell type category 
    and update adata.obs.

    Parameters
    ----------
    adata : anndata.AnnData
        The annotated data matrix.
    k_neighbors : int
        The number of nearest neighbors to consider.
    celltype_label : str
        The cell type label to count as neighbors.
    
    Returns
    -------
    None
    
    -------
    ##E.g     for celltype in np.unique(adata.obs['max_pred_celltype']):
        # adata = calculate_celltype_neighbors(adata, 300, celltype)
    """
    # Get coordinates of spots
    coordinates = adata.obsm['spatial']

    # Initialize KNN model
    knn = NearestNeighbors(n_neighbors=k_neighbors+1, algorithm='auto')  # k+1 to exclude the spot itself
    knn.fit(coordinates)

    # Initialize a list to store neighbor counts for the specified cell type
    neighbor_counts = []
    mean_neighbors = []

    # Iterate over each spot
    for spot_coord in coordinates:
        # Find K nearest neighbors (excluding the spot itself)
        _, indices = knn.kneighbors([spot_coord])

        # Get the cell type labels of the neighbors
        neighbor_cell_types = adata.obs.iloc[indices[0][1:]][cell_obs]

        # Count the occurrences of the specified cell type label
        celltype_count = (neighbor_cell_types == celltype_label).sum()
        neighbor_counts.append(celltype_count)


    # Update adata.obs with the neighbor counts 
    adata.obs[f'{celltype_label}_neighbors'] = neighbor_counts

    return adata

# Function to compare datasets
def compare_datasets(data1, data2, n_comparisons=1):
    t_stat, t_p_value = ttest_ind(data1, data2, equal_var=False)
    mw_stat, mw_p_value = mannwhitneyu(data1, data2)
    wr_stat, wr_p_value = ranksums(data1, data2)

    t_p_value_c = t_p_value * n_comparisons
    mw_p_value_c = mw_p_value * n_comparisons
    wr_p_value_c = wr_p_value * n_comparisons

    formatted_t_p_value = "{:.2e}".format(t_p_value_c)
    formatted_mw_p_value = "{:.2e}".format(mw_p_value_c)
    formatted_wr_p_value = "{:.2e}".format(wr_p_value_c)

    results = {
        'T-Test P-Value': formatted_t_p_value,
        'Mann-Whitney U P-Value': formatted_mw_p_value,
        'Wilcoxon Rank-Sum P-Value': formatted_wr_p_value,
        'Mean Data1': np.mean(data1),
        'Mean Data2': np.mean(data2),
        'Median Data1': np.median(data1),
        'Median Data2': np.median(data2)
    }

    return results

# Function to plot box plots
def plot_boxplot_for_arrays(array1, array2, save_path=None, array_1='Young', array_2='Geriatric'):
    concatenated_array = np.concatenate([array1, array2])
    labels = [array_1] * len(array1) + [array_2] * len(array2)

    data = {'Value': concatenated_array, 'Array': labels}
    df = pd.DataFrame(data)

    plt.figure(figsize=(1.5, 3))
    sns.boxplot(x='Array', y='Value', data=df)
    plt.ylabel('Senescence score')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

# Function to extract data
def extract_data(adata, condition, age, location):
    df = adata[condition].obs[['Senescence_score']].copy()
    df['cell_name'] = adata[condition].obs_names
    df['age'] = age
    df['location'] = location
    return df


def plot_spatial_data(adata_dict, obs, titles=None, crop=None, cmap=None, vmin=None, vmax=None, spot_size=15, scale_bar_length=500, scale_bar_label='500 um', output_file=None):
    """
    Plot spatial data with options for different observations, custom colormaps, and zoom-in capabilities.
    Legend is only being plot for Category obs in a similar way on how squidpy/scanpy is handling color associations (accessing uns and serially asigning colors to obs)
    Parameters:
    - adata_dict: Dictionary of AnnData objects to plot.
    - obs: List of observations to plot.
    - titles: List of titles for each subplot.
    - crop: Dictionary of coordinates to crop for each subplot.
    - cmap: Colormap for the plots.
    - vmin, vmax: Min and max values for colormap normalization.
    - spot_size: Size of the spots in the plot.
    - scale_bar_length: Length of the scale bar.
    - scale_bar_label: Label for the scale bar.
    - output_file: File name to save the figure (default is None, which does not save).
    """

    if not isinstance(obs, list):
        obs = [obs] * len(adata_dict)

    num_samples = len(adata_dict)
    num_obs = len(obs)
    nums = num_samples  # number of columns
    num_rows = num_obs  # number of rows

    # Create the figure with subplots
    fig, axs = plt.subplots(num_rows, nums, figsize=(6.4 * nums, 3.2 * num_rows), squeeze=False)

    # Define font properties for the scale bar
    fontprops = fm.FontProperties(size=7)

    for row in range(num_rows):
        for col, (key, adata) in enumerate(adata_dict.items()):
            partaxs = axs[row, col]

            # Plot the spatial data
            sc.pl.spatial(adata, color=obs[row], wspace=0.0, hspace=2.0, cmap=cmap, vmin=vmin, vmax=vmax,
                          spot_size=spot_size, frameon=False, show=False, ax=partaxs, legend_loc=None,
                          crop_coord=crop[col] if crop and col in crop else None)

            # Set the title for the plot
            if titles and col < len(titles):
                partaxs.set_title(titles[col])
            else:
                partaxs.set_title(key)

            # Add a scale bar
            scalebar = AnchoredSizeBar(partaxs.transData, scale_bar_length, scale_bar_label, 'lower right',
                                       pad=1, color='black', frameon=False, size_vertical=2,
                                       fontproperties=fontprops)
            partaxs.add_artist(scalebar)

    # Hide unused axes if there are any
    for ax in axs.flat[num_obs * num_samples:]:
        ax.set_visible(False)

    # Adjust layout
    plt.subplots_adjust(wspace=0.1, hspace=0.3)

    # Save the figure if output_file is specified
    if output_file:
        plt.savefig(output_file, format='pdf', bbox_inches='tight', dpi=300)

    # Show the figure
    plt.show()

    # Create separate legend figures for each categorical observation
    for row, observation in enumerate(obs):
        adata = list(adata_dict.values())[0]  # Assuming all adata have the same categories and colors
        if adata.obs[observation].dtype.name == 'category':
            fig_legend, ax_legend = plt.subplots(figsize=(12, 2))  # Adjust figsize as needed

            # Extract cell types and their corresponding colors from the AnnData object
            cell_types = adata.obs[observation].cat.categories
            colors = adata.uns[f'{observation}_colors']

            # Ensure the number of colors matches the number of cell types
            assert len(cell_types) == len(colors), "The number of colors must match the number of cell types."

            # Create patches for each cell type and add them to the legend
            patches = [mpatches.Patch(color=colors[i], label=cell_type) for i, cell_type in enumerate(cell_types)]

            # Add the legend to the axis
            ax_legend.legend(handles=patches, loc='center', frameon=False, ncol=len(patches)//2)

            # Turn off the axis
            ax_legend.axis('off')

            # Show the legend figure
            plt.show()

            # Save the legend figure if output_file is specified
            if output_file:
                legend_output_file = output_file.replace('.pdf', f'_{observation}_legend.pdf')
                fig_legend.savefig(legend_output_file, format='pdf', bbox_inches='tight', dpi=300)


