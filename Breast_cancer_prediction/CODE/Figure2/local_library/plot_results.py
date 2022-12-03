#!/usr/bin/env python
# coding: utf-8
"""
Created on Thu Apr 25 08:45:38 2019

@author: anupt

Script to run in the results folder of Agents x Width
Produces Heatmaps and CSVs of results
"""

# %%Import system modules and libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from local_library.initialize_outputs import keywise_template


# %%
def main(
        results, cond_agents, cond_trainEXs,
        model_titles, perf_metrics, class_labels, dist_titles,
        num_train_total, results_folder_relpath
        ):
    # %% Create folder to store Heatmaps and CSVs
    savedir = results_folder_relpath

    # Heatmaps
    savedir_Heatmap = {}
    savedir_Heatmap['mean'] = os.path.join(savedir, 'Heatmap_MEAN')
    savedir_Heatmap['stdev'] = os.path.join(savedir, 'Heatmap_STDEV')
    for t_type in savedir_Heatmap.keys():
        if not os.path.exists(savedir_Heatmap[t_type]):
            os.mkdir(savedir_Heatmap[t_type])

    # CSVs
    savedir_CSV = {}
    savedir_CSV['mean'] = os.path.join(savedir, 'CSV_MEAN')
    savedir_CSV['stdev'] = os.path.join(savedir, 'CSV_STDEV')
    for t_type in savedir_CSV.keys():
        if not os.path.exists(savedir_CSV[t_type]):
            os.mkdir(savedir_CSV[t_type])

    # %% Create arrays to save and plot

    # Initialize
    arrayplot = dict()
    for dist in dist_titles.keys():
        arrayplot[dist] = keywise_template(
                keywise_template(
                        keywise_template([], savedir_Heatmap),
                        perf_metrics
                ),
                class_labels)
    # N.B. savedir_Heatmap is used only for its keys (mean, stdev), so that
    # the deepest nested template uses these keys

    # Initialize dictionary to store the maximum and minimum values
    # for each set of arrays (e.g. macro / F1)
    cbar_lim = dict(
            min=keywise_template(
                    keywise_template(
                            keywise_template(100., savedir_Heatmap),
                            perf_metrics
                    ),
                    class_labels),
            max=keywise_template(
                    keywise_template(
                            keywise_template(-100., savedir_Heatmap),
                            perf_metrics
                    ),
                    class_labels),
                )
    # N.B. savedir_Heatmap is used only for its keys (mean, stdev), so that
    # the deepest nested template uses these keys

    # Get the number of models tested, which is = to number of rows in array
    num_models = len(results.keys())

    # Created array of results, on a class-wise, metric-wise and type-wise
    # basis (i.e. same dict structure that the results are in)
    for j in class_labels.keys():
        for k in perf_metrics.keys():
            for t in savedir_Heatmap.keys():
                for d in arrayplot.keys():
                    for r in results.keys():
                        arrayplot[d][j][k][t].append(
                                results[r][d][j][k][t])
                    arrayplot[d][j][k][t] = np.concatenate(
                            arrayplot[d][j][k][t])
                    arrayplot[d][j][k][t] = np.reshape(
                            arrayplot[d][j][k][t],
                            (num_models, cond_agents['num']))

                array_mins = list()
                array_maxs = list()
                for d in dist_titles.keys():
                    array_mins.append(np.min(arrayplot[d][j][k][t]))
                    array_maxs.append(np.max(arrayplot[d][j][k][t]))
#                agent_min = np.min(arrayplot['agent'][j][k][t])
#                GEC_min = np.min(arrayplot['GEC'][j][k][t])
#                agent_max = np.max(arrayplot['agent'][j][k][t])
#                GEC_max = np.max(arrayplot['GEC'][j][k][t])
                cbar_lim['min'][j][k][t] = np.min(array_mins)
                cbar_lim['max'][j][k][t] = np.max(array_maxs)
    del array_mins, array_maxs

    # %% Create x-axis labels
    xticks = cond_agents['values'].to_list()
    if xticks[0] == 1:
        # Central learning tested within each agent
        xticks[0] = 'Central'

    for i in range(cond_agents['num']):
        xticks[i] = str(xticks[i]) + \
            '\n(' + str(cond_trainEXs['values'][i]) + ')'

    xlabel = cond_agents['name'] + '\n(' + cond_trainEXs['name'] + ')'

    yticks = []
    for d in results.keys():
        yticks.append(model_titles[d])
    ylabel = 'Models'

    # %% Save CSVs

    # Per model type (agent or GEC)
    for d in arrayplot.keys():
        # Per class
        for j in class_labels.keys():
            # Per metric
            for k in perf_metrics:
                # Per output type
                for t in savedir_CSV.keys():
                    csv_tosave = pd.DataFrame(
                            arrayplot[d][j][k][t],
                            index=yticks,
                            columns=xticks
                            )
                    csv_filename = \
                        d + '_' + \
                        k + '_' + \
                        j + '_' + \
                        t.upper() + \
                        '.csv'
                    csv_tosave.to_csv(
                            os.path.join(savedir_CSV[t], csv_filename)
                            )
    # %% Condition matrix
    '''
    Save an image of the condition matrix
    '''
    dummy = np.zeros([num_models, cond_agents['num']])
    fig, ax = plt.subplots(figsize=(16, 6),
                           nrows=1, ncols=1,
                           sharex=False, sharey=False,
                           constrained_layout=True)
    ax = sns.heatmap(dummy, linewidths=2, cbar=False,
                     vmin=-1, vmax=1, cmap="coolwarm")
#    ax.set_aspect('equal')

    ax.set_xticklabels(xticks, fontsize=12, rotation=0)
    ax.set_xlabel(xlabel, fontsize=16, labelpad=10)
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.set_ticks_position('bottom')

    ax.set_yticklabels(yticks, fontsize=12, rotation=0)
#    ax.set_ylabel(ylabel, fontsize=16, labelpad=6)
    ax.yaxis.set_label_position('left')
    ax.yaxis.set_ticks_position('left')

    # Save figure
    plt.savefig(os.path.join(savedir, 'conditions.png'), bbox_inches='tight')

    # Close figures
    plt.clf()
    plt.close('all')

    # %% Plot mean and stdevs
    #####################################################
    '''
    Display heatmaps of the means
    If files are being saved, then save the heatmaps of mean and stdev

    In general, it'll be a Nx2 grid (i.e. no more than 2 wide)
        Ideally, the item of interest is compared along rows
        i.e. Central vs Permutation tested Central (scrambled)

    The solution for having only a single colorbar appear was taken from here:
        https://stackoverflow.com/questions/28356359/one-colorbar-for-seaborn-heatmaps-in-subplot
    Possible alternates:
        - https://stackoverflow.com/questions/43363389/share-axis-and-remove-unused-in-matplotlib-subplots
        - https://stackoverflow.com/questions/13784201/matplotlib-2-subplots-1-colorbar
    '''

    # A total of 12 models have been trained
    # The output will be a 6x2 subplot grid
    # Thus, models will be paired in gropus of two, based on this plot order
    # I would say, put the agent on the left and the distributed on the right

    # Plotting parameters
    plot_height = 16  # Height
    plot_width = 30  # Width
    # Define the colormap. Alt: "RdBu_r"
    colormap = 'coolwarm'
    # Width of line separating instances on heatmap
    gapwidth = 1
    # Annotation font size
    antsz = 14
    # Annotation format
    antfmt = '.2f'
    # Axis label size
    axnamesize = 16
    axlabelsize = 14

    # First, loop through the 2 types of results: mean and stdev
    for t_type in savedir_Heatmap.keys():
        # Second, loop through the classes (Macro average, C0, C1, etc)
        for j_class in class_labels.keys():
            # Third, loop through the metrics (acc, F1, mcc, etc)
            for k_perf in perf_metrics.keys():
                # Create the subplot array
                fig, axs = plt.subplots(figsize=(plot_width, plot_height),
                                        nrows=1, ncols=3,
                                        sharex=False, sharey=False,
                                        constrained_layout=True)

                # Loop through the subplots and output them
                # Left is "agent" and right is "GEC"
                # For a given metric (e.g. F1) in a given class (e.g. Macro)
                # and output type (e.g. mean),
                # using the same color legend (i.e. max and min)
                for i, ax in enumerate(axs.flat):
                    # Use seaborn to create heatmap

                    # If it's the left plot, don't draw the colorbar
                    if i == 0:
                        # Plot the agent on the left
                        dmod = 'agent'
                        sns.heatmap(
                                # The data
                                arrayplot[dmod][j_class][k_perf][t_type],
                                # The axis on the subplot
                                ax=ax,
                                # Set min and max of colorbar (same for all)
                                vmin=cbar_lim['min'][j_class][k_perf][t_type],
                                vmax=cbar_lim['max'][j_class][k_perf][t_type],
                                # Don't draw colorbar for left subplot
                                cbar=False,
                                # Define the colormap. Alt: "RdBu_r"
                                cmap=colormap,
                                # Width of line separating instances on heatmap
                                linewidths=gapwidth,
                                # Annotate with value plotted
                                annot=True,
                                annot_kws={"size": antsz}, fmt=antfmt
                        )
                        # Only show y-axis label on left plot
                        ax.set_ylabel(ylabel, fontsize=axnamesize)

                    # If it's the Middle plot, don't draw colorbar or
                    # the y-axis label
                    elif i == 1:
                        # Plot the GEC on the right
                        dmod = 'GEC_AP'
                        sns.heatmap(
                                # The data
                                arrayplot[dmod][j_class][k_perf][t_type],
                                # The axis on the subplot
                                ax=ax,
                                # Set min and max of colorbar (same for all)
                                vmin=cbar_lim['min'][j_class][k_perf][t_type],
                                vmax=cbar_lim['max'][j_class][k_perf][t_type],
                                # Don't draw colorbar for left subplot
                                cbar=False,
                                # Define the colormap. Alt: "RdBu_r"
                                cmap=colormap,
                                # Width of line separating instances on heatmap
                                linewidths=gapwidth,
                                # Annotate with value plotted
                                annot=True,
                                annot_kws={"size": antsz}, fmt=antfmt
                        )

                    # If it's the right plot, draw the colorbar
                    elif i == 2:
                        # Plot the GEC on the right
                        dmod = 'GEC_MV'
                        sns.heatmap(
                                # The data
                                arrayplot[dmod][j_class][k_perf][t_type],
                                # The axis on the subplot
                                ax=ax,
                                # Set min and max of colorbar (same for all)
                                vmin=cbar_lim['min'][j_class][k_perf][t_type],
                                vmax=cbar_lim['max'][j_class][k_perf][t_type],
                                # Draw colorbar for right subplot
                                cbar=True,
                                # Define the colormap. Alt: "RdBu_r"
                                cmap=colormap,
                                # Width of line separating instances on heatmap
                                # Define colorbar size and aspect ratio
                                cbar_kws={"shrink": 0.9},
                                linewidths=gapwidth,
                                # Annotate with value plotted
                                annot=True,
                                annot_kws={"size": antsz}, fmt=antfmt
                        )

                    # Set subplot title
                    ax.set_title(dist_titles[dmod], fontsize=20)
                    # Make the units on the heatmap squares
                    ax.set_aspect('equal')

                    # Define the x-axis labels for each subplot
                    # Label it based on the values of the conditions
                    ax.set_xlabel(xlabel, fontsize=axnamesize)
                    ax.set_xticklabels(xticks,
                                       fontsize=axlabelsize, rotation=0)
                    ax.xaxis.set_label_position('bottom')  # Labels on bottoms
                    ax.xaxis.set_ticks_position('bottom')  # Ticks on bottom

                    # Define the y-axis labels for each subplot
                    # Using the list of strings created above
                    ax.set_yticklabels(
                            yticks, fontsize=axlabelsize, rotation=0)

                # Master title for subplot
                fig.suptitle(
                        perf_metrics[k_perf] + ' ('+t_type+')\n' +
                        class_labels[j_class]['title'],
                        fontsize=24,
                        y=1.05)

                # Save figure
                plt.savefig(
                    os.path.join(
                            # Folder (different folder for mean and stdev)
                            savedir_Heatmap[t_type],
                            # Filename
                            perf_metrics[k_perf] + \
                            '_' + j_class + '_' + \
                            t_type.upper() + \
                            '.png'
                    ),
                    bbox_inches='tight',  # Reduce white space
                    dpi=150
                )

                # Close figures
                plt.clf()
                plt.close('all')
# %% done
    return

# %% Define some functions
