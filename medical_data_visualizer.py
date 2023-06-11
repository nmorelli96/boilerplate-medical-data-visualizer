import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv('medical_examination.csv')

# Add 'overweight' column
df['overweight'] = np.where((df['weight'] / ((df['height'] / 100) **2)) > 25, 1, 0)

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.

df['cholesterol'] = np.where(df['cholesterol'] == 1, 0 ,1)
df['gluc'] = np.where(df['gluc'] == 1, 0 ,1)

# Draw Categorical Plot
def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    df_cat = pd.melt(df, value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'], id_vars=['cardio'])


    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the columns for the catplot to work correctly.
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')
    

    # Draw the catplot with 'sns.catplot()'
    catplot = sns.catplot(data=df_cat, x='variable', y='total', hue='value', col='cardio', kind='bar')


    # Get the figure for the output
    catplot.fig.set_size_inches(15,8)
    fig = catplot.fig


    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig


# Draw Heat Map
def draw_heat_map():
    # Clean the data
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) & 
         df['height'].between(df['height'].quantile(0.025), df['height'].quantile(0.975), inclusive='both') & 
         df['weight'].between(df['weight'].quantile(0.025), df['weight'].quantile(0.975), inclusive='both')].copy()

    # Calculate the correlation matrix
    corr = df_heat.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(corr)



    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(15,12))
    ax.set_facecolor("white")
    # Draw the heatmap with 'sns.heatmap()'
    sns.set(font_scale=1.3)
    chart = sns.heatmap(df_heat.corr(), annot=True, mask=mask, fmt=".1f", center = 0.0, 
                cbar_kws={"shrink":0.5, "ticks":[-0.08, 0.00, 0.08, 0.16, 0.24]}, 
                linewidths=1, vmax=0.30, vmin=-0.14
    )
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)


    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig
