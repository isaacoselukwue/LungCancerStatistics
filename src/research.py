import sklearn as sk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

def descriptive_statistics(data):
    """
    Computes comprehensive descriptive statistics for the dataset.
    
    Parameters:
    data (pd.DataFrame): The dataset to analyze
    
    Returns:
    dict: Dictionary containing all descriptive statistics results
    """
    results = {}
    
    continuous_vars = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_vars = data.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    for col in continuous_vars.copy():
        if data[col].nunique() < 10 and data[col].dtype != 'float64':
            categorical_vars.append(col)
            continuous_vars.remove(col)
    
    results['continuous'] = analyze_continuous_variables(data, continuous_vars)
    
    results['categorical'] = analyze_categorical_variables(data, categorical_vars)
    
    correlation_result = correlation_analysis(data)
    results['correlation'] = correlation_result
    
    results['visualisations'] = create_visualisations(data, continuous_vars, categorical_vars, correlation_result)
    
    return results

def analyze_continuous_variables(data, continuous_vars):
    """
    Computes statistics for continuous variables.
    
    Parameters:
    data (pd.DataFrame): The dataset
    continuous_vars (list): List of continuous variable column names
    
    Returns:
    dict: Dictionary with statistics for each continuous variable
    """
    results = {}
    
    for var in continuous_vars:
        var_stats = {
            'mean': data[var].mean(),
            'median': data[var].median(),
            'mode': data[var].mode()[0],
            'std_dev': data[var].std(),
            'variance': data[var].var(),
            'min': data[var].min(),
            'max': data[var].max(),
            'quartiles': [
                data[var].quantile(0.25),
                data[var].quantile(0.50),
                data[var].quantile(0.75)
            ],
            'IQR': data[var].quantile(0.75) - data[var].quantile(0.25)
        }
        results[var] = var_stats
    
    return results

def analyze_categorical_variables(data, categorical_vars):
    """
    Computes statistics for categorical variables.
    
    Parameters:
    data (pd.DataFrame): The dataset
    categorical_vars (list): List of categorical variable column names
    
    Returns:
    dict: Dictionary with statistics for each categorical variable
    """
    results = {}
    
    for var in categorical_vars:
        value_counts = data[var].value_counts()
        percentages = data[var].value_counts(normalize=True) * 100
        
        freq_df = pd.DataFrame({
            'frequency': value_counts,
            'percentage': percentages
        })
        
        mode_value = data[var].mode()[0]
        mode_count = value_counts.max()
        mode_percentage = percentages.max()
        
        results[var] = {
            'frequency_table': freq_df,
            'mode': {
                'value': mode_value,
                'count': mode_count,
                'percentage': mode_percentage
            },
            'unique_values': data[var].nunique()
        }
    
    return results

def create_visualisations(data, continuous_vars, categorical_vars, correlation_result):
    """
    Creates visualisation functions for the dataset.
    
    Parameters:
    data (pd.DataFrame): The dataset
    continuous_vars (list): List of continuous variable columns
    categorical_vars (list): List of categorical variable columns
    
    Returns:
    dict: Dictionary with visualisation functions for each type
    """
    viz_functions = {}
    
    def plot_histograms(columns=None):
        if columns is None:
            columns = continuous_vars
        elif isinstance(columns, str):
            columns = [columns]
        
        n = len(columns)
        fig, axes = plt.subplots(n, 1, figsize=(10, 4*n))
        if n == 1:
            axes = [axes]
            
        for i, col in enumerate(columns):
            axes[i].hist(data[col].dropna(), bins=30, alpha=0.7)
            axes[i].set_title(f'Histogram of {col}')
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Frequency')
        
        plt.tight_layout()
        return fig

    def plot_boxplots(columns=None):
        if columns is None:
            columns = continuous_vars
        elif isinstance(columns, str):
            columns = [columns]
            
        fig, ax = plt.subplots(figsize=(12, 6))
        data[columns].boxplot(ax=ax)
        ax.set_title('Boxplots of Continuous Variables')
        ax.set_ylabel('Value')
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig
    
    def plot_bar_charts(columns=None):
        if columns is None:
            columns = categorical_vars
        elif isinstance(columns, str):
            columns = [columns]
            
        n = len(columns)
        fig, axes = plt.subplots(n, 1, figsize=(10, 4*n))
        if n == 1:
            axes = [axes]
            
        for i, col in enumerate(columns):
            data[col].value_counts().plot(kind='bar', ax=axes[i])
            axes[i].set_title(f'Frequency of {col}')
            axes[i].set_ylabel('Count')
        
        plt.tight_layout()
        return fig
    
    def plot_correlation_heatmap():
        corr_matrix = data[continuous_vars].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
        ax.set_title('Correlation Heatmap')
        plt.tight_layout()
        return fig
    
    def plot_lung_cancer_correlations():
        """Plot correlations between different factors and lung cancer."""
        lung_corrs = correlation_result['lung_cancer_correlations']
        
        vars_list = [x[0] for x in lung_corrs]
        corr_values = [x[1] for x in lung_corrs]
        
        sorted_indices = np.argsort(corr_values)
        sorted_vars = [vars_list[i] for i in sorted_indices]
        sorted_corrs = [corr_values[i] for i in sorted_indices]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        bars = ax.barh(sorted_vars, sorted_corrs)
        
        for i, bar in enumerate(bars):
            corr = sorted_corrs[i]
            if corr > 0:
                bar.set_color('royalblue')
            else:
                bar.set_color('lightcoral')
        
        ax.set_title('Correlations with Lung Cancer')
        ax.set_xlabel('Pearson Correlation Coefficient')
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        for i, v in enumerate(sorted_corrs):
            ax.text(v + (0.01 if v >= 0 else -0.01), 
                    i, 
                    f'{v:.2f}', 
                    color='black', 
                    va='center',
                    ha='left' if v >= 0 else 'right')
        
        plt.tight_layout()
        return fig

    def plot_factor_correlations():
        """Create a focused correlation heatmap for key factors and lung cancer."""
        key_vars = ['LUNG_CANCER', 'SMOKING', 'ALCOHOL_CONSUMING', 'ANXIETY', 'PEER_PRESSURE', 'AGE', 'GENDER']
        
        key_vars = [var for var in key_vars if var in data.columns]
        
        corr_matrix = data[key_vars].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        ax.set_title('Correlations Between Key Factors and Lung Cancer')
        
        plt.tight_layout()
        return fig
    
    viz_functions['histograms'] = plot_histograms
    viz_functions['boxplots'] = plot_boxplots
    viz_functions['bar_charts'] = plot_bar_charts
    viz_functions['correlation_heatmap'] = plot_correlation_heatmap
    viz_functions['lung_cancer_correlations'] = plot_lung_cancer_correlations
    viz_functions['factor_correlations'] = plot_factor_correlations
    
    return viz_functions

def correlation_analysis(data):
    """
    Performs correlation analysis on the dataset.
    
    Parameters:
    data (pd.DataFrame): The dataset
    
    Returns:
    dict: Dictionary with correlation results
    """
    numerical_data = data.select_dtypes(include=['float64', 'int64'])
    
    pearson_corr = numerical_data.corr(method='pearson')
    # Spearman correlation is not needed for this analysis, but included for completeness
    spearman_corr = numerical_data.corr(method='spearman')
    
    corr_pairs = []
    for i in range(len(pearson_corr.columns)):
        for j in range(i):
            col1 = pearson_corr.columns[i]
            col2 = pearson_corr.columns[j]
            corr_value = pearson_corr.iloc[i, j]
            corr_pairs.append((col1, col2, corr_value))
    
    corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    
    top_correlations = corr_pairs[:10] if len(corr_pairs) > 10 else corr_pairs
    
    if 'LUNG_CANCER' in pearson_corr.columns:
        lung_cancer_correlations = []
        for col in pearson_corr.index:
            if col != 'LUNG_CANCER':
                corr_value = pearson_corr.loc[col, 'LUNG_CANCER']
                lung_cancer_correlations.append((col, corr_value))
        
        lung_cancer_correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        
        top_lung_cancer_correlations = lung_cancer_correlations[:10]
        
        psychosocial_correlations = {}
        for factor in ['ANXIETY', 'PEER_PRESSURE']:
            if factor in pearson_corr.columns:
                psychosocial_correlations[factor] = pearson_corr.loc[factor, 'LUNG_CANCER']
        
        behavioral_correlations = {}
        for habit in ['SMOKING', 'ALCOHOL_CONSUMING']:
            if habit in pearson_corr.columns:
                behavioral_correlations[habit] = pearson_corr.loc[habit, 'LUNG_CANCER']
    else:
        lung_cancer_correlations = []
        top_lung_cancer_correlations = []
        psychosocial_correlations = {}
        behavioral_correlations = {}
    
    return {
        'pearson_correlation': pearson_corr,
        'spearman_correlation': spearman_corr,
        'top_correlations': top_correlations,
        'lung_cancer_correlations': lung_cancer_correlations,
        'top_lung_cancer_correlations': top_lung_cancer_correlations,
        'psychosocial_correlations': psychosocial_correlations,
        'behavioral_correlations': behavioral_correlations
    }

def load_and_preprocess_data(file_name):
    dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(dir, file_name)
    data = pd.read_csv(file_path)
    
    print(data.head())
    print(f"Original shape: {data.shape}")
    
    data = data.drop_duplicates()
    print(f"\nAfter dropping duplicates: {data.shape}")
    
    data.columns = data.columns.str.strip()
    data.columns = data.columns.str.replace(' ', '_')
    
    binary_cols = [
        'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 
        'CHRONIC_DISEASE', 'FATIGUE', 'ALLERGY', 'WHEEZING', 
        'ALCOHOL_CONSUMING', 'COUGHING', 'SHORTNESS_OF_BREATH', 
        'SWALLOWING_DIFFICULTY', 'CHEST_PAIN'
    ]
    
    binary_mapping = {1: 0, 2: 1}
    
    for col in binary_cols:
        if col in data.columns:
            data[col] = data[col].map(binary_mapping)
    
    data['LUNG_CANCER'] = data['LUNG_CANCER'].map({'YES': 1, 'NO': 0})
    data['GENDER'] = data['GENDER'].map({'M': 1, 'F': 0})
    
    print("\nPreprocessed data:")
    print(data.head())
    
    print("\nBasic statistics:")
    print(data.describe())
    
    print("\nColumns:")
    print(data.columns)
    
    return data

def lung_cancer_features(data):
    """
    Analyzes lung cancer distribution across various features.
    
    Parameters:
    data (pd.DataFrame): The preprocessed dataset
    
    Returns:
    tuple: (data, cancer_by_feature, psychosocial_dist, behavioral_dist)
    """
    print("\nCancer Distribution:")
    cancer_dist = data['LUNG_CANCER'].value_counts(normalize=True) * 100
    print(f"Lung Cancer (Yes): {cancer_dist[1]:.2f}%")
    print(f"No Lung Cancer (No): {cancer_dist[0]:.2f}%")
    
    print("\nCancer Distribution by Features:")
    cancer_by_feature = {}
    
    binary_cols = [
        'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 
        'CHRONIC_DISEASE', 'FATIGUE', 'ALLERGY', 'WHEEZING', 
        'ALCOHOL_CONSUMING', 'COUGHING', 'SHORTNESS_OF_BREATH', 
        'SWALLOWING_DIFFICULTY', 'CHEST_PAIN'
    ]
    
    for col in binary_cols:
        if col in data.columns:
            feature_cancer_dist = data.groupby(col)['LUNG_CANCER'].mean() * 100
            cancer_by_feature[col] = feature_cancer_dist
            print(f"\n{col}:")
            for val, pct in feature_cancer_dist.items():
                feature_label = "Yes" if val == 1 else "No"
                print(f"  {feature_label}: {pct:.2f}% have lung cancer")
    
    print("\nComplex Cancer Distribution - Psychosocial and Behavioral Factors:")
    
    print("\nPsychosocial Factors and Lung Cancer:")
    
    data['psychosocial'] = data['ANXIETY'].astype(str) + "_" + data['PEER_PRESSURE'].astype(str)
    psychosocial_dist = data.groupby('psychosocial')['LUNG_CANCER'].agg(['count', 'mean'])
    psychosocial_dist['percent'] = psychosocial_dist['mean'] * 100
    psychosocial_dist['group_percent'] = psychosocial_dist['count'] / len(data) * 100
    
    print("\nAnxiety_PeerPressure | Count | % of Total | % with Lung Cancer")
    print("--------------------------------------------------------")
    for group, row in psychosocial_dist.iterrows():
        anxiety = "Yes" if group.split("_")[0] == "1" else "No"
        peer = "Yes" if group.split("_")[1] == "1" else "No"
        print(f"{anxiety}_{peer} | {row['count']} | {row['group_percent']:.2f}% | {row['percent']:.2f}%")
    
    print("\nBehavioral Habits and Lung Cancer:")
    
    data['behavioral'] = data['SMOKING'].astype(str) + "_" + data['ALCOHOL_CONSUMING'].astype(str)
    behavioral_dist = data.groupby('behavioral')['LUNG_CANCER'].agg(['count', 'mean'])
    behavioral_dist['percent'] = behavioral_dist['mean'] * 100
    behavioral_dist['group_percent'] = behavioral_dist['count'] / len(data) * 100
    
    print("\nSmoking_Alcohol | Count | % of Total | % with Lung Cancer")
    print("--------------------------------------------------------")
    for group, row in behavioral_dist.iterrows():
        smoking = "Yes" if group.split("_")[0] == "1" else "No"
        alcohol = "Yes" if group.split("_")[1] == "1" else "No"
        print(f"{smoking}_{alcohol} | {row['count']} | {row['group_percent']:.2f}% | {row['percent']:.2f}%")
    
    return data, cancer_by_feature, psychosocial_dist, behavioral_dist

def plot_cancer_distribution(data):
    """
    Plots the overall distribution of lung cancer cases.
    
    Parameters:
    data (pd.DataFrame): The preprocessed dataset
    
    Returns:
    matplotlib.figure.Figure: The figure containing the plot
    """
    cancer_counts = data['LUNG_CANCER'].value_counts()
    labels = ['No', 'Yes']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie(cancer_counts, labels=labels, autopct='%1.1f%%', 
           colors=['lightblue', 'salmon'], startangle=90)
    ax.set_title('Lung Cancer Distribution')
    plt.tight_layout()
    return fig

def plot_cancer_by_feature(data, feature):
    """
    Plots the relationship between a feature and lung cancer.
    
    Parameters:
    data (pd.DataFrame): The preprocessed dataset
    feature (str): The feature to analyze
    
    Returns:
    matplotlib.figure.Figure: The figure containing the plot
    """
    ct = pd.crosstab(data[feature], data['LUNG_CANCER'])
    ct.columns = ['No Cancer', 'Cancer']
    ct.index = ['No', 'Yes']
    
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    ct.plot(kind='bar', ax=ax[0], color=['lightblue', 'salmon'])
    ax[0].set_title(f'Cancer Count by {feature}')
    ax[0].set_ylabel('Count')
    ax[0].set_xlabel(feature)
    
    ct_pct.plot(kind='bar', ax=ax[1], color=['lightblue', 'salmon'])
    ax[1].set_title(f'Cancer Percentage by {feature}')
    ax[1].set_ylabel('Percentage (%)')
    ax[1].set_xlabel(feature)
    
    for a in ax:
        for container in a.containers:
            a.bar_label(container, fmt='%.1f')
    
    plt.tight_layout()
    return fig

def plot_combined_factors(data, dist_data, factor_type='psychosocial'):
    """
    Plots the relationship between combined factors and lung cancer.
    
    Parameters:
    data (pd.DataFrame): The preprocessed dataset
    factor_type (str): Either 'psychosocial' or 'behavioral'
    
    Returns:
    matplotlib.figure.Figure: The figure containing the plot
    """
    dist = dist_data
    if factor_type == 'psychosocial':
        title = 'Anxiety and Peer Pressure'
    else:
        title = 'Smoking and Alcohol'
    
    new_index = []
    for idx in dist.index:
        parts = idx.split('_')
        if factor_type == 'psychosocial':
            anxiety = "Anxiety" if parts[0] == "1" else "No Anxiety"
            peer = "Peer Pressure" if parts[1] == "1" else "No Peer Pressure"
            new_idx = f"{anxiety}\n{peer}"
        else:
            smoking = "Smoking" if parts[0] == "1" else "No Smoking"
            alcohol = "Alcohol" if parts[1] == "1" else "No Alcohol"
            new_idx = f"{smoking}\n{alcohol}"
        new_index.append(new_idx)
    
    plot_data = dist[['percent']].copy()
    plot_data.index = new_index
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(plot_data.index, plot_data['percent'], color='lightcoral')
    ax.set_ylabel('Percentage with Lung Cancer (%)')
    ax.set_title(f'Lung Cancer by {title}')
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

def cancer_calcs_who():
    lung_cancer = 2210000
    breast_cancer = 2260000
    colon_cancer = 1930000
    prostate_cancer = 1410000
    skin_cancer = 1200000
    stomach_cancer = 1090000
    
    total_cancer = lung_cancer + breast_cancer + colon_cancer + prostate_cancer + skin_cancer + stomach_cancer
    
    lung_cancer_percentage = (lung_cancer / total_cancer) * 100
    return lung_cancer_percentage
    

def main():
    
    # print ("Lung Cancer Percentage: ", cancer_calcs_who())
    data = load_and_preprocess_data('lung_cancer_survey.csv')

    stats_results = descriptive_statistics(data)
    
    print("\nSummary of Continuous Variables:")
    for var, stats in stats_results['continuous'].items():
        print(f"\n{var}:")
        print(f"  Mean: {stats['mean']:.2f}, Median: {stats['median']:.2f}")
        print(f"  Std Dev: {stats['std_dev']:.2f}, Variance: {stats['variance']:.2f}")
    
    print("\nExamples of visualisations can be generated with:")
    print("stats_results['visualisations']['histograms']('AGE')")
    print("stats_results['visualisations']['correlation_heatmap']()")
    
    stats_results['visualisations']['histograms']('AGE')
    stats_results['visualisations']['correlation_heatmap']()
    

if __name__ == "__main__":
    main()