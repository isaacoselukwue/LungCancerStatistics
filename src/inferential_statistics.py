import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class inferential_statistics:
    def logistic_regression_analysis(data):
        """
        Performs logistic regression analysis on lung cancer data with interaction terms
        and moderation effects.
        
        Parameters:
        data (pd.DataFrame): The preprocessed dataset
        
        Returns:
        tuple: (model, model_results, interaction_results, visualisation_functions)
        """
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
        from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
        from scipy import stats
        
        print("\n Logistic Regression Analysis with Interaction Terms")
        
        model_data = data.copy()
        
        model_data['LUNG_CANCER'] = model_data['LUNG_CANCER'].astype(int)
        
        print("\n Creating Interaction Terms")
        model_data['SMOKING_ANXIETY'] = model_data['SMOKING'] * model_data['ANXIETY']
        model_data['SMOKING_PEER_PRESSURE'] = model_data['SMOKING'] * model_data['PEER_PRESSURE']
        model_data['ALCOHOL_ANXIETY'] = model_data['ALCOHOL_CONSUMING'] * model_data['ANXIETY']
        model_data['ALCOHOL_PEER_PRESSURE'] = model_data['ALCOHOL_CONSUMING'] * model_data['PEER_PRESSURE']
        
        model_data['AGE_C'] = model_data['AGE'] - model_data['AGE'].mean()
        model_data['AGE_SMOKING'] = model_data['AGE_C'] * model_data['SMOKING']
        model_data['AGE_ALCOHOL'] = model_data['AGE_C'] * model_data['ALCOHOL_CONSUMING']
        model_data['AGE_ANXIETY'] = model_data['AGE_C'] * model_data['ANXIETY']
        model_data['AGE_PEER_PRESSURE'] = model_data['AGE_C'] * model_data['PEER_PRESSURE']
        
        model_data['GENDER_SMOKING'] = model_data['GENDER'] * model_data['SMOKING']
        model_data['GENDER_ALCOHOL'] = model_data['GENDER'] * model_data['ALCOHOL_CONSUMING']
        model_data['GENDER_ANXIETY'] = model_data['GENDER'] * model_data['ANXIETY']
        model_data['GENDER_PEER_PRESSURE'] = model_data['GENDER'] * model_data['PEER_PRESSURE']
        
        formula_base = "LUNG_CANCER ~ AGE_C + GENDER + SMOKING + ALCOHOL_CONSUMING + ANXIETY + PEER_PRESSURE"
        
        formula_psychosocial = (formula_base + 
                            " + SMOKING_ANXIETY + SMOKING_PEER_PRESSURE + " + 
                            "ALCOHOL_ANXIETY + ALCOHOL_PEER_PRESSURE")
        
        formula_full = (formula_psychosocial + 
                    " + AGE_SMOKING + AGE_ALCOHOL + AGE_ANXIETY + AGE_PEER_PRESSURE + " +
                    "GENDER_SMOKING + GENDER_ALCOHOL + GENDER_ANXIETY + GENDER_PEER_PRESSURE")
        
        print("\n Fitting Logistic Regression Models")
        model_base = smf.logit(formula=formula_base, data=model_data).fit(disp=0)
        model_psychosocial = smf.logit(formula=formula_psychosocial, data=model_data).fit(disp=0)
        model_full = smf.logit(formula=formula_full, data=model_data).fit(disp=0)
        
        print("\n Likelihood Ratio Tests")
        lr_test_psychosocial = stats.chi2.sf(
            -2 * (model_base.llf - model_psychosocial.llf), 
            df=model_psychosocial.df_model - model_base.df_model
        )
        
        lr_test_full = stats.chi2.sf(
            -2 * (model_psychosocial.llf - model_full.llf), 
            df=model_full.df_model - model_psychosocial.df_model
        )
        
        print(f"Base vs Psychosocial Interactions: Chi2={-2*(model_base.llf-model_psychosocial.llf):.2f}, df={model_psychosocial.df_model-model_base.df_model}, p-value={lr_test_psychosocial:.4f}")
        print(f"Psychosocial vs Full Model: Chi2={-2*(model_psychosocial.llf-model_full.llf):.2f}, df={model_full.df_model-model_psychosocial.df_model}, p-value={lr_test_full:.4f}")
        
        if lr_test_full < 0.05:
            final_model = model_full
            print("\nSelected Model: Full model with all interactions and moderations")
        elif lr_test_psychosocial < 0.05:
            final_model = model_psychosocial
            print("\nSelected Model: Model with psychosocial interactions")
        else:
            final_model = model_base
            print("\nSelected Model: Base model (no significant improvement with interactions)")
        
        model_data['predicted_prob'] = final_model.predict()
        model_data['predicted_class'] = (model_data['predicted_prob'] > 0.5).astype(int)
        
        conf_matrix = confusion_matrix(model_data['LUNG_CANCER'], model_data['predicted_class'])
        
        fpr, tpr, thresholds = roc_curve(model_data['LUNG_CANCER'], model_data['predicted_prob'])
        roc_auc = auc(fpr, tpr)
        
        class_report = classification_report(model_data['LUNG_CANCER'], model_data['predicted_class'])
        
        interaction_vars = [var for var in final_model.params.index if '_' in var and var not in ['AGE_C']]
        interaction_results = pd.DataFrame({
            'Variable': interaction_vars,
            'Coefficient': [final_model.params[var] for var in interaction_vars],
            'Odds Ratio': [np.exp(final_model.params[var]) for var in interaction_vars],
            'Std Error': [final_model.bse[var] for var in interaction_vars],
            'z-value': [final_model.tvalues[var] for var in interaction_vars],
            'p-value': [final_model.pvalues[var] for var in interaction_vars]
        })
        interaction_results = interaction_results.sort_values('p-value')
        
        print("\n Model Summary")
        print(f"Log-Likelihood: {final_model.llf:.2f}")
        print(f"AIC: {final_model.aic:.2f}")
        print(f"BIC: {final_model.bic:.2f}")
        print(f"Pseudo R-squared: {final_model.prsquared:.4f}")
        
        print("\n Key Coefficients and Wald Tests")
        print(final_model.summary().tables[1])
        
        print("\n Classification Report")
        print(class_report)
        
        print("\n Significant Interactions")
        print(interaction_results[interaction_results['p-value'] < 0.05])
        
        viz_functions = {}
        
        # 1. Coefficient plot
        def plot_coefficients():
            params = final_model.params[1:]
            conf = final_model.conf_int()
            conf.columns = ['Lower', 'Upper']
            conf_intervals = conf.loc[params.index]
            
            coef_df = pd.DataFrame({
                'Coefficient': params,
                'Lower': conf_intervals['Lower'],
                'Upper': conf_intervals['Upper']
            })
            
            if len(coef_df) > 10:
                main_effects = ['AGE_C', 'GENDER', 'SMOKING', 'ALCOHOL_CONSUMING', 'ANXIETY', 'PEER_PRESSURE']
                sig_interactions = interaction_results[interaction_results['p-value'] < 0.05].Variable.tolist()
                important_vars = main_effects + sig_interactions
                important_vars = [var for var in important_vars if var in coef_df.index]
                coef_df = coef_df.loc[important_vars]
            
            coef_df = coef_df.reindex(coef_df['Coefficient'].abs().sort_values(ascending=False).index)
            
            fig, ax = plt.subplots(figsize=(12, len(coef_df) * 0.5 + 2))
            for i, (idx, row) in enumerate(coef_df.iterrows()):
                color = 'red' if row['Coefficient'] < 0 else 'blue'
                ax.errorbar(
                    row['Coefficient'], 
                    i, 
                    xerr=[[row['Coefficient'] - row['Lower']], [row['Upper'] - row['Coefficient']]],
                    fmt='o', 
                    color=color,
                    ecolor='black',
                    capsize=3
                )
                ax.text(0, i, f" {idx}", va='center', ha='right' if row['Coefficient'] > 0 else 'left', fontsize=9)
            
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            
            ax.set_yticks(range(len(coef_df)))
            ax.set_yticklabels([''] * len(coef_df))
            ax.set_xlabel('Coefficient Value (Log Odds)')
            ax.set_title('Logistic Regression Coefficients with 95% Confidence Intervals')
            
            plt.tight_layout()
            return fig
        
        # 2. ROC curve
        def plot_roc_curve():
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Receiver Operating Characteristic (ROC) Curve')
            ax.legend(loc="lower right")
            plt.tight_layout()
            return fig
        
        # 3. Confusion matrix
        def plot_confusion_matrix():
            fig, ax = plt.subplots(figsize=(8, 8))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
            ax.set_title('Confusion Matrix')
            ax.set_xticklabels(['No Cancer', 'Cancer'])
            ax.set_yticklabels(['No Cancer', 'Cancer'])
            plt.tight_layout()
            return fig
        
        # 4. Interaction effects plot
        def plot_interaction_effects():
            if len(interaction_results) > 0:
                most_sig_interaction = interaction_results.iloc[0]['Variable']
                parts = most_sig_interaction.split('_')
                
                if 'SMOKING' in most_sig_interaction and 'ANXIETY' in most_sig_interaction:
                    var1, var2 = 'SMOKING', 'ANXIETY'
                elif 'SMOKING' in most_sig_interaction and 'PEER_PRESSURE' in most_sig_interaction:
                    var1, var2 = 'SMOKING', 'PEER_PRESSURE'
                elif 'ALCOHOL' in most_sig_interaction and 'ANXIETY' in most_sig_interaction:
                    var1, var2 = 'ALCOHOL_CONSUMING', 'ANXIETY'
                elif 'ALCOHOL' in most_sig_interaction and 'PEER_PRESSURE' in most_sig_interaction:
                    var1, var2 = 'ALCOHOL_CONSUMING', 'PEER_PRESSURE'
                elif 'AGE' in most_sig_interaction:
                    if 'SMOKING' in most_sig_interaction:
                        var1, var2 = 'AGE_C', 'SMOKING'
                    elif 'ALCOHOL' in most_sig_interaction:
                        var1, var2 = 'AGE_C', 'ALCOHOL CONSUMING'
                    elif 'ANXIETY' in most_sig_interaction:
                        var1, var2 = 'AGE_C', 'ANXIETY'
                    else:
                        var1, var2 = 'AGE_C', 'PEER_PRESSURE'
                elif 'GENDER' in most_sig_interaction:
                    if 'SMOKING' in most_sig_interaction:
                        var1, var2 = 'GENDER', 'SMOKING'
                    elif 'ALCOHOL' in most_sig_interaction:
                        var1, var2 = 'GENDER', 'ALCOHOL_CONSUMING'
                    elif 'ANXIETY' in most_sig_interaction:
                        var1, var2 = 'GENDER', 'ANXIETY'
                    else:
                        var1, var2 = 'GENDER', 'PEER_PRESSURE'
                else:
                    var1, var2 = 'SMOKING', 'ANXIETY'
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                if var1 != 'AGE_C' and var2 != 'AGE_C':
                    interaction_data = pd.crosstab(
                        index=model_data[var1], 
                        columns=model_data[var2], 
                        values=model_data['predicted_prob'], 
                        aggfunc='mean'
                    )
                    
                    interaction_data.plot(kind='bar', ax=ax)
                    ax.set_xlabel(var1)
                    ax.set_ylabel('Predicted Probability of Lung Cancer')
                    ax.set_title(f'Interaction Effect: {var1} × {var2}')
                    ax.legend(title=var2)
                    
                elif var1 == 'AGE_C' or var2 == 'AGE_C':
                    cont_var = 'AGE_C'
                    cat_var = var2 if var1 == 'AGE_C' else var1
                    
                    model_data['AGE_Group'] = pd.cut(model_data['AGE_C'], bins=3, labels=['Low', 'Medium', 'High'])
                    
                    grouped_data = model_data.groupby(['AGE_Group', cat_var])['predicted_prob'].mean().unstack()
                    
                    grouped_data.plot(kind='bar', ax=ax)
                    ax.set_xlabel('Age Group')
                    ax.set_ylabel('Predicted Probability of Lung Cancer')
                    ax.set_title(f'Moderation Effect: Age × {cat_var}')
                    ax.legend(title=cat_var)
                    
                plt.tight_layout()
                return fig
            else:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.text(0.5, 0.5, "No significant interactions found", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=14)
                ax.axis('off')
                return fig
            
        def get_coefficient_intervals():
            """
            Creates a DataFrame with coefficient estimates, 95% confidence intervals, odds ratios and p-values.
            Focuses on displaying both main effects and interaction terms.
            
            Returns:
            pd.DataFrame: Table with coefficient statistics and confidence intervals
            """
            params = final_model.params
            conf = final_model.conf_int()
            conf.columns = ['Lower CI', 'Upper CI']
            
            coef_table = pd.DataFrame({
                'Coefficient': params,
                'Lower CI': conf['Lower CI'],
                'Upper CI': conf['Upper CI'],
                'Odds Ratio': np.exp(params),
                'OR Lower CI': np.exp(conf['Lower CI']),
                'OR Upper CI': np.exp(conf['Upper CI']),
                'Std Error': final_model.bse,
                'z-value': final_model.tvalues,
                'p-value': final_model.pvalues
            })
            
            coef_table['95% CI'] = coef_table.apply(lambda row: f"({row['Lower CI']:.4f}, {row['Upper CI']:.4f})", axis=1)
            coef_table['OR 95% CI'] = coef_table.apply(lambda row: f"({row['OR Lower CI']:.4f}, {row['OR Upper CI']:.4f})", axis=1)
            
            coef_table['Type'] = 'Main Effect'
            for idx in coef_table.index:
                if '_' in idx and idx != 'Intercept' and idx != 'AGE_C':
                    coef_table.loc[idx, 'Type'] = 'Interaction'
            
            display_cols = ['Coefficient', '95% CI', 'Odds Ratio', 'OR 95% CI', 'p-value', 'Type']
            display_table = coef_table[display_cols].copy()
            
            display_table = display_table.sort_values(['Type', 'p-value'])
            
            return display_table

        viz_functions['coefficient_intervals'] = get_coefficient_intervals
        viz_functions['coefficients'] = plot_coefficients
        viz_functions['roc_curve'] = plot_roc_curve
        viz_functions['confusion_matrix'] = plot_confusion_matrix
        viz_functions['interaction_effects'] = plot_interaction_effects
        
        return final_model, model_data, interaction_results, viz_functions


    def anova_analysis(data, dependent_var='LUNG_CANCER', create_age_groups=True, risk_clustering=True):
        """
        Performs Analysis of Variance (ANOVA) to test for significant differences
        in lung cancer prevalence across different groups.
        
        Parameters:
        data (pd.DataFrame): The preprocessed dataset
        dependent_var (str): The dependent variable to analyze (default: 'LUNG_CANCER')
        create_age_groups (bool): Whether to create age groups for analysis
        risk_clustering (bool): Whether to create risk profile clusters
        
        Returns:
        tuple: (results_dict, visualisation_functions)
        """
        import scipy.stats as stats
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        
        print("\n Analysis of Variance (ANOVA)")
        
        anova_data = data.copy()
        results_dict = {}
        
        if create_age_groups:
            print("\n Creating Age Groups")
            anova_data['AGE_Group'] = pd.cut(
                anova_data['AGE'], 
                bins=[0, 40, 60, 100],
                labels=['Young', 'Middle', 'Older']
            )
            
            print("\n ANOVA: Lung Cancer by Age Group")
            age_results = inferential_statistics.analyze_group_differences(
                anova_data, 
                group_var='AGE_Group', 
                dependent_var=dependent_var
            )
            results_dict['age_groups'] = age_results
        
        if risk_clustering:
            print("\n Creating Risk Profiles via Clustering")
            risk_features = [
                'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE',
                'CHRONIC_DISEASE', 'FATIGUE', 'ALLERGY', 'WHEEZING',
                'ALCOHOL_CONSUMING', 'COUGHING', 'SHORTNESS_OF_BREATH',
                'SWALLOWING_DIFFICULTY', 'CHEST_PAIN'
            ]
            
            risk_features = [col for col in risk_features if col in anova_data.columns]
            
            scaler = StandardScaler()
            X = scaler.fit_transform(anova_data[risk_features])
            
            kmeans = KMeans(n_clusters=3, random_state=42)
            anova_data['Risk_Profile'] = kmeans.fit_predict(X)
            
            cluster_cancer_rates = anova_data.groupby('Risk_Profile')[dependent_var].mean()
            cluster_mapping = {
                cluster_cancer_rates.idxmin(): 'Low Risk',
                cluster_cancer_rates.index[int(cluster_cancer_rates.rank()[1]-1)]: 'Medium Risk',
                cluster_cancer_rates.idxmax(): 'High Risk'
            }
            anova_data['Risk_Level'] = anova_data['Risk_Profile'].map(cluster_mapping)
            
            print("\n ANOVA: Lung Cancer by Risk Profile")
            risk_results = inferential_statistics.analyze_group_differences(
                anova_data, 
                group_var='Risk_Level', 
                dependent_var=dependent_var
            )
            results_dict['risk_profiles'] = risk_results
            
            cluster_features = anova_data.groupby('Risk_Level')[risk_features].mean()
            results_dict['cluster_characteristics'] = cluster_features
        
        # 3. Additional custom group comparisons
        print("\n ANOVA: Lung Cancer by Gender")
        gender_results = inferential_statistics.analyze_group_differences(
            anova_data, 
            group_var='GENDER', 
            dependent_var=dependent_var
        )
        results_dict['gender'] = gender_results
        
        print("\n ANOVA: Lung Cancer by Smoking and Alcohol Status")
        anova_data['Smoking_Alcohol'] = 'Neither'
        anova_data.loc[(anova_data['SMOKING'] == 1) & (anova_data['ALCOHOL_CONSUMING'] == 0), 'Smoking_Alcohol'] = 'Smoking Only'
        anova_data.loc[(anova_data['SMOKING'] == 0) & (anova_data['ALCOHOL_CONSUMING'] == 1), 'Smoking_Alcohol'] = 'Alcohol Only'
        anova_data.loc[(anova_data['SMOKING'] == 1) & (anova_data['ALCOHOL_CONSUMING'] == 1), 'Smoking_Alcohol'] = 'Both'
        
        smoking_alcohol_results = inferential_statistics.analyze_group_differences(
            anova_data, 
            group_var='Smoking_Alcohol', 
            dependent_var=dependent_var
        )
        results_dict['smoking_alcohol'] = smoking_alcohol_results
        
        viz_functions = {}
        
        # 1. Box plot function for group comparisons
        def plot_group_boxplots(group_var=None):
            """Creates boxplots showing the distribution of the dependent variable across groups."""
            if group_var is None or group_var not in results_dict:
                group_var = list(results_dict.keys())[0]
            
            group_data = results_dict[group_var]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            actual_group_var = group_data['group_var']
            
            sns.boxplot(
                data=anova_data, 
                x=actual_group_var, 
                y=dependent_var,
                ax=ax
            )
            
            sns.stripplot(
                data=anova_data, 
                x=actual_group_var, 
                y=dependent_var,
                color='black', 
                size=3, 
                alpha=0.3,
                ax=ax
            )
            
            anova_result = f"ANOVA: F={group_data['f_value']:.2f}, p={group_data['p_value']:.4f}"
            ax.text(0.5, 0.01, anova_result, 
                    horizontalalignment='center', 
                    transform=ax.transAxes, 
                    bbox=dict(facecolor='white', alpha=0.8))
            
            ax.set_title(f'Distribution of {dependent_var} by {actual_group_var}')
            plt.tight_layout()
            return fig
        
        # 2. Mean comparison plot with error bars
        def plot_group_means(group_var=None):
            """Creates a plot of group means with confidence intervals."""
            if group_var is None or group_var not in results_dict:
                group_var = list(results_dict.keys())[0]
                
            group_data = results_dict[group_var]
            
            actual_group_var = group_data['group_var']
            
            group_stats = anova_data.groupby(actual_group_var)[dependent_var].agg(['mean', 'std', 'count'])
            group_stats['se'] = group_stats['std'] / np.sqrt(group_stats['count'])
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            bars = ax.bar(
                x=np.arange(len(group_stats)), 
                height=group_stats['mean'],
                yerr=group_stats['se'],
                capsize=5,
                color='lightblue',
                width=0.6
            )
            
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2., 
                    height + group_stats['se'].iloc[i] + 0.01,
                    f'{height:.2f}', 
                    ha='center', 
                    va='bottom',
                    fontsize=9
                )
            
            anova_result = f"ANOVA: F={group_data['f_value']:.2f}, p={group_data['p_value']:.4f}"
            ax.text(0.5, 0.01, anova_result, 
                    horizontalalignment='center', 
                    transform=ax.transAxes, 
                    bbox=dict(facecolor='white', alpha=0.8))
            
            ax.set_xticks(np.arange(len(group_stats)))
            ax.set_xticklabels(group_stats.index)
            
            ax.set_ylabel(f'Mean {dependent_var}')
            ax.set_title(f'Mean {dependent_var} by {actual_group_var} with 95% Confidence Intervals')
            
            if 'posthoc_result' in group_data:
                posthoc = group_data['posthoc_result']
                significant_pairs = []
                
                for i in range(len(posthoc.pvalues)):
                    if posthoc.pvalues[i] < 0.05:
                        print(f'posthoc.data[i]: {posthoc.data[i]}')
                        row = posthoc._results_table.data[i+1]
                        group1, group2 = row[0], row[1]
                        significant_pairs.append((group1, group2))
                
                if significant_pairs:
                    y_max = group_stats['mean'].max() + 3 * group_stats['se'].max()
                    for i, (group1, group2) in enumerate(significant_pairs):
                        idx1 = np.where(group_stats.index == group1)[0][0]
                        idx2 = np.where(group_stats.index == group2)[0][0]
                        
                        line_y = y_max + 0.02 * i
                        ax.plot([idx1, idx2], [line_y, line_y], 'k-', lw=1)
                        
                        ax.text((idx1 + idx2) / 2, line_y, '*', ha='center', va='bottom')
            
            plt.tight_layout()
            return fig
        
        # 3. Post-hoc test visualisation
        def plot_posthoc_results(group_var=None):
            """Creates a visualisation of post-hoc test results."""
            if group_var is None or group_var not in results_dict:
                group_var = list(results_dict.keys())[0]
                
            group_data = results_dict[group_var]
            
            if 'posthoc_result' not in group_data:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.text(0.5, 0.5, "No post-hoc test results available\n(ANOVA was not significant)", 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes, fontsize=14)
                ax.axis('off')
                return fig
            
            posthoc = group_data['posthoc_result']
            
            fig, ax = plt.subplots(figsize=(10, len(posthoc.pvalues) * 0.3 + 2))
            
            y_positions = np.arange(len(posthoc.pvalues))
            
            labels = []
            for i in range(len(posthoc.pvalues)):
                row = posthoc._results_table.data[i+1]  # +1 to skip header
                group1, group2 = row[0], row[1]
                labels.append(f"{group1} vs {group2}")
            
            colors = ['red' if pval < 0.05 else 'blue' for pval in posthoc.pvalues]
            ax.scatter(posthoc.meandiffs, y_positions, color=colors)
            
            for i, (lower, upper) in enumerate(zip(posthoc.confint[:, 0], posthoc.confint[:, 1])):
                ax.plot([lower, upper], [i, i], 'k-', alpha=0.6)
                
                p_text = f"p={posthoc.pvalues[i]:.4f}"
                ax.text(upper + 0.05 * (upper - lower), i, p_text, va='center', fontsize=9)
            
            ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
            
            ax.set_yticks(y_positions)
            ax.set_yticklabels(labels)
            ax.set_xlabel('Mean Difference')
            ax.set_title(f"Tukey's HSD Post-hoc Test Results\nRed = Significant at p<0.05")
            
            plt.tight_layout()
            return fig
        
        # 4. If risk clustering was performed, create a profile comparison
        def plot_risk_profiles():
            """Creates a radar chart to visualise the characteristics of risk profiles."""
            if 'cluster_characteristics' not in results_dict:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.text(0.5, 0.5, "Risk profiles were not created", 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes, fontsize=14)
                ax.axis('off')
                return fig
                
            cluster_features = results_dict['cluster_characteristics']
            
            categories = cluster_features.columns
            N = len(categories)
            
            fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
            
            angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
            angles += angles[:1] 
            
            for risk_level, values in cluster_features.iterrows():
                values_list = values.tolist()
                values_list += values_list[:1]
                
                ax.plot(angles, values_list, linewidth=2, label=risk_level)
                ax.fill(angles, values_list, alpha=0.1)
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, fontsize=8)
            
            ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            ax.set_title('Risk Profile Characteristics', size=15)
            
            plt.tight_layout()
            return fig
        
        viz_functions['group_boxplots'] = plot_group_boxplots
        viz_functions['group_means'] = plot_group_means
        viz_functions['posthoc_results'] = plot_posthoc_results
        
        if risk_clustering:
            viz_functions['risk_profiles'] = plot_risk_profiles
        
        return results_dict, viz_functions


    def analyze_group_differences(data, group_var, dependent_var='LUNG_CANCER'):
        """
        Helper function to perform ANOVA and post-hoc tests for group differences.
        
        Parameters:
        data (pd.DataFrame): The dataset
        group_var (str): The grouping variable
        dependent_var (str): The dependent variable (default: 'LUNG_CANCER')
        
        Returns:
        dict: Results of the analysis
        """
        import scipy.stats as stats
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
        
        results = {
            'group_var': group_var,
            'dependent_var': dependent_var
        }
        
        groups = data[group_var].unique()
        
        print(f"\nGroup statistics for {group_var}:")
        group_stats = data.groupby(group_var)[dependent_var].agg(['count', 'mean', 'std'])
        print(group_stats)
        
        groups_data = [data[data[group_var] == group][dependent_var] for group in groups]
        f_value, p_value = stats.f_oneway(*groups_data)
        
        results['f_value'] = f_value
        results['p_value'] = p_value
        
        print(f"\nANOVA results: F({len(groups)-1}, {len(data)-len(groups)}) = {f_value:.4f}, p = {p_value:.4f}")
        
        if p_value < 0.05:
            print("\nANOVA is significant. Performing Tukey's HSD post-hoc test...")
            
            endog = data[dependent_var]
            groups = data[group_var]
            
            tukey_results = pairwise_tukeyhsd(endog=endog, groups=groups, alpha=0.05)
            
            print("\nTukey's HSD results:")
            print(tukey_results)
            
            results['posthoc_result'] = tukey_results
            
            print("\nSignificant pairwise comparisons (p < 0.05):")
            tukey_data = tukey_results._results_table.data
            header = tukey_data[0]  # This is ['group1', 'group2', 'meandiff', 'p-adj', 'lower', 'upper', 'reject']
            for row in tukey_data[1:]:
                group1, group2, meandiff, p_adj, lower, upper, reject = row
                print(f"{group1} vs {group2}: Mean diff = {float(meandiff):.4f}, p = {float(p_adj):.4f}")
        else:
            print("\nANOVA is not significant. No post-hoc tests needed.")
        
        return results