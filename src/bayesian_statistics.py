import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pymc as pm
import arviz as az
from scipy import stats
import pytensor.tensor as pt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings
warnings.filterwarnings('ignore')

class bayesian_statistics:
    def bayesian_regression_analysis(data, chains=2, draws=2000, tune = 1000, target_accept=0.95):
        """
        Performs Bayesian logistic regression analysis on lung cancer data.
        
        Parameters:
        data (pd.DataFrame): The preprocessed dataset
        chains (int): Number of MCMC chains to run
        draws (int): Number of samples to draw
        
        Returns:
        tuple: (trace, results_dict, visualisation_functions)
        """
        print("\n  Bayesian Logistic Regression Analysis")
        
        model_data = data.copy()
        
        model_data['LUNG_CANCER'] = model_data['LUNG_CANCER'].astype(int)
        
        print("\n  Creating Interaction Terms")
        model_data['SMOKING_ANXIETY'] = model_data['SMOKING'] * model_data['ANXIETY']
        model_data['SMOKING_PEER_PRESSURE'] = model_data['SMOKING'] * model_data['PEER_PRESSURE']
        model_data['ALCOHOL_ANXIETY'] = model_data['ALCOHOL_CONSUMING'] * model_data['ANXIETY']
        model_data['ALCOHOL_PEER_PRESSURE'] = model_data['ALCOHOL_CONSUMING'] * model_data['PEER_PRESSURE']
        
        model_data['AGE_C'] = (model_data['AGE'] - model_data['AGE'].mean()) / model_data['AGE'].std()
        model_data['AGE_SMOKING'] = model_data['AGE_C'] * model_data['SMOKING']
        model_data['AGE_ALCOHOL'] = model_data['AGE_C'] * model_data['ALCOHOL_CONSUMING']
        model_data['AGE_ANXIETY'] = model_data['AGE_C'] * model_data['ANXIETY']
        model_data['AGE_PEER_PRESSURE'] = model_data['AGE_C'] * model_data['PEER_PRESSURE']
        
        model_data['GENDER_SMOKING'] = model_data['GENDER'] * model_data['SMOKING']
        model_data['GENDER_ALCOHOL'] = model_data['GENDER'] * model_data['ALCOHOL_CONSUMING']
        model_data['GENDER_ANXIETY'] = model_data['GENDER'] * model_data['ANXIETY']
        model_data['GENDER_PEER_PRESSURE'] = model_data['GENDER'] * model_data['PEER_PRESSURE']
        
        X_base = pd.DataFrame({
            'Intercept': np.ones(len(model_data)),
            'AGE_C': model_data['AGE_C'],
            'GENDER': model_data['GENDER'],
            'SMOKING': model_data['SMOKING'],
            'ALCOHOL_CONSUMING': model_data['ALCOHOL_CONSUMING'],
            'ANXIETY': model_data['ANXIETY'],
            'PEER_PRESSURE': model_data['PEER_PRESSURE']
        })
        
        X_interaction = X_base.copy()
        X_interaction['SMOKING_ANXIETY'] = model_data['SMOKING_ANXIETY']
        X_interaction['SMOKING_PEER_PRESSURE'] = model_data['SMOKING_PEER_PRESSURE']
        X_interaction['ALCOHOL_ANXIETY'] = model_data['ALCOHOL_ANXIETY']
        X_interaction['ALCOHOL_PEER_PRESSURE'] = model_data['ALCOHOL_PEER_PRESSURE']
        
        X_full = X_interaction.copy()
        X_full['AGE_SMOKING'] = model_data['AGE_SMOKING']
        X_full['AGE_ALCOHOL'] = model_data['AGE_ALCOHOL']
        X_full['AGE_ANXIETY'] = model_data['AGE_ANXIETY']
        X_full['AGE_PEER_PRESSURE'] = model_data['AGE_PEER_PRESSURE']
        X_full['GENDER_SMOKING'] = model_data['GENDER_SMOKING']
        X_full['GENDER_ALCOHOL'] = model_data['GENDER_ALCOHOL'] 
        X_full['GENDER_ANXIETY'] = model_data['GENDER_ANXIETY']
        X_full['GENDER_PEER_PRESSURE'] = model_data['GENDER_PEER_PRESSURE']
        
        y = model_data['LUNG_CANCER'].values
        
        results_dict = {
            'model_data': model_data,
            'X_base': X_base,
            'X_interaction': X_interaction,
            'X_full': X_full,
            'y': y
        }
        
        print("\n  Fitting Base Bayesian Model (Main Effects Only)")
        with pm.Model() as base_model:
            intercept = pm.Normal('Intercept', mu=0, sigma=3)
            age_c = pm.Normal('AGE_C', mu=0, sigma=1)
            gender = pm.Normal('GENDER', mu=0, sigma=1)
            smoking = pm.Normal('SMOKING', mu=0, sigma=1)
            alcohol = pm.Normal('ALCOHOL_CONSUMING', mu=0, sigma=1)
            anxiety = pm.Normal('ANXIETY', mu=0, sigma=1)
            peer_pressure = pm.Normal('PEER_PRESSURE', mu=0, sigma=1)
            
            beta = pt.stack([intercept, age_c, gender, smoking, alcohol, anxiety, peer_pressure])
            eta = pm.math.dot(X_base.values, beta)
            
            p = pm.math.sigmoid(eta)
            pm.Bernoulli('y', p=p, observed=y)
            
            base_trace = pm.sample(draws=draws, chains=chains, return_inferencedata=True, tune=tune)#, nuts={'target_accept': target_accept, 'max_treedepth': 12}
            
            with base_model:
                pm.compute_log_likelihood(base_trace)
            
        base_waic = az.waic(base_trace)
        
        print("\n  Fitting Interaction Bayesian Model")
        with pm.Model() as interaction_model:
            intercept = pm.Normal('Intercept', mu=0, sigma=3)
            age_c = pm.Normal('AGE_C', mu=0, sigma=1)
            gender = pm.Normal('GENDER', mu=0, sigma=1)
            smoking = pm.Normal('SMOKING', mu=0, sigma=1)
            alcohol = pm.Normal('ALCOHOL_CONSUMING', mu=0, sigma=1)
            anxiety = pm.Normal('ANXIETY', mu=0, sigma=1)
            peer_pressure = pm.Normal('PEER_PRESSURE', mu=0, sigma=1)
            
            smoking_anxiety = pm.Normal('SMOKING_ANXIETY', mu=0, sigma=0.5)
            smoking_peer = pm.Normal('SMOKING_PEER_PRESSURE', mu=0, sigma=0.5)
            alcohol_anxiety = pm.Normal('ALCOHOL_ANXIETY', mu=0, sigma=0.5)
            alcohol_peer = pm.Normal('ALCOHOL_PEER_PRESSURE', mu=0, sigma=0.5)
            
            beta = pt.stack([intercept, age_c, gender, smoking, alcohol, anxiety, peer_pressure,
                          smoking_anxiety, smoking_peer, alcohol_anxiety, alcohol_peer])
            eta = pm.math.dot(X_interaction.values, beta)
            
            p = pm.math.sigmoid(eta)
            pm.Bernoulli('y', p=p, observed=y)
            
            interaction_trace = pm.sample(draws=draws, chains=chains, return_inferencedata=True, tune=tune)#3, nuts={'target_accept': target_accept, 'max_treedepth': 12}
            
            with interaction_model:
                pm.compute_log_likelihood(interaction_trace)
            
        interaction_waic = az.waic(interaction_trace)
        
        print("\n  Fitting Full Bayesian Model (with Age Moderations)")
        with pm.Model() as full_model:
            intercept = pm.Normal('Intercept', mu=0, sigma=3)
            age_c = pm.Normal('AGE_C', mu=0, sigma=1)
            gender = pm.Normal('GENDER', mu=0, sigma=1)
            smoking = pm.Normal('SMOKING', mu=0, sigma=1)
            alcohol = pm.Normal('ALCOHOL_CONSUMING', mu=0, sigma=1)
            anxiety = pm.Normal('ANXIETY', mu=0, sigma=1)
            peer_pressure = pm.Normal('PEER_PRESSURE', mu=0, sigma=1)
            
            smoking_anxiety = pm.Normal('SMOKING_ANXIETY', mu=0, sigma=0.5)
            smoking_peer = pm.Normal('SMOKING_PEER_PRESSURE', mu=0, sigma=0.5)
            alcohol_anxiety = pm.Normal('ALCOHOL_ANXIETY', mu=0, sigma=0.5)
            alcohol_peer = pm.Normal('ALCOHOL_PEER_PRESSURE', mu=0, sigma=0.5)
            
            age_smoking = pm.Normal('AGE_SMOKING', mu=0, sigma=0.5)
            age_alcohol = pm.Normal('AGE_ALCOHOL', mu=0, sigma=0.5)
            age_anxiety = pm.Normal('AGE_ANXIETY', mu=0, sigma=0.5)
            age_peer = pm.Normal('AGE_PEER_PRESSURE', mu=0, sigma=0.5)
            gender_smoking = pm.Normal('GENDER_SMOKING', mu=0, sigma=0.5)
            gender_alcohol = pm.Normal('GENDER_ALCOHOL', mu=0, sigma=0.5)
            gender_anxiety = pm.Normal('GENDER_ANXIETY', mu=0, sigma=0.5)
            gender_peer = pm.Normal('GENDER_PEER_PRESSURE', mu=0, sigma=0.5)
            
            beta = pt.stack([intercept, age_c, gender, smoking, alcohol, anxiety, peer_pressure,
                          smoking_anxiety, smoking_peer, alcohol_anxiety, alcohol_peer,
                          age_smoking, age_alcohol, age_anxiety, age_peer,
                          gender_smoking, gender_alcohol, gender_anxiety, gender_peer])
            eta = pm.math.dot(X_full.values, beta)
            
            p = pm.math.sigmoid(eta)
            pm.Bernoulli('y', p=p, observed=y)
            
            full_trace = pm.sample(draws=draws, chains=chains, return_inferencedata=True, tune=tune)#, nuts={'target_accept': target_accept, 'max_treedepth': 12}
            
            with full_model:
                pm.compute_log_likelihood(full_trace)
            
        full_waic = az.waic(full_trace)
        
        print("\n  Calculating LOO-CV")
        try:
            base_loo = az.loo(base_trace)
            interaction_loo = az.loo(interaction_trace)
            full_loo = az.loo(full_trace)
            
            print("\n  LOO Comparison")
            compare_dict = {'base': base_trace, 'interaction': interaction_trace, 'full': full_trace}
            loo_compare = az.compare(compare_dict, ic='loo', scale='deviance')
            print(loo_compare)

            results_dict['base_loo'] = base_loo
            results_dict['interaction_loo'] = interaction_loo
            results_dict['full_loo'] = full_loo
            results_dict['loo_compare'] = loo_compare

            best_model_name_loo = loo_compare.index[0]
            print(f"\nSelected model based on LOO: {best_model_name_loo}")
        except Exception as e:
            print(f"\nError calculating or comparing LOO: {e}")
            print("Proceeding with WAIC-based selection.")
            results_dict.pop('base_loo', None)
            results_dict.pop('interaction_loo', None)
            results_dict.pop('full_loo', None)
            results_dict.pop('loo_compare', None)
        
        if 'loo_compare' in results_dict:
            best_model_name_final = results_dict['loo_compare'].index[0]
            print(f"\nSelecting final model based on LOO rank 0: {best_model_name_final}")
        else:
            waic_scores = {
                'base': base_waic.elpd_waic,
                'interaction': interaction_waic.elpd_waic,
                'full': full_waic.elpd_waic
            }
            best_model_name_final = min(waic_scores, key=waic_scores.get)
            if best_model_name_final == 'Base Model': best_model_name_final = 'base'
            elif best_model_name_final == 'Interaction Model': best_model_name_final = 'interaction'
            else: best_model_name_final = 'full'
            print(f"\nSelecting final model based on WAIC (LOO failed): {best_model_name_final}")

        
        if best_model_name_final == 'base':
            final_trace = base_trace
            final_model = base_model
            X_final = X_base
        elif best_model_name_final == 'interaction':
            final_trace = interaction_trace
            final_model = interaction_model
            X_final = X_interaction
        else:
            final_trace = full_trace
            final_model = full_model
            X_final = X_full

        results_dict['final_model_name'] = best_model_name_final
        results_dict['final_trace'] = final_trace
        
        print(f"\n  Model Comparison Summary")
        print(f"Base Model WAIC: {base_waic.elpd_waic:.2f}")
        print(f"Interaction Model WAIC: {interaction_waic.elpd_waic:.2f}")
        print(f"Full Model WAIC: {full_waic.elpd_waic:.2f}")
        if 'loo_compare' in results_dict:
            print("\n  LOO Comparison Table:")
            print(results_dict['loo_compare'])
        
        base_bic = -2 * base_trace.log_likelihood.y.mean(dim=["chain", "draw"]).sum() + X_base.shape[1] * np.log(len(y))
        interaction_bic = -2 * interaction_trace.log_likelihood.y.mean(dim=["chain", "draw"]).sum() + X_interaction.shape[1] * np.log(len(y))
        full_bic = -2 * full_trace.log_likelihood.y.mean(dim=["chain", "draw"]).sum() + X_full.shape[1] * np.log(len(y))
        
        bf_base_vs_interaction = np.exp((base_bic - interaction_bic) / 2)
        bf_interaction_vs_full = np.exp((interaction_bic - full_bic) / 2)
        
        print(f"\n  Approximate Bayes Factors")
        print(f"Interaction vs Base: {1/bf_base_vs_interaction:.2f}")
        print(f"Full vs Interaction: {1/bf_interaction_vs_full:.2f}")
        
        results_dict.update({
            'base_trace': base_trace,
            'interaction_trace': interaction_trace,
            'full_trace': full_trace,
            'waic_scores': {
                 'base': base_waic.elpd_waic,
                 'interaction': interaction_waic.elpd_waic,
                 'full': full_waic.elpd_waic
             },
            'bayes_factors': {
                'Interaction vs Base': 1/bf_base_vs_interaction,
                'Full vs Interaction': 1/bf_interaction_vs_full
            }
        })
        
        results_dict.pop('best_model', None)
        print(f"\n  Proceeding with selected model: {best_model_name_final}")
        posterior_samples = az.extract(final_trace, var_names=list(X_final.columns))
        print("\n  Extracted Posterior Samples")
        model_data['predicted_prob_mean'] = 0
        
        n_samples = min(500, posterior_samples.sizes["sample"])
        subset_indices = np.random.choice(posterior_samples.sizes["sample"], n_samples, replace=False)
        print(f"\n  Using {n_samples} samples for prediction")
        pred_probs = []
        for i in range(n_samples):
            sample_idx = subset_indices[i]
            params = {var: posterior_samples[var].isel(sample=sample_idx).values for var in X_final.columns}
            linear_pred = np.zeros(len(model_data))
            
            for var in X_final.columns:
                linear_pred += X_final[var].values * params[var]
                
            pred_probs.append(1 / (1 + np.exp(-linear_pred)))
        print("\n  Calculated Predicted Probabilities")
        pred_probs_array = np.array(pred_probs)
        model_data['predicted_prob_mean'] = pred_probs_array.mean(axis=0)
        model_data['predicted_prob_lower'] = np.percentile(pred_probs_array, 2.5, axis=0)
        model_data['predicted_prob_upper'] = np.percentile(pred_probs_array, 97.5, axis=0)
        
        model_data['predicted_class'] = (model_data['predicted_prob_mean'] > 0.5).astype(int)
        
        from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
        
        conf_matrix = confusion_matrix(model_data['LUNG_CANCER'], model_data['predicted_class'])
        
        fpr, tpr, _ = roc_curve(model_data['LUNG_CANCER'], model_data['predicted_prob_mean'])
        roc_auc = auc(fpr, tpr)
        
        class_report = classification_report(model_data['LUNG_CANCER'], model_data['predicted_class'])
        
        print("\n  Model Evaluation")
        print(f"AUC: {roc_auc:.4f}")
        print("\n  Classification Report")
        print(class_report)
        
        results_dict.update({
            'confusion_matrix': conf_matrix,
            'roc_curve': {'fpr': fpr, 'tpr': tpr},
            'auc': roc_auc,
            'classification_report': class_report
        })
        
        viz_functions = {}
        
        # 1. Posterior distributions 
        def plot_posterior_distributions(params=None, figsize=(12, 10)):
            """Plot posterior distributions for specified parameters."""
            if params is None:
                params = ['Intercept', 'SMOKING', 'ALCOHOL_CONSUMING', 'ANXIETY', 'PEER_PRESSURE']
                
                for param in list(X_final.columns):
                    if '_' in param and param != 'AGE_C' and param in final_trace.posterior:
                        params.append(param)
            
            n_params = len(params)
            n_cols = 2
            n_rows = int(np.ceil(n_params / n_cols))
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
            axes = axes.flatten()
            
            for i, param in enumerate(params):
                if i < len(axes) and param in final_trace.posterior:
                    az.plot_posterior(final_trace, var_names=[param], ax=axes[i])
                    axes[i].set_title(f"Posterior for {param}")
                    
                    axes[i].axvline(x=0, color='red', linestyle='--', alpha=0.5)
                    
                    interval = az.hdi(final_trace.posterior[param])
                    print("Intervals are: \n")                    
                    print(interval)

                    var_name = list(interval.data_vars)[0]
                    lower = interval[var_name].sel(hdi='lower').values
                    upper = interval[var_name].sel(hdi='higher').values
                    print(f"{var_name}: [{lower:.3f}, {upper:.3f}]")
                    within_zero = (lower <= 0 <= upper)
                    
                    if within_zero:
                        axes[i].set_title(f"Posterior for {param} (Not Significant)")
                    else:
                        axes[i].set_title(f"Posterior for {param} (Significant)")
            
            for j in range(i+1, len(axes)):
                axes[j].set_visible(False)
                
            plt.tight_layout()
            return fig
        
        # 2. Coefficient plot with credible intervals
        def plot_coefficient_intervals(figsize=(10, 8)):
            """Plot coefficient estimates with 95% credible intervals."""
            var_names = list(X_final.columns)
            
            stats_dict = {}
            for var in var_names:
                if var in final_trace.posterior:
                    posterior_samples = final_trace.posterior[var].values.flatten()
                    mean = posterior_samples.mean()
                    hdi_interval = az.hdi(posterior_samples)
                    stats_dict[var] = {
                        'mean': mean,
                        'hdi_lower': hdi_interval[0],
                        'hdi_upper': hdi_interval[1]
                    }
            
            stats_df = pd.DataFrame(stats_dict).T
            stats_df = stats_df.sort_values('mean', key=abs, ascending=False)
            
            if 'Intercept' in stats_df.index:
                stats_df = stats_df.drop('Intercept')
            
            fig, ax = plt.subplots(figsize=figsize)
            
            params = stats_df.index
            y_pos = np.arange(len(params))
            ax.scatter(stats_df['mean'], y_pos, color='blue', s=60, zorder=3)
            
            for i, (idx, row) in enumerate(stats_df.iterrows()):
                ax.plot([row['hdi_lower'], row['hdi_upper']], [i, i], 'b-', alpha=0.6, linewidth=2)
                
                if not (row['hdi_lower'] <= 0 <= row['hdi_upper']):
                    ax.plot([row['hdi_lower'], row['hdi_upper']], [i, i], 'r-', linewidth=3, alpha=0.7)
            
            ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(params)
            ax.set_xlabel('Coefficient Value')
            ax.set_title('Bayesian Logistic Regression Coefficients with 95% Credible Intervals')
            
            plt.tight_layout()
            return fig
            
        # 3. ROC curve 
        def plot_roc_curve(figsize=(8, 8)):
            """Plot ROC curve with confidence band."""
            fig, ax = plt.subplots(figsize=figsize)
            
            ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            
            ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
            
            n_curves = 100
            subset_indices = np.random.choice(posterior_samples.sizes["sample"], n_curves, replace=False)
            
            all_fprs = []
            all_tprs = []
            
            for i in range(n_curves):
                sample_idx = subset_indices[i]
                params = {var: posterior_samples[var].isel(sample=sample_idx).values for var in X_final.columns}
                linear_pred = np.zeros(len(model_data))
                
                for j, var in enumerate(X_final.columns):
                    linear_pred += X_final[var].values * params[var]
                    
                prob = 1 / (1 + np.exp(-linear_pred))
                sample_fpr, sample_tpr, _ = roc_curve(model_data['LUNG_CANCER'], prob)
                all_fprs.append(sample_fpr)
                all_tprs.append(sample_tpr)
            
            std_fpr = np.linspace(0, 1, 100)
            interp_tprs = []
            
            for i in range(n_curves):
                interp_tpr = np.interp(std_fpr, all_fprs[i], all_tprs[i])
                interp_tpr[0] = 0.0
                interp_tprs.append(interp_tpr)
            
            mean_tpr = np.mean(interp_tprs, axis=0)
            lower_tpr = np.percentile(interp_tprs, 2.5, axis=0)
            upper_tpr = np.percentile(interp_tprs, 97.5, axis=0)
            
            ax.fill_between(std_fpr, lower_tpr, upper_tpr, color='blue', alpha=0.2, label='95% CI')
            
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Bayesian ROC Curve with 95% Credible Interval')
            ax.legend(loc="lower right")
            
            plt.tight_layout()
            return fig
            
        # 4. Confusion matrix
        def plot_confusion_matrix(figsize=(8, 8)):
            """Plot confusion matrix."""
            fig, ax = plt.subplots(figsize=figsize)
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
            ax.set_title('Confusion Matrix')
            ax.set_xticklabels(['No Cancer', 'Cancer'])
            ax.set_yticklabels(['No Cancer', 'Cancer'])
            plt.tight_layout()
            return fig
            
        # 5. Model comparison plot
        # 5a. WAIC comparison plot
        def plot_waic_comparison(figsize=(10, 6)):
            """Plot WAIC comparison for model selection."""
            fig, ax = plt.subplots(figsize=figsize)

            if 'waic_scores' not in results_dict:
                ax.text(0.5, 0.5, "WAIC scores not available.",
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes, fontsize=12)
                ax.axis('off')
                return fig

            models = list(results_dict['waic_scores'].keys())
            waics = list(results_dict['waic_scores'].values())

            best_waic = min(waics)
            relative_waic = [w - best_waic for w in waics]

            sorted_indices = np.argsort(relative_waic)
            sorted_models = [models[i] for i in sorted_indices]
            sorted_waic = [relative_waic[i] for i in sorted_indices]
            original_waics_sorted = [waics[i] for i in sorted_indices]

            bars = ax.barh(sorted_models, sorted_waic, color=['green'] + ['lightblue'] * (len(models) - 1))
            bars[0].set_color('green')

            for i, v in enumerate(sorted_waic):
                if v == 0:
                    ax.text(v + 0.5, i, f"Best (WAIC={original_waics_sorted[i]:.1f})", va='center')
                else:
                    ax.text(v + 0.5, i, f"+{v:.1f}", va='center')

            ax.set_xlabel('WAIC Difference (Relative to Best Model, Lower is Better)')
            ax.set_title('Model Comparison using WAIC')
            ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
            ax.invert_yaxis()

            plt.tight_layout()
            return fig

        # 5b. LOO comparison plot
        def plot_loo_comparison(figsize=(10, 6)):
            """Plot LOO comparison results from az.compare."""
            fig, ax = plt.subplots(figsize=figsize)

            if 'loo_compare' not in results_dict:
                ax.text(0.5, 0.5, "LOO comparison results not available.",
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes, fontsize=12)
                ax.axis('off')
                return fig

            loo_compare_df = results_dict['loo_compare']

            az.plot_compare(loo_compare_df, insample_dev=False, figsize=figsize, plot_standard_error=True, ax=ax)
            ax.set_title('Model Comparison using LOO-CV (Lower is Better)')

            plt.tight_layout()
            return fig
        
            
        # 6. Effect size plot for interactions
        def plot_interaction_effects(figsize=(12, 8)):
            """Plot the effect sizes for interactions with uncertainty."""
            interaction_terms = [col for col in X_final.columns if '_' in col and col != 'AGE_C']
            
            if not interaction_terms:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.text(0.5, 0.5, "No interaction terms in the selected model", 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes, fontsize=14)
                ax.axis('off')
                return fig
                
            effect_data = {
                'Term': [],
                'Odds Ratio': [],
                'Lower CI': [],
                'Upper CI': []
            }
            
            for term in interaction_terms:
                if term in final_trace.posterior:
                    samples = final_trace.posterior[term].values.flatten()
                    odds_ratio = np.exp(samples.mean())
                    hdi_interval = az.hdi(np.exp(samples))
                    
                    effect_data['Term'].append(term)
                    effect_data['Odds Ratio'].append(odds_ratio)
                    effect_data['Lower CI'].append(hdi_interval[0])
                    effect_data['Upper CI'].append(hdi_interval[1])
            
            effect_df = pd.DataFrame(effect_data)
            
            effect_df['Effect Size'] = abs(np.log(effect_df['Odds Ratio']))
            effect_df = effect_df.sort_values('Effect Size', ascending=False)
            
            fig, ax = plt.subplots(figsize=figsize)
            
            ax.axvline(x=1, color='gray', linestyle='--', alpha=0.7)
            
            y_pos = np.arange(len(effect_df))
            ax.errorbar(
                effect_df['Odds Ratio'], 
                y_pos,
                xerr=[effect_df['Odds Ratio'] - effect_df['Lower CI'], 
                      effect_df['Upper CI'] - effect_df['Odds Ratio']],
                fmt='o',
                color='blue',
                capsize=5,
                markeredgewidth=2
            )
            
            for i, (_, row) in enumerate(effect_df.iterrows()):
                if not (row['Lower CI'] <= 1 <= row['Upper CI']):
                    color = 'red' if row['Odds Ratio'] > 1 else 'purple'
                    ax.plot([row['Lower CI'], row['Upper CI']], [i, i], '-', color=color, linewidth=3, alpha=0.7)
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(effect_df['Term'])
            ax.set_xscale('log')
            ax.set_xlabel('Odds Ratio (log scale)')
            ax.set_title('Interaction Effect Sizes with 95% Credible Intervals')
            
            ax.text(
                0.02, 0.02, 
                "Red/Purple: Significant effect (95% CI excludes 1.0)\nValues > 1: Increased odds\nValues < 1: Decreased odds",
                transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.8)
            )
            
            plt.tight_layout()
            return fig
            
        # 7. Trace plots for convergence diagnostics
        def plot_trace_diagnostics(params=None, figsize=(12, 10)):
            """Plot MCMC trace plots to diagnose convergence."""
            if params is None:
                params = ['Intercept', 'SMOKING', 'ALCOHOL_CONSUMING', 'ANXIETY', 'PEER_PRESSURE']
                
                for param in list(X_final.columns):
                    if '_' in param and param != 'AGE_C' and param in final_trace.posterior:
                        params.append(param)
                        if len(params) >= 8:
                            break
            
            n_params = len(params)
            n_cols = 2
            n_rows = int(np.ceil(n_params / n_cols))
            
            fig = plt.figure(figsize=figsize)
            for i, param in enumerate(params):
                if param in final_trace.posterior:
                    az.plot_trace(final_trace, var_names=[param])
            
            plt.tight_layout()
            return fig
            
        # 8. Calibration plot
        def plot_calibration_curve(figsize=(10, 8)):
            """Plot calibration curve to assess predicted probabilities."""
            n_bins = 10
            bin_edges = np.linspace(0, 1, n_bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            binned_preds = np.digitize(model_data['predicted_prob_mean'], bin_edges[1:-1])
            binned_preds = np.clip(binned_preds, 0, n_bins-1)
            bin_counts = np.zeros(n_bins)
            bin_positive_counts = np.zeros(n_bins)
            
            for i in range(len(model_data)):
                bin_idx = binned_preds[i]
                bin_counts[bin_idx] += 1
                bin_positive_counts[bin_idx] += model_data['LUNG_CANCER'].iloc[i]
            
            bin_fractions = np.zeros(n_bins)
            bin_errors = np.zeros(n_bins)
            
            for i in range(n_bins):
                if bin_counts[i] > 0:
                    bin_fractions[i] = bin_positive_counts[i] / bin_counts[i]
                    z = 1.96  # 95% CI
                    n = bin_counts[i]
                    p = bin_fractions[i]
                    bin_errors[i] = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
            
            fig, ax = plt.subplots(figsize=figsize)
            
            ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
            
            valid_bins = bin_counts > 0
            ax.errorbar(
                bin_centers[valid_bins],
                bin_fractions[valid_bins],
                yerr=bin_errors[valid_bins],
                fmt='o',
                capsize=5,
                label='Observed Fraction'
            )
            
            ax.set_xlabel('Predicted Probability')
            ax.set_ylabel('Observed Fraction')
            ax.set_title('Calibration Plot')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            divider = make_axes_locatable(ax)
            ax_hist = divider.append_axes("bottom", size="20%", pad=0.1)
            ax_hist.hist(model_data['predicted_prob_mean'], bins=bin_edges, alpha=0.5)
            ax_hist.set_xlim(0, 1)
            ax_hist.set_yticks([])
            ax_hist.set_xlabel('Predicted Probability')
            
            plt.tight_layout()
            return fig
            
        # 9. Credible interval table
        def get_credible_intervals():
            """Generate a DataFrame with posterior statistics including 95% credible intervals."""
            var_names = list(X_final.columns)
            
            results = {
                'Mean': [],
                '2.5%': [],
                '97.5%': [],
                'HDI Low': [],
                'HDI High': [],
                'Odds Ratio': [],
                'OR 2.5%': [],
                'OR 97.5%': [],
                'Significant': []
            }
            
            param_names = []
            
            for var in var_names:
                if var in final_trace.posterior:
                    param_names.append(var)
                    posterior_samples = final_trace.posterior[var].values.flatten()
                    
                    mean = posterior_samples.mean()
                    percentile_2_5 = np.percentile(posterior_samples, 2.5)
                    percentile_97_5 = np.percentile(posterior_samples, 97.5)
                    hdi_interval = az.hdi(posterior_samples)
                    
                    or_samples = np.exp(posterior_samples)
                    or_mean = or_samples.mean()
                    or_2_5 = np.percentile(or_samples, 2.5)
                    or_97_5 = np.percentile(or_samples, 97.5)
                    
                    significant = not (percentile_2_5 <= 0 <= percentile_97_5)
                    
                    results['Mean'].append(mean)
                    results['2.5%'].append(percentile_2_5)
                    results['97.5%'].append(percentile_97_5)
                    results['HDI Low'].append(hdi_interval[0])
                    results['HDI High'].append(hdi_interval[1])
                    results['Odds Ratio'].append(or_mean)
                    results['OR 2.5%'].append(or_2_5)
                    results['OR 97.5%'].append(or_97_5)
                    results['Significant'].append(significant)
            
            df = pd.DataFrame(results, index=param_names)
            df['95% CI'] = df.apply(lambda row: f"({row['2.5%']:.4f}, {row['97.5%']:.4f})", axis=1)
            df['95% HDI'] = df.apply(lambda row: f"({row['HDI Low']:.4f}, {row['HDI High']:.4f})", axis=1)
            df['OR 95% CI'] = df.apply(lambda row: f"({row['OR 2.5%']:.4f}, {row['OR 97.5%']:.4f})", axis=1)
            
            df['Type'] = 'Main Effect'
            for idx in df.index:
                if '_' in idx and idx != 'AGE_C':
                    df.loc[idx, 'Type'] = 'Interaction'
            
            df = df.sort_values(['Type', 'Significant', 'Mean'], ascending=[True, False, False])
            
            display_cols = ['Mean', '95% CI', '95% HDI', 'Odds Ratio', 'OR 95% CI', 'Significant', 'Type']
            return df[display_cols]
        
        viz_functions['posterior_distributions'] = plot_posterior_distributions
        viz_functions['coefficient_intervals'] = plot_coefficient_intervals
        viz_functions['roc_curve'] = plot_roc_curve
        viz_functions['confusion_matrix'] = plot_confusion_matrix
        viz_functions['waic_comparison'] = plot_waic_comparison
        viz_functions['loo_comparison'] = plot_loo_comparison
        viz_functions['interaction_effects'] = plot_interaction_effects
        viz_functions['trace_diagnostics'] = plot_trace_diagnostics
        viz_functions['calibration_curve'] = plot_calibration_curve
        viz_functions['credible_intervals'] = get_credible_intervals
        
        return final_trace, results_dict, viz_functions