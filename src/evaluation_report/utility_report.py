from typing import Dict
import ipywidgets as widgets
import pandas as pd
import plotly.graph_objects as go

from data_evaluator.plot_evaluation import histo_plot_utility_compare
from data_synthesizer.pipeline.pipeline_results import PipelineResults


class UtilityReport :
    
    def __init__(self, real_data : pd.DataFrame, categorical_columns : list, numerical_columns : list, pipeline_results : Dict[str, PipelineResults] ) :
        self._pipeline_results = pipeline_results
        self._real_data = real_data
        self._categorical_columns = categorical_columns
        self._numerical_columns = numerical_columns
            
       
    def get_report(self) : 

        sub_sub_tab = [widgets.VBox() for _ in range(6)]        

        plot = {}
        f_i = dict()
        shap_importances = dict()
        acc_means = dict()
        for j_acc, j in zip(self._pipeline_results, range(len(self._pipeline_results))) :

            utility_evaluation_results = self._pipeline_results[j_acc]['utility_evaluation_results']
            train_synthetic_test_real_results = utility_evaluation_results['train_synthetic_test_real_results']                
            permutation_importance_tstr = utility_evaluation_results['permutation_importance_tstr']                
            shap_importance_tstr = utility_evaluation_results['shap_importance_tstr']
            accuracy_mean_tstr = utility_evaluation_results['accuracy_mean_tstr']                

            plot[j_acc] = train_synthetic_test_real_results
            f_i[j_acc] = permutation_importance_tstr
            shap_importances[j_acc] = shap_importance_tstr
            acc_means[j_acc] = accuracy_mean_tstr

        utility_evaluation_results = self._pipeline_results[j_acc]['utility_evaluation_results']
        train_real_test_real_results = utility_evaluation_results['train_real_test_real_results']
        permutation_importance_trtr = utility_evaluation_results['permutation_importance_trtr']
        shap_importance_trtr = utility_evaluation_results['shap_importance_trtr']
        accuracy_mean_trtr = utility_evaluation_results['accuracy_mean_trtr']

        acc_means['real'] = accuracy_mean_trtr
        acc_means = { key: abs(value - accuracy_mean_trtr) for key, value in acc_means.items()}
        plot['real'] = train_real_test_real_results


        # Create and display a DataFrame for accuracy means
        acc_means_df = pd.DataFrame(acc_means, index=[0])

        # Add the DataFrame and plot to the corresponding VBox
        vbox = sub_sub_tab[0]
        vbox.children = [
            widgets.VBox([
                widgets.Label('Accuracy Means:'),
                widgets.HTML(acc_means_df.to_html()),
                widgets.Label('Utility Comparison:'),
                histo_plot_utility_compare(plot, 'f1'),
                histo_plot_utility_compare(plot, 'accuracy'),
            ])
        ]

        f_i['real'] = permutation_importance_trtr
        shap_importances['real'] = shap_importance_trtr
        features_key = list(self._pipeline_results)[0]
        f_i['variable'] = self._real_data.columns[:-1]
        shap_importances['variable'] = self._real_data.columns[:-1]
            
    #---------------------------------------------------------------------------------           
        # Create sub-accordion for models
        model_names = [i for i in utility_evaluation_results['train_synthetic_test_real_results'].index]
        sub_sub_sub_tab = [widgets.VBox() for _ in model_names]
        acc_sub_1 = widgets.Accordion(children=sub_sub_sub_tab)

        for id, model_name in enumerate(model_names):
            if model_name == 'XGBoost' : 
                acc_sub_1.set_title(id, model_name)
                f_i_dict = dict()
                f_i_dict['real'] = f_i['real'][model_name]
                for utility_key in self._pipeline_results:
                    f_i_dict[utility_key] = f_i[utility_key][model_name]
                f_i_df = pd.DataFrame(f_i_dict).sort_values(by=['real'], ascending=False)
                f_i_html = widgets.HTML(value=f_i_df.to_html())

                sub_sub_sub_tab[id].children = [f_i_html]

        vbox = sub_sub_tab[1]
        vbox.children = [widgets.VBox([acc_sub_1])]
    # #---------------------------------------------------------------------------------

        scatters = []
        for utility_key in self._pipeline_results:
                scatters.append(go.Scatter(mode='lines+markers', x=f_i['variable'], y=f_i[utility_key]['XGBoost'], name=utility_key))
        scatters.append(go.Scatter(mode='lines+markers', x=f_i['variable'], y=f_i['real']['XGBoost'], name='real'))
        fig = go.FigureWidget(data=scatters)
        vbox = sub_sub_tab[2]
        vbox.children = [widgets.VBox([fig])]
    # #---------------------------------------------------------------------------------

        for id, model_name in enumerate(model_names):
            if model_name == 'XGBoost' : 
                acc_sub_1.set_title(id, model_name)
                shap_importances_dict = dict()
                shap_importances_dict['real'] = shap_importances['real'][model_name]
                for utility_key in self._pipeline_results:
                    shap_importances_dict[utility_key] = shap_importances[utility_key][model_name]
                shap_importances_df = pd.DataFrame(shap_importances_dict).sort_values(by=['real'], ascending=False)
                shap_importances_html = widgets.HTML(value=shap_importances_df.to_html())

                sub_sub_sub_tab[id].children = [shap_importances_html]

        vbox = sub_sub_tab[3]
        vbox.children = [widgets.VBox([acc_sub_1])]
    # #---------------------------------------------------------------------------------

        scatters = []
        for utility_key in self._pipeline_results:
            scatters.append(go.Scatter(mode='lines+markers', x=shap_importances['variable'], y=shap_importances[utility_key]['XGBoost'], name=utility_key))
        scatters.append(go.Scatter(mode='lines+markers', x=shap_importances['variable'], y=shap_importances['real']['XGBoost'], name='real'))
        fig = go.FigureWidget(data=scatters)
        vbox = sub_sub_tab[4]
        vbox.children = [widgets.VBox([fig])]
    # #---------------------------------------------------------------------------------
                
        correlation_values = dict() 
        for utility_key in self._pipeline_results:
            utility_evaluation_results = self._pipeline_results[utility_key]['utility_evaluation_results']

            rbo_pi = utility_evaluation_results['rbo_permutation_importance']
            rbo_shap = utility_evaluation_results['rbo_shap']
            spearman_pi = utility_evaluation_results['spearman_permutation_importance']
            spearman_shap = utility_evaluation_results['spearman_shap']
            kendall_pi = utility_evaluation_results['kendall_permutation_importance']
            kendall_shap = utility_evaluation_results['kendall_shap']

            correlation_values[utility_key] = [rbo_pi, rbo_shap, spearman_pi, spearman_shap, kendall_pi, kendall_shap]
            
            
        correlations_df = pd.DataFrame.from_dict(correlation_values, orient='index', columns = ['rbo permutation-importance', 'rbo shap-importance', 
                                                                                            'spearman permutation-importance', 'spearman shap-importance', 
                                                                                            'kendall permutation-importance', 'kendall shap-importance'])
            
        
        vbox = sub_sub_tab[5]
        vbox.children = [widgets.VBox([widgets.HTML(value=correlations_df.to_html())])]

        acc = widgets.Accordion(sub_sub_tab)
        acc.set_title(0 , 'bar comparison') 
        acc.set_title(1 , 'permutation feature importance')
        acc.set_title(2 , 'permutation feature importance plot') 
        acc.set_title(3 , 'shap feature importance') 
        acc.set_title(4 , 'shap feature importance plot') 
        acc.set_title(5 , 'rbo evaluation feature importance')        

        return acc
