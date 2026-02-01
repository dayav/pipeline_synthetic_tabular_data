import io
import pickle
from typing import Dict

from data_synthesizer.pipeline.pipeline_results import PipelineResults
import ipywidgets as widgets
import pandas as pd
from matplotlib import pyplot as plt
from IPython.display import display, HTML

from data_evaluator.univariate_evaluator import UnivariateEvaluator
from data_evaluator.plot_evaluation import plot_correlation_diff_, plot_correlation_ratio_diff_


class ResemblanceReport :

    def __init__(self, real_data : pd.DataFrame, categorical_columns : list, numerical_columns : list, pipeline_results : Dict[str, PipelineResults], ) :
        self._pipeline_results = pipeline_results
        self._real_data = real_data
        self._categorical_columns = categorical_columns
        self._numerical_columns = numerical_columns
    
    def get_categorical_univariate_report(self) :
        accordions = []
        box_plot_data = {}
        kl_means = {}
        mode_collapse_values = {}
            
        for j_acc, j in zip(self._pipeline_results, range(len(self._pipeline_results))) :
            categorical_univariate_result = self._pipeline_results[j_acc]['resemblance_evaluation_results']['categorical_univariate']
            synthetic_data = self._pipeline_results[j_acc]['generation_results']['synthetic_data']


            chi_test = categorical_univariate_result['chi_test'].style.applymap(UnivariateEvaluator.style_negative, props='color:red;')
            jensen_shanon = categorical_univariate_result['jensen_shanon'].style.applymap(UnivariateEvaluator.style_binary, props='color:red;', subset= pd.IndexSlice[:, ['JS_divergence']])
            jensen_shanon_mean = categorical_univariate_result['jensen_shanon_data_mean']

            tabular_output_1 = widgets.HTML(chi_test.to_html())
            tabular_output_2 = widgets.HTML(jensen_shanon.to_html())
            tabular_output_3 = widgets.HTML(jensen_shanon_mean.to_frame().to_html())
            # tabular_output_4 = widgets.HTML(pd.DataFrame.from_dict(mode_collapse_values[j_acc], orient='index').to_html())
            box_plot_data[j_acc] = jensen_shanon.data['JS_divergence']
            kl_means[j_acc] = jensen_shanon_mean
            dropdown = widgets.Dropdown(options=self._categorical_columns, description='Columns:')
            output = widgets.interactive_output(self.display_dataframe_and_plot,
                                                    {'real':widgets.fixed(self._real_data),'synthetic': widgets.fixed(synthetic_data), 'column': dropdown})

            accordion = widgets.Accordion(children=[widgets.VBox([widgets.HBox([tabular_output_1, tabular_output_2, tabular_output_3]), dropdown, output])])
            accordion.set_title(0, j_acc)
            accordions.append(accordion)

        tabular_output_1 = widgets.Button(description="Show Boxplot")
        tabular_output_1.on_click(lambda b : self.plot_boxplot( box_plot_data, "JS divergence Boxplots comparison "))
        tabular_output_2 = widgets.HTML(pd.DataFrame(kl_means).transpose().to_html())
        tabular_output_3 = widgets.HTML(pd.DataFrame(mode_collapse_values).to_html())
        
        accordion = widgets.Accordion(children=[widgets.VBox([widgets.HBox([tabular_output_1, tabular_output_2, tabular_output_3])])])
        accordion.set_title(0, 'Summary')
        accordions.append(accordion)
        return widgets.VBox(accordions)
    

    def get_numerical_univariate_report(self) :
        accordions = []
        box_plot_data = dict()
        cohens_means = dict()

        for j_acc, j in zip(self._pipeline_results, range(len(self._pipeline_results))) :
            numerical_univariate_result = self._pipeline_results[j_acc]['resemblance_evaluation_results']['numerical_univariate']
            synthetic_data = self._pipeline_results[j_acc]['generation_results']['synthetic_data']
            
            statistical_test = numerical_univariate_result['univariate_num_s'].style.applymap(UnivariateEvaluator.style_negative, props='color:red;') 
            divergences = numerical_univariate_result['univariate_num_js'].style.applymap(UnivariateEvaluator.style_binary, props='color:red;', subset= pd.IndexSlice[:, ['JS_divergence']])  
            divergences_mean = numerical_univariate_result['univariate_num_js_data_mean']  
                        
            tabular_output_1 = widgets.HTML(statistical_test.to_html())
            tabular_output_2 = widgets.HTML(divergences.to_html())
            tabular_output_3 = widgets.HTML(divergences_mean.to_frame().to_html())
            box_plot_data[j_acc] = divergences.data['cohen_s_d']
            cohens_means[j_acc] = divergences_mean
            dropdown = widgets.Dropdown(options=self._numerical_columns, description='Columns:')
            output = widgets.interactive_output(self.display_dataframe_and_plot,
                                                    {'real':widgets.fixed(self._real_data),'synthetic': widgets.fixed(synthetic_data), 'column': dropdown})

            accordion = widgets.Accordion(children=[widgets.VBox([widgets.HBox([tabular_output_1, tabular_output_2, tabular_output_3]), dropdown, output])])
            accordion.set_title(0, j_acc)
            accordions.append(accordion)

        tabular_output_1 = widgets.Button(description="Show Boxplot")
        tabular_output_1.on_click(lambda b : self.plot_boxplot( box_plot_data, "Cohen's d Boxplots comparison "))
        tabular_output_2 = widgets.HTML(pd.DataFrame(cohens_means).transpose().to_html())
        
        accordion = widgets.Accordion(children=[widgets.VBox([widgets.HBox([tabular_output_1, tabular_output_2])])])
        accordion.set_title(0, 'Summary')
        accordions.append(accordion)
        return widgets.VBox(accordions)
    
    def get_categorical_multivariate_report(self):
        sub_sub_tab = [widgets.VBox() for _ in range(len(self._pipeline_results))]
        acc = widgets.Accordion(children=sub_sub_tab)
        
        for j_acc, j in zip(self._pipeline_results, range(len(self._pipeline_results))) :
            categorical_multivariate_result = self._pipeline_results[j_acc]['resemblance_evaluation_results']['categorical_multivariate']
            acc.set_title(j, j_acc)
            
            corr_real = categorical_multivariate_result['cramer_real']
            corr_synth = categorical_multivariate_result['cramer_synth']
            norm = categorical_multivariate_result['diff_norm_cramer']
            spearman =  categorical_multivariate_result['cramer_spearman_correlation_coefficient']
            
            # Plotting the correlation difference
            fig = plot_correlation_diff_(corr_real, corr_synth, norm, spearman)
            
            # Convert the Matplotlib figure to an image
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            
            # Create an Image widget
            plot_img = widgets.Image(value=buf.read(), format='png', width=2000, height=1000)
            
            # Add the plot and other widgets to the VBox
            sub_sub_tab[j].children = [plot_img]
            
        return acc


    def get_numerical_multivariate_report(self) :
        sub_sub_tab = [widgets.VBox() for _ in range(len(self._pipeline_results))]
        acc = widgets.Accordion(children=sub_sub_tab)
        
        for j_acc, j in zip(self._pipeline_results, range(len(self._pipeline_results))) :
            numerical_multivariate_result = self._pipeline_results[j_acc]['resemblance_evaluation_results']['numerical_multivariate']
            acc.set_title(j, j_acc)
            
            corr_real = numerical_multivariate_result['pearson_real']
            corr_synth = numerical_multivariate_result['pearson_synth']
            norm = numerical_multivariate_result['pearson_norm_diff'] #pearson_spearman_correlation_coefficient
            spearman = numerical_multivariate_result['pearson_spearman_correlation_coefficient']
            # Plotting the correlation difference
            fig = plot_correlation_diff_(corr_real, corr_synth, norm, spearman)
            
            # Convert the Matplotlib figure to an image
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            
            # Create an Image widget
            plot_img = widgets.Image(value=buf.read(), format='png', width=2000, height=1000)
            
            # Add the plot and other widgets to the VBox
            sub_sub_tab[j].children = [plot_img]
        return acc
    
    def get_numcat_multivariate_report(self) :
        sub_sub_tab = [widgets.VBox() for _ in range(len(self._pipeline_results))]
        acc = widgets.Accordion(children=sub_sub_tab)
        
        for j_acc, j in zip(self._pipeline_results, range(len(self._pipeline_results))) :
            categorical_numerical_multivariate_result = self._pipeline_results[j_acc]['resemblance_evaluation_results']['categorical_numerical_multivariate']

            acc.set_title(j , j_acc) 

            corr_real = categorical_numerical_multivariate_result['corr_ratio_real']
            corr_synth = categorical_numerical_multivariate_result['corr_ratio_synth']
            norm = categorical_numerical_multivariate_result['diff_norm_corr_ratio'] #corr_ratio_spearman_correlation_coefficient
            spearman = categorical_numerical_multivariate_result['corr_ratio_spearman_correlation_coefficient']
            # Plotting the correlation difference
            fig = plot_correlation_ratio_diff_(corr_real, corr_synth, norm, spearman)
            
            # Convert the Matplotlib figure to an image
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            
            # Create an Image widget
            plot_img = widgets.Image(value=buf.read(), format='png', width=2000, height=1000)
            
            # Add the plot and other widgets to the VBox
            sub_sub_tab[j].children = [plot_img]
        return acc

   
    
    def display_dataframe_and_plot(self, real, synthetic, column):
        
        # Display a horizontal line as a ribbon
        display(HTML('<hr>'))
        
        # Plot the distribution
        self.plot_distribution(real, synthetic, column)

    def plot_distribution(self, real, df, column):
        if df[column].dtype == 'float64':
            # Plot overlaid histograms for numerical data
            plt.figure(figsize=(8, 5))
            real[column].plot(kind='hist', density=True, alpha=0.5, label='Real')
            df[column].plot(kind='hist', density=True, alpha=0.5, label='Synthetic')
            plt.title('Histogram Comparison')
            plt.legend()
            plt.show()
        else:
            # For categorical data, use bar plots
            plt.figure(figsize=(8, 5))
            real_counts = real[column].value_counts(normalize=True)
            synthetic_counts = df[column].value_counts(normalize=True)
            
            # Align the indexes for comparison
            combined_counts = pd.DataFrame({'Real': real_counts, 'Synthetic': synthetic_counts}).fillna(0)
            combined_counts.plot(kind='bar', alpha=0.7)
            plt.title('Categorical Distribution Comparison')
            plt.ylabel('Proportion')
            plt.show()



