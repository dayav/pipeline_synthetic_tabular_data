import io
from typing import Dict
import pandas as pd
import numpy as np
import ipywidgets as widgets
from matplotlib import pyplot as plt

from data_synthesizer.pipeline.pipeline_results import PipelineResults


class PrivacyReport :
    
    def __init__(self, real_data : pd.DataFrame, categorical_columns : list, numerical_columns : list, pipeline_results : Dict[str, PipelineResults] ) :
        self._pipeline_results = pipeline_results
        self._real_data = real_data
        self._categorical_columns = categorical_columns
        self._numerical_columns = numerical_columns

    def _primary_metric(self, attribute: str, metrics):
        """
        Extract a single comparable metric from a metrics dict. For categorical
        attributes we favor Accuracy; for numerical ones we favor RMSE.
        """
        if isinstance(metrics, dict):
            if attribute in self._categorical_columns:
                return metrics.get("Accuracy")
            if attribute in self._numerical_columns:
                return metrics.get("RMSE") or metrics.get("MSE")
        return metrics

 
    def get_report(self) :

        self.results_synthetic = {}
        self.results_real = {}
        self.gower = {}
        self.dissimilarity = {}
        self.real_dissimilarity = {}
        self.share_dissimilarity = {}
        self.epsilon_dissimilarity = {}
        self.jensen_shanon_categorical = {}
        self.jensen_shanon_numerical = {}

        for j_acc, j in zip(self._pipeline_results, range(len(self._pipeline_results))) :

            privacy_evaluation_results = self._pipeline_results[j_acc]['privacy_evaluation_results']
            attribute_synthetic_prediction = privacy_evaluation_results['attribute_synthetic_prediction']
            attribute_real_prediction = privacy_evaluation_results['attribute_real_prediction']
            dissimilarity_synthetic_real = privacy_evaluation_results['dissimilarity_synthetic_real']
            dissimilarity_synthetic_test = privacy_evaluation_results['dissimilarity_synthetic_test']          
            self.jensen_shanon_categorical[j_acc] = privacy_evaluation_results['jensen_shanon_categorical']
            self.jensen_shanon_numerical[j_acc] = privacy_evaluation_results['jensen_shanon_numerical']
            self.share_dissimilarity[j_acc] = privacy_evaluation_results['share']

            self.results_synthetic[j_acc] = attribute_synthetic_prediction
            self.results_real[j_acc] = attribute_real_prediction

            self.dissimilarity[j_acc] = dissimilarity_synthetic_real
            id_test = j_acc+'-test'
            self.dissimilarity[id_test] = dissimilarity_synthetic_test


        accordions = []
        
        differences = {}
        primary_results_real = {}
        primary_results_synth = {}
        for key in self.results_synthetic :
            if key in self.results_real :
                primary_results_synth[key] = {att: self._primary_metric(att, val) for att, val in self.results_synthetic[key].items()}
                primary_results_real[key] = {att: self._primary_metric(att, val) for att, val in self.results_real[key].items()}
                diffs = {}
                for att, synth_val in primary_results_synth[key].items():
                    real_val = primary_results_real[key].get(att)
                    if synth_val is None or real_val is None:
                        continue
                    diffs[att] = abs(synth_val - real_val)
                differences[key] = diffs


        dissimilarity_df = pd.DataFrame(self.dissimilarity)
        dict_of_means = {key: np.mean(value) for key, value in dissimilarity_df.items() if key in self._pipeline_results}
        share_dissimilarity_df = pd.DataFrame([dict_of_means, self.share_dissimilarity])
        share_dissimilarity_df.index = ['Minimum means', 'share']

        for j_acc, j in zip(self._pipeline_results, range(len(self._pipeline_results))) :

            difference = differences[j_acc]
            result_real = primary_results_real.get(j_acc, {})
            result_synth = primary_results_synth.get(j_acc, {})

            js_div = {}
            for key in difference :
                if key in self.jensen_shanon_categorical[j_acc].index :
                    js_div[key] = self.jensen_shanon_categorical[j_acc].at[key, 'JS_divergence']
                if key in self.jensen_shanon_numerical[j_acc].index :
                    js_div[key] = self.jensen_shanon_numerical[j_acc].at[key, 'JS_divergence']

            difference_percent = {key: difference[key]/result_real[key] for key in difference if key in result_real and result_real[key]}
            inference_result_df = pd.DataFrame.from_dict({'Accuracy/MSE - real' : result_real,
                                    'Accuracy/MSE - synthetic' : result_synth,
                                    'difference' : difference,
                                    'difference_percent' : difference_percent,
                                    'JS divergence' : js_div})
            

            tabular_output_1 = widgets.HTML(inference_result_df.to_html())        

            accordion = widgets.Accordion(children=[widgets.VBox([widgets.HBox([tabular_output_1])])])
            accordion_title = f"{j_acc} Inference"
            accordion.set_title(0, accordion_title)
            accordions.append(accordion)
            
            #to update
            # Create the dissimilarity plot
            fig, ax = plt.subplots(figsize=(10, 6))
            dissimilarity_df.boxplot(ax=ax)
            ax.set_title('Min - Dissimilarity Distance')
            ax.set_ylabel('Values')
      
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close(fig)         
            
            output_diss = widgets.Image(value=buf.read(), format='png')          

        accordion_boxplot = widgets.Accordion(children=[widgets.VBox([output_diss, widgets.VBox([widgets.HTML(share_dissimilarity_df.to_html())])])])
        accordion_boxplot_title = "dissimilarity distance"
        accordion_boxplot.set_title(0, accordion_boxplot_title)
        accordions.append(accordion_boxplot)

        return widgets.VBox(accordions)
