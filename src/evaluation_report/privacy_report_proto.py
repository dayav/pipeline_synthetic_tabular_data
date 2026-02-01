import pickle
from typing import Dict
import pandas as pd
import numpy as np
import ipywidgets as widgets
from matplotlib import pyplot as plt
from IPython.display import display, HTML

from data_evaluator.univariate_evaluator import UnivariateEvaluator
from data_evaluator.privacy_evaluation import PrivacyEvaluator, SimilarityType
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

            
    def build_report(self, no_gower = True) :
        self.results_synthetic = {}
        self.results_real = {}
        self.gower = {}
        self.dissimilarity = {}
        self.real_dissimilarity = {}
        self.share_dissimilarity = {}
        self.dissimilarity_idx = {}
        self.dissimilarity_idx_test = {}
        self.dissimilarity_categorical = {}
        self.dissimilarity_numerical= {}
        self.dissimilarity_idx_categorical = {}
        self.dissimilarity_idx_numerical = {}
        self.dissimilarity_mean_imbalanced = {}
        self.dissimilarity_idx_mean_imbalanced = {}
        self.dissimilarity_preprocess = {}
        self.dissimilarity_idx_preprocess = {}
        self.epsilon_dissimilarity = {}
        self.epsilon_dissimilarity_idx = {}


        for j_acc, j in zip(self._pipeline_results, range(len(self._pipeline_results))) :

            privacy_evaluation_results = self._pipeline_results[j_acc]['privacy_evaluation_results']
            attribute_synthetic_prediction = privacy_evaluation_results['attribute_synthetic_prediction']
            attribute_real_prediction = privacy_evaluation_results['attribute_real_prediction']

            self.results_synthetic[j_acc] = attribute_synthetic_prediction
            self.results_real[j_acc] = attribute_real_prediction

            self._uni_evaluators[j_acc].evaluate_categorical_stat_evaluation()
            self._uni_evaluators[j_acc].evaluate_numerical_stat_evaluation()


            if no_gower :
                if (SimilarityType.DISSIMILARITY in self._dissimilarities) :
                    diss = self._privacy_evaluator[j_acc].evaluate_similarity_stdg(SimilarityType.DISSIMILARITY)
                    self.dissimilarity[j_acc] = diss[0]
                    id_test = j_acc+'-test'
                    self.dissimilarity[id_test] = diss[1]
                    self.dissimilarity_idx[j_acc] = diss[2]
                    self.dissimilarity_idx[id_test] = diss[3]
                    self.share_dissimilarity[j_acc] = diss[4]

                    
                if (SimilarityType.EPSILON_DISSIMILARITY in self._dissimilarities) :
                    real_dissimilarity = self._privacy_evaluator[j_acc].evaluate_similarity_stdg(SimilarityType.EPSILON_DISSIMILARITY)
                    self.real_dissimilarity[j_acc] = real_dissimilarity[0]
                    id_synth = j_acc+'-synth'
                    self.real_dissimilarity[id_synth] = real_dissimilarity[1]
                    self.epsilon_dissimilarity_idx[j_acc] = real_dissimilarity[2]
                    self.epsilon_dissimilarity_idx[id_synth] = real_dissimilarity[3]
                    self.epsilon_dissimilarity[j_acc] = real_dissimilarity[4]

                if (SimilarityType.DISSIMILARITY_CATEGORICAL in self._dissimilarities) :
                    diss_categorical = self._privacy_evaluator[j_acc].evaluate_similarity_stdg(SimilarityType.DISSIMILARITY_CATEGORICAL)
                    self.dissimilarity_categorical[j_acc] = diss_categorical[0]
                    self.dissimilarity_idx_categorical[j_acc] = diss_categorical[1]

                if (SimilarityType.DISSIMILARITY_NUMERICAL in self._dissimilarities) :
                    diss_numerical = self._privacy_evaluator[j_acc].evaluate_similarity_stdg(SimilarityType.DISSIMILARITY_NUMERICAL)
                    self.dissimilarity_numerical[j_acc] = diss_numerical[0]
                    self.dissimilarity_idx_numerical[j_acc] = diss_numerical[1]

                if (SimilarityType.DISSIMILARITY_MEAN_IMBALANCED in self._dissimilarities) :   
                    diss_mean_imbalanced = self._privacy_evaluator[j_acc].evaluate_similarity_stdg(SimilarityType.DISSIMILARITY_MEAN_IMBALANCED)
                    self.dissimilarity_mean_imbalanced[j_acc] = diss_mean_imbalanced[0]
                    self.dissimilarity_idx_mean_imbalanced[j_acc] = diss_mean_imbalanced[1] 

                
                if (SimilarityType.DISSIMILARITY_WITH_PREPROCESSING_CAT in self._dissimilarities) :   
                    diss_preprocess = self._privacy_evaluator[j_acc].evaluate_similarity_stdg(SimilarityType.DISSIMILARITY_WITH_PREPROCESSING_CAT)
                    self.dissimilarity_preprocess[j_acc] = diss_preprocess[0]
                    self.dissimilarity_idx_preprocess[j_acc] = diss_preprocess[1]
         
            else :
                gower = self._privacy_evaluator[j_acc].evaluate_similarity_stdg(SimilarityType.GOWER)
                self.gower[j_acc] = gower

 
    def get_report(self) :
        accordions = []
        output_most_sim = []
        dropdown_most_sim = []
        output_most_sim_test = []
        dropdown_most_sim_test = []
        dropdown_most_sim_categorical = []
        dropdown_most_sim_mean_imbalanced = []
        output_most_sim_categorical = []
        dropdown_most_sim_numerical = []
        output_most_sim_numerical = []
        output_most_sim_mean_imbalanced = []
        output_most_sim_preprocess = []
        dropdown_most_sim_preprocess = []
        dropdown_most_sim_epsilon = []      
        output_most_sim_epsilon = [] 
        # difference = {key: abs(self.results_synthetic[key] - self.results_real[key]) for key in self.results_synthetic if key in self.results_real}
        
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

        # difference_percent = {key: difference[key]/self.results_real[key] for key in difference if key in self.results_real}
        # differences_percent = {}

        # for key in differences :
        #     if key in self.results_real :
        #         differences_percent[key] = {key_2: differences[key][key_2]/self.results_real[key][key_2] for key_2 in differences[key]}

        if not self.gower :
            if (SimilarityType.DISSIMILARITY in self._dissimilarities) : 
                dissimilarity_df = pd.DataFrame(self.dissimilarity)
                dict_of_means = {key: np.mean(value) for key, value in dissimilarity_df.items() if key in self._evaluation_dict}
                share_dissimilarity_df = pd.DataFrame([dict_of_means, self.share_dissimilarity])
                share_dissimilarity_df.index = ['Minimum means', 'share']
            if (SimilarityType.DISSIMILARITY_CATEGORICAL in self._dissimilarities) : dissimilarity_categorical_df = pd.DataFrame(self.dissimilarity_categorical)
            if (SimilarityType.DISSIMILARITY_NUMERICAL in self._dissimilarities) : dissimilarity_numerical_df = pd.DataFrame(self.dissimilarity_numerical)
            if (SimilarityType.DISSIMILARITY_MEAN_IMBALANCED in self._dissimilarities) :  dissimilarity_mean_imbalanced_df = pd.DataFrame(self.dissimilarity_mean_imbalanced)
            if (SimilarityType.DISSIMILARITY_WITH_PREPROCESSING_CAT in self._dissimilarities) :  dissimilarity_preprocess_df = pd.DataFrame(self.dissimilarity_preprocess)
            if (SimilarityType.EPSILON_DISSIMILARITY in self._dissimilarities) :
                dissimilarity_epsilon_df = pd.DataFrame(self.real_dissimilarity)
                dict_of_means = {key: np.mean(value) for key, value in dissimilarity_epsilon_df.items() if key in self._evaluation_dict}
                epsilon_dissimilarity_df = pd.DataFrame([dict_of_means, self.epsilon_dissimilarity])
                epsilon_dissimilarity_df.index = ['Minimum means', 'epsilon']

        for j_acc, j in zip(self._evaluation_dict, range(len(self._evaluation_dict))) :

            eval_cat = self._uni_evaluators[j_acc].get_categorical_stat_evaluation()
            eval_num = self._uni_evaluators[j_acc].get_numerical_stat_evaluation()
            difference = differences[j_acc]
            result_real = primary_results_real.get(j_acc, {})
            result_synth = primary_results_synth.get(j_acc, {})

            js_div = {}
            for key in difference :
                if key in eval_cat[1].data.index :
                    js_div[key] = eval_cat[1].data.at[key, 'JS_divergence']
                if key in eval_num[1].data.index :
                    js_div[key] = eval_num[1].data.at[key, 'JS_divergence']

            difference_percent = {key: difference[key]/result_real[key] for key in difference if key in result_real and result_real[key]}
            inference_result_df = pd.DataFrame.from_dict({'Accuracy/MSE - real' : result_real,
                                    'Accuracy/MSE - synthetic' : result_synth,
                                    'difference' : difference,
                                    'difference_percent' : difference_percent,
                                    'JS divergence' : js_div})
            



            tabular_output_1 = widgets.HTML(inference_result_df.to_html())        

            accordion = widgets.Accordion(children=[widgets.VBox([widgets.HBox([tabular_output_1])])])
            #to update
            accordion_title = f"{j_acc} Inference"
            accordion.set_title(0, accordion_title)
            accordions.append(accordion)


            if not self.gower : 
                if (SimilarityType.DISSIMILARITY in self._dissimilarities) :               
                    synthetic_ids = [i for i in range(20)]

                    id_test = j_acc+'-test'
                    dropdown = widgets.Dropdown(options=synthetic_ids, description=j_acc)
                    dropdown_most_sim.append(dropdown)
                    df_real, df_synth = self.creat_datframe_for_vizualization(j_acc, self.dissimilarity_idx[j_acc])
                    output_most_sim.append(widgets.interactive_output(self.display_most_similar ,
                                                            {'real':widgets.fixed(df_real),
                                                            'synth': widgets.fixed(df_synth), 
                                                            'synthetic_id': dropdown, 
                                                            'idx':widgets.fixed(self.dissimilarity_idx[j_acc][:20]),
                                                            'dissimilarity' : widgets.fixed(self.dissimilarity[j_acc][:20]) }))

                    dropdown = widgets.Dropdown(options=synthetic_ids, description=id_test)
                    dropdown_most_sim.append(dropdown)
                    df_real, df_synth = self.creat_datframe_for_vizualization(j_acc, self.dissimilarity_idx[id_test])
                    output_most_sim.append(widgets.interactive_output(self.display_most_similar ,
                                                            {'real':widgets.fixed(df_real),
                                                            'synth': widgets.fixed(df_synth), 
                                                            'synthetic_id': dropdown, 
                                                            'idx':widgets.fixed(self.dissimilarity_idx[id_test][:20]),
                                                            'dissimilarity' : widgets.fixed(self.dissimilarity[id_test][:20]) }))
                    
                if (SimilarityType.EPSILON_DISSIMILARITY in self._dissimilarities) :               
                    real_ids = [i for i in range(20)]

                    id_synth = j_acc+'-synth'
                    dropdown = widgets.Dropdown(options=real_ids, description=j_acc)
                    dropdown_most_sim_epsilon.append(dropdown)
                    df_real_other, df_real  = self.creat_datframe_for_vizualization_epsilon(j_acc, self.epsilon_dissimilarity_idx[j_acc], False)
                    output_most_sim_epsilon.append(widgets.interactive_output(self.display_most_similar_epsilon ,
                                                            {'real':widgets.fixed(df_real),
                                                            'other': widgets.fixed(df_real_other), 
                                                            'real_id': dropdown, 
                                                            'idx':widgets.fixed(self.epsilon_dissimilarity_idx[j_acc][:20]),
                                                            'dissimilarity' : widgets.fixed(self.real_dissimilarity[j_acc][:20]) }))

                    dropdown = widgets.Dropdown(options=real_ids, description=id_synth)
                    dropdown_most_sim_epsilon.append(dropdown)
                    df_synth, df_real  = self.creat_datframe_for_vizualization_epsilon(j_acc, self.epsilon_dissimilarity_idx[id_synth], True)
                    output_most_sim_epsilon.append(widgets.interactive_output(self.display_most_similar_epsilon ,
                                                            {'real':widgets.fixed(df_real),
                                                            'other': widgets.fixed(df_synth), 
                                                            'real_id': dropdown, 
                                                            'idx':widgets.fixed(self.epsilon_dissimilarity_idx[id_synth][:20]),
                                                            'dissimilarity' : widgets.fixed(self.real_dissimilarity[id_synth][:20]) }))

                if (SimilarityType.DISSIMILARITY_CATEGORICAL in self._dissimilarities) :
                    synthetic_ids_categorical = [i for i in range(20)]
                    dropdown_categorical = widgets.Dropdown(options=synthetic_ids_categorical, description=j_acc)
                    dropdown_most_sim_categorical.append(dropdown_categorical)
                    df_real, df_synth = self.creat_datframe_for_vizualization(j_acc, self.dissimilarity_idx_categorical[j_acc])
                    output_most_sim_categorical.append(widgets.interactive_output(self.display_most_similar ,
                                                            {'real':widgets.fixed(df_real),
                                                            'synth': widgets.fixed(df_synth), 
                                                            'synthetic_id': dropdown_categorical, 
                                                            'idx':widgets.fixed(self.dissimilarity_idx_categorical[j_acc][:20]),
                                                            'dissimilarity' : widgets.fixed(self.dissimilarity_categorical[j_acc][:20]) }))

                if (SimilarityType.DISSIMILARITY_NUMERICAL in self._dissimilarities) :
                    synthetic_ids_numerical = [i for i in range(20)]
                    dropdown_numerical = widgets.Dropdown(options=synthetic_ids_numerical, description=j_acc)
                    dropdown_most_sim_numerical.append(dropdown_numerical)
                    df_real, df_synth = self.creat_datframe_for_vizualization(j_acc, self.dissimilarity_idx_numerical[j_acc])
                    output_most_sim_numerical.append(widgets.interactive_output(self.display_most_similar ,
                                                            {'real':widgets.fixed(df_real),
                                                            'synth': widgets.fixed(df_synth), 
                                                            'synthetic_id': dropdown_numerical, 
                                                            'idx':widgets.fixed(self.dissimilarity_idx_numerical[j_acc][:20]),
                                                            'dissimilarity' : widgets.fixed(self.dissimilarity_numerical[j_acc][:20]) }))

                if (SimilarityType.DISSIMILARITY_MEAN_IMBALANCED in self._dissimilarities) :
                    synthetic_ids_mean_imbalanced = [i for i in range(20)]
                    dropdown_mean_imbalanced = widgets.Dropdown(options=synthetic_ids_mean_imbalanced, description=j_acc)
                    dropdown_most_sim_mean_imbalanced.append(dropdown_mean_imbalanced)

                    df_real, df_synth = self.creat_datframe_for_vizualization(j_acc, self.dissimilarity_idx_mean_imbalanced[j_acc])
                    output_most_sim_mean_imbalanced.append(widgets.interactive_output(self.display_most_similar ,
                                                            {'real':widgets.fixed(df_real),
                                                            'synth': widgets.fixed(df_synth), 
                                                            'synthetic_id': dropdown_mean_imbalanced, 
                                                            'idx':widgets.fixed(self.dissimilarity_idx_mean_imbalanced[j_acc][:20]),
                                                            'dissimilarity' : widgets.fixed(self.dissimilarity_mean_imbalanced[j_acc][:20]) }))

                if (SimilarityType.DISSIMILARITY_WITH_PREPROCESSING_CAT in self._dissimilarities) :
                    synthetic_ids_preprocess = [i for i in range(20)]
                    dropdown_preprocess = widgets.Dropdown(options=synthetic_ids_preprocess, description=j_acc)
                    dropdown_most_sim_preprocess.append(dropdown_preprocess)

                    df_real, df_synth = self.creat_datframe_for_vizualization(j_acc, self.dissimilarity_idx_preprocess[j_acc])
                    output_most_sim_preprocess.append(widgets.interactive_output(self.display_most_similar ,
                                                            {'real':widgets.fixed(df_real),
                                                            'synth': widgets.fixed(df_synth), 
                                                            'synthetic_id': dropdown_preprocess, 
                                                            'idx':widgets.fixed(self.dissimilarity_idx_preprocess[j_acc][:20]),
                                                            'dissimilarity' : widgets.fixed(self.dissimilarity_preprocess[j_acc][:20]) }))
                    
                if (SimilarityType.DISSIMILARITY_WITH_PREPROCESSING_CAT in self._dissimilarities) :
                    synthetic_ids_preprocess = [i for i in range(20)]
                    dropdown_preprocess = widgets.Dropdown(options=synthetic_ids_preprocess, description=j_acc)
                    dropdown_most_sim_preprocess.append(dropdown_preprocess)

                    df_real, df_synth = self.creat_datframe_for_vizualization(j_acc, self.dissimilarity_idx_preprocess[j_acc])
                    output_most_sim_preprocess.append(widgets.interactive_output(self.display_most_similar ,
                                                            {'real':widgets.fixed(df_real),
                                                            'synth': widgets.fixed(df_synth), 
                                                            'synthetic_id': dropdown_preprocess, 
                                                            'idx':widgets.fixed(self.dissimilarity_idx_preprocess[j_acc][:20]),
                                                            'dissimilarity' : widgets.fixed(self.dissimilarity_preprocess[j_acc][:20]) }))             

        
        if not self.gower :
            if (SimilarityType.DISSIMILARITY in self._dissimilarities) :
                output_diss = widgets.Output()           
                
                with output_diss :
                    plt.figure(figsize=(10, 6))  
                    dissimilarity_df.boxplot()
                    plt.title(' min - dissimilarity distance')
                    plt.ylabel('Values')
                    plt.show() 

            if (SimilarityType.EPSILON_DISSIMILARITY in self._dissimilarities) :
                output_diss_epsilon = widgets.Output()           
                
                with output_diss_epsilon :
                    plt.figure(figsize=(10, 6))  
                    dissimilarity_epsilon_df.boxplot()
                    plt.title(' min - real dissimilarity distance')
                    plt.ylabel('Values')
                    plt.show()  

            if (SimilarityType.DISSIMILARITY_CATEGORICAL in self._dissimilarities) :
                output_diss_cat = widgets.Output()
                
                with output_diss_cat :
                    plt.figure(figsize=(10, 6))  
                    dissimilarity_categorical_df.boxplot()
                    plt.title(' min - dissimilarity categorical distance')
                    plt.ylabel('Values')
                    plt.show()

            if (SimilarityType.DISSIMILARITY_NUMERICAL in self._dissimilarities) :
                output_diss_num = widgets.Output()
            
                with output_diss_num :
                    plt.figure(figsize=(10, 6))  
                    dissimilarity_numerical_df.boxplot()
                    plt.title(' min - dissimilarity numerical distance')
                    plt.ylabel('Values')
                    plt.show()

            if (SimilarityType.DISSIMILARITY_MEAN_IMBALANCED in self._dissimilarities) :
                output_diss_mean_imbalanced = widgets.Output()
            
                with output_diss_mean_imbalanced :
                    plt.figure(figsize=(10, 6))  
                    dissimilarity_mean_imbalanced_df.boxplot()
                    plt.title(' min - dissimilarity mean imbalanced distance')
                    plt.ylabel('Values')
                    plt.show()

            if (SimilarityType.DISSIMILARITY_WITH_PREPROCESSING_CAT in self._dissimilarities) :
                output_diss_preprocess = widgets.Output()
            
                with output_diss_preprocess :
                    plt.figure(figsize=(10, 6))  
                    dissimilarity_preprocess_df.boxplot()
                    plt.title(' min - dissimilarity with preprocess distance')
                    plt.ylabel('Values')
                    plt.show()

        if self.gower :
        
            gower_diff_df = pd.DataFrame(self.gower)
            output = widgets.Output()

            with output :
                plt.figure(figsize=(10, 6)) 
                gower_diff_df.boxplot()
                plt.title(' min - gower distance')
                plt.ylabel('Values')
                plt.show()

        if (SimilarityType.DISSIMILARITY in self._dissimilarities) :
            accordion_1 = widgets.Accordion(children=[widgets.VBox([output_diss, widgets.VBox([widgets.HTML(share_dissimilarity_df.to_html())]), widgets.HBox(dropdown_most_sim), widgets.HBox(output_most_sim)])])
            accordions.append(accordion_1)

        if (SimilarityType.DISSIMILARITY_CATEGORICAL in self._dissimilarities) :
            accordion_2 = widgets.Accordion(children=[widgets.VBox([output_diss_cat, widgets.HBox(dropdown_most_sim_categorical), widgets.HBox(output_most_sim_categorical)])])
            accordions.append(accordion_2)
        if (SimilarityType.DISSIMILARITY_NUMERICAL in self._dissimilarities) :
            accordion_3 = widgets.Accordion(children=[widgets.VBox([output_diss_num, widgets.HBox(dropdown_most_sim_numerical), widgets.HBox(output_most_sim_numerical)])])
            accordions.append(accordion_3)
        if (SimilarityType.DISSIMILARITY_MEAN_IMBALANCED in self._dissimilarities) :
            accordion_4 = widgets.Accordion(children=[widgets.VBox([output_diss_mean_imbalanced, widgets.HBox(dropdown_most_sim_mean_imbalanced), widgets.HBox(output_most_sim_mean_imbalanced)])])
            accordions.append(accordion_4) 

        if (SimilarityType.DISSIMILARITY_WITH_PREPROCESSING_CAT in self._dissimilarities) :
            accordion_5 = widgets.Accordion(children=[widgets.VBox([output_diss_preprocess, widgets.HBox(dropdown_most_sim_preprocess), widgets.HBox(output_most_sim_preprocess)])])
            accordions.append(accordion_5)

        if (SimilarityType.EPSILON_DISSIMILARITY in self._dissimilarities) :
            accordion_6 = widgets.Accordion(children=[widgets.VBox([output_diss_epsilon, widgets.VBox([widgets.HTML(epsilon_dissimilarity_df.to_html())]), widgets.HBox(dropdown_most_sim_epsilon), widgets.HBox(output_most_sim_epsilon)])])
            accordions.append(accordion_6)


        return widgets.VBox(accordions)
    
    def display_most_similar(self, real, synth, synthetic_id, idx, dissimilarity):
        # Display a horizontal line as a ribbon
        display(HTML('<hr>'))
        
        synth_row = synth.iloc[synthetic_id]
        print('real id : ',  idx[synthetic_id])
        print('dissimilarity : ', dissimilarity[synthetic_id])
        real_row = real[real['index'] == idx[synthetic_id]]

        new_df = pd.DataFrame({'Features': synth.columns, 
                       'Synth': synth_row.values,
                       'Real': real_row.values[0][:-1]})
        
        display(new_df)
        del new_df

    def display_most_similar_epsilon(self, real, other, real_id, idx, dissimilarity):
        # Display a horizontal line as a ribbon
        display(HTML('<hr>'))
        
        real_row = real.iloc[real_id]
        print('other id : ',  idx[real_id])
        print('dissimilarity : ', dissimilarity[real_id])
        other_row = other[other['index'] == idx[real_id]]

        new_df = pd.DataFrame({'Features': real.columns, 
                       'Real': real_row.values,
                       'Other': other_row.values[0][:-1]})
        
        display(new_df)
        del new_df


    
    def save_evaluation(self, file_name) :

        saved = {'evaluation_dict' : self._evaluation_dict,
                'uni_evaluators' : self._uni_evaluators , 
                'privacy_evaluator' : self._privacy_evaluator,
                'dissimilarities' : self._dissimilarities,
                'results_synthetic' : self.results_synthetic,
                'results_real' : self.results_real,
                'gower' : self.gower,
                'dissimilarity' : self.dissimilarity,
                'dissimilarity_idx' : self.dissimilarity_idx,
                'share_dissimilarity' : self.share_dissimilarity,
                'dissimilarity_idx_test' : self.dissimilarity_idx_test,
                'dissimilarity_categorical' : self.dissimilarity_categorical, 
                'dissimilarity_idx_categorical' : self.dissimilarity_idx_categorical,
                'dissimilarity_numerical' : self.dissimilarity_numerical,
                'dissimilarity_idx_numerical' : self.dissimilarity_idx_numerical,
                'dissimilarity_mean_imbalanced' : self.dissimilarity_mean_imbalanced,
                'dissimilarity_idx_mean_imbalanced' : self.dissimilarity_idx_mean_imbalanced,
                'dissimilarity_preprocess' : self.dissimilarity_preprocess,
                'dissimilarity_idx_preprocess' : self.dissimilarity_idx_preprocess,
                'real_dissimilarity': self.real_dissimilarity,
                'epsilon_dissimilarity': self.epsilon_dissimilarity,
                'epsilon_dissimilarity_idx': self.epsilon_dissimilarity_idx }
        
        with open(file_name +'.pkl', 'wb') as file:
            pickle.dump(saved, file)

    @staticmethod
    def load_evaluation(file_name) :
        with open(file_name, 'rb') as file:
            loaded_object = pickle.load(file)
        
        if 'gower' in loaded_object : 
            report = PrivacyReport(evaluation_dict=None, loaded=loaded_object)
        else : 
            report = None

        return report

    def creat_datframe_for_vizualization(self,j_acc, idx):

        real_df = self._evaluation_dict[j_acc][0].iloc[idx[:20]].copy()
        real_df['index'] = idx[:20]
        synth_df = self._evaluation_dict[j_acc][1].iloc[:20].copy()
        return real_df, synth_df
    
    def creat_datframe_for_vizualization_epsilon(self,j_acc, idx, is_synth):

        if (is_synth) :
            other_df = self._evaluation_dict[j_acc][1].iloc[idx[:20]].copy()
        else :
            other_df = self._evaluation_dict[j_acc][0].iloc[idx[:20]].copy()
        other_df['index'] = idx[:20]
        real_df = self._evaluation_dict[j_acc][0].iloc[:20].copy()
        return other_df, real_df
