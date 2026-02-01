from typing import Dict
import pandas as pd
import ipywidgets as widgets

from data_synthesizer.pipeline.pipeline_results import PipelineResults

class PrivacyAnonymeterReport :
    
    def __init__(self, real_data : pd.DataFrame, categorical_columns : list, numerical_columns : list, pipeline_results : Dict[str, PipelineResults],) :
        
        self._pipeline_results = pipeline_results
        self._real_data = real_data
        self._categorical_columns = categorical_columns
        self._numerical_columns = numerical_columns
            

    def _create_report_dataframe(self) :

        self.result_singling_uni = {'attacks_numbers' : [], 
                                'attacks_succeeded': [],
                                'privacy_risk_original': [],
                                'privacy_risk_control': [],
                                'privacy_risk_naive': [],
                                'specific_privacy': [] }
        
        self.result_singling_multi = {'attacks_numbers' : [], 
                        'attacks_succeeded': [],
                        'privacy_risk_original': [],
                        'privacy_risk_control': [],
                        'privacy_risk_naive': [],
                        'specific_privacy': [] }
        
        self.result_linkability_multi = {'attacks_numbers' : [], 
                        'privacy_risk_original': [],
                        'privacy_risk_control': [],
                        'privacy_risk_naive': [],
                        'specific_privacy': [] }

        for j_acc, j in zip(self._pipeline_results, range(len(self._pipeline_results))) :          

            privacy_anonymeter_evaluation_results = self._pipeline_results[j_acc]['privacy_anonymeter_results']
            singling_univariate = privacy_anonymeter_evaluation_results['singling_univariate']
            singling_multivariate = privacy_anonymeter_evaluation_results['singling_multivariate']
            linkability_attacks = privacy_anonymeter_evaluation_results['linkability_attacks']

            self.result_singling_uni['attacks_numbers'].append(singling_univariate.attacks_numbers)
            self.result_singling_uni['attacks_succeeded'].append(singling_univariate.attacks_succeeded)
            self.result_singling_uni['privacy_risk_original'].append(singling_univariate.privacy_risk_original)
            self.result_singling_uni['privacy_risk_control'].append(singling_univariate.privacy_risk_control)
            self.result_singling_uni['privacy_risk_naive'].append(singling_univariate.privacy_risk_naive)
            self.result_singling_uni['specific_privacy'].append(singling_univariate.specific_privacy)

            self.result_singling_multi['attacks_numbers'].append(singling_multivariate.attacks_numbers)
            self.result_singling_multi['attacks_succeeded'].append(singling_multivariate.attacks_succeeded)
            self.result_singling_multi['privacy_risk_original'].append(singling_multivariate.privacy_risk_original)
            self.result_singling_multi['privacy_risk_control'].append(singling_multivariate.privacy_risk_control)
            self.result_singling_multi['privacy_risk_naive'].append(singling_multivariate.privacy_risk_naive)
            self.result_singling_multi['specific_privacy'].append(singling_multivariate.specific_privacy)

            self.result_linkability_multi['attacks_numbers'].append(linkability_attacks.attacks_numbers)
            self.result_linkability_multi['privacy_risk_original'].append(linkability_attacks.privacy_risk_original)
            self.result_linkability_multi['privacy_risk_control'].append(linkability_attacks.privacy_risk_control)
            self.result_linkability_multi['privacy_risk_naive'].append(linkability_attacks.privacy_risk_naive)
            self.result_linkability_multi['specific_privacy'].append(linkability_attacks.specific_privacy)

        result_singling_uni_df = pd.DataFrame.from_dict({'Attacks numbers NA' : self.result_singling_uni['attacks_numbers'],
                                                         'Attacks succeeded' : self.result_singling_uni['attacks_succeeded'],
                                                         'Privacy Risk original' : self.result_singling_uni['privacy_risk_original'],
                                                         'Privacy Risk control' : self.result_singling_uni['privacy_risk_control'],
                                                         'Privacy Risk naive' : self.result_singling_uni['privacy_risk_naive'],
                                                         'Specific Privacy ' : self.result_singling_uni['specific_privacy']}, orient='index', columns=self._pipeline_results.keys())
        
        result_singling_multi_df = pd.DataFrame.from_dict({'Attacks numbers NA' : self.result_singling_multi['attacks_numbers'],
                                                    'Attacks succeeded' : self.result_singling_multi['attacks_succeeded'],
                                                    'Privacy Risk original' : self.result_singling_multi['privacy_risk_original'],
                                                    'Privacy Risk control' : self.result_singling_multi['privacy_risk_control'],
                                                    'Privacy Risk naive' : self.result_singling_multi['privacy_risk_naive'],
                                                    'Specific Privacy ' : self.result_singling_multi['specific_privacy']}, orient='index', columns=self._pipeline_results.keys())

        result_linkability_df = pd.DataFrame.from_dict({'Attacks numbers NA' : self.result_linkability_multi['attacks_numbers'],
                                                    'Privacy Risk original' : self.result_linkability_multi['privacy_risk_original'],
                                                    'Privacy Risk control' : self.result_linkability_multi['privacy_risk_control'],
                                                    'Privacy Risk naive' : self.result_linkability_multi['privacy_risk_naive'],
                                                    'Specific Privacy ' : self.result_linkability_multi['specific_privacy']}, orient='index', columns=self._pipeline_results.keys())
        
        return result_singling_uni_df, result_singling_multi_df, result_linkability_df



    def get_report(self) :
        accordions = []

        result_singling_uni_df, result_singling_multi_df, result_linkability_df = self._create_report_dataframe()

        tabular_output_1 = widgets.HTML(result_singling_uni_df.to_html())
        accordion_1 = widgets.Accordion(children=[widgets.VBox([tabular_output_1])])
        accordion_1.set_title(0, 'singling univariate')
        accordions.append(accordion_1)

 
        tabular_output_2 = widgets.HTML(result_singling_multi_df.to_html())
        accordion_2 = widgets.Accordion(children=[widgets.VBox([tabular_output_2])])
        accordion_2.set_title(0, 'singling multivariate')
        accordions.append(accordion_2)     

        tabular_output_3 = widgets.HTML(result_linkability_df.to_html())
        accordion_3 = widgets.Accordion(children=[widgets.VBox([tabular_output_3])])
        accordion_3.set_title(0, 'linkability')
        accordions.append(accordion_3)      

        return widgets.VBox(accordions)

