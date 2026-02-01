import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display
import json
from enum import Enum

class ModelType(Enum):
    CTGAN = 1
    DP_CGANS = 2
    TVAE = 3

def plot_training_loss(training_dict) :

    accordions = []

    for model in training_dict :

        path_list = training_dict[model]
        accordions.append(_get_model_figs(path_list, model))

    tab = widgets.Tab(accordions)
    for title, i in zip(training_dict, range(len(training_dict))) :
        tab.set_title(i, str(title)) 
    return tab



def _get_DPCGANS_params_training(loss_values) :
        
        wanted_keys = ['enforce_min_max_values',
                       'enforce_rounding',
                       'field_names', 
                       'field_types',
                       'field_transformers', 
                       'anonymize_fields',
                       'primary_key',
                       'constraints',
                       'table_metadata',
                       'embedding_dim',
                       'generator_dim',
                       'discriminator_dim',
                       'generator_lr',
                       'generator_decay',
                       'log_frequency',
                       'discriminator_lr',
                       'discriminator_decay',
                       'batch_size',
                       'discriminator_steps',
                       'log_frequency',
                       'pac',
                       'cuda',
                       'rounding',
                       'min_value',
                       'max_value',
                       'private',
                       'compress_dims',
                       'decompress_dims',
                       'l2scale',
                       'loss_factor'
                       ]
        
        def my_filtering_function(pair):
            key, value = pair
            if key in wanted_keys:
                return True  # keep pair in the filtered dictionary
            else:
                return False  # filter pair out of the dictionary
        
        filtered_params = dict(filter(my_filtering_function, loss_values.items()))

        return filtered_params

def _get_model_figs(path_list, model_type) :
    widgets_accordion = []
    loss_1, loss_2 = _get_loss_string(model_type)
    for path in path_list :

        with open(path, "r") as fp:
        # Load the dictionary from the file
            loss_values = json.load(fp)


        fig = go.FigureWidget(data=[go.Scatter(x=loss_values['Epoch'], y=loss_values[loss_1], name=loss_1),
                            go.Scatter(x=loss_values['Epoch'], y=loss_values[loss_2], name=loss_2)])

        fig.update_layout(title="Training loss", xaxis_title='Epoch', yaxis_title='Loss')
        out = widgets.Output()
        with out:
            display(_get_DPCGANS_params_training(loss_values))
        widget_accordion = widgets.VBox([fig, out])
    
    
        widgets_accordion.append(widget_accordion)

    acc = widgets.Accordion(widgets_accordion)

    for path, i in zip(path_list, range(len(path_list))) :
        acc.set_title(i, path)

    return acc

def _get_loss_string(model_type) :
    if (model_type == ModelType.CTGAN or model_type == ModelType.DP_CGANS) :
        return 'Generator Loss', 'Discriminator Loss'
    elif (model_type == ModelType.TVAE) :
        return 'Reconstruct Loss', 'KLD Loss'
    else :
        return '', ''

