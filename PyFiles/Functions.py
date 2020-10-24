from tqdm import tqdm 
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


def stacked_model(models):
    """Creates a stacked model given a dictionary of SciKitLearn models
    -----------------------------------------
    Input: 
        models: Dictionary containing the model name and function.
    
    Output: 
        stack_model: A new dictionary containing a SciKitLearn StackingClassifier object
    -----------------------------------------"""

    stack_m = [] 
    for model, m in models.items(): 
        stack_m.append((model, m))
    stack_model = StackingClassifier(estimators = stack_m, final_estimator = LogisticRegression(max_iter = 2500), cv = 3)
    models['Stacked'] = stack_model
    
    return models


def test_models(x_train, y_train, models, n_jobs = 2):
    """
    Test all models given.
    
    This will test each model on its own using RepeatedStratifiedKFold then it will test a stacking classifier with every single model in the dictionary.  
    
    returns: vanilla_dict (contains results and model names)"""
    results = []
    model_names = []
    pbar = tqdm(models.items())
    
    for model, m in pbar: 
        pbar.set_description(f'Evaluating {model.upper()}')
        cv = RepeatedStratifiedKFold(n_splits = 10, n_repeats = 10)
        scores = cross_val_score(m, x_train, y_train, scoring = 'accuracy', cv = cv, n_jobs = n_jobs, 
                                 error_score = 'raise')
        results.append(scores)
        model_names.append(model)
    vanilla_dict = {i:y for i,y in zip(model_names, results)}
   
    return vanilla_dict


def plot_model_results(model_dict, figure_title= 'Model Results', filepath = None, figsize = (10, 8)):
    
    """Plots and saves an image of the plot.
    Input:
    results: the results of the models, gained from test_models()
    model_names: the names of models used, gained from test_models()
    filepath: the filepath for the graph image to be saved to
    """
    model_names = [i for i in model_dict.keys()]
    results = [i for k,i in model_dict.items()]
    plt.figure(figsize = figsize)
    plt.boxplot(results, labels = model_names, showmeans = True)
    plt.title(f'{figure_title}')
    plt.ylabel('Accuracy'); plt.xlabel('Model')
    if filepath:
        plt.savefig(filepath)
        plt.show()
           
        
