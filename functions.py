import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import cross_val_score

def unbalanced_features(table: pd.DataFrame) -> pd.DataFrame:
    """Takes in a table and returns another table with column names 
    and information about most frequent values in those columns.
    """
    
    feature_names = [column for column in table]
    qty_most_freq_val = [table[column].value_counts().iloc[0] for column in table]
    qty_most_freq_val_perc = [table[column].value_counts().iloc[0] / len(table) * 100 for column in table]
    
    most_freq_val_table = pd.DataFrame({"Feature Name": feature_names, "QTY of most freq. value": qty_most_freq_val,
                                        "% of Total Values": qty_most_freq_val_perc}).sort_values(by="% of Total Values", ascending=False)
    
    return most_freq_val_table


def feature_selector(eli5_df: pd.DataFrame, baseline_score: float, model: object, train_evaluation_df: pd.DataFrame, 
                     target: str, start=1, end=11, step=1, divider=10000) -> pd.DataFrame:
    """Takes in an eli5 dataframe with weight of features, iterates through various thresholds of weight, 
    finds the lowest score and returns dataframe with selected features.
    
    Keyword arguments:
    eli5_df -- eli5 dataframe with weight of features.
    baseline_score -- score for comparison.
    model -- sklearn model.
    train_evaulation_df -- dataframe used for cross-validation.
    target -- target feature name.
    start -- start point of the iteration (default 1).
    end -- end point of the iteration (default 11).
    step -- step of the iteration (default 1). 
    divider -- transforms int to float (default 10000).
    
    Whereas Python range() function works only with integers divider is used to transform
    integers to float numbers with required decimal places. 
    """

    final_score = baseline_score
    delta_score = 0
    final_selected_features = []
    avg_scores = []
    
    for i in range(start, end, step):
        threshold = i / divider
        selected_features_df = eli5_df[eli5_df["weight"] > threshold]
        selected_features = []
        rows = len(selected_features_df)
        
        for row in range(rows):
            selected_features.append(selected_features_df["feature"][row])
            
        X_eval_selected = train_evaluation_df[selected_features]
        y_eval = train_evaluation_df[target]
            
        scores = cross_val_score(model, X_eval_selected, y_eval, scoring="neg_root_mean_squared_error")
        avg_score = abs(scores.mean())
        avg_scores.append(avg_score)
        
        if avg_score < final_score:
            final_score = avg_score
            final_selected_features = selected_features

    if final_score < baseline_score:
        delta_score = baseline_score - final_score
        print("The score was improved by", round(delta_score, 5))
        print("The best score:", round(final_score, 5))
        return final_selected_features
    else:
        print("The score was not improved.")
        print("The lowest achieved score:", round(min(avg_scores), 5))
        return final_selected_features
    
    
def alpha_selector(model_name: str, X_eval: pd.DataFrame, y_eval: pd.Series, normalize=False) -> float:
    """Takes in a sklearn model name and iterates through sequence from 0.1 to 10 with step 0.1,
    where each number each time is assigned to Alpha attribute.
    
    Keyword arguments:
    model_name -- sklearn model name: Ridge or Lasso.
    X_eval -- variables as a dataframe which are used in cross-validation.
    y_eval -- target as a series which are used in cross-validation.
    normalize -- parameter used in sklearn model (default False).
    """
     
    alpha_sequence = np.arange(0.1, 10.1, 0.1)
    lowest_score = 1
    final_alpha = 0

    if model_name == "Ridge":
        for alpha in alpha_sequence:
            regression = Ridge(alpha=alpha, normalize=normalize)
            scores = cross_val_score(regression, X_eval, y_eval, scoring="neg_root_mean_squared_error")
            avg_score = round(abs(scores.mean()), 5)
            if avg_score < lowest_score:
                lowest_score = avg_score
                final_alpha = alpha
        print("The lowest reached score:", lowest_score)
        print("Optimal alpha:", final_alpha)
    
    elif model_name == "Lasso":
        for alpha in alpha_sequence:
            regression = Lasso(alpha=alpha, normalize=normalize)
            scores = cross_val_score(regression, X_eval, y_eval, scoring="neg_root_mean_squared_error")
            avg_score = round(abs(scores.mean()), 5)
            if avg_score < lowest_score:
                lowest_score = avg_score
                final_alpha = alpha
        print("The lowest reached score:", lowest_score)
        print("Optimal alpha:", final_alpha)
        
    return final_alpha