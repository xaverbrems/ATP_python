import pickle
import sys
import pandas as pd

def predict_from_ingredients(ingredients: list, model) -> dict:
    
    features_n = len(model.feature_names_in_)
    features_names = list(model.feature_names_in_)

    ingredients_dict = {ingredient : features_names.index(ingredient)
                        if ingredient in features_names else None
                        for ingredient in ingredients}
    
    if any(ingredients_dict.values()):
        
        X_bool = [1 if el in ingredients_dict.values() else 0 for el in range(features_n)]
        res_df = pd.DataFrame([X_bool], columns=model.feature_names_in_)
        
        return {'status': 'success',
                'predicted_energy-kcal_100g': model.predict(res_df)[0],
                'number of features' : features_n,
                'used_ingredients' : [ingredient for ingredient in ingredients_dict if ingredients_dict[ingredient]],
                'not_found_ingredients' : [ingredient for ingredient in ingredients_dict if not ingredients_dict[ingredient]],}
                
    else: return {"status": "failure",
                  "error": f"None of the following ingredients are in the features : {ingredients}"}

ingredients = sys.argv[1:]
model = pickle.load(open('../models/off_rf_model.pickle', 'rb'))
print(predict_from_ingredients(ingredients=ingredients, model=model))