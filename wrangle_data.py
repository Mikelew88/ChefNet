from pymongo import MongoClient
import pandas as pd

if __name__ == '__main__':
    #Store results in mongo
    db_client = MongoClient()
    db = db_client['allrecipes']
    recipe_db = db['recipe_data']

    df = pd.DataFrame(list(recipe_db.find()))

    ingred_list = df[['id','ingred_list']]
