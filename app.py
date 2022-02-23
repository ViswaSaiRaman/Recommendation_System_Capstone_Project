from flask import Flask
from flask import request,render_template,  jsonify

# Other imports
import numpy as np 
import pandas as pd 
import pickle

# Load the pickle files generated 

final_ratings = pd.read_pickle('models/user_item_ratings.pkl')
lr_model_obj = pd.read_pickle('models/best_model_lr.pkl')
tfidf_obj = pd.read_pickle('models/tfidf_vec.pkl')

# Load the final dataframe which was saves as rating_df.csv
df = pd.read_csv('reviews_df.csv')

# Combine all last statements we used in the ipynb file 
# used for recommending a product to a user into a function 

def top5recommend(username):
    top20_ratings = final_ratings.loc[username].sort_values(ascending=False)[0:20]
    products = []
    sentiments = []


    for item in top20_ratings.index.tolist():
        # get the item name
        item_name = item
        # split the review into tokens to send it as features to the model
        item_review = df[df['name']==item_name]['text'].tolist()
    
        # get the features of the review
        tfidf_features = tfidf_obj.transform(item_review)
        # get the name and the sentiment of the review
        products.append(item_name)
        sentiments.append(lr_model_obj.predict(tfidf_features).mean())

    # Create a dataframe of the products and the sentiments
    products_df = {'Product' : products, 'Sentiment' : sentiments}
    products_df = pd.DataFrame(products_df)

    # Sort the dataframe based on the sentiment score in the descending order
    products_df = products_df.sort_values(by=['Sentiment'], ascending=False)

    # Get the top 5 products 
    top5_products = products_df['Product'][0:5].tolist()

    return top5_products


# Write the flask application from here 

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/recommend", methods=['POST'])
def recommend():
    
    # Get the username 
    username = str(request.form.get('username'))
    print(username)

    # recommend top5 products for that user
    try:
        top5_products = top5recommend(username)
        print(top5_products)

        results = ""
        for i,item in enumerate(top5_products):
            results = results+str(i+1)+". "+item
            results = results+"\n"

        # Render them in the webpage 
        return render_template('index.html', items_list='Top 5 recommendations are:\n\n {0}'.format(results))

    except Exception:
        return render_template('index.html', items_list='Username {0} does not exist.\n'.format(username))


if __name__ == "__main__":
    # Set the debug to True
    app.debug = True
    # Run the app
    app.run()
    

