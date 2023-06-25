Input Dataset source:
https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset/code

In the input data I am considering a nromalised column of both ratings and watch, to accomodate the fact with less watch time ratings importance will scaled accrodingly and with more watch time ratings will be scaled differntly 

This code performs the following steps:

Reads the dataset from the file 'ratings.csv' into a Pandas DataFrame.

Adds a new column 'TotalTime' and sets its value to 100.

Generates random values for the 'WatchTime' column using apply and random.randint functions, ensuring that NaN values are replaced with random integers between 0 and 100. As these values were not present in the input dataset

Renames the columns of the DataFrame for better readability.

Calculates the adjusted rating based on the watch time using apply and a lambda function.

Reshapes the DataFrame into a pivot table using pivot function, with 'User' as the index, 'MovieID' as the columns, and 'AdjustedRating' as the values.
Splits the data into training and test sets using train_test_split.

Computes the item-item similarity matrix based on the adjusted ratings using cosine similarity from cosine_similarity.

Defines the get_similar_movies function that takes a movie ID and returns a list of similar movies based on their similarity values.

Calls the get_similar_movies function with a movie ID and prints the similar movies.

Evaluates the model's performance by calculating the mean absolute error (MAE) between the predicted ratings




