Input dataset:
https://www.kaggle.com/datasets/marian447/retail-store-sales-transactions


The code provided performs the following tasks:

Loading Libraries: This code imports the necessary libraries, specifically pandas for data manipulation and warnings to suppress any warnings that may arise.

Reading the Dataset: The code reads the dataset from the 'scanner_data.csv' file using the pd.read_csv() function and assigns it to the variable df.

Finding the Most Purchased SKU: The code groups the DataFrame by SKU and calculates the sum of the Quantity for each SKU using groupby() and sum(). Then, it identifies the SKU with the maximum total Quantity using idxmax(). The result is stored in the variable most_purchased_sku.

Top 3 Products Purchased After the Most Purchased SKU: The code filters the DataFrame to select transactions that occurred after the date of the most purchased SKU. It uses the df['Date'] > df[df['SKU'] == most_purchased_sku]['Date'].max() condition to filter the rows. The filtered DataFrame is stored in after_purchase_df. It then groups the filtered DataFrame by SKU and calculates the sum of Quantity for each SKU using groupby() and sum(). Finally, it selects the top 3 SKUs with the largest total Quantity using nlargest(3). The result is stored in the variable top_3_purchased as a list.

Probability of Purchasing the Top 3 Products: The code filters the transactions DataFrame for customers who purchased the most purchased SKU using df[df['SKU'] == most_purchased_sku] and stores it in customer_transactions. It then groups the customer transactions by Customer_ID and calculates the total number of transactions for each customer using groupby() and size(). The result is stored in the customer_transaction_count DataFrame.

Next, the code filters the transactions DataFrame for customers who purchased the top 3 SKUs using df[df['SKU'].isin(top_3_purchased)] and stores it in conditional_transactions. It also groups the conditional transactions by Customer_ID and calculates the total number of transactions for each customer, storing the result in the conditional_transaction_count DataFrame.

The code merges the customer_transaction_count and conditional_transaction_count DataFrames on the common column Customer_ID using pd.merge(), resulting in the customer_probabilities DataFrame. This DataFrame contains the transaction counts for each customer who purchased the most purchased SKU and also purchased the top 3 SKUs.

The code calculates the probability for each customer by dividing the Conditional_Transaction_Count by the Transaction_Count and stores the result in the Probability column of the customer_probabilities DataFrame.

Finally, the code sorts the customer_probabilities DataFrame by the Probability column in descending order using sort_values(). The resulting DataFrame is printed, displaying the probability of purchasing the top 3 products for each customer who purchased the most purchased SKU.