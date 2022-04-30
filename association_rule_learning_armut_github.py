#########################
# Business Problem
#########################

# Armut, Turkey's largest online service platform, brings together service providers and those who want to receive service.
# You can easily access services such as cleaning, renovation, transportation with a few touches on your computer or smart phone.
# provides access.
# By using the data set containing the service users and the services and categories these users have received.
# It is desired to create a product recommendation system with Association Rule Learning.


#########################
# Data set
#########################
# The data set consists of the services customers receive and the categories of these services.
# It contains the date and time information of each service received.

# UserId: Customer number
# ServiceId: Anonymized services belonging to each category. (Example: Upholstery washing service under the cleaning category)
# A ServiceId can be found under different categories and refers to different services under different categories.
# (Example: Service with CategoryId 7 and ServiceId 4 is honeycomb cleaning, while service with CategoryId 2 and ServiceId 4 is furniture assembly)
# CategoryId: Anonymized categories. (Example: Cleaning, transportation, renovation category)
# CreateDate: The date the service was purchased



import pandas as pd
pd.set_option('display.max_columns', None)
from mlxtend.frequent_patterns import apriori, association_rules

#########################
# TASK 1: Preparing the Data
#########################

# Step 1: Read your pear_data.csv file.
df_ = pd.read_csv("week_5/armut_data.csv")
df = df_.copy()


# Step 2: ServiceID represents a different service for each CategoryID.
# Combine ServiceID and CategoryID with "_" to create a new variable to represent the services.
df["Hizmet"] = [str(row[1]) + "_" + str(row[2]) for row in df.values]
df.head()
df["ServiceId"].value_counts()
df["CategoryId"].value_counts()
df[df["UserId"]== 9195].head()

# Step 3: The data set consists of the date and time of the receipt of services, there is no basket definition (invoice, etc.).
# In order to apply Association Rule Learning, a basket (invoice, etc.) definition must be created.
# Here, the definition of basket is the services that each customer receives monthly.
# For example, a customer with id 25446 received a basket of 4_5, 48_5, 6_7, 47_7 services received in the 8th month of 2017; In the 9th month of 2017
# 17_5, 14_7 services it receives represent another basket.
# Baskets must be identified with a unique ID. To do this, first create a new date variable containing only the year and month.
# Combine UserID and the newly created date variable with "_" on a user basis and assign it to a new variable named ID.

df["CreateDate"] = pd.to_datetime(df["CreateDate"])
df.head()
df["NEW_DATE"] = df["CreateDate"].dt.strftime("%Y-%m")
df.head()

df["SepetID"] = [str(row[0]) + "_" + str(row[5]) for row in df.values]
df.head()

df["SepetID"].value_counts().hist(bins=50).plot()

#########################
# TASK 2: Create Association Rules
#########################

# Step 1: Create cart service pivot table as below.

# Hizmet         0_8  10_9  11_11  12_7  13_11  14_7  15_1  16_8  17_5  18_4..
# SepetID
# 0_2017-08        0     0      0     0      0     0     0     0     0     0..
# 0_2017-09        0     0      0     0      0     0     0     0     0     0..
# 0_2018-01        0     0      0     0      0     0     0     0     0     0..
# 0_2018-04        0     0      0     0      0     1     0     0     0     0..
# 10000_2017-08    0     0      0     0      0     0     0     0     0     0..
df.head()
invoice_product_df = df.groupby(['SepetID', 'Hizmet'])['Hizmet'].count().unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)
invoice_product_df.head()

df["Hizmet_copy"] = df["Hizmet"].copy()
invoice_pivot = pd.pivot_table(df, values=['Hizmet_copy'], index=['SepetID'],
                    columns=['Hizmet'], aggfunc="count").fillna(0).applymap(lambda x: 1 if x > 0 else 0)
invoice_pivot.columns = [i[1] for i in invoice_pivot.columns]

invoice_pivot.head()
invoice_product_df.head()

# Step 2: Create association rules.

invoice_product_df.shape[0] * 0.005
frequent_itemsets = apriori(invoice_product_df, min_support=0.005, use_colnames=True)
frequent_itemsets.tail(100)

rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.005)
rules.head()
rules.to_csv("rules1.csv")

#Step 3: Using the arl_recommender function, suggest a service to a user who has received the 2_0 service in the last 1 month.
def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in sorted_rules["antecedents"].items():
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"]))
    recommendation_list = list({item for item_list in recommendation_list for item in item_list})
    return recommendation_list[:rec_count]


a = arl_recommender(rules, "2_0", 1)


sorted_rules = rules.sort_values("lift", ascending=False)