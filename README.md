# Automobile-Insurance-claims-prediction
## Project Aim
The aim is to build a classification  model that can determine the likelihood of a claim being filed as fraudulent or legitimate. 
This model will help insurance companies:
1. Detect Fraud: By identifying patterns and predicting the probability of fraudulent claims, companies can reduce losses due to fraudulent activity.
2. Optimize Claims Processing: A predictive model can assist in flagging claims that require more scrutiny, allowing for a more efficient and focused investigation process.
3. Improve Customer Satisfaction: By speeding up the handling of legitimate claims, insurance companies can improve customer satisfaction and loyalty.
4. Risk Management: The model can provide insights into high-risk claims or customers, helping companies better manage their risk portfolios.
5. Cost Reduction: Efficient claim predictions can lead to reduced investigative and operational costs.
   ## Tools used
   1. Python 
## Data 
The data used in this project was downloaded from kaggle and and be gotten [here](https://www.kaggle.com/datasets/buntyshah/auto-insurance-claims-data?select=insurance_claims.csv)
### Data description 
The dataset has the following columns: 
1. months_as_customer - This describes the number of months a customer has been in business with the insurance company. 
2. age - This describes the age of the customer. 
3. policy_number - Unique identifier for the insurance policy.
4. policy_bind_date - This refers to the date when the insurance coverage officially begins.
5. policy_state - The state where the insurance policy was issued.
6. policy_csl - Combined single limit, which is the maximum amount the insurer will pay for damages under the policy. refers to a type of liability coverage that combines both bodily injury and property damage into a single overall limit
8. policy_deductable - The amount the policyholder is required to pay out-of-pocket before your insurance company starts to cover a claim.
9. policy_annual_premium - The annual cost of the insurance policy.
10. umbrella_limit - refers to the maximum amount of coverage provided by an umbrella insurance policy. Umbrella insurance offers additional liability coverage beyond the limits of your existing policies, such as auto or homeowners insurance
11. insured_zip - ZIP code of the insured customer.
12. insured_sex - Gender of the insured customer.
13. insured_education_level - Educational level of the insured customer.
14. insured_occupation - Occupation of the insured customer.
15. insured_hobbies - Hobbies of the insured customer.
16. insured_relationship - refers to the connection between the insured individual and the policyholder. 
17. capital-gains - Any capital gains reported by the insured.Capital gains refer to the profit you make from selling a capital asset, such as stocks, bonds, real estate, or other investments, for more than you originally paid for it.
18. capital-loss - Any capital losses reported by the insured. A capital loss occurs when a capital asset, such as an investment or real estate, is sold for less than its purchase price.
19. incident_date - Date of the incident that led to the insurance claim.
20. incident_type - Type of incident
21. collision_type - specific type of collision involved in the incident (if applicable).
22. incident_severity - The severity of the incident
23. authorities_contacted - Which authorities (e.g., police) were contacted after the incident.
24. incident_state - The state where the incident occurred.
25. incident_city - The city where the incident occurred.
26. incident_location - The specific location of the incident.
27. incident_hour_of_the_day - he hour of the day when the incident occurred.
28. number_of_vehicles_involved - The number of vehicles involved in the incident.
29. property_damage - Indicates whether there was property damage in the incident.
30. bodily_injuries - Number of bodily injuries reported.
31. witnesses - Number of witnesses to the incident.
32. police_report_available - Whether a police report is available for the incident.
33. total_claim_amount - Total amount claimed by the insured.
34. injury_claim - Amount claimed for injuries.
35. property_claim - Amount claimed for property damage.
36. vehicle_claim - Amount claimed for vehicle damage.
37. auto_make - Make of the vehicle involved in the incident.
38. auto_model - Model of the vehicle involved in the incident.
39. auto_year - Year of the vehicle involved in the incident.
40. fraud_reported - Indicates whether fraud was reported for the claim.
41. _c39 - Null 
    ## Data cleaning
    1. The last column is itrrelevant since it has all null values hence should be deleted.
    2. Checking for missing values - missing values have been denoted by '?' in the dataset hence we replace that with null.
    3. Checking for null values :
       1. 'authorities_contacted' has 91 null values.
       2. 'collision_type' has 178 null values
       3. 'property_damage' 360 has null values
       4.  'police_report_available' has 343 null values.
        Since most of our data is categorical data, we will fill the null values with the mode.
    ## Data insights 
    1. The youngest customer is aged 19 and the eldest is aged 64
    2. The lowest policy deductible is 500 an dthe highest is 2000.
    3. The lowest annual premium is 433 an dthe highest is 2048.
    4. Most accidents have happend at 5pm.(Maybe because of rush hour)
    5. The lowest total claim amount is 100 and the highest is 114920.

    ## Exploratory data analysis
    1. ### Univariate EDA
       1. Distribution of Customer Age
          - Question: What is the distribution of the ages of insured customers? Are there any noticeable trends or anomalies?
          
          ![image](https://github.com/user-attachments/assets/f4178fa5-ba3a-4917-adc0-c12ec3246c08)

       2. Distribution of Policy Annual Premium
          - Question: What is the distribution of annual premiums? Are there any outliers or unusual patterns?
          
          ![image](https://github.com/user-attachments/assets/e2639602-89ff-4539-b707-5d19d14923a3)

       3. Distribution of total claim amount
          - Question: What is the distribution of the total claim amounts? Are there any significant outliers or trends?

          ![image](https://github.com/user-attachments/assets/8bd1a831-ff4d-4b1d-baa1-2bb6e612c390)

       4. Proportion of fraud_reported
          - Question: What proportion of claims have fraud reported? Is the fraud rate high or low?

          ![image](https://github.com/user-attachments/assets/37c836dd-50be-43ea-959f-6ebfb2bdd440)

        5. Distribution of incident_severity
            - Question: How severe are the incidents reported? Are there more incidents with higher or lower severity
           
           ![image](https://github.com/user-attachments/assets/d2752c67-f3e7-40a1-92d0-6a52fbcbe11b)


    2. ### Multivariate EDA
       1. Age vs. total_claim_amount
          - Question: Is there a correlation between the age of the insured and the total claim amount? Does age influence claim amounts?

           ![image](https://github.com/user-attachments/assets/d3495f0e-af68-49fe-b5ee-ce12869c421b)


       2. policy_annual_premium vs. total_claim_amount
          - Question: How does the policy annual premium relate to the total claim amount? Are higher premiums associated with higher claims?

           ![image](https://github.com/user-attachments/assets/5de37d20-ba44-432b-b293-ebb911bed731)


       3. insured_sex vs. fraud_reported
          - Question: Is there a difference in fraud reporting between different genders? Does gender have an impact on fraud detection?

           ![image](https://github.com/user-attachments/assets/1073573d-e4cd-4dee-8b19-135eaf743b3f)


       4. policy_state vs. fraud_reported
          - Question: Are there any states where fraud is reported more frequently? Does the state have an impact on fraud incidence?

           ![image](https://github.com/user-attachments/assets/f646f0c8-5af6-4c37-a726-cd25f179b4fb)
   
       5. incident state vs. fraud_reported
          - Question: Are there any states where fraud is reported more frequently? Does the state have an impact on fraud incidence?
            
           ![image](https://github.com/user-attachments/assets/ae3a4247-e99c-489e-bfa9-c42f53518a1f)



       6. incident_type vs. total_claim_amount
           - Question: How does the type of incident affect the total claim amount? Are certain types of incidents associated with higher claims?

           ![image](https://github.com/user-attachments/assets/6633ee50-7f58-4442-ad91-42b5979662fa)


       7. insured_education_level vs. fraud_reported
           - Question: Does the education level of the insured have any correlation with fraud reporting? Are higher education levels associated with different fraud rates?

            ![image](https://github.com/user-attachments/assets/1bc70538-664b-4202-a858-3253dad27492)

          
       8. incident_hour_of_the_day vs. total_claim_amount
           - Question: Are there specific hours of the day that are associated with higher claim amounts? Does the time of day influence claim size?

           ![image](https://github.com/user-attachments/assets/f8fcdf56-76c2-45ab-bcca-c6c6e05b028a)


       9. incident type vs fraud reported
           - Question : Are certain types of incidents more likely to be associated with fraud?
           
            ![image](https://github.com/user-attachments/assets/1610fe6c-f5cd-430c-acc5-16a74854db0d)
          
       10. What is the distribution of males and females?
           
            ![image](https://github.com/user-attachments/assets/bc53180d-9570-44c1-bc6a-c3c70cf7d4cd)
       
       11. What is the distribution of the relationship of the insured.

            ![image](https://github.com/user-attachments/assets/eb1cd27e-5268-470b-bacf-af9eb42eac3e)
           
       ## Model training
       1. Data preparation
          - Since some of our columns have an object data type and machine learning models only works with numerical data types we have to encode these colums.
            1. Data encoding
          - The data was encoded using label encoder.
            2. Feature selection
          - A correlation matrix was created showing correlation coefficients between variables
          - The p values for each feature was calculated
          -  Using a logic test with a significance value of 5% (p-value < 0.05), we only keep the significant features
            3. Finding and removing the highly correlated features
          - We also need to look for predictor variable pairs which have a high correlation with each other to avoid autocorrelation.
          - Vehicle claim and total claim amount are highly correlated, considering which predictor variable to drop, vehicle claim is slightly better correlated (and lower
            p-value) to the dependent variable fraud reported, so let's drop total claim amount from the feature dataframe.
            4. Train and test set
          - The ratio of 70:30 was used.
            ### Models
            In this project we will use the following models:
            1. Suport Vector Classifier
               - The SVC is a supervised learning model used for classification tasks. It finds the hyperplane that best 
                 separates different classes in the feature space. The goal is to maximize the margin between the
                 hyperplane and the closest data points (called support vectors) from each class. It can also handle non
                 linear data using kernel tricks.
            2. KNN(K nearest neighbours)
               - KNN is a simple, non-parametric classification method. It classifies new data points based on the majority
                 class of its K nearest neighbors in the training dataset. The distance between data points is usually
                 measured using Euclidean distance. It’s easy to implement but can be slow with large datasets and
                 sensitive to irrelevant features.
            3. Decision Tree classifier
               - This is a tree-based classifier where decisions are made by splitting the data at each node based on a
                 feature that best separates the classes. Each leaf node represents a class label, and the branches
                 represent the combination of features leading to that class. It’s easy to interpret but prone to
                 overfitting.
            4. Random forest classifier
               - Random Forest is an ensemble method that builds multiple decision trees during training and outputs the
                 mode (most frequent) of the class predictions from all trees. It reduces overfitting by averaging multiple
                 trees, making it more robust and accurate compared to a single decision tree.
            5. Ada Boost Classifier
               - AdaBoost (Adaptive Boosting) is another ensemble technique that combines multiple weak classifiers
                 (usually decision trees with a single split, called stumps) to create a strong classifier. It adjusts the
                 weights of misclassified samples after each iteration to focus more on difficult cases. It's sensitive to
                 noisy data but performs well on clean datasets.
            6. XgBoost Classifier
                - XGBoost (Extreme Gradient Boosting) is a powerful, efficient implementation of gradient boosting that
                  improves performance through techniques like regularization, parallelization, and handling missing
                  values. It builds models sequentially, with each model correcting the errors of its predecessor, and it’s
                  particularly popular in machine learning competitions due to its high accuracy.
            7. Voting Classifier
                - A Voting Classifier is an ensemble method that combines the predictions of different models (e.g., SVC,
                  KNN, Random Forest) by majority voting for classification tasks. There are two types of voting: hard
                  voting (majority class wins) and soft voting (average probabilities of predictions). It improves overall
                  model performance by leveraging the strengths of different algorithms.
               and them compare them.
## Model coparison 
The models are compared based on their performance. 
### Model Performance

| Model               | Score   |
|---------------------|---------|
| Ada Boost           | 0.853333|
| Random Forest       | 0.823333|
| XgBoost             | 0.816667|
| Decision Tree       | 0.810000|
| Voting Classifier   | 0.800000|
| SVC                 | 0.760000|
| KNN                 | 0.760000|


 ![image](https://github.com/user-attachments/assets/471763af-256c-43f8-8c9e-c3dac5a19554)
 
Based on the scores above, Ada Boost is the best model with the highest performance score of 0.853333. This indicates that Ada Boost has the best classification accuracy compared to the other models for my dataset. 
      
            

           

 

            
    
       
       
    
       

    
    
