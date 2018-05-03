# Rental-Income-Prediction-for-Multifamily-properties
This is the capstone project that is executed as part of the program requirements of Master of Science in Analytics Degree, The University of Chicago. The project is executed for a real estate mortgage lender. The objective is to predict rental income for multifamily properties.

Data
1. The data has been scraped off real estate websites. Given that the nature of scraping is confidential, the respective python codes are not being provided. However, scraping is executed using Scrapy and Crawlera.
2. The data is scraped across 9 markets (4 small and 5 large), the details of which are available in the project PPT. 
3. The data is categorized as:
   a. Property characteristics
   b. Rent (dependent variable)
   c. Zip code characteristics
   
Codes
1.  Python codes used for predicting rents are categorized and made available:
   a. Data merging
   b. Data Cleaning
   c. Categorical Dummy Variable creation
   d. Imputation, Principal Component Analysis and Similarity measures
   e. Models - Variants of Gradient Boosting, Random Forests, K-nearest neighbors and support vector machines
   
Feature Engineering Types:
The following are the different types of independent variables that would be used to test the predictive accuracy of rental incomes
   1. Rents of top 30 properties weighted by their similarity score
   2. Rents of top 30 properties (unweighted)
   3. Log scaled rents of top 30 properties (dep var also transformed)
   4. Rents of top 30 properties as % of median rent in the zip code (dep var also transformed)
   5. Rents of top 30 properties as % of mean of top 30 properties (dep var also transformed)
   6. Combination of 4 & 5 (dep var also transformed)

Results
1. A PPT is attached that describes data extraction, analysis of descriptive statistics, feature engineering and error rates of predictive models.

