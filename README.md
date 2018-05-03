# Rental-Income-Prediction-for-Multifamily-properties
This is the capstone project that is executed as part of the program requirements of Master of Science in Analytics Degree, The University of Chicago. The project is executed for a real estate mortgage lender. The objective is to predict rental income for multifamily properties.

Data
1. The data has been scraped off real estate websites. Given that the nature of scraping is confidential, the respective python codes are not being provided. However, scraping is executed using Scrapy and Crawlera.
2. The data is scraped across 9 markets (4 small and 5 large), the details of which are available in the project PPT. 
3. The data is categorized as:
   a. Property characteristics
   b. Rent (dependent variable)
   c. Zip code characteristics
   
Codes & Input Files
1. Python codes used for predicting rents are categorized and made available:
   a. Data merging
   b. Data Cleaning
   c. Categorical Dummy Variable creation
   d. Imputation, Principal Component Analysis and Similarity measures
   e. Models - Variants of Gradient Boosting, Random Forests, K-nearest neighbors and support vector machines
2. Input Files 1.rar contains market-wise input files for the code 1a. Merge_Zillow_ZipCode.py
3. Input Files 2.rar contains market-wise input files for the code 1b. Merge_Apartments.py
4. Input Files 3.rar contains the merged pre-imputed dataset. It also contains stratified train and test datasets used to build predictive models.
   
Feature Engineering Types:
The following are the different types of independent variables that would be used to test the predictive accuracy of rental incomes
   1. Rents of top 30 properties weighted by their similarity score
   2. Rents of top 30 properties (unweighted)
   3. Log scaled rents of top 30 properties (dependent variable also transformed)
   4. Rents of top 30 properties as % of median rent in the zip code (dependent variable also transformed)
   5. Rents of top 30 properties as % of mean of top 30 properties (dependent variable also transformed)
   6. Combination of 4 & 5 (dependent variable also transformed)

Results
1. A PPT is attached that describes data extraction, analysis of descriptive statistics, feature engineering and error rates of predictive models.

