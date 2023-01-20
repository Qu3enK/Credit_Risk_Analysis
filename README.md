# Credit_Risk_Analysis

## Overview
The purpose of the Credit Risk analysis is to analyze credit card credit dataset from LendingClub utilizing machine learning to apply various techniques to train and evaluate model with unbalanced classes.  As part of the effort, use `imbalanced-learn` and `scikit-learn` libraries to build and evaluate models using resampling.

In order to perform the analysis, we will utilize `RandomOverSampler`, `SMOTE`, `ClusterCentroids`, `SMOTEENN` algorithms to oversample, undersample and combination sample the dataset.  Then utilize the  `BalancedRandomForestClassifier` and `EasyEnsembleClassifier` models to predict credit risk. [^1]


## Results
The LendingClub dataset contained over 115,000 records, as such we utilized a few techniques to train and evaluate models based on the **loan_status**. In applying the train and test methodology to the dataset, the data was split using a 76/25 split and as a results applications were catergorized into a Training set; 51,352 'low risk' & 260 'high risk'.

![7_traintest_model](https://user-images.githubusercontent.com/112449480/213764355-2ed6c288-211d-4806-b6a6-89b7b3d14da2.png)



### Naive Random Oversampling
Resampling the data via Naive Random Oversampling balanced the sample dataset for 'high' and 'low' risk  status to 51,352.

![1a_oversampl_smplcnt](https://user-images.githubusercontent.com/112449480/213613579-dbd1a305-db3b-4b0a-8c54-b972820c77de.png)


- Balance Accuracy Score: 65%

![1b_oversmpl_score](https://user-images.githubusercontent.com/112449480/213613560-affa4dcd-8faf-49e4-838a-4935dcd61461.png)

- Precision and Recall
  -  High Risk: precision: 1%, recall: 61%
  -  Low Risk: precision: 100%, recall: 69%
  -  F1: 2%

![1c_oversmpl_imbalrprt](https://user-images.githubusercontent.com/112449480/213613654-84ab6df5-ea06-4114-b052-9b86f563ebbf.png)



### SMOTE Oversampling
Resampling the data via SMOTE Oversampling balanced the sample dataset for 'high' and 'low' risk  status to 51,352.

![2a_smote_smplcnt](https://user-images.githubusercontent.com/112449480/213613742-009166e4-3e1a-445d-bf5b-9ed9d716c68c.png)


- Balance Accuracy Score: 62%

![2b_smote_score](https://user-images.githubusercontent.com/112449480/213613792-77b605f4-2e43-4104-984d-28b20b16be13.png)


- Precision and Recall
  -  High Risk: precision: 1%, recall: 61%
  -  Low Risk: precision: 100%, recall: 64%
  -  F1: 2%
  -  
![2c_smote_imbalrpt](https://user-images.githubusercontent.com/112449480/213613818-1b9d05be-c7c0-461c-8035-663aa3f5d0ee.png)


### Undersampling
Resampling the data via Undersampling lowered the sample dataset and balanced it for 'high' and 'low' risk  status to 260.

![3a_undersmpl_smplcnt](https://user-images.githubusercontent.com/112449480/213613843-5fdf9119-388c-42c8-b32a-4dfe6f9f48bb.png)

- Balance Accuracy Score: 52%

![3b_undersmpl_score](https://user-images.githubusercontent.com/112449480/213613872-b658b422-a2b1-47b7-b4c8-444501a66c3d.png)

- Precision and Recall
  -  High Risk: precision: 1%, recall: 61%
  -  Low Risk: precision: 100%, recall: 44%
  -  F1: 1%

![3c_undersmpl_imbalrpt](https://user-images.githubusercontent.com/112449480/213613907-c2ff425c-c093-41c8-a24c-a354a298220a.png)


### Combination Sampling (Over & Under)
Resampling the data via Combination Sampling set the status for the sample dataset for 'high risk' to 68,460 and 'low risk' to 62,011.

![4a_combosmpl_smplcnt](https://user-images.githubusercontent.com/112449480/213614019-bb1dc8b0-ce81-4057-908e-e8ef8ac04db0.png)

- Balance Accuracy Score: 64%

![4b_combosmpl_score](https://user-images.githubusercontent.com/112449480/213614048-74d7837d-99db-4721-99e1-58d8d823eacc.png)

- Precision and Recall
  -  High Risk: precision: 1%, recall: 70%%
  -  Low Risk: precision: 100%, recall: 58%%
  -  F1: 2%

![4c_combosmpl_imbalrpt](https://user-images.githubusercontent.com/112449480/213614075-e457875c-c838-4b15-ad35-a047f1806880.png)


### Balanced Random Forest Classifier
Resampling the data via Balanced Random Forest Classifier set the status for the sample dataset for 'high risk' to 260 and 'low risk' to 51,352.

![5a_rndmforest_smplcnt](https://user-images.githubusercontent.com/112449480/213614090-9aa003a8-1f4e-4b2c-aef8-e49e5e54c46e.png)

- Balance Accuracy Score: 78%

![5b_rndmforest_score](https://user-images.githubusercontent.com/112449480/213614109-26b830d0-b1cf-46ae-97df-fb6ff96f8116.png)

- Precision and Recall
  -  High Risk: precision: 4%, recall: 67%
  -  Low Risk: precision: 100%, recall: 91%
  -  F1: 7%

![5c_rndmforest_imbalrpt](https://user-images.githubusercontent.com/112449480/213614452-9c06a0f3-7f2b-4701-abaa-54d68ece30ad.png)

- Features Ranking: top 5 features

![5d_rndmforest_featurerank](https://user-images.githubusercontent.com/112449480/213614471-41fb6687-d711-4b3c-a164-2992f88539c7.png)


### Easy Ensemble AdaBoost Classifier
Resampling the data via Easy Ensemble AdaBoost Classifier set the status for the sample dataset for 'high risk' to 260 and 'low risk' to 51,352.

![6a_adaboost_smplcnt](https://user-images.githubusercontent.com/112449480/213614485-e5d6d82f-3bb1-4bd3-ba41-5e3e00a46a86.png)

- Balance Accuracy Score: 93%

![6b_adaboost_score](https://user-images.githubusercontent.com/112449480/213614497-733e9853-558f-44ba-8eba-f4fe3127b931.png)


- Precision and Recall
  -  High Risk: precision: 7%, recall: 91%
  -  Low Risk: precision: 100%, recall: 94%
  -  F1: 14%

![6c_adaboost_imbalrpt](https://user-images.githubusercontent.com/112449480/213614509-77a099c1-df0f-4666-999f-a56a640b13f7.png)



## Summary
In reviewing the results of the various resampling methodologies, they appear to be in the same **precision** (1-4) and **recall** (60-70) range for 'high risk' status. However the Easy Ensemble AdaBoost Classifier shows a higher accuracy score: 93%, precision: 7% and recall: 91% and F1 rating: 14%. With such a big jump in the numbers, I would not recommend using any of the models.  Note that the original dataset,only 0.5% (347/68,817) of the loan status was high risk.



[^1]: Utilize some of description from Module 18 work to assist in writing my background for the Challenge
