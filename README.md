This repository contains a Python script that performs classification using the Iris dataset. The script utilizes the scikit-learn library to implement a decision tree classifier. 
The main objective is to accurately predict the species of Iris flowers based on their sepal and petal measurements.

The script first loads the Iris dataset, which is a popular benchmark dataset in machine learning. It then splits the data into training and testing sets, with a 75:25 ratio. 
The decision tree classifier is trained on the training set using different values of the max_depth parameter. The accuracy of the model is evaluated for each max_depth value to determine 
the optimal depth that results in the best performance.

A plot is generated to visualize the accuracy scores for different max_depth values, allowing you to identify the optimal depth. After that, a decision tree classifier is trained with 
the chosen max_depth and tested on the independent testing set.

The script prints the predicted and actual values of the Iris species for the testing set. Additionally, it displays the accuracy score of the model, indicating the percentage of correctly 
classified instances.

Feel free to use and modify the code in this repository under the MIT License. Contributions are welcome, so if you encounter any issues or have suggestions for improvement, please open an
issue or submit a pull request.

Thank you for your interest in this project!
