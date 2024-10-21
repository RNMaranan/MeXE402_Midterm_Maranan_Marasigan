![Blue Green Pixel Game Streamer YouTube Banner (1)](https://github.com/user-attachments/assets/f951bfa3-c9ef-417c-8b10-7d4d6a770c50)

###


![i](https://github.com/user-attachments/assets/da1e0dbc-5226-4023-a8ce-74ac9857ba0c)

![i](https://github.com/user-attachments/assets/519d83b8-7c26-414f-a05a-24313248282b)

 <h1 align="center"> 🌟🦖All About Linear Regression🦖🌟

 ###
 
## 🕹️ Linear Regression
&nbsp;&nbsp;&nbsp;&nbsp; Linear regression analysis is used to predict the value of a variable based on the value of another variable. The variable you want to predict is called the dependent variable. The variable you are using to predict the other variable's value is called the independent variable. The case of one explanatory variable is called simple linear regression; for more than one, the process is called multiple linear regression.

&nbsp;&nbsp;&nbsp;&nbsp; This form of analysis estimates the coefficients of the linear equation, involving one or more independent variables that best predict the value of the dependent variable. Linear regression fits a straight line or surface that minimizes the discrepancies between predicted and actual output values. 


## 🎮 Purpose
- To predict continuous outcomes based on the relationship between independent and dependent variables.
- To estimate the expected value of the dependent variable when given a specific value for the independent variable.
- To understand and quantify the strength and nature (positive/negative) of relationships between variables.

## 🔧 Linear Regression Formula

The Linear regression equation is represented as:

$$
\huge
ŷ = b_0 + b_1X_1 + b_2X_2 + ... + b_nX_n
$$

### Where:

- $$ŷ$$: Dependent variable. 
- $$b_0$$: y-intercept (constant term). 
- $$b_1, b_2, ..., b_n$$: Slope coefficients. 
- $$X_1, X_2, ..., X_n$$: Independent variables.

This formula is used in linear regression to model the relationship between a dependent variable and one or more independent variables by fitting a straight line.

## 🏁 Some Use Cases

- Estimating student performance based on hours studied, attendance rate, and previous test scores.
- Forecasting monthly electricity bills based on household size and appliance usage.
- Forecasting the number of product sales based on advertising spend.


![i (2)](https://github.com/user-attachments/assets/c6ba8522-4b52-4eb8-bcf1-3fafe818b392)

 <h1 align="center"> 🌟🦖All About Logistic Regression🦖🌟

 ###
 
## 🕹️ Logistic Regression
&nbsp;&nbsp;&nbsp;&nbsp; Logistic regression, known as a logit model, is a data analysis technique that uses mathematics to predict a binary outcome based on prior observations of a data set. This is used to describe data and to explain the relationship between one dependent binary variable and one or more nominal, ordinal, interval or ratio-level independent variables.

## 🎮 Purpose
- To predict binary or categorical outcomes.
- To estimate the probability that a given input point belongs to a particular category.

## 🔧 Logistic Regression Formula

The logistic regression equation is represented as:

$$
\huge
\ln \left(\frac{p}{1 - p}\right) = b_0 + b_1X_1 + b_2X_2 + b_3X_3 + b_4X_4
$$

### Where:
- $$p$$: Probability of the event occurring.
- $$\(\frac{p}{1 - p}\)$$: Odds of the event occurring.
- $$\(\ln \left(\frac{p}{1 - p}\right)\)$$: Natural logarithm of the odds.
- $$\(b_0\)$$: Intercept.
- $$\(b_1, b_2, b_3, b_4\)$$: Coefficients for the independent variables.
- $$\(X_1, X_2, X_3, X_4\)$$: Independent variables.

This formula is used in logistic regression to model the probability of a binary outcome based on one or more independent variables.


## 🏁 Some Use Cases
- Classifying emails as spam or not spam.
- Predicting whether a customer will buy a product based on demographic features.
- Diagnosing diseases as positive or negative based on symptoms and test results.


![i (4)](https://github.com/user-attachments/assets/83180a46-dd58-4f3c-b9f2-1db8534d5b17)


![i (3)](https://github.com/user-attachments/assets/2d029eef-b02d-4324-a5a3-aaee115fbaff)
## 🪻About Iris Dataset

## 🪻Content
&nbsp;&nbsp;&nbsp;&nbsp; The Iris flower data set or Fisher's Iris data set is a multivariate data set introduced by the British statistician and biologist Ronald Fisher in his 1936 paper. The use of multiple measurements in taxonomic problems as an example of linear discriminant analysis.[1] It is sometimes called Anderson's Iris data set because Edgar Anderson collected the data to quantify the morphologic variation of Iris flowers of three related species.[2] Two of the three species were collected in the Gaspé Peninsula "all from the same pasture, and picked on the same day and measured at the same time by the same person with the same apparatus".[3]

## 🪻Dataset Details
- **Number of Instances:** 150

- **Number of Attributes:** 5

- **Independent Variables:** SL: Sepal Length SW: Sepal Width PL: Petal Length PW: Petal Width these variables will be used to predict the outcome
  
- **Dependent Variable:** Class (species: Iris-Setosa, Iris-Versicolour, Iris-Virginica) the outcome variable that the analysis aims to predict.

## 🪻Attributes

Iris dataset is described by the following features:

<div align="center">
 
| **Attributes**      | **Description**                                                                                   |
|-------------------|---------------------------------------------------------------------------------------------------|
| Sepal Length      | Continuous variable representing the length of the sepal in centimeters.                        |
| Sepal Width       | Continuous variable representing the width of the sepal in centimeters.                         |
| Petal Length      | Continuous variable representing the length of the petal in centimeters.                        |
| Petal Width       | Continuous variable representing the width of the petal in centimeters.                         |
| Species           | Categorical variable indicating the species of the iris flower (Iris setosa, Iris versicolor, Iris virginica). |

</div>

## 🪻Source
&nbsp;&nbsp;&nbsp;&nbsp; The Iris Dataset was introduced by the British biologist and statistician Ronald A. Fisher in 1936 as part of his work on discriminant analysis. The dataset is available from various sources, including UCI Machine Learning Repository and Kaggle.
![i (3)](https://github.com/user-attachments/assets/108164eb-0f40-4a79-b6fc-eb8d0dd63971)

## 🪻Project Objectives

**1. Converting Letters into Numerical Values:** 
- Transform categorical attributes into numerical values to facilitate the application of machine learning algorithms.
  
**2. Classification Accuracy Using Programming:**
- Evaluate and improve the classification accuracy of Class as Iris setosa, Iris versicolor, or Iris virginica using various programming techniques and machine learning algorithms.
  
**3. Interpretation:**
- Discuss the model's ability to classify and the importance of features.

![i (5)](https://github.com/user-attachments/assets/6b65602f-102d-4c84-8893-b9978c22622e)

## 🪻Step-by-Step Linear Regression Analysis

### 🪻 Data Preprocessing

**🪻Step 1. Converting letters into numerical values**

&nbsp;&nbsp;&nbsp;&nbsp; In this step, every Class such as Iris setosa, Iris versicolor, or Iris virginica in the data will be converted into numerical values because the model is designed to establish a linear relationship between a dependent variable (the output) and one or more independent variables (the inputs).

&nbsp;&nbsp;&nbsp;&nbsp; For this Iris dataset, we assigned the following numerical values to the corresponding Iris class:

<div align="center">
 
### Class Conversion
***Iris setosa, Iris versicolor, or Iris virginica***

| Class         | Numerical Value |
|---------------|-----------------|
|Iris setosa    | 10              |
|Iris versicolor| 9               |
|Iris virginica | 8               |

</div>

**🪻 Step 2. Classification Accuracy Using Programming**

&nbsp;&nbsp;&nbsp;&nbsp; In this part of the analysis, these are the sections of the code where you import libraries and modules that facilitate the implementation of linear regression. Specifically, it involves using libraries like Pandas for data manipulation and Scikit-learn (sklearn) for model training and evaluation.

The following libraries are used for model implementation:

```python
# Importing the Dataset
import pandas as pd  # For data manipulation

# Creating the Training Set and Test Set
from sklearn.model_selection import train_test_split  # For splitting the dataset into train and test sets

# Building the Model
from sklearn.linear_model import LinearRegression  # For linear regression implementation

```

Explanation:

**1. import pandas as pd:** Imports the Pandas library, which is essential for data manipulation and analysis, particularly for loading and examining the dataset.

**2. from sklearn.model_selection import train_test_split:** Imports the function used to split your dataset into training and testing subsets.

**3. from sklearn.linear_model import LinearRegression:** Imports the LogisticRegression class, which is necessary for creating and training the logistic regression model.
**4. R-Squared and Adjusted R-Squared**

### 🪻 Evaluation Metrics

&nbsp;&nbsp;&nbsp;&nbsp; After performing data preprocessing and importing the necessary libraries for analysis and visualization, we can now proceed to obtain the inputs and outputs of the given dataset.

**1. Getting the Inputs and Outputs**
![inputs](https://github.com/user-attachments/assets/0a202734-c4e0-4da3-bae3-4375bee48aa8)

```python
#row,column
X = dataset.iloc[:,1:].values #selecting all columns starting from the second column (index 1) to the last column (independent variables)
y = dataset.iloc[:,0].values #selecting the first column only (dependent variable)

X #showing the array of all independent variables

y #showing the array of all the dependent variable
```

**2. Creating the Training Set and Test Set**
![train test](https://github.com/user-attachments/assets/b5c359ce-b722-4fdf-8821-45a62f80f34c)

```python
#Imports the function used to split your dataset into training and testing subsets.
from sklearn.model_selection import train_test_split #library, module, function
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

X_train
X_test

y_train
y_test

#test_size=0.2: This parameter specifies the proportion of the dataset to include in the test split. In this case, 0.2 means that 20% of the data will be reserved for testing, while 80% will be used for training the model.
#random_state=0: This parameter is used to control the randomness of the data splitting. Setting a specific integer (like 0) ensures that the split will be the same each time you run the code, which is useful for reproducibility. If you use a different integer or do not set it, the split may vary with each execution, leading to different results.

```
**3. Building and Training the Model**

![build and train](https://github.com/user-attachments/assets/50790980-5205-4b50-93c7-e9984a35728e)

```python
#Building the Model
#Imports the LinearRegression class, which is necessary for creating and training the linear regression model.
from sklearn.linear_model import LinearcRegression
model = LinearRegression()

#Training the Model
model.fit(X_train,y_train)
```

**4. R-Squared and Adjusted R-Squared**

```python

```

### 🪻 Interpretation

**🪻 Step 3. Interptetation**

**Model's Classification Ability**

**Feature Contributions**

###
###

![i (5)](https://github.com/user-attachments/assets/302af36b-5a61-4e26-b985-7327976d1e39)

## 🍄About Mushroom Dataset

## 🍄Content
&nbsp;&nbsp;&nbsp;&nbsp; This dataset includes descriptions of hypothetical samples corresponding to 23 species of gilled mushrooms in the Agaricus and Lepiota Family Mushroom drawn from The Audubon Society Field Guide to North American Mushrooms (1981). Each species is identified as definitely edible, definitely poisonous, or of unknown edibility and not recommended. This latter class was combined with the poisonous one. The Guide clearly states that there is no simple rule for determining the edibility of a mushroom; no rule like "leaflets three, let it be'' for Poisonous Oak and Ivy.

## 🍄Dataset Details
- **Number of Instances:** 8,124

- **Number of Attributes:** 22

- **Independent Variables:** All the attributes except for the class (e.g., cap shape, cap color, odor, etc.) that will be used to predict the outcome.
  
- **Dependent Variable:** Class (Edible or Poisonous) - the outcome variable that the analysis aims to predict.


## 🍄Attributes

Each mushroom in the dataset is described by the following features:

<div align="center">

| **Attribute**                   | **Description**                                                |
|----------------------------------|------------------------------------------------------------------|
| Class                            | Edible (e) or Poisonous (p)                                     |
| Cap-Shape                       | Bell (b), Conical (c), Convex (x), Flat (f), Knobbed (k), or Sunken (s)                |
| Cap Surface                     | Fibrous (f), Grooves (g), Scaly (y), or Smooth (s)                              |
| Cap-Color                       | Brown (n), Buff (b), Cinnamon (c), Gray (g), Green (r), Pink (p), Purple (u), Red (e), White (w), or Yellow (y) |
| Bruises                         | Yes (t) or No (f)                                              |
| Odor                            | Almond (a), Anise (l), Creosote (c), Fishy (y), Foul (f), Musty (m), None (n), Pungent (p), or Spicy (s) |
| Gill-Attachment                 | Attached (a), descending (d), Free (f), or Notched (n)                         |
| Gill-Spacing                      | Close (c), Crowded (w), distant (d)                         |
| Gill-Size                       | Broad (b), or Narrow (n) |
| Gill-Color                      | Black (k), Brown (n), Buff (b), Chocolate (h), Gray (g), Green (r), Orange (o), Pink (p), Purple (u), Red (e), White (w), or Yellow (y) |
| Stalk-Shape                     | Enlarging (e) or Tapering (t)                                  |
| Stalk-Root                      | Bulbous (b), Club (c), Cup (u), Equal (e), rhizomorphs (z), or Rooted (r)                 |
| Stalk-Surface-Above-Ring        | Fibrous (f), Scaly (y), Silky (k), or Smooth (s)                                |
| Stalk-Surface-Below-Ring        | Fibrous (f), Scaly (y), Silky (k), or Smooth (s)                                |
| Stalk-Color-Above-Ring          | Brown (n), Buff (b), Cinnamon (c), Gray (g), Orange (o), Pink (p), Red (e), White (w), or Yellow (y) |
| Stalk-Color-Below-Ring          | Brown (n), Buff (b), Cinnamon (c), Gray (g), Orange (o), Pink (p), Red (r), White (w), or Yellow (y) |
| Veil-Type                       | Partial (p) or Universal (u)                                   |
| Veil-Color                      | Brown (n), Orange (o), White (w), or Yellow (y)                                 |
| Ring-Number                     | None (n), One (o), or Two (t)                                  |
| Ring-Type                       | Cobwebby (c), Evanescent (e), Flaring (f), Large (l), None (n), Pendant (p), Sheathing (s), or Zone (z) |
| Spore-Print-Color               | Black (k), Brown (n), Buff (b), Chocolate (h), Green (r), Orange (o), Purple (u), White (w), or Yellow (y) |
| Population                      | Abundant (a), Clustered (c), Numerous (n), Scattered (s), Several (v), or Solitary (y) |
| Habitat                         | Grasses (g), Leaves (l), Meadows (m), Paths (p), Urban (u), Waste (w), Woods (d)           |

</div>

## 🍄Source
UCI Machine Learning Repository

![i (2)](https://github.com/user-attachments/assets/0121a6dc-0fea-4caf-b7bb-f0689e9050ce)

## 🍄Project Objectives

**1. Converting Letters into Numerical Values:** 
- Transform categorical attributes (e.g., class, cap shape, odor) into numerical values to facilitate the application of machine learning algorithms.

**2. Perform Data Cleaning:**
- Conduct data cleaning to handle missing values, remove duplicates, and ensure data consistency, preparing the dataset for analysis.

**3. Classification Accuracy Using Programming:**
- Evaluate and improve the classification accuracy of mushrooms as edible or poisonous using various programming techniques and machine learning algorithms.

**4. Interpretation:**
- Discuss the model's ability to classify and the importance of features.

###

![i (6)](https://github.com/user-attachments/assets/9e26c38c-7b88-42a8-8a19-c445fb36c4a0)

## 🍄Step-by-Step Logistic Regression Analysis

### 🍄 Data Preprocessing

**🍄Step 1. Converting letters into numerical values**

![conv](https://github.com/user-attachments/assets/17e1d350-b681-4aee-9df3-d8900450b8cf)

&nbsp;&nbsp;&nbsp;&nbsp; In this step, every letter in the data will be converted into numerical values because the model is designed to establish a linear relationship between a dependent variable (the output) and one or more independent variables (the inputs).

&nbsp;&nbsp;&nbsp;&nbsp; For this mushroom dataset, we assigned the following numerical values to the corresponding letters:

<div align="center">
 
### Class Conversion
***Poisonous (p) or Edible (e)***

| Letter        | Numerical Value |
|---------------|-----------------|
| p             | 0               |
| e             | 1               |

</div>

<div align="center">
 
### Other Attributes Conversion

#### This includes:
***Cap-shape, Cap-surface, Cap-color, Bruises, Odor, Gill-attachment, Gill-spacing, Gill-size, Gill-color,***  
***Stalk-shape, Stalk-root, Stalk-surface-above-ring, Stalk-surface-below-ring, Stalk-color-above-ring,***  
***Stalk-color-below-ring, Veil-type, Veil-color, Ring-number, Ring-type, Spore-print-color, Population, Habitat***

| Letter | Numerical Value |   | Letter | Numerical Value |
|--------|-----------------|---|--------|-----------------|
| a      | 1               |   | n      | 14              |
| b      | 2               |   | o      | 15              |
| c      | 3               |   | p      | 16              |
| d      | 4               |   | r      | 18              |
| e      | 5               |   | s      | 19              |
| f      | 6               |   | t      | 20              |
| g      | 7               |   | u      | 21              |
| h      | 8               |   | v      | 22              |
| k      | 11              |   | w      | 23              |
| l      | 12              |   | x      | 24              |
| m      | 13              |   | y      | 25              |

</div>

###

**🍄 Step 2. Perform Data Cleaning**

![vs](https://github.com/user-attachments/assets/d7cd3670-f80f-4b43-91f9-66055d2c7653)

&nbsp;&nbsp;&nbsp;&nbsp; After converting letters to numbers, we can perform data cleaning if we find any missing values in our given dataset.
In the case of the mushroom dataset, we observed that there are many missing values under column L, which is the stalk-root column.

Aside from checking manually, we can also use VS Code to locate missing values in our dataset by inputting the following code:

```python
import pandas as pd

# Load the dataset
data = pd.read_csv('mushroom.csv')  # The file name depends on your assigned file name

# Display the first few rows of the dataset
print("Preview of the dataset:")
print(data.head())

# Check for missing values in each column
missing_values = data.isnull().sum()

# Display the count of missing values for each column
print("\nMissing values in each column:")
print(missing_values)

# Find the rows with missing values
missing_rows = data[data.isnull().any(axis=1)]

# Display the rows with missing values
print("Rows with missing values:")
print(missing_rows)

# Step 1: Handle missing values
# Fill missing numeric values with the mode of the respective columns
data['stalk-root'].fillna(data['stalk-root'].mode()[0], inplace=True)

# Save the cleaned dataset
cleaned_file_path = 'C:/Users/Ralph/Downloads/Cleaned_Mushroom.csv' #Depends on your assigned file name
data.to_csv(cleaned_file_path, index=False)

print("Data cleaning process completed and saved to:", cleaned_file_path)

```
**Explanation of the Code:**

**1. Load the Dataset:** The dataset is loaded from a specified CSV file.

**2. Preview the Dataset:** The first few rows of the dataset are displayed to give an overview of its contents.

**3. Check for Missing Values:** The code checks for missing values in each column and prints the count.

**4. Identify Rows with Missing Values:** It identifies and prints the rows that contain any missing values.

**5. Handle Missing Values:** We use the mode for handling missing values because we are dealing with categorical data in the stalk-root attribute. The mode, representing the most frequently occurring value, is more suitable in this context as it preserves the characteristics of the dataset while effectively addressing missing values.

**6. Save the Cleaned Dataset:** Finally, the cleaned dataset is saved to a new CSV file, and a confirmation message is printed.

###

### 🍄Model Implementation

**🍄 Step 3. Classification Accuracy Using Programming**

&nbsp;&nbsp;&nbsp;&nbsp; In this part of the analysis, these are the sections of the code where you import libraries and modules that facilitate the implementation of logistic regression. Specifically, it involves using libraries like Pandas for data manipulation and Scikit-learn (sklearn) for model training and evaluation.

The following libraries are used for model implementation:

```python
# Importing the Dataset
import pandas as pd  # For data manipulation

# Creating the Training Set and Test Set
from sklearn.model_selection import train_test_split  # For splitting the dataset into train and test sets

# Feature Scaling
from sklearn.preprocessing import StandardScaler  # For standardizing features

# Building the Model
from sklearn.linear_model import LogisticRegression  # For logistic regression implementation

# Confusion Matrix
from sklearn.metrics import confusion_matrix  # For evaluating the model performance
from sklearn.metrics import accuracy_score  # For calculating the accuracy of the model

# Visualization Libraries
import matplotlib.pyplot as plt  # For plotting graphs
import seaborn as sns  # For enhanced visualizations
```

Explanation:

**1. import pandas as pd:** Imports the Pandas library, which is essential for data manipulation and analysis, particularly for loading and examining the dataset.

**2. from sklearn.model_selection import train_test_split:** Imports the function used to split your dataset into training and testing subsets.

**3. from sklearn.preprocessing import StandardScaler:** Imports the StandardScaler class for feature scaling, which normalizes the data to improve model performance.

**4. from sklearn.linear_model import LogisticRegression:** Imports the LogisticRegression class, which is necessary for creating and training the logistic regression model.

**5. from sklearn.metrics import confusion_matrix:**  Import functions of the model by calculating the confusion matrix 

**6. from sklearn.metrics import accuracy_score:** Import functions to evaluate the performance of the model by calculating accuracy score.

**7. import matplotlib.pyplot as plt:** Imports the Matplotlib library for creating static, animated, and interactive visualizations in Python.

**8. import seaborn as sns:** Imports the Seaborn library, which provides a high-level interface for drawing attractive statistical graphics.

### 🍄 Evaluation Metrics

&nbsp;&nbsp;&nbsp;&nbsp; After performing data preprocessing and importing the necessary libraries for analysis and visualization, we can now proceed to obtain the inputs and outputs of the given dataset.

**1. Getting the Inputs and Outputs**
![inputs](https://github.com/user-attachments/assets/0a202734-c4e0-4da3-bae3-4375bee48aa8)

```python
#row,column
X = dataset.iloc[:,1:].values #selecting all columns starting from the second column (index 1) to the last column (independent variables)
y = dataset.iloc[:,0].values #selecting the first column only (dependent variable)

X #showing the array of all independent variables

y #showing the array of all the dependent variable
```

**2. Creating the Training Set and Test Set**
![train test](https://github.com/user-attachments/assets/b5c359ce-b722-4fdf-8821-45a62f80f34c)

```python
#Imports the function used to split your dataset into training and testing subsets.
from sklearn.model_selection import train_test_split #library, module, function
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

X_train
X_test

y_train
y_test

#test_size=0.2: This parameter specifies the proportion of the dataset to include in the test split. In this case, 0.2 means that 20% of the data will be reserved for testing, while 80% will be used for training the model.
#random_state=0: This parameter is used to control the randomness of the data splitting. Setting a specific integer (like 0) ensures that the split will be the same each time you run the code, which is useful for reproducibility. If you use a different integer or do not set it, the split may vary with each execution, leading to different results.

```
**3. Feature Scaling**
![feature scaling](https://github.com/user-attachments/assets/4c5ca32b-1828-45bf-8651-714ff6d971b0)

```python
#Imports the StandardScaler class for feature scaling, which normalizes the data to improve model performance.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)

X_train #show the values for X_train
```
**4. Building and Training the Model**

![build and train](https://github.com/user-attachments/assets/50790980-5205-4b50-93c7-e9984a35728e)

```python
#Building the Model
#Imports the LogisticRegression class, which is necessary for creating and training the logistic regression model.
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state=0)

#Training the Model
model.fit(X_train,y_train)
```

**5. Confusion Matrix and Accuracy**

![confu](https://github.com/user-attachments/assets/720ccf7a-8deb-43e1-9ee1-16abde497655)

```python
#Confusion Matrix
#Import functions of the model by calculating the confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix (y_test, y_pred)

#Accuracy
#Import functions to evaluate the performance of the model by calculating accuracy score.
from sklearn.metrics import accuracy_score
accuracy_score (y_test, y_pred)
```
### 🍄 Visualization: Confusion Matrices

![visuals](https://github.com/user-attachments/assets/283c442e-364b-49f6-9f09-3b7a62e6f5ca)


<p align="center">
  <strong>Visualization: Plot Confusion Matrices</strong>
  <br>
  <img src="https://github.com/user-attachments/assets/cdb5bb82-24ba-4433-b066-4de0ae3aacf7" alt="Visualization: Plot Confusion Matrices" width="500"/>
</p>

```python 
import matplotlib.pyplot as plt # For plotting graphs
import seaborn as sns # For enhanced visualizations
from sklearn.metrics import confusion_matrix

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Set up the matplotlib figure
plt.figure(figsize=(8, 6))

# Create a heatmap for the confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', 
            xticklabels=['Edible', 'Poisonous'], 
            yticklabels=['Edible', 'Poisonous'])

# Labeling the axes and title
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix')

# Show the plot
plt.show()
```

### 🍄 Interpretation

**🍄 Step 4. Interptetation**

**Model's Classification Ability**

&nbsp;&nbsp;&nbsp;&nbsp; The logistic regression model developed for mushroom classification effectively distinguishes between edible and poisonous mushrooms based on various features. The confusion matrix generated provides a comprehensive view of the model's performance, summarizing both correct and incorrect predictions made during the classification process.

**Confusion Matrix Overview**
- True Positives (TP): The number of edible mushrooms correctly classified as edible.
- True Negatives (TN): The number of poisonous mushrooms correctly classified as poisonous.
- False Positives (FP): The number of poisonous mushrooms incorrectly classified as edible.
- False Negatives (FN): The number of edible mushrooms incorrectly classified as poisonous.
  
For example, if the confusion matrix shows a high number of TP (e.g., 742) and TN (e.g., 818), it indicates that the model performs well in identifying both edible and poisonous mushrooms. Conversely, a high number of FP and FN may highlight potential weaknesses in the model, necessitating further evaluation or feature adjustment.

**Accuracy Calculation**

The overall accuracy of the model can be calculated using the following formula:

$$
\huge
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

In this case, if the confusion matrix reveals an accuracy of around 96%, it signifies that the model effectively identifies mushroom classes in the majority of instances.

**Feature Contributions**

Each feature (e.g., cap shape, cap color, gill attachment, etc.) contributes to the prediction of the class labels. Analyzing feature significance can help identify which characteristics most affect the model's predictions. For example, features such as Cap-Shape, Cap-color, Odor, gill-attachment, and spore print color may have higher correlations with edibility, making them critical for classification.

![i (6)](https://github.com/user-attachments/assets/9c125ee0-0a9b-4038-a82b-54eb4677e378)

![Blue Green Pixel Game Streamer YouTube Banner](https://github.com/user-attachments/assets/0ae8ee0e-7e83-4867-b1a6-c12229683f84)


### References
<sub>IBM. (2023). About Linear Regression | IBM. Www.ibm.com; IBM. https://www.ibm.com/topics/linear-regression</sub>

<sub>“Iris Dataset,” Kaggle, Aug. 03, 2017. https://www.kaggle.com/datasets/vikrishnan/iris-dataset</sub>

<sub>Statistics Solutions, “What is Logistic Regression? - Statistics Solutions,” Statistics Solutions, Apr. 22, 2024. https://www.statisticssolutions.com/free-resources/directory-of-statistical-analyses/what-is-logistic-regression/</sub>

<sub>“What is Logistic Regression? - Logistic Regression Model Explained - AWS,” Amazon Web Services, Inc. https://aws.amazon.com/what-is/logistic-regression/</sub>

<sub>Statistics Solutions, “What is Logistic Regression? - Statistics Solutions,” Statistics Solutions, Apr. 22, 2024. https://www.statisticssolutions.com/free-resources/directory-of-statistical-analyses/what-is-logistic-regression/</sub>

<sub>“Mushroom classification,” Kaggle, Dec. 01, 2016. https://www.kaggle.com/datasets/uciml/mushroom-classification</sub>




