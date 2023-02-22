
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("countries-of-the-world.csv", decimal = ",")
# print(df)

# 1. Summary Statistics of Population
# * Mean, Median, Quartiles, Variance
# * Create a boxplot of this data

# Calculate summary statistics of the population column
population_mean = df['Population'].mean()
population_median = df['Population'].median()
population_quartiles = df['Population'].quantile([0.25, 0.5, 0.75])
population_variance = df['Population'].var()

# Print the summary statistics of the population column
print("Mean population: ", population_mean)
print("Median population: ", population_median)
print("Population quartiles: ", population_quartiles)
print("Population variance: ", population_variance)

# Create a boxplot of the population data
plt.boxplot(df['Population'])
plt.title('Population Distribution')
plt.show()
print('====================================================================================================================================================================================')
                                        ## End of Part 1 ##

# 2. Summary Statistics of GDP
# * Mean, Median, Quartiles, Variance
# * Create a boxplot of this data

#dropping the nan values 
df.dropna(inplace = True)

# Calculate summary statistics of the GDP column
GDP_mean = df ['GDP ($ per capita)'].mean()
GDP_median = df['GDP ($ per capita)'].median()
GDP_quartiles = df ['GDP ($ per capita)'].quantile([0.25, 0.5, 0.75])
GDP_variance = df ['GDP ($ per capita)'].var()

# Print the summary statistics of the GDP column
print("Mean GDP: ", GDP_mean)
print("Median GDP: ", GDP_median)
print("GDP quartiles: ", GDP_quartiles)
print("GDP variance: ", GDP_variance)

# Create a boxplot of the GDP data
plt.boxplot(df['GDP ($ per capita)'])
plt.title('GDP Distribution')
plt.show()
print('====================================================================================================================================================================================')
                                        ## End of Part 2 ##

# * Find correlation coefficient
correlation = df['Population'].corr(df['GDP ($ per capita)'])
# Print the correlation coefficient
print("Correlation between population and GDP: ", correlation)
print('====================================================================================================================================================================================')

# * Calculate the line of best fit

# Calculate the slope and intercept of the line of best fit
x = df['Population']
y = df['GDP ($ per capita)']

X_bar = x - np.mean(x)
Y_bar = y - np.mean(y)

slope_a = X_bar.dot(Y_bar) / X_bar.dot(X_bar)
intercept_b = np.mean(y) - slope_a*np.mean(x)

if np.sign(intercept_b) == 1.0:
        print("Linear Regression Model:  y = {0:.2f}x + {1:.2f}".format(slope_a, intercept_b))
elif np.sign(intercept_b) == -1.0:
        print("Linear Regression Model:  y = {0:.2f}x - {1:.2f}".format(slope_a, abs(intercept_b)))

# Print the intercept and slope of the line of best fit
print('Intercept of Best fit: ', intercept_b)
print('Slope of Best fit: ', slope_a)
print('====================================================================================================================================================================================')

# * Make a scatterplot of the data and include the line of best fit
plt.scatter(x, y)
plt.plot(x, intercept_b + slope_a*x, color = 'red')
plt.xlabel('Population')
plt.ylabel('GDP ($ per capita)')
plt.title('Population vs GDP')
plt.show()

# * Predict the GDP for a country with a population of 5, 250, 000
population = 5250000
predicted_gdp = intercept_b + slope_a * population

print("Predicted GDP for a population of 5,250,000:", predicted_gdp)
print('====================================================================================================================================================================================')

# * Extra: Calculate the Root Mean Square Error of all prediction values
prediction_GDP_all = intercept_b + slope_a*df['Population'].values
differenceOF_GDP_and_Population = df['GDP ($ per capita)'].values - prediction_GDP_all
square_each_and_takeMean = np.mean(differenceOF_GDP_and_Population**2)
squared_difference = np.sqrt(square_each_and_takeMean)
print('The square root of the mean of the squared differences: ', squared_difference)
# * Questions to answer:
#     * How good is this prediction?
        # There is really low relation ship between the two, it is going to be between positive 1 and negative one and 0 means there is no agreement 
#     * Can you find any other variables with a better correlation?
        # It will be replacing the Population and replace it with other variables on the table 
print('====================================================================================================================================================================================')
                                                ## End of Part 3 ##

# 4. Multivariable Linear Regression
#    * Find the vector of parameters $\hat{\theta}$ that will model the data using the following variables:
#       * Population, Population Density, Literacy, Birthrate, Deathrate
#     * Predict the GDP for a country with
#       * Population = 5, 250,000
#       * Population Density = 72.0
#       * Literacy = 56.0%
#       * Birthrate = 22.0
#       * Deathrate = 12.0
x_data = np.array(df[["Population", "Pop. Density (per sq. mi.)", "Literacy (%)", "Birthrate", "Deathrate"]])
x_data = np.insert(x_data, 0, 1, axis= 1)
y_data = np.array(df['GDP ($ per capita)'])

# Theta = np.matmul(slope_a, Z)
Theta = np.matmul(x_data.transpose(), y_data)
Theta = np.matmul(np.linalg.inv(np.matmul(x_data.transpose(), x_data)), Theta)

x_test = np.array([1, 5250000, 72.0, 56.0, 22.0, 12.0])

prediction = np.dot(Theta, x_test)
print("This is my prediction: ", prediction)
print('====================================================================================================================================================================================')

#     * Extra: Calculate the Root Mean Square Error of all prediction values - Did it improve from the 2-D regression model?
y_hat = slope_a*x_test + intercept_b
print("this is y_hat :", y_hat)
print('====================================================================================================================================================================================')

p = np.abs(prediction - slope_a)
print("This the Root Mean Error: ", p)
print('====================================================================================================================================================================================')
                                                ## End of Part 3 ##

# 5. Questions to answer:
#     * Which variable from the data has the smallest weight in the prediction?
#       - Population is having the smallest weight in the prediction.
#     * Which variable from the data has the largest weight in the prediction? Would it be positively or negatively correlated
#       - Birthrate is the having the largest weight in the prediction and it is in a negatively. 
#     * Are there any other variables with a higher correlation that we should consider including in our model?
#       - My be the birthrate or agriculture.






