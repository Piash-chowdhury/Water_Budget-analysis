
from google.colab import files
uploaded = files.upload()


import pandas as pd

# Assuming your file is named 'your_file.xlsx'
df = pd.read_excel('ML.xlsx')


df.head()


>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> IMAGE 01 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> IMAGE 01 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

import matplotlib.pyplot as plt
import seaborn as sns

# Generate a heatmap with a different color palette
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='YlGnBu', fmt=".2f")  # 'YlGnBu' is a yellow-green-blue palette
plt.title('Correlation Heatmap')
plt.show()



>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> IMAGE 01 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> IMAGE 01 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


# Generate box plots for each variable in the dataset with light blue color
plt.figure(figsize=(10, 8))

# Create a boxplot for each column with light blue color and names on the left side (horizontal orientation)
sns.boxplot(data=df, orient='h', color='#ADD8E6')  # Light blue color hex code

plt.title('Box Plots of Variables')
plt.xlabel('Values')
plt.ylabel('Variables')
plt.show()



>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> IMAGE 01 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> IMAGE 01 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Generate scatter plots between each pair of variables in the dataset
sns.pairplot(df, diag_kind='kde', plot_kws={'alpha':0.7})

plt.suptitle('Scatter Plots Between Components', y=1.02)
plt.show()


>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> IMAGE 01 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> IMAGE 01 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Generate frequency distribution histograms for each variable in the dataset
df.hist(figsize=(12, 10), bins=15, color='skyblue', edgecolor='black')

plt.suptitle('Frequency Distribution Histograms', y=1.02)
plt.show()




>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> IMAGE 01 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> IMAGE 01 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>




import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Load the data (assuming df is your DataFrame)
df = pd.read_excel('ML.xlsx')  # Replace 'your_file.xlsx' with your actual file name

# Split the data into input features (X) and output target (y)
X = df.drop(columns=['Runoff'])  # All columns except 'Runoff' are input features
y = df['Runoff']  # 'Runoff' is the target variable

# Split the data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Dictionary to store the results
results = {}

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
results['Linear Regression'] = {
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_lr)),
    'MAE': mean_absolute_error(y_test, y_pred_lr),
    'R2': r2_score(y_test, y_pred_lr)
}

# Decision Tree Regressor
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
results['Decision Tree'] = {
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_dt)),
    'MAE': mean_absolute_error(y_test, y_pred_dt),
    'R2': r2_score(y_test, y_pred_dt)
}

# Random Forest Regressor
rf = RandomForestRegressor(random_state=42, n_estimators=100)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
results['Random Forest'] = {
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_rf)),
    'MAE': mean_absolute_error(y_test, y_pred_rf),
    'R2': r2_score(y_test, y_pred_rf)
}

# K-Nearest Neighbors Regressor
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
results['K-Nearest Neighbors'] = {
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_knn)),
    'MAE': mean_absolute_error(y_test, y_pred_knn),
    'R2': r2_score(y_test, y_pred_knn)
}

# Convert results to a DataFrame for easy visualization
results_df = pd.DataFrame(results).T

# Determine common Y-axis range
y_min = min(y_test.min(), min([min(y_pred_lr), min(y_pred_dt), min(y_pred_rf), min(y_pred_knn)]))
y_max = max(y_test.max(), max([max(y_pred_lr), max(y_pred_dt), max(y_pred_rf), max(y_pred_knn)]))

# Plotting the actual vs predicted values for each model with enhanced visualization

plt.figure(figsize=(20, 16))

# Function to add metrics text on the plots
def add_metrics_text(ax, model_name):
    metrics = results[model_name]
    textstr = f'RMSE: {metrics["RMSE"]:.2f}\nMAE: {metrics["MAE"]:.2f}\nR²: {metrics["R2"]:.2%}'
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.5))

# Linear Regression
ax1 = plt.subplot(2, 2, 1)
sns.regplot(x=y_test, y=y_pred_lr, ci=95, color='blue', scatter_kws={'s':50, 'alpha':0.7}, line_kws={'lw':2})
ax1.set_title('Linear Regression')
ax1.set_xlim(y_min, y_max)
ax1.set_ylim(y_min, y_max)
ax1.set_xlabel('Actual')
ax1.set_ylabel('Predicted')
add_metrics_text(ax1, 'Linear Regression')

# Decision Tree
ax2 = plt.subplot(2, 2, 2)
sns.regplot(x=y_test, y=y_pred_dt, ci=95, color='red', scatter_kws={'s':50, 'alpha':0.7}, line_kws={'lw':2})
ax2.set_title('Decision Tree')
ax2.set_xlim(y_min, y_max)
ax2.set_ylim(y_min, y_max)
ax2.set_xlabel('Actual')
ax2.set_ylabel('Predicted')
add_metrics_text(ax2, 'Decision Tree')

# Random Forest
ax3 = plt.subplot(2, 2, 3)
sns.regplot(x=y_test, y=y_pred_rf, ci=95, color='green', scatter_kws={'s':50, 'alpha':0.7}, line_kws={'lw':2})
ax3.set_title('Random Forest')
ax3.set_xlim(y_min, y_max)
ax3.set_ylim(y_min, y_max)
ax3.set_xlabel('Actual')
ax3.set_ylabel('Predicted')
add_metrics_text(ax3, 'Random Forest')

# K-Nearest Neighbors
ax4 = plt.subplot(2, 2, 4)
sns.regplot(x=y_test, y=y_pred_knn, ci=95, color='purple', scatter_kws={'s':50, 'alpha':0.7}, line_kws={'lw':2})
ax4.set_title('K-Nearest Neighbors')
ax4.set_xlim(y_min, y_max)
ax4.set_ylim(y_min, y_max)
ax4.set_xlabel('Actual')
ax4.set_ylabel('Predicted')
add_metrics_text(ax4, 'K-Nearest Neighbors')

plt.tight_layout()
plt.show()
