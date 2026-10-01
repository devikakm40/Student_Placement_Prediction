import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error

# 1. Load Dataset
df = pd.read_csv('students_placement_v2.csv')

# 2. Feature Engineering Function
def add_engineered_features(data):
    df_feat = data.copy()
    df_feat['academic_avg'] = (df_feat['score_10th'] * 0.2) + (df_feat['puc_score'] * 0.2) + (df_feat['cgpa'] * 6)
    df_feat['skill_index'] = (df_feat['aptitude_score'] * 0.6) + (df_feat['soft_skills_score'] * 0.4)
    df_feat['practical_exp'] = (df_feat['internships'] * 2.5) + df_feat['projects']
    return df_feat

# Apply Feature Engineering
df_engineered = add_engineered_features(df)

# 3. Separate Features (X) and Targets (y)
X = df_engineered.drop(columns=['sl_no', 'placed', 'salary_lpa'])
y_placed = df_engineered['placed']
y_salary = df_engineered['salary_lpa']

# 4. Categorical and Numerical Columns (Including engineered columns)
categorical_cols = ['puc_stream', 'degree_branch', 'target_role']
numerical_cols = [
    'score_10th', 'puc_score', 'cgpa', 'internships', 'projects', 
    'aptitude_score', 'soft_skills_score', 'academic_avg', 'skill_index', 'practical_exp'
]

# 5. Preprocessor Setup
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
        ('num', 'passthrough', numerical_cols)
    ]
)

# Transform Features
X_transformed = preprocessor.fit_transform(X)

# 6. Train-Test Split & Model Training
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_transformed, y_placed, test_size=0.2, random_state=42, stratify=y_placed
)

# Gradient Boosting Classifier for higher accuracy
clf_model = GradientBoostingClassifier(n_estimators=150, learning_rate=0.08, max_depth=4, random_state=42)
clf_model.fit(X_train_c, y_train_c)

# Evaluate Classifier
y_pred_c = clf_model.predict(X_test_c)
accuracy = accuracy_score(y_test_c, y_pred_c)
print(f"=== PLACEMENT CLASSIFIER ACCURACY: {accuracy * 100:.2f}% ===")

# 7. Train Salary Regressor on Placed candidates
placed_mask = df_engineered['placed'] == 1
X_placed = X_transformed[placed_mask]
y_placed_salary = y_salary[placed_mask]

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_placed, y_placed_salary, test_size=0.2, random_state=42
)

reg_model = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42)
reg_model.fit(X_train_r, y_train_r)

y_pred_r = reg_model.predict(X_test_r)
mae = mean_absolute_error(y_test_r, y_pred_r)
print(f"=== SALARY REGRESSOR MEAN ABSOLUTE ERROR: ±{mae:.2f} LPA ===")

# 8. Save Model Objects
pipeline_data = {
    'preprocessor': preprocessor,
    'classifier': clf_model,
    'regressor': reg_model
}

joblib.dump(pipeline_data, 'placement_model_pipeline.pkl')
print("\nNew high-accuracy model pipeline saved to 'placement_model_pipeline.pkl'!")