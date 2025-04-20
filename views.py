
from django.shortcuts import render
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load MySQL Data
def load_data():
    engine = create_engine("mysql+pymysql://root:1Amthegreatest@localhost/minor")
    df = pd.read_sql("SELECT * FROM minor", con=engine)
    df.drop(columns=["ID"], errors="ignore", inplace=True)
    df["WEEKLY_SELF_STUDY_HOURS"] *= 3
    return df

# EDA (Save visualizations to static folder)
def perform_eda(df):
    # Correlation heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm', annot_kws={"size": 10})
    plt.title('Correlation Heatmap', fontsize=16)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('predictor/static/correlation_heatmap.png', dpi=150)
    plt.close()

    # Histogram of study hours
    plt.figure(figsize=(8, 5))
    sns.histplot(df["WEEKLY_SELF_STUDY_HOURS"], kde=True, bins=20, color='skyblue')
    plt.title("Weekly Study Hours Distribution")
    plt.xlabel("Hours")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("predictor/static/study_hours_distribution.png", dpi=150)
    plt.close()

    # Gender countplot
    plt.figure(figsize=(6, 4))
    sns.countplot(x="GENDER", data=df, palette="pastel")
    plt.title("Gender Distribution")
    plt.xticks([0, 1], ["Female", "Male"])
    plt.tight_layout()
    plt.savefig("predictor/static/gender_distribution.png", dpi=150)
    plt.close()

    # Boxplot of scores
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df[["MATH", "PHYSICS", "CHEMISTRY", "BIOLOGY", "ENGLISH", "HISTORY", "GEOGRAPHY"]], palette="Set2")
    plt.title("Score Distribution by Subject")
    plt.tight_layout()
    plt.savefig("predictor/static/subject_score_boxplot.png", dpi=150)
    plt.close()

    # Feature importance
    features = ["GENDER", "PART_TIME_JOB", "DAYS_ABSENT", "EXTRACURICULAR", "WEEKLY_SELF_STUDY_HOURS"]
    X = df[features]
    y = df[["MATH", "HISTORY", "PHYSICS", "CHEMISTRY", "BIOLOGY", "ENGLISH", "GEOGRAPHY"]]
    model = RandomForestRegressor(random_state=42)
    model.fit(X, y)
    importances = model.feature_importances_

    plt.figure(figsize=(8, 6))
    sns.barplot(x=importances, y=features, palette="mako")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig("predictor/static/feature_importance.png", dpi=150)
    plt.close()

    # Residual plot for MATH
    y_math = y["MATH"]
    y_pred_math = model.predict(X)[:, 0]
    residuals = y_math - y_pred_math

    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=y_pred_math, y=residuals, color="coral")
    plt.axhline(0, linestyle='--', color='gray')
    plt.xlabel("Predicted MATH Scores")
    plt.ylabel("Residuals")
    plt.title("Residual Plot for MATH")
    plt.tight_layout()
    plt.savefig("predictor/static/residual_plot.png", dpi=150)
    plt.close()

# Train ML model
def train_model(df):
    X = df[["GENDER", "PART_TIME_JOB", "DAYS_ABSENT", "EXTRACURICULAR", "WEEKLY_SELF_STUDY_HOURS"]]
    y = df[["MATH", "HISTORY", "PHYSICS", "CHEMISTRY", "BIOLOGY", "ENGLISH", "GEOGRAPHY"]]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model

# Load & prepare
df = load_data()
perform_eda(df)
model = train_model(df)

# View logic for home + prediction
def index(request):
    if request.method == "POST":
        gender = int(request.POST["gender"])
        job = int(request.POST["job"])
        absent = int(request.POST["absent"])
        extra = int(request.POST["extra"])
        study = float(request.POST["study"])

        input_df = pd.DataFrame([[gender, job, absent, extra, study]],
                                columns=["GENDER", "PART_TIME_JOB", "DAYS_ABSENT", "EXTRACURICULAR", "WEEKLY_SELF_STUDY_HOURS"])
        predictions = model.predict(input_df)[0]
        subjects = ["MATH", "HISTORY", "PHYSICS", "CHEMISTRY", "BIOLOGY", "ENGLISH", "GEOGRAPHY"]

        def score_to_grade(score):
            if score >= 90: return 'A+'
            elif score >= 80: return 'A'
            elif score >= 70: return 'B'
            elif score >= 60: return 'C'
            elif score >= 50: return 'D'
            else: return 'F'

        grades = [score_to_grade(s) for s in predictions]
        results = zip(subjects, predictions, grades)

        return render(request, "predictor/result.html", {"results": results})

    return render(request, "predictor/index.html")
