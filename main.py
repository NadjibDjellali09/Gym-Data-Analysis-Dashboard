import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Set up the Streamlit page
st.set_page_config(page_title="Gym Data Analysis Dashboard", layout="wide")

# Title and description
st.title("Gym Data Analysis Dashboard")
st.markdown("""
This dashboard provides insights into fitness data, including univariate, bivariate, and multivariate analyses. Use the sidebar to navigate between sections and apply filters dynamically.
""")

# Load predefined dataset
data = pd.read_csv("gym_members_exercise_tracking.csv")

# Clean column names: remove spaces, special characters, and ensure uniqueness
data.columns = data.columns.str.replace(r"[()]", "", regex=True)  # Remove parentheses
data.columns = data.columns.str.replace(r"[^a-zA-Z0-9_]", "_", regex=True)  # Replace special chars with "_"
data.columns = data.columns.str.strip()  # Remove leading/trailing spaces
data.columns = data.columns.str.replace("__", "_")  # Replace double underscores

# Sidebar navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Overview", "Univariate Analysis", "Bivariate Analysis", "Multivariate Analysis", "Personalized Recommendation"])

# Apply filters dynamically
st.sidebar.title("Filters")
if st.sidebar.checkbox("Enable Filters"):
    filter_columns = st.sidebar.multiselect("Select columns to filter", data.columns)
    for col in filter_columns:
        if data[col].dtype in ['int64', 'float64']:
            min_val, max_val = st.sidebar.slider(f"Filter {col}", float(data[col].min()), float(data[col].max()), (float(data[col].min()), float(data[col].max())))
            data = data[(data[col] >= min_val) & (data[col] <= max_val)]
        else:
            unique_vals = data[col].unique()
            selected_vals = st.sidebar.multiselect(f"Filter {col}", unique_vals, default=unique_vals)
            data = data[data[col].isin(selected_vals)]

if section == "Overview":
    st.header("Dataset Overview")
    st.write("### First 10 rows of the dataset:")
    st.write(data.head(10))

    st.write("### Dataset Summary:")
    st.write(data.describe())

elif section == "Univariate Analysis":
    st.header("Univariate Analysis")
    column = st.selectbox("Select a column", data.columns)

    if data[column].dtype in ['int64', 'float64']:
        st.write(f"### Statistics for {column}:")
        st.write(data[column].describe())

        st.write(f"### Histogram of {column}:")
        fig = px.histogram(data, x=column, marginal="box", title=f"Histogram and Boxplot of {column}", hover_data=data.columns)
        st.plotly_chart(fig)

    elif data[column].dtype == 'object':
        st.write(f"### Distribution of {column}:")
        fig = px.pie(data, names=column, title=f"Pie Chart of {column}")
        st.plotly_chart(fig)

elif section == "Bivariate Analysis":
    st.header("Bivariate Analysis")

    # Allow duplicate column selection using unique keys
    x_col = st.selectbox("Select X-axis", data.columns, key="x_column")
    y_col = st.selectbox("Select Y-axis", data.columns, key="y_column")

    # If the same column is selected for both X and Y, create a duplicate column for plotting
    if x_col == y_col:
        data["temp_y"] = data[y_col]  # Create a temporary column

        st.write(f"### Scatter Plot: {x_col} vs {y_col}")
        fig = px.scatter(data, x=x_col, y="temp_y", trendline="ols", 
                         title=f"Scatter Plot of {x_col} vs {y_col}", hover_data=data.columns)

        # Drop temp column after plotting
        data.drop(columns=["temp_y"], inplace=True)

    else:
        st.write(f"### Scatter Plot: {x_col} vs {y_col}")
        fig = px.scatter(data, x=x_col, y=y_col, trendline="ols", 
                         title=f"Scatter Plot of {x_col} vs {y_col}", hover_data=data.columns)

    st.plotly_chart(fig)


elif section == "Multivariate Analysis":
    st.header("Multivariate Analysis")
    st.write("### Principal Component Analysis (PCA)")

    # Standardize the data
    numerical_data = data.select_dtypes(include=['int64', 'float64'])
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(numerical_data)

    # Perform PCA
    pca = PCA()
    pca_data = pca.fit_transform(standardized_data)

    # Scree plot
    st.write("#### Scree Plot")
    explained_variance = np.cumsum(pca.explained_variance_ratio_)
    fig = px.line(
        x=range(1, len(explained_variance) + 1),
        y=explained_variance,
        markers=True,
        labels={"x": "Number of Components", "y": "Cumulative Explained Variance"},
        title="Scree Plot"
    )
    st.plotly_chart(fig)

    # Correlation circle
    st.write("#### Correlation Circle")
    components = pca.components_[:2]
    fig = px.scatter(
        x=components[0],
        y=components[1],
        text=numerical_data.columns,
        labels={"x": "Component 1", "y": "Component 2"},
        title="Correlation Circle",
        hover_name=numerical_data.columns
    )
    for i, (x, y) in enumerate(zip(components[0], components[1])):
        fig.add_shape(type="line", x0=0, y0=0, x1=x, y1=y, line=dict(color="red", width=2))
    st.plotly_chart(fig)

elif section == "Personalized Recommendation":
    st.header("Personalized Recommendations")

    # Input fields for user characteristics
    age = st.number_input("Enter your age:", min_value=10, max_value=100, value=25)
    weight = st.number_input("Enter your weight (kg):", min_value=30, max_value=200, value=70)
    height = st.number_input("Enter your height (m):", min_value=1.0, max_value=2.5, value=1.75)
    workout_type = st.selectbox("Select workout type:", ["Yoga", "Cardio", "HIIT", "Strength"])

    # Example recommendation logic
    if st.button("Get Recommendations"):
        fat_percentage = round((1.2 * (weight / (height ** 2))) + (0.23 * age) - 5.4, 2)
        water_intake = round(weight * 0.033, 2)
        session_duration = {"Yoga": 1.0, "Cardio": 1.5, "HIIT": 0.75, "Strength": 1.25}[workout_type]
        calories_burned = round(session_duration * 500, 2)

        st.subheader("Your Personalized Recommendations:")
        st.write(f"**Optimal Fat Percentage:** {fat_percentage}%")
        st.write(f"**Recommended Water Intake:** {water_intake} liters/day")
        st.write(f"**Suggested Session Duration:** {session_duration} hours")
        st.write(f"**Estimated Calories Burned per Session:** {calories_burned} kcal")