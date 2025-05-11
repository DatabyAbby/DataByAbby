import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import os
from data_analysis_tool import DataAnalysisTool
from auth import init_db, show_login_page
from subscription import init_subscription_db, create_free_subscription, check_subscription, get_plan_limits
from analytics import init_analytics_db, log_action, get_user_stats, get_daily_usage, init_daily_stats
from dashboard import show_user_dashboard

# Set page config
st.set_page_config(
    page_title="AI Data Analysis Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize databases
init_db()
init_subscription_db()
init_analytics_db()
init_daily_stats()

# Check if user is authenticated
if 'is_authenticated' not in st.session_state:
    st.session_state.is_authenticated = False

# Check if user has initialized subscription
if 'subscription_initialized' not in st.session_state:
    st.session_state.subscription_initialized = False

# Show login page if not authenticated
if not st.session_state.is_authenticated:
    show_login_page()
    st.stop()  # Stop execution here if not logged in
    
# Create a free subscription for new users
if not st.session_state.subscription_initialized:
    create_free_subscription(st.session_state.user.email)
    st.session_state.subscription_initialized = True
    # Log user signup
    log_action(st.session_state.user.email, "signup", {"source": "web"})

# Initialize session state variables
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'tool' not in st.session_state:
    st.session_state.tool = DataAnalysisTool()
if 'cleaning_info' not in st.session_state:
    st.session_state.cleaning_info = None
if 'data_overview' not in st.session_state:
    st.session_state.data_overview = None
if 'eda_results' not in st.session_state:
    st.session_state.eda_results = None
if 'visualization_data' not in st.session_state:
    st.session_state.visualization_data = None
if 'model_results' not in st.session_state:
    st.session_state.model_results = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = "data_import"

# Sidebar navigation
st.sidebar.title("📊 Data Analysis Tool")

# User info in sidebar
st.sidebar.success(f"Logged in as {st.session_state.user.email}")
if st.sidebar.button("Logout"):
    st.session_state.is_authenticated = False
    st.rerun()

st.sidebar.markdown("---")

# Navigation options
pages = {
    "data_import": "Data Import",
    "data_overview": "Data Overview",
    "data_cleaning": "Data Cleaning",
    "exploratory_analysis": "Exploratory Analysis",
    "visualization": "Visualization",
    "modeling": "Predictive Modeling",
    "insights": "Summary Insights",
    "dashboard": "My Dashboard"
}

# Create sidebar navigation
st.sidebar.subheader("Navigation")
selected_page = st.sidebar.radio("Go to", list(pages.values()))

# Convert selected page name back to key
for key, value in pages.items():
    if value == selected_page:
        st.session_state.current_page = key

# Subscription status in sidebar
subscription = check_subscription(st.session_state.user.email)
if subscription['active']:
    plan_display = subscription['plan'].replace('_', ' ').title()
    st.sidebar.success(f"✅ {plan_display} Plan - {subscription['days_left']} days left")
else:
    st.sidebar.warning("⚠️ Subscription expired")

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.info(
    """
    This tool provides comprehensive data analysis capabilities without requiring 
    external API dependencies. Upload your data and explore through cleaning, 
    visualization, and modeling.
    """
)

# Helper function to display base64 images
def display_base64_image(base64_str, caption=None):
    """Display a base64 encoded image in Streamlit"""
    st.image(base64.b64decode(base64_str), caption=caption, use_container_width=True)

# Data Import Page
def render_data_import():
    st.title("Data Import")
    st.write("Upload your dataset (CSV or Excel) to begin analysis.")
    
    # Check subscription limits
    subscription = check_subscription(st.session_state.user.email)
    plan = subscription['plan'] if subscription['active'] else 'free_trial'
    limits = get_plan_limits(plan)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls"])
        
        if uploaded_file is not None:
            # Determine file type
            file_type = uploaded_file.name.split('.')[-1].lower()
            
            # Check file size against subscription limit
            file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
            
            if file_size_mb > limits['max_file_size']:
                st.error(f"File too large: {file_size_mb:.1f} MB. Your {plan.replace('_', ' ').title()} plan allows files up to {limits['max_file_size']} MB.")
                st.info("Upgrade your plan to analyze larger files.")
                return
            
            # Load the data
            try:
                file_content = uploaded_file.read()
                success, message = st.session_state.tool.load_data(file_content, file_type=file_type)
                
                if success:
                    # Check if data has too many rows for current plan
                    if len(st.session_state.tool.data) > limits['max_rows']:
                        st.error(f"Dataset too large: {len(st.session_state.tool.data)} rows. Your {plan.replace('_', ' ').title()} plan allows up to {limits['max_rows']} rows.")
                        st.info("Upgrade your plan to analyze larger datasets.")
                        return
                    
                    st.session_state.data_loaded = True
                    st.success(message)
                    
                    # Log the data upload action
                    log_action(st.session_state.user.email, "data_upload", {
                        "file_name": uploaded_file.name,
                        "file_size_mb": file_size_mb,
                        "rows": len(st.session_state.tool.data),
                        "columns": len(st.session_state.tool.data.columns)
                    })
                    
                    # Get initial overview
                    st.session_state.data_overview = st.session_state.tool.get_data_overview()
                    
                    # Display preview
                    st.subheader("Data Preview")
                    df_preview = st.session_state.tool.data.head(5)
                    st.dataframe(df_preview)
                    
                    # Suggest next steps
                    st.info("✅ Data loaded successfully! Navigate to 'Data Overview' to explore your dataset.")
                else:
                    st.error(message)
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
    
    with col2:
        st.subheader("Sample Dataset")
        st.write("Don't have a dataset? Use our sample data.")
        
        if st.button("Load Sample Data"):
            # Create a sample dataset
            np.random.seed(42)
            n_samples = 1000
            
            # Create features
            age = np.random.normal(35, 10, n_samples).astype(int)
            age = np.clip(age, 18, 70)
            
            income = 30000 + age * 1000 + np.random.normal(0, 10000, n_samples)
            
            education_years = np.random.randint(8, 22, n_samples)
            
            # More features with correlations
            experience = age - education_years - 5 + np.random.randint(-3, 4, n_samples)
            experience = np.clip(experience, 0, 50)
            
            debt = np.random.normal(20000, 15000, n_samples) + (70 - age) * 500
            debt = np.clip(debt, 0, 100000)
            
            # Categorical features
            job_categories = ['Technical', 'Management', 'Sales', 'Support', 'Administrative']
            job = np.random.choice(job_categories, n_samples)
            
            region_categories = ['North', 'South', 'East', 'West', 'Central']
            region = np.random.choice(region_categories, n_samples)
            
            # Target: Credit score influenced by other variables
            credit_score = 300 + (income / 20000) * 200 + education_years * 10 - (debt / 20000) * 100
            credit_score = credit_score + np.random.normal(0, 50, n_samples)
            credit_score = np.clip(credit_score, 300, 850).astype(int)
            
            # Create DataFrame
            df = pd.DataFrame({
                'Age': age,
                'Income': income,
                'Education_Years': education_years,
                'Experience': experience,
                'Debt': debt,
                'Job_Category': job,
                'Region': region,
                'Credit_Score': credit_score
            })
            
            # Add some missing values
            for col in df.columns:
                mask = np.random.random(n_samples) < 0.05  # 5% missing values
                df.loc[mask, col] = np.nan
            
            # Convert to CSV in memory
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_bytes = csv_buffer.getvalue().encode()
            
            # Load the sample data
            success, message = st.session_state.tool.load_data(csv_bytes, file_type='csv')
            
            if success:
                st.session_state.data_loaded = True
                st.success("Sample data loaded successfully!")
                
                # Get initial overview
                st.session_state.data_overview = st.session_state.tool.get_data_overview()
                
                # Display preview
                st.subheader("Data Preview")
                df_preview = st.session_state.tool.data.head(5)
                st.dataframe(df_preview)
                
                # Suggest next steps
                st.info("✅ Sample data loaded! Navigate to 'Data Overview' to explore the dataset.")
            else:
                st.error(message)

# Data Overview Page
def render_data_overview():
    st.title("Data Overview")
    
    if not st.session_state.data_loaded:
        st.warning("No data loaded. Please import data first.")
        return
    
    overview = st.session_state.data_overview
    
    # Display basic information
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", overview['shape'][0])
    with col2:
        st.metric("Columns", overview['shape'][1])
    with col3:
        st.metric("Numeric Columns", len(overview['numeric_columns']))
    with col4:
        st.metric("Categorical Columns", len(overview['categorical_columns']))
    
    # Display column details
    st.subheader("Column Information")
    
    # Create a dataframe with column info
    col_info = []
    for col in overview['columns']:
        missing = overview['missing_values'].get(col, 0)
        missing_pct = missing / overview['shape'][0] * 100 if overview['shape'][0] > 0 else 0
        
        col_info.append({
            "Column": col,
            "Type": overview['dtypes'].get(col, ""),
            "Missing Values": missing,
            "Missing %": f"{missing_pct:.2f}%"
        })
    
    col_df = pd.DataFrame(col_info)
    st.dataframe(col_df)
    
    # Show data preview
    st.subheader("Data Preview")
    st.dataframe(st.session_state.tool.data.head(10))
    
    # Show numeric statistics if available
    if 'numeric_stats' in overview and overview['numeric_stats']:
        st.subheader("Numeric Column Statistics")
        
        # Convert the nested dictionaries to a more streamlit-friendly format
        stats_df = pd.DataFrame.from_dict({k: v for k, v in overview['numeric_stats'].items()})
        st.dataframe(stats_df)
    
    # Show categorical statistics if available
    if 'categorical_stats' in overview and overview['categorical_stats']:
        st.subheader("Categorical Column Distributions")
        
        for col, stats in overview['categorical_stats'].items():
            st.write(f"**{col}** - {stats['unique_count']} unique values")
            
            # Convert top values to a dataframe
            top_values = pd.DataFrame.from_dict(stats['top_values'], orient='index', columns=['Count'])
            top_values.index.name = 'Value'
            top_values.reset_index(inplace=True)
            
            # Show as bar chart
            st.bar_chart(data=top_values, x='Value', y='Count')

# Data Cleaning Page
def render_data_cleaning():
    st.title("Data Cleaning")
    
    if not st.session_state.data_loaded:
        st.warning("No data loaded. Please import data first.")
        return
    
    st.write("Clean your dataset by handling missing values, removing duplicates, and standardizing column names.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Cleaning Options")
        
        remove_duplicates = st.checkbox("Remove duplicate rows", value=True)
        
        handle_missing = st.radio(
            "How to handle missing values",
            ["Fill with appropriate values", "Drop rows with missing values", "Drop columns with many missing values"],
            index=0
        )
        
        # Map the radio options to the function parameters
        missing_option_map = {
            "Fill with appropriate values": "fill",
            "Drop rows with missing values": "drop_rows",
            "Drop columns with many missing values": "drop_cols"
        }
        
        missing_option = missing_option_map[handle_missing]
        
        # Only show this option if dropping columns
        drop_threshold = 0.5
        if missing_option == "drop_cols":
            drop_threshold = st.slider(
                "Threshold for dropping columns (% missing)",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                format="%d%%"
            )
        
        if st.button("Clean Data"):
            with st.spinner("Cleaning data..."):
                cleaning_info = st.session_state.tool.clean_data(
                    remove_duplicates=remove_duplicates,
                    handle_missing=missing_option,
                    drop_threshold=drop_threshold
                )
                
                st.session_state.cleaning_info = cleaning_info
                
                # Update data overview after cleaning
                st.session_state.data_overview = st.session_state.tool.get_data_overview()
                
                st.success("Data cleaning completed!")
    
    with col2:
        st.subheader("Cleaning Results")
        
        if st.session_state.cleaning_info:
            cleaning_info = st.session_state.cleaning_info
            
            st.write(f"Original shape: {cleaning_info['original_shape']}")
            st.write(f"Final shape: {cleaning_info['final_shape']}")
            st.write(f"Duplicates removed: {cleaning_info['duplicates_removed']}")
            
            if cleaning_info['missing_values_handled']:
                st.write("Missing values handled:")
                
                if 'rows_dropped' in cleaning_info['missing_values_handled']:
                    st.write(f"- Rows dropped: {cleaning_info['missing_values_handled']['rows_dropped']}")
                
                if 'columns_dropped' in cleaning_info['missing_values_handled']:
                    cols_dropped = cleaning_info['missing_values_handled']['columns_dropped']
                    st.write(f"- Columns dropped: {', '.join(cols_dropped) if cols_dropped else 'None'}")
                
                # For each column with filled values
                for col, info in cleaning_info['missing_values_handled'].items():
                    if isinstance(info, dict) and 'method' in info:
                        st.write(f"- {col}: {info['original_missing']} missing values filled using {info['method']}")
            
            if cleaning_info['columns_renamed']:
                st.write("Column names standardized:")
                for original, new in cleaning_info['columns_renamed'].items():
                    if original != new:
                        st.write(f"- '{original}' → '{new}'")
        else:
            st.info("No cleaning has been performed yet. Use the options on the left to clean your data.")
    
    # Show data preview after cleaning
    if st.session_state.cleaning_info:
        st.subheader("Cleaned Data Preview")
        st.dataframe(st.session_state.tool.data.head(10))

# Exploratory Analysis Page
def render_exploratory_analysis():
    st.title("Exploratory Data Analysis")
    
    if not st.session_state.data_loaded:
        st.warning("No data loaded. Please import data first.")
        return
    
    st.write("Explore your data with statistical analysis and identify patterns and relationships.")
    
    if st.button("Perform Exploratory Analysis"):
        with st.spinner("Analyzing data..."):
            eda_results = st.session_state.tool.perform_eda()
            st.session_state.eda_results = eda_results
            st.success("Analysis complete!")
    
    if st.session_state.eda_results:
        eda_results = st.session_state.eda_results
        
        # Display correlation matrix if available
        if 'correlation_matrix' in eda_results and eda_results['correlation_matrix']:
            st.subheader("Correlation Matrix")
            corr_matrix = pd.DataFrame(eda_results['correlation_matrix'])
            
            # Display as a heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            cax = ax.matshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            fig.colorbar(cax)
            
            # Set ticks and labels
            ax.set_xticks(range(len(corr_matrix.columns)))
            ax.set_yticks(range(len(corr_matrix.columns)))
            ax.set_xticklabels(corr_matrix.columns, rotation=90)
            ax.set_yticklabels(corr_matrix.columns)
            
            # Add correlation values to the heatmap
            for i in range(len(corr_matrix.columns)):
                for j in range(len(corr_matrix.columns)):
                    ax.text(i, j, f"{corr_matrix.iloc[j, i]:.2f}", ha="center", va="center", 
                           color="white" if abs(corr_matrix.iloc[j, i]) > 0.5 else "black")
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Also display as a table for reference
            st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm', axis=None, vmin=-1, vmax=1))
        
        # Display skewness if available
        if 'skewness' in eda_results and eda_results['skewness']:
            st.subheader("Distribution Skewness")
            skew_data = pd.DataFrame.from_dict(eda_results['skewness'], orient='index', columns=['Skewness'])
            skew_data.index.name = 'Column'
            skew_data.reset_index(inplace=True)
            
            # Highlight highly skewed columns
            def highlight_skew(val):
                if abs(val) > 1:
                    return 'background-color: #ffcccb'  # Light red
                elif abs(val) > 0.5:
                    return 'background-color: #ffffcc'  # Light yellow
                else:
                    return ''
            
            # Apply styling
            styled_skew = skew_data.style.format({'Skewness': '{:.3f}'})
            styled_skew = styled_skew.applymap(highlight_skew, subset=['Skewness'])
            
            st.dataframe(styled_skew)
            
            st.info("""
            **Interpretation:**
            * |Skewness| < 0.5: Approximately symmetric
            * 0.5 < |Skewness| < 1: Moderately skewed
            * |Skewness| > 1: Highly skewed
            """)
        
        # Display outliers if available
        if 'outliers' in eda_results and eda_results['outliers']:
            st.subheader("Outlier Detection")
            
            outlier_data = []
            for col, info in eda_results['outliers'].items():
                outlier_data.append({
                    'Column': col,
                    'Outlier Count': info['count'],
                    'Percentage': f"{info['percentage']:.2f}%",
                    'Lower Bound': f"{info['lower_bound']:.2f}",
                    'Upper Bound': f"{info['upper_bound']:.2f}"
                })
            
            if outlier_data:
                st.dataframe(pd.DataFrame(outlier_data))
            else:
                st.write("No significant outliers detected in the dataset.")
            
            st.info("""
            **Note:** Outliers are detected using the IQR method:
            * Lower bound = Q1 - 1.5 * IQR
            * Upper bound = Q3 + 1.5 * IQR
            """)
        
        # Display summary statistics if available
        if 'summary_statistics' in eda_results and eda_results['summary_statistics']:
            st.subheader("Summary Statistics")
            stats_df = pd.DataFrame.from_dict({k: v for k, v in eda_results['summary_statistics'].items()})
            st.dataframe(stats_df)
        
        # Display categorical distributions if available
        if 'categorical_distributions' in eda_results and eda_results['categorical_distributions']:
            st.subheader("Categorical Distributions")
            
            for col, distribution in eda_results['categorical_distributions'].items():
                st.write(f"**{col}**")
                
                # Convert to dataframe
                dist_df = pd.DataFrame.from_dict(distribution, orient='index', columns=['Count'])
                dist_df.index.name = 'Value'
                dist_df.reset_index(inplace=True)
                
                # Show as bar chart
                st.bar_chart(data=dist_df, x='Value', y='Count')

# Visualization Page
def render_visualization():
    st.title("Data Visualization")
    
    if not st.session_state.data_loaded:
        st.warning("No data loaded. Please import data first.")
        return
    
    st.write("Generate visualizations to better understand your data.")
    
    if st.button("Generate Visualizations"):
        with st.spinner("Creating visualizations..."):
            viz_data = st.session_state.tool.generate_visualizations()
            st.session_state.visualization_data = viz_data
            st.success("Visualizations created!")
    
    if st.session_state.visualization_data:
        viz_data = st.session_state.visualization_data
        
        # Create tabs for different visualization types
        viz_tabs = st.tabs([
            "Histograms", "Box Plots", "Correlation Heatmap", "Scatter Plots", "Bar Charts"
        ])
        
        # Histograms
        with viz_tabs[0]:
            st.subheader("Distributions (Histograms)")
            
            if 'histograms' in viz_data and viz_data['histograms']:
                for col, img_str in viz_data['histograms'].items():
                    st.write(f"**Distribution of {col}**")
                    display_base64_image(img_str)
                    st.markdown("---")
            else:
                st.info("No histograms were generated. This may be because there are no suitable numeric columns in the dataset.")
        
        # Box Plots
        with viz_tabs[1]:
            st.subheader("Box Plots (Outlier Detection)")
            
            if 'boxplots' in viz_data and viz_data['boxplots']:
                for col, img_str in viz_data['boxplots'].items():
                    st.write(f"**Box Plot of {col}**")
                    display_base64_image(img_str)
                    st.markdown("---")
            else:
                st.info("No box plots were generated. This may be because there are no suitable numeric columns in the dataset.")
        
        # Correlation Heatmap
        with viz_tabs[2]:
            st.subheader("Correlation Heatmap")
            
            if 'correlation_heatmap' in viz_data and viz_data['correlation_heatmap']:
                display_base64_image(viz_data['correlation_heatmap'], "Correlation Matrix Heatmap")
            else:
                st.info("No correlation heatmap was generated. This may be because there are not enough numeric columns for correlation analysis.")
        
        # Scatter Plots
        with viz_tabs[3]:
            st.subheader("Scatter Plots")
            
            if 'scatter_plots' in viz_data and viz_data['scatter_plots']:
                for plot_name, img_str in viz_data['scatter_plots'].items():
                    st.write(f"**{plot_name.replace('_vs_', ' vs ')}**")
                    display_base64_image(img_str)
                    st.markdown("---")
            else:
                st.info("No scatter plots were generated. This may be because there are not enough numeric columns in the dataset.")
        
        # Bar Charts
        with viz_tabs[4]:
            st.subheader("Bar Charts (Categorical Data)")
            
            if 'bar_charts' in viz_data and viz_data['bar_charts']:
                for col, img_str in viz_data['bar_charts'].items():
                    st.write(f"**Distribution of {col}**")
                    display_base64_image(img_str)
                    st.markdown("---")
            else:
                st.info("No bar charts were generated. This may be because there are no suitable categorical columns in the dataset.")

# Modeling Page
def render_modeling():
    st.title("Predictive Modeling")
    
    if not st.session_state.data_loaded:
        st.warning("No data loaded. Please import data first.")
        return
    
    st.write("Build predictive models based on your data.")
    
    # Get column list for selection
    columns = st.session_state.tool.data.columns.tolist()
    numeric_columns = st.session_state.tool.numeric_columns
    
    # Choose modeling approach
    model_type = st.radio(
        "Select modeling approach:",
        ["Supervised Learning (Prediction)", "Unsupervised Learning (Clustering)"]
    )
    
    if model_type == "Supervised Learning (Prediction)":
        st.subheader("Supervised Learning")
        
        col1, col2 = st.columns(2)
        
        with col1:
            target_column = st.selectbox(
                "Select target column to predict:",
                columns
            )
            
            algorithm = st.radio(
                "Select algorithm:",
                ["Random Forest", "Linear/Logistic Regression"]
            )
            
            algorithm_map = {
                "Random Forest": "random_forest",
                "Linear/Logistic Regression": "linear_regression"
            }
            
            test_size = st.slider(
                "Test set size (%):",
                min_value=10,
                max_value=50,
                value=20,
                step=5
            ) / 100
            
            if st.button("Build Model"):
                with st.spinner("Building model..."):
                    try:
                        model_results = st.session_state.tool.build_model(
                            target_column=target_column.lower(),
                            model_type=algorithm_map[algorithm],
                            test_size=test_size
                        )
                        
                        st.session_state.model_results = model_results
                        st.success("Model built successfully!")
                    except Exception as e:
                        st.error(f"Error building model: {str(e)}")
        
        with col2:
            if st.session_state.model_results and 'task_type' in st.session_state.model_results:
                model_results = st.session_state.model_results
                
                st.write(f"**Model Type:** {model_results['model_type'].replace('_', ' ').title()}")
                st.write(f"**Task:** {model_results['task_type'].title()}")
                
                if model_results['task_type'] == 'regression':
                    metrics = model_results['metrics']
                    st.metric("R² Score (Test)", f"{metrics['test_r2']:.4f}")
                    st.metric("RMSE (Test)", f"{metrics['test_rmse']:.4f}")
                else:  # classification
                    metrics = model_results['metrics']
                    st.metric("Accuracy (Test)", f"{metrics['test_accuracy']:.4f}")
    
    else:  # Clustering
        st.subheader("Unsupervised Learning (Clustering)")
        
        st.write("K-means clustering will be performed on the numeric columns of your dataset.")
        
        if len(numeric_columns) < 2:
            st.warning("K-means clustering requires at least 2 numeric columns. Your dataset doesn't have enough numeric columns.")
        else:
            if st.button("Perform Clustering"):
                with st.spinner("Performing clustering..."):
                    try:
                        clustering_results = st.session_state.tool.build_model(model_type='kmeans')
                        st.session_state.model_results = clustering_results
                        st.success("Clustering completed!")
                    except Exception as e:
                        st.error(f"Error performing clustering: {str(e)}")
    
    # Display model results
    if st.session_state.model_results:
        model_results = st.session_state.model_results
        
        st.subheader("Model Results")
        
        # Create tabs based on model type
        if model_results['model_type'] == 'kmeans':
            # K-means results
            st.write(f"**Optimal number of clusters (k):** {model_results['optimal_k']}")
            
            # Show cluster sizes
            st.write("**Cluster sizes:**")
            for cluster, size in model_results['cluster_sizes'].items():
                st.write(f"- Cluster {cluster}: {size} samples")
            
            # Show visualizations
            if 'visualization' in model_results:
                viz = model_results['visualization']
                
                if 'cluster_plot' in viz:
                    st.write("**Cluster Visualization:**")
                    display_base64_image(viz['cluster_plot'])
                
                if 'elbow_plot' in viz:
                    st.write("**Elbow Method Plot:**")
                    display_base64_image(viz['elbow_plot'])
        
        else:
            # Supervised learning results
            result_tabs = st.tabs(["Performance Metrics", "Visualizations"])
            
            with result_tabs[0]:
                if model_results['task_type'] == 'regression':
                    # Regression metrics
                    st.write("**Regression Performance:**")
                    metrics = model_results['metrics']
                    
                    metrics_df = pd.DataFrame({
                        'Metric': ['R² (Training)', 'R² (Test)', 'RMSE (Training)', 'RMSE (Test)'],
                        'Value': [
                            f"{metrics['train_r2']:.4f}",
                            f"{metrics['test_r2']:.4f}",
                            f"{metrics['train_rmse']:.4f}",
                            f"{metrics['test_rmse']:.4f}"
                        ]
                    })
                    
                    st.dataframe(metrics_df, hide_index=True)
                
                else:
                    # Classification metrics
                    st.write("**Classification Performance:**")
                    metrics = model_results['metrics']
                    
                    metrics_df = pd.DataFrame({
                        'Metric': ['Accuracy (Training)', 'Accuracy (Test)'],
                        'Value': [
                            f"{metrics['train_accuracy']:.4f}",
                            f"{metrics['test_accuracy']:.4f}"
                        ]
                    })
                    
                    st.dataframe(metrics_df, hide_index=True)
            
            with result_tabs[1]:
                if 'visualization' in model_results:
                    viz = model_results['visualization']
                    
                    if model_results['task_type'] == 'regression':
                        if 'prediction_plot' in viz:
                            st.write("**Actual vs Predicted Values:**")
                            display_base64_image(viz['prediction_plot'])
                    else:
                        if 'confusion_matrix' in viz:
                            st.write("**Confusion Matrix:**")
                            display_base64_image(viz['confusion_matrix'])
                    
                    if 'feature_importance' in viz and viz['feature_importance']:
                        st.write("**Feature Importance/Coefficients:**")
                        display_base64_image(viz['feature_importance'])

# Summary Insights Page
def render_summary_insights():
    st.title("Summary Insights")
    
    if not st.session_state.data_loaded:
        st.warning("No data loaded. Please import data first.")
        return
    
    st.write("Get a summary of key insights and findings from your data analysis.")
    
    if st.button("Generate Summary"):
        with st.spinner("Analyzing data and generating summary..."):
            summary = st.session_state.tool.get_data_summary()
            st.session_state.summary = summary
            st.success("Summary generated!")
    
    if 'summary' in st.session_state and st.session_state.summary:
        summary = st.session_state.summary
        
        # Dataset overview
        st.subheader("Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", summary['dataset']['shape'][0])
        with col2:
            st.metric("Columns", summary['dataset']['shape'][1])
        with col3:
            st.metric("Numeric Columns", summary['dataset']['numeric_columns'])
        with col4:
            st.metric("Categorical Columns", summary['dataset']['categorical_columns'])
        
        # Key insights
        st.subheader("Key Insights")
        
        if 'key_insights' in summary and summary['key_insights']:
            for i, insight in enumerate(summary['key_insights']):
                with st.expander(f"{i+1}. {insight['title']}", expanded=True):
                    st.write(insight['details'])
        else:
            st.info("No key insights were generated. This may be because not enough analysis has been performed on the dataset.")
        
        # Data export options
        st.subheader("Export Processed Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export as CSV"):
                csv_data = st.session_state.tool.export_processed_data(format='csv')
                if csv_data:
                    csv_str = csv_data.decode('utf-8')
                    
                    # Create a download link
                    b64 = base64.b64encode(csv_data).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="processed_data.csv">Download CSV File</a>'
                    st.markdown(href, unsafe_allow_html=True)
        
        with col2:
            if st.button("Export as Excel"):
                excel_data = st.session_state.tool.export_processed_data(format='excel')
                if excel_data:
                    # Create a download link
                    b64 = base64.b64encode(excel_data).decode()
                    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="processed_data.xlsx">Download Excel File</a>'
                    st.markdown(href, unsafe_allow_html=True)

# Add usage tracking to each function
def track_page_view(page_name):
    if 'is_authenticated' in st.session_state and st.session_state.is_authenticated:
        log_action(st.session_state.user.email, "page_view", {"page": page_name})

# Render appropriate page based on navigation
if st.session_state.current_page == "data_import":
    track_page_view("data_import")
    render_data_import()
elif st.session_state.current_page == "data_overview":
    track_page_view("data_overview")
    render_data_overview()
elif st.session_state.current_page == "data_cleaning":
    track_page_view("data_cleaning")
    render_data_cleaning()
elif st.session_state.current_page == "exploratory_analysis":
    track_page_view("exploratory_analysis")
    render_exploratory_analysis()
elif st.session_state.current_page == "visualization":
    track_page_view("visualization")
    render_visualization()
elif st.session_state.current_page == "modeling":
    track_page_view("modeling")
    render_modeling()
elif st.session_state.current_page == "insights":
    track_page_view("insights")
    render_summary_insights()
elif st.session_state.current_page == "dashboard":
    track_page_view("dashboard")
    show_user_dashboard(st.session_state.user.email)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center">
        <p>AI Data Analysis Tool | v1.0</p>
    </div>
    """,
    unsafe_allow_html=True
)