# AI Data Analysis Tool SaaS

A comprehensive data analysis tool that allows users to analyze CSV and Excel files, generate visualizations, and build predictive models - all packaged as a complete SaaS (Software as a Service) solution ready for commercial deployment.

## Features

### Core Data Analysis Features
- **Data Import**: Upload CSV and Excel files or use sample data
- **Data Cleaning**: Handle missing values, remove duplicates, standardize column names
- **Exploratory Data Analysis**: Generate statistics, correlations, and distributions
- **Data Visualization**: Create histograms, box plots, scatter plots, and more
- **Predictive Modeling**: Build regression, classification, and clustering models
- **Summary Insights**: Get AI-generated insights about your data

### SaaS Features
- **User Authentication**: Secure login and signup system
- **Subscription Plans**: Free trial, Basic, Premium, and Enterprise tiers
- **Usage Tracking**: Monitor data uploads, model creation, and other activities
- **User Dashboard**: Personal dashboard showing usage statistics and subscription status
- **Usage Limits**: Plan-specific limits on file size, number of models, etc.
- **Payment Integration**: Ready for integration with payment processors

## Subscription Plans

| Feature | Free Trial | Basic | Premium | Enterprise |
|---------|------------|-------|---------|------------|
| Max File Size | 5 MB | 25 MB | 100 MB | 1 GB |
| Max Dataset Rows | 10,000 | 50,000 | 500,000 | 5 million |
| Models Per Day | 3 | 10 | 50 | Unlimited |
| Visualizations | ✓ | ✓ | ✓ | ✓ |
| Data Export | ✓ | ✓ | ✓ | ✓ |
| API Access | ✗ | ✗ | ✓ | ✓ |
| Support | Community | Email | Priority | Dedicated |
| Duration | 14 days | Monthly | Annual | Annual |

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/your-username/data-analysis-tool.git
   cd data-analysis-tool
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```
   streamlit run streamlit_app.py
   ```

## SaaS Deployment

For production deployment, it's recommended to:

1. **Replace SQLite with PostgreSQL**:
   ```python
   # Replace database connection in auth.py, analytics.py, and subscription.py
   import psycopg2
   conn = psycopg2.connect(DATABASE_URL)
   ```

2. **Set up Payment Processing**:
   ```python
   # Add to subscription.py
   import stripe
   stripe.api_key = os.environ["STRIPE_API_KEY"]
   ```

3. **Add Email Verification**:
   ```python
   # Add to auth.py
   from sendgrid import SendGridAPIClient
   ```

4. **Configure a Production Server**:
   - Use Gunicorn or uWSGI in front of Streamlit
   - Set up HTTPS with SSL certificates
   - Configure domain name

## Deployment Options

### Streamlit Cloud (Development)

1. Create a repository on GitHub with these files
2. Sign up for [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your GitHub account and deploy the app

### AWS Elastic Beanstalk (Production)

Perfect for production SaaS deployment:
1. Create an Elastic Beanstalk environment
2. Configure RDS PostgreSQL database
3. Set up load balancer and auto-scaling
4. Connect to Stripe for payments

### Google Cloud Run (Production)

Serverless option for production:
1. Build a Docker container
2. Deploy to Cloud Run
3. Connect to Cloud SQL PostgreSQL
4. Set up Cloud IAM for authentication

## Requirements

- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- streamlit
- plotly
- sqlite3 (development) / PostgreSQL (production)
- stripe (for payment processing)
- sendgrid (for email notifications)