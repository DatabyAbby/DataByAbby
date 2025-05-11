import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import io
import base64
from analytics import get_user_stats
from subscription import check_subscription, get_plan_limits, upgrade_subscription

def show_user_dashboard(user_email):
    st.title("My Dashboard")
    
    # Get subscription info
    subscription = check_subscription(user_email)
    
    # Get usage stats
    stats = get_user_stats(user_email)
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Overview", "Usage", "Subscription"])
    
    with tab1:
        show_overview_tab(user_email, subscription, stats)
    
    with tab2:
        show_usage_tab(user_email, stats, subscription)
    
    with tab3:
        show_subscription_tab(user_email, subscription)
        
def show_overview_tab(user_email, subscription, stats):
    # Check if user has an active subscription
    if subscription['active']:
        st.success(f"✅ Your {subscription['plan'].replace('_', ' ').title()} plan is active")
    else:
        st.warning("⚠️ Your subscription has expired")
    
    # Display key stats
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Datasets Analyzed", stats['uploads'])
    with col2:
        st.metric("Models Built", stats['models'])
    with col3:
        st.metric("Days Remaining", subscription['days_left'])
    
    # Recent activity
    st.subheader("Recent Activity")
    
    if stats['recent_activity']:
        activities = []
        for activity in stats['recent_activity']:
            # Format the timestamp
            dt = datetime.strptime(activity['timestamp'], '%Y-%m-%d %H:%M:%S')
            date_str = dt.strftime('%b %d, %Y')
            time_str = dt.strftime('%H:%M')
            
            # Format the action
            action = activity['action'].replace('_', ' ').title()
            
            # Get a summary from details
            details = activity['details']
            summary = "N/A"
            if isinstance(details, dict):
                if 'file_name' in details:
                    summary = details['file_name']
                elif 'model_type' in details:
                    summary = details['model_type']
            
            activities.append({
                "Date": date_str,
                "Time": time_str,
                "Action": action,
                "Details": summary
            })
        
        activities_df = pd.DataFrame(activities)
        st.dataframe(activities_df, hide_index=True)
    else:
        st.info("No recent activity to display")

def show_usage_tab(user_email, stats, subscription):
    st.subheader("Usage Statistics")
    
    # Get the limits based on the subscription plan
    plan = subscription['plan'] if subscription['active'] else 'free_trial'
    limits = get_plan_limits(plan)
    
    # Create usage progress bars
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Daily Model Usage")
        if limits['models_per_day'] > 0:
            progress = min(1.0, stats['models'] / limits['models_per_day'])
            st.progress(progress)
            st.caption(f"{stats['models']} of {limits['models_per_day']} models used today")
        else:
            st.info("Unlimited models available")
    
    with col2:
        st.subheader("Total Analysis Actions")
        st.metric("Total Actions Performed", stats['total_actions'])
    
    # Show usage trends (mock data - would be real in production)
    st.subheader("Usage Trends")
    
    # Create some sample data for the chart
    dates = [(datetime.now() - timedelta(days=i)).strftime('%b %d') for i in range(7, 0, -1)]
    usage = [min(limits['models_per_day'], int(stats['models'] * (0.5 + i/10))) for i in range(7)]
    
    # Plot the data
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(dates, usage, color='#ff4b4b')
    ax.set_xlabel('Date')
    ax.set_ylabel('Models Created')
    ax.set_title('Models Created Last 7 Days')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Convert plot to base64 string for display
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    st.image(f"data:image/png;base64,{img_str}")

def show_subscription_tab(user_email, subscription):
    st.subheader("Subscription Details")
    
    # Current plan details
    current_plan = subscription['plan'] if subscription['active'] else 'None'
    if current_plan:
        current_plan_name = current_plan.replace('_', ' ').title()
        st.info(f"Current Plan: **{current_plan_name}**")
        
        if subscription['active']:
            st.success(f"Your subscription is active for {subscription['days_left']} more days")
        else:
            st.error("Your subscription has expired")
    else:
        st.warning("You don't have an active subscription")
    
    # Available plans
    st.subheader("Available Plans")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Basic")
        st.markdown("$9.99/month")
        st.markdown("- Up to 25MB files")
        st.markdown("- 50,000 rows max")
        st.markdown("- 10 models per day")
        st.markdown("- Email support")
        if st.button("Upgrade to Basic"):
            # In a real app, this would go to a payment page
            st.session_state.temp_plan = 'basic'
            st.session_state.show_payment = True
    
    with col2:
        st.markdown("### Premium")
        st.markdown("$29.99/month")
        st.markdown("- Up to 100MB files") 
        st.markdown("- 500,000 rows max")
        st.markdown("- 50 models per day")
        st.markdown("- Priority support")
        st.markdown("- API access")
        if st.button("Upgrade to Premium"):
            st.session_state.temp_plan = 'premium'
            st.session_state.show_payment = True
    
    with col3:
        st.markdown("### Enterprise")
        st.markdown("$99.99/month")
        st.markdown("- Up to 1GB files")
        st.markdown("- 5 million rows max")
        st.markdown("- Unlimited models")
        st.markdown("- Dedicated support")
        st.markdown("- Custom features")
        if st.button("Contact Sales"):
            st.info("For enterprise plans, please contact sales@yourcompany.com")
    
    # Payment form (would be replaced with a real payment processor in production)
    if 'show_payment' in st.session_state and st.session_state.show_payment:
        st.subheader("Payment Information")
        st.markdown(f"Upgrading to **{st.session_state.temp_plan.title()}** plan")
        
        # Mock payment form
        st.text_input("Name on Card")
        st.text_input("Card Number")
        col1, col2 = st.columns(2)
        with col1:
            st.text_input("Expiration Date")
        with col2:
            st.text_input("CVC")
        
        if st.button("Complete Payment"):
            # This would process payment in a real app
            upgrade_subscription(user_email, st.session_state.temp_plan)
            st.success("Subscription upgraded successfully!")
            st.session_state.show_payment = False
            st.rerun()