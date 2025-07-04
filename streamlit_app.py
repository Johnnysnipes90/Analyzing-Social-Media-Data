import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score
from collections import Counter

# -------- Streamlit page configuration --------
st.set_page_config(
    page_title="📈 Social Media Engagement Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📊 Social Media Engagement Rate Dashboard")
st.markdown("""
Welcome to your interactive social media analytics dashboard! Here you can:
- Explore your dataset
- View feature importance
- Compare predicted vs actual engagement rates
- Predict engagement for new posts
""")

# -------- Load dataset & model --------
df = pd.read_csv('data/raw_data.csv')
model = joblib.load('best_gradient_boosting_model.joblib')

# -------- Feature engineering (same as notebook) --------
df['post_date'] = pd.to_datetime(df['post_date'])
df['day_of_week'] = df['post_date'].dt.day_name()
df['hour'] = df['post_date'].dt.hour
df['engagement'] = df['likes'] + df['comments'] + df['shares']
df['engagement_rate'] = df['engagement'] / df['followers']

df['hashtag_count'] = df['hashtags'].apply(lambda x: len(x.split()))
df['like_rate'] = df['likes'] / df['impressions']
df['comment_rate'] = df['comments'] / df['impressions']
df['share_rate'] = df['shares'] / df['impressions']
df['link_click_rate'] = df['link_clicks'] / df['followers']
df['caption_length'] = df['caption_text'].apply(lambda x: len(x.split()))

# One-hot encode
df_model = pd.get_dummies(df, columns=['platform', 'content_type', 'day_of_week'], drop_first=True)
target = 'engagement_rate'
features = [col for col in df_model.columns if col not in [
    'post_id', 'post_date', 'caption_text', 'hashtags', 'engagement_rate', 'engagement', 
    'likes', 'comments', 'shares', 'impressions'
]]
X = df_model[features]
y = df_model[target]

# -------- Sidebar --------
theme = st.sidebar.radio("🎨 Choose Theme", ['Light', 'Dark'])
st.sidebar.markdown(f"🎨 **Current Theme:** {theme}")

if theme == "Dark":
    st.markdown("""
    <style>
        body {background-color: #0E1117; color: white;}
        .st-emotion-cache-1v0mbdj {background-color: #0E1117;}
    </style>
    """, unsafe_allow_html=True)

st.sidebar.caption(f"⏰ Session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

charts_to_show = st.sidebar.multiselect(
    "📊 Choose Charts to Display", 
    ["Platform Boxplot", "Day of Week Bar", "Hour Line", "Feature Importance", "Predicted vs Actual"],
    default=["Platform Boxplot", "Day of Week Bar", "Feature Importance"]
)

# -------- Data exploration --------
st.markdown("## 📑 Dataset Preview")
st.dataframe(df.head(10))

if "Platform Boxplot" in charts_to_show:
    st.markdown("## 📈 Engagement Rate Distribution by Platform")
    fig1, ax1 = plt.subplots(figsize=(8,6))
    sns.boxplot(data=df, x='platform', y='engagement_rate', palette='Set2', ax=ax1, hue='platform')
    st.pyplot(fig1)
    st.caption("💡 **Insight:** Compare engagement rates across platforms to inform platform strategy.")

if "Day of Week Bar" in charts_to_show:
    st.markdown("## 📅 Average Engagement by Day of Week")
    avg_engagement_day = df.groupby('day_of_week')['engagement_rate'].mean().reindex(
        ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    )
    fig2, ax2 = plt.subplots(figsize=(10,6))
    sns.barplot(x=avg_engagement_day.index, y=avg_engagement_day.values, ax=ax2, color='skyblue')
    ax2.set_ylabel("Average Engagement Rate")
    st.pyplot(fig2)
    st.caption("💡 **Tip:** Post on days with higher average engagement to boost performance.")

if "Hour Line" in charts_to_show:
    st.markdown("## ⏰ Average Engagement by Posting Hour")
    avg_engagement_hour = df.groupby('hour')['engagement_rate'].mean()
    fig3, ax3 = plt.subplots(figsize=(10,6))
    avg_engagement_hour.plot(kind='line', marker='o', ax=ax3)
    ax3.set_xlabel("Hour of Day")
    ax3.set_ylabel("Average Engagement Rate")
    ax3.grid(True)
    st.pyplot(fig3)
    st.caption("💡 **Observation:** Identify the best times to post based on engagement trends.")

# -------- Top hashtags --------
all_hashtags = " ".join(df['hashtags']).split()
hashtag_counts = Counter(all_hashtags)
top_hashtags = pd.DataFrame(hashtag_counts.most_common(10), columns=['Hashtag', 'Count'])

st.markdown("## 🏷️ Top 10 Hashtags")
st.dataframe(top_hashtags)
st.caption("💡 **Use popular hashtags strategically to increase reach and engagement.**")

# -------- Feature Importance --------
if "Feature Importance" in charts_to_show:
    st.markdown("## ⭐ Feature Importance from Best Model")
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({'feature': X.columns, 'importance': importances}).sort_values(
        by='importance', ascending=False).reset_index(drop=True)
    st.dataframe(feature_importance_df.head(10))

    fig4, ax4 = plt.subplots(figsize=(10,6))
    sns.barplot(data=feature_importance_df.head(10), x='importance', y='feature', palette='viridis', ax=ax4)
    ax4.set_title('Top 10 Influential Features')
    st.pyplot(fig4)

# -------- Predicted vs Actual --------
if "Predicted vs Actual" in charts_to_show:
    st.markdown("## 🔎 Predicted vs Actual Engagement Rate")
    y_pred = model.predict(X)
    fig5, ax5 = plt.subplots(figsize=(8,6))
    ax5.scatter(y, y_pred, alpha=0.6)
    ax5.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    ax5.set_xlabel('Actual Engagement Rate')
    ax5.set_ylabel('Predicted Engagement Rate')
    ax5.set_title('Predicted vs Actual Engagement Rate')
    st.pyplot(fig5)

    # Display model performance metrics
    rmse_test = np.sqrt(mean_squared_error(y, y_pred))
    r2_test = r2_score(y, y_pred)
    st.info(f"📊 **Model Performance on Data:** RMSE = {rmse_test:.4f}, R² = {r2_test:.4f}")

# -------- Predict new data --------
st.sidebar.header("🔮 Predict New Engagement Rate")
with st.sidebar.form("prediction_form"):
    followers = st.number_input("Followers", min_value=1, value=1000)
    hashtag_count = st.number_input("Hashtag Count", min_value=0, value=2)
    like_rate = st.number_input("Like Rate (0-1)", min_value=0.0, max_value=1.0, value=0.1)
    comment_rate = st.number_input("Comment Rate (0-1)", min_value=0.0, max_value=1.0, value=0.05)
    share_rate = st.number_input("Share Rate (0-1)", min_value=0.0, max_value=1.0, value=0.02)
    link_click_rate = st.number_input("Link Click Rate (0-1)", min_value=0.0, max_value=1.0, value=0.01)
    caption_length = st.number_input("Caption Length (words)", min_value=0, value=15)
    hour = st.slider("Hour of Post", min_value=0, max_value=23, value=12)
    platform_instagram = st.checkbox("Platform: Instagram")
    platform_twitter = st.checkbox("Platform: Twitter")
    content_type_text = st.checkbox("Content Type: Text")
    content_type_video = st.checkbox("Content Type: Video")
    day_of_week = st.selectbox("Day of Week", [
        "Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"
    ])
    submit = st.form_submit_button("Predict")

if submit:
    input_dict = {
        'followers': followers,
        'hashtag_count': hashtag_count,
        'like_rate': like_rate,
        'comment_rate': comment_rate,
        'share_rate': share_rate,
        'link_click_rate': link_click_rate,
        'caption_length': caption_length,
        'hour': hour,
    }
    input_dict['platform_Instagram'] = int(platform_instagram)
    input_dict['platform_Twitter'] = int(platform_twitter)
    input_dict['content_type_text'] = int(content_type_text)
    input_dict['content_type_video'] = int(content_type_video)
    for day in ['Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']:
        input_dict[f'day_of_week_{day}'] = int(day == day_of_week)
    for col in X.columns:
        if col not in input_dict:
            input_dict[col] = 0
    input_df = pd.DataFrame([input_dict])
    missing_cols = [col for col in X.columns if col not in input_df.columns]
    for col in missing_cols:
        input_df[col] = 0
    extra_cols = [col for col in input_df.columns if col not in X.columns]
    input_df.drop(columns=extra_cols, inplace=True)
    input_df = input_df[X.columns]

    with st.spinner("🔮 Predicting engagement..."):
        prediction = model.predict(input_df)[0]
        st.sidebar.success(
            f"""
            📈 **Predicted Engagement Rate: {prediction:.4f}**

            👉 This means for every 1,000 followers, your post is expected to receive roughly
            **{prediction * 100:.1f} engagements per 1,000 followers** (likes, comments, shares combined).

            🔎 Consider improving timing, hashtags, and content type to increase engagement.
            """
        )
