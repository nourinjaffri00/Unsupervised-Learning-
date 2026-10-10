import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(
    page_title= "Unsupervised Pharma",
    page_icon= "⚕️",
    layout='wide'
)
st.title ("⚕️ Getz Pharma Customer Segmentation ")
st.subheader ("Unsupervised Learning with K-Means ")
st.caption ("")
st.info("")

df=pd.read_csv ('getz.csv')

metric_1, metric_2, metric_3, metric_4= st.columns(4)
with metric_1:
    st.metric("Customers", len(df))
with metric_2:
    st.metric("Cities", df['City'].nunique())
with metric_3:
    st.metric("Avg Monthly Orders", f'{df['Monthly_Orders'].mean():.1f}')
with metric_4:
    st.metric("Avg Monthly Spend", f'{df["Monthly_Spend_PKR"].mean():,.0f}')

tab1, tab2, tab3, tab4= st.tabs(
    [
        'Dataset',
        'Choose K',
        'Customer Segments', 
        "What we learnt? "
    ]
)

with tab1:
    st.header(
        "Step 1: Understand Data"
    )
    st.write(
        "Before Machine learning, **Understand** columns "
    )
    st.dataframe(df.head(10), width='stretch')
    st.subheader ("Missing values")

    m= df.isna().sum()
    st.dataframe (
        m.reset_index().rename(
            columns= {'index': "Column", 0: "Missing_Values"}
        ),
        width='stretch',
        hide_index=True
    )

features= [
    'Monthly_Orders', 
    'Avg_Order_Value_PKR',
    'Products_Purchased',
    'Payment_Delay_Days',
    'Days_Since_Last_Order'
]
x= df[features]
scalar= StandardScaler ()
x_scaled= scalar.fit_transform(x)

with tab2:
    st.header("Step 2: How many groups? ")
    st.subheader ("Elbow Method")
    st.write(
    " Train K-Mean with different values of K"
    "Inertia: Inertia tells us how tightly customers sit around clusters centers"
    "Elbow position show improvement"
    )
    k_val= list(range(1,8))
    inertia_val= []
    for k in k_val:
        elbow_model= KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )
        elbow_model.fit(x_scaled)
        inertia_val.append(elbow_model.inertia_)

    elbow_df= pd.DataFrame(
        {
            'K': k_val,
            'Inertia': inertia_val
        }
    )

    st.line_chart(
        elbow_df,
        x='K',
        y='Inertia',
        x_label= '# of Clusters (K)',
        y_label='Inertia'
    )
    st.success ("Chart bend @ K=")

    select_k= st.slider(
        "Choose cluster size", 
        min_value=2,
        max_value=6,
        value=3,
        step=1
    )

model= KMeans(
    n_clusters=select_k,
    random_state=42,
    n_init=10
)
cluster_labels= model.fit_predict(x_scaled)
df_clustered= df.copy()
df_clustered['Cluster']= cluster_labels

cluster_sum= (
    df_clustered.groupby('Cluster').agg(
        Customers= ("Customer_ID", "count"),
        Monthly_Orders= ("Avg_Order_Value_PKR", 'mean'),
        Product_Purchased = ("Products_Purchased", "mean"),
        Payment_Delay_Days=("Days_Since_Last_Order","mean"),
        Monthly_Spend_PKR= ("Monthly_Spend_PKR","mean")

    ).round(1)
)

if select_k==3:
    order_clusters= (
        cluster_sum ['Monthly_Spend_PKR'].sort_values().index.tolist()
    )

    seg_name= {
        order_clusters[0]: 'Low Engagement',
        order_clusters[1]: "Regular",
        order_clusters[2]: 'Premium'

    }
    df_clustered['Segment']=df_clustered['Cluster'].map(seg_name)
else: 
    df_clustered["Segment"]=(
        'Cluster'+df_clustered['Cluster'].astype(str)
    )

with tab3:
    st.header ("Step 3: Explore the Groups: ***K-Means Discovered*")
    st.subheader(" Customers in Each Group")

    seg_counts= (
        df_clustered['Segment'].value_counts().rename_axis('Segment').reset_index(name='Customers')
    )
    st.bar_chart(
        seg_counts,
        x='Segment',
        y= 'Customers',
        x_label= 'Customer Segments', 
        y_label= '# of Customers' 
    )
    st.subheader ("Visualization")
    st.scatter_chart(
        df_clustered,
        x="Monthly_Orders",
        y="Avg_Order_Value_PKR",
        color="Segment",
        size="Products_Purchased",
        x_label="Monthly Orders",
        y_label="Average Order Value (PKR)"
    )
    st.subheader("Cluster Summary")
    seg_sum=(
    df_clustered.groupby("Segment").agg(
        Customers=("Customer_ID","count"),
        Avg_Monthly_Orders=("Monthly_Orders","mean"),
        Avg_Order_Value_PKR=("Avg_Order_Value_PKR","mean"),
        Avg_Products=("Products_Purchased","mean"),
        Avg_Payment_Delay = ("Payment_Delay_Days","mean"),
        Avg_Days_Since_Last_Order=("Days_Since_Last_Order","mean"),
        Avg_Monthly_Spend_PKR=("Monthly_Spend_PKR", "mean"),
        ).round(1).reset_index()
    )

    st.dataframe(seg_sum, width="stretch", hide_index=True)
    if select_k == 3:
        st.success(
            "Great! The model has discovered three useful customer"
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            st.subheader("⭐Premium")
            st.write(
                "Usually place more orders, buy more products and spend more"
            )
        with c2:
            st.subheader("👤 Regular Customers")
            st.write ("Habitual Buyers with Discounts")

        with c3:
            st.subheader("😞Low Engagement")
            st.write ("Always Complain!!!")

    st.subheader("Explore Individual Customers")

    segment_filter = st.selectbox(
        "Choose a Segment",
        options=sorted(df_clustered["Segment"].unique()),
    )

    filtered = df_clustered[
        df_clustered["Segment"] == segment_filter
    ]

    st.dataframe(filtered, width="stretch",hide_index=True)




