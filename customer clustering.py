### LINK DỮ LIỆU ####
# https://www.kaggle.com/datasets/ilkeryildiz/online-retail-listing

import streamlit as st
import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score, accuracy_score, confusion_matrix, classification_report, roc_curve, auc, davies_bouldin_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import time




# Thêm CSS để căn giữa đoạn văn bản
st.markdown(
    """
    <style>
    .center {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sử dụng lớp center để căn giữa
st.markdown('<div class="center"> <h2 style="text-align:center;">HỆ HỖ TRỢ QUYẾT ĐỊNH - NHÓM 22</h2> </div>', unsafe_allow_html=True)
st.markdown('<div class="center"> <h1 style="text-align:center;">DỰ ĐOÁN TẦN SUẤT MUA HÀNG CỦA KHÁCH HÀNG</h1> </div>', unsafe_allow_html=True)

# Bài toán demo và data
df= pd.read_csv("D:\Downloads\online_retail_listing.csv (1)\online_retail_listing (1).csv")
st.subheader("Dữ liệu lịch sử giao dịch của công ty thương mại điện tử")
st.write(df)


                            # Xử lý dữ liệu và thống kê mô tả
# Bỏ giá trị Null
df = df.dropna()
# Chuyển cột InvoiceDate sang định dạng datetiem
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%d.%m.%Y %H:%M').dt.date
# Tính tổng tiền mỗi đơn hàng
df['total_price']=df['Quantity']*df['Price']
df['total_price'] = df['total_price'].apply(lambda x: max(x, 0))
#Ngày mua hàng gần nhất
max_day = df["InvoiceDate"].max() + timedelta(days =1)
df['diff']=(max_day-df.InvoiceDate)

@st.cache_data
def loading_data():
    data = df.groupby('Customer ID')['Invoice'].nunique().reset_index(name='order_count')  #Tính tổng số đơn hàng của mỗi khách hàng
    data = pd.merge(data, df.groupby('Customer ID')['total_price'].sum().reset_index(name='total_price'), on='Customer ID')  #Tính tổng tiền của mỗi khách hàng
    
    data = pd.merge(data, df.groupby('Customer ID')['InvoiceDate'].min().reset_index(name = 'begin_date'), on = 'Customer ID')  #Tính ngày đầu tiên mua hàng của mỗi khách hàng
    data = pd.merge(data, df.groupby('Customer ID')['InvoiceDate'].max().reset_index(name = 'end_date'), on = 'Customer ID')     # Ngày cuối cùng mua hàng
    data['begin_date'] = pd.to_datetime(data['begin_date'], format='%d.%m.%Y')
    data['end_date'] = pd.to_datetime(data['end_date'], format='%d.%m.%Y')
    data['avg_date'] = (((data['end_date'] - data['begin_date']).dt.days/ data['order_count'])).round(0)    #Số ngày trung bình để phát sinh 1 đơn hàng
    data= pd.merge(data, df.groupby('Customer ID')['diff'].min().reset_index(name = 'recency'), on = 'Customer ID')       #Ngày mua hàng cuối cùng
    data['recency'] = data['recency'].apply(lambda x: x.days)
    data = data.drop(columns = ['begin_date', 'end_date'])

    # Tính số đơn hàng bị hủy
    cancel_data = df[df['Invoice'].str.startswith('C', na=False)].drop_duplicates()
    data = pd.merge(data, cancel_data.groupby('Customer ID')['Invoice'].nunique().reset_index(name = 'cancel_order'), on = 'Customer ID')

    # Tỷ lệ đơn hàng bị hủy
    data['order_success_rate'] = data['cancel_order']/data['order_count']
    data['success_order'] = data['order_count'] - data['cancel_order']

    return data


# data_load_state = st.text("Loading data...")
data = loading_data()
# data_load_state.text("Loading data.... done!!")
# time.sleep(2)
# data_load_state.empty()

options = st.selectbox(
    'Lựa chọn yêu cầu:',
    ('Thống kê dữ liệu', 'Dự báo tần số mua hàng của khách hàng') 
)

if options == 'Thống kê dữ liệu':
    prediction_button = st.button("Đồng ý")
    if prediction_button:
        st.header("Các thống kê giao dịch của công ty")
        # Hiển thị các thống kê
        st.write("Tổng số đơn hàng đã thực hiện: {}".format(data['order_count'].sum()))
        st.write("Tỷ lệ đơn hoàn thành: {}%".format(round(100 * (1 - data['cancel_order'].sum() / data['order_count'].sum()), 2)))
        st.write("Tổng số khách hàng đã giao dịch: {}".format(data['Customer ID'].count()))
        st.write("Tổng doanh thu: {}£".format(data['total_price'].sum()))
        st.write("Doanh thu trung bình cho mỗi đơn hàng thành công: {}£".format(round(data['total_price'].sum() / (data['order_count'].sum() - data['cancel_order'].sum()))))
        st.write("Số tiền trung bình khách hàng phải trả: {}£".format(round(data['total_price'].sum() / data['Customer ID'].count(), 2)))
        st.write("Số lượng sản phẩm của công ty: {}".format(df['StockCode'].nunique()))
        #Nhóm dữ liệu theo quốc gia và tính tổng doanh thu cho mỗi quốc gia
        country_revenue = df.groupby('Country')['total_price'].sum().reset_index()
        #Xác định quốc gia có doanh thu cao nhất
        max_revenue_country = country_revenue.loc[country_revenue['total_price'].idxmax()]
        st.write("Quốc gia có số daonh thu lớn nhất là {} với doanh thu {}£".format(max_revenue_country['Country'], max_revenue_country['total_price']))
        

elif options == 'Dự báo tần số mua hàng của khách hàng':
    prediction= st.selectbox(
        'Lựa chọn mô hình:',
        ( 'Logistic Regression','K-means','Agglomerative clustering') 
    )
    prediction_button2 = st.button("Dự báo")
    if prediction_button2:
        ############################################# K-means ########################################
        if prediction == 'K-means':
            X = data[['total_price', 'success_order']]

            # Chuẩn hóa dữ liệu
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Hiển thị dữ liệu sau khi phân cụm
            def data_after(df, lable):
                df = data
                label = pd.DataFrame(lable, columns=['label'])
                df = pd.concat([data, label], axis = 1)
                df = df[["Customer ID", "order_count",	"total_price	avg_date",	"recency", "label"]]
                st.write("Bảng dữ liệu sau khi phân cụm")
                st.dataframe(df)
            # Đánh giá kết quả phân cụm
            def result(X, label):
                st.write('Đánh giá kết quả phân cụm:')
                silhouette_avg = silhouette_score(X, label)
                davies_bouldin = davies_bouldin_score(X, label)
                st.write("Silhouette Score: {}".format(silhouette_avg))
                st.write("Davies-Bouldin Index: {}".format(davies_bouldin))
            # Vẽ biểu đồ phân cụm
            def plot_clusters(X, labels, title):
                fig, ax = plt.subplots(figsize=(12, 10))
                scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow', alpha=0.6, edgecolors='w', s=50)
                
                # Vẽ biên giới của từng cụm
                unique_labels = np.unique(labels)
                for label in unique_labels:
                    points = X[labels == label]
                    if len(points) >= 3:  
                        hull = ConvexHull(points)
                        for simplex in hull.simplices:
                            ax.plot(points[simplex, 0], points[simplex, 1], 'k-', alpha=0.5)
                        
                legend = ax.legend(*scatter.legend_elements(), title="Clusters")
                ax.add_artist(legend)
                ax.set_title(title)
                ax.set_xlabel('Total Price')
                ax.set_ylabel('Success Order')
                st.pyplot(fig)
            ################## Mô hình 1: Mô hình cơ bản
            st.header("Mô hình 1: Mô hình cơ bản")
            # Thực hiện phân cụm sử dụng K-means với số cụm K = 5
            kmeans1 = KMeans(n_clusters= 5, init='k-means++', max_iter=100, n_init=10, tol= 0.0001, random_state=42)
            y_kmeans1 = kmeans1.fit_predict(X_scaled)
            col1, col2 = st.columns(2)
            with col1:
                result(X_scaled,y_kmeans1)
                plot_clusters(X_scaled, y_kmeans1, "Mô hình 1: Mô hình cơ bản")
            with col2:
                data_after(X_scaled, y_kmeans1)

            ##################### Mô hình 2: Thay đổi số vòng lặp 'max_iter' từ 100 lên 300
            st.header("Mô hình 2: Thay đổi số vòng lặp 'max_iter' từ 100 lên 300")
            # Thực hiện phân cụm sử dụng K-means với số cụm K = 5
            kmeans2 = KMeans(n_clusters= 5, init='k-means++', max_iter=300, n_init=10, tol= 0.0001, random_state=42)
            y_kmeans2 = kmeans2.fit_predict(X_scaled)
            col1, col2 = st.columns(2)
            with col1:
                result(X_scaled,y_kmeans2)
                plot_clusters(X_scaled, y_kmeans2, "Mô hình 2: Thay đổi số vòng lặp 'max_iter' từ 100 lên 300")
            with col2:
                data_after(X_scaled, y_kmeans2)

            ########################### Mô hình 3: Thay đổi 'n_init' từ 10 lên 30
            st.header("Mô hình 3: Thay đổi 'n_init' từ 10 lên 30")
            # Thực hiện phân cụm sử dụng K-means với số cụm K = 5
            kmeans3 = KMeans(n_clusters= 5, init='k-means++', max_iter=100, n_init=30, tol= 0.0001, random_state=42)

            y_kmeans3 = kmeans3.fit_predict(X_scaled)
            col1, col2 = st.columns(2)
            with col1:
                result(X_scaled,y_kmeans3)
                plot_clusters(X_scaled, y_kmeans3, "Mô hình 3: Thay đổi 'n_init' từ 10 lên 30")
            with col2:
                data_after(X_scaled, y_kmeans3)

            ################### Mô hình 4: Thay đổi 'init' từ 'k-means++' thành 'random'
            st.header("Mô hình 4: Thay đổi 'init' từ 'k-means++' thành 'random'")
            # Thực hiện phân cụm sử dụng K-means với số cụm K = 5
            kmeans4 = KMeans(n_clusters= 5, init='random', max_iter=100, n_init=10, tol= 0.0001, random_state=42)

            y_kmeans4 = kmeans4.fit_predict(X_scaled)
            col1, col2 = st.columns(2)
            with col1:
                result(X_scaled,y_kmeans4)
                plot_clusters(X_scaled, y_kmeans4, "Mô hình 4: Thay đổi 'init' từ 'k-means++' thành 'random'")
            with col2:
                data_after(X_scaled, y_kmeans4)

            ################ Mô hình 5: Thay đổi 'tol' từ 0.0001 lên 0.1
            st.header("Mô hình 5: Thay đổi 'tol' từ 0.0001 lên 0.1")
            # Thực hiện phân cụm sử dụng K-means với số cụm K = 5
            kmeans5 = KMeans(n_clusters= 5, init='k-means++', max_iter=100, n_init=10, tol= 0.1, random_state=42)
            y_kmeans5 = kmeans5.fit_predict(X_scaled)
            col1, col2 = st.columns(2)
            with col1:
                result(X_scaled,y_kmeans5)
                plot_clusters(X_scaled, y_kmeans5, "Mô hình 5: Thay đổi 'tol' từ 0.0001 lên 0.1")
            with col2:
                data_after(X_scaled, y_kmeans5)

##################################################################################################################################################################
        ################################################ Hồi quy logistic ####################################
        elif prediction == 'Logistic Regression':
            # Gán nhãn dữ liệu
            X = data
            total_price_quantiles = X['total_price'].quantile([0.2, 0.4, 0.6, 0.8])
            success_order_quantiles = X['order_count'].quantile([0.2, 0.4, 0.6, 0.8])
            def classify_customer(row):
                if (row['total_price'] <= total_price_quantiles[0.2] and row['order_count'] <= success_order_quantiles[0.2]) or (row['total_price'] > total_price_quantiles[0.2] and row['order_count'] <= success_order_quantiles[0.2]) or (row['total_price'] <= total_price_quantiles[0.2] and row['order_count'] > success_order_quantiles[0.2]):
                    return 0
                elif (row['total_price'] <= total_price_quantiles[0.4] and row['order_count'] <= success_order_quantiles[0.4]) or (row['total_price'] > total_price_quantiles[0.4] and row['order_count'] <= success_order_quantiles[0.4]) or (row['total_price'] <= total_price_quantiles[0.4] and row['order_count'] > success_order_quantiles[0.4]):
                    return 1
                elif (row['total_price'] <= total_price_quantiles[0.6] and row['order_count'] <= success_order_quantiles[0.6]) or (row['total_price'] > total_price_quantiles[0.6] and row['order_count'] <= success_order_quantiles[0.6]) or (row['total_price'] <= total_price_quantiles[0.6] and row['order_count'] > success_order_quantiles[0.6]):
                    return 2
                elif (row['total_price'] <= total_price_quantiles[0.8] and row['order_count'] <= success_order_quantiles[0.8]) or (row['total_price'] > total_price_quantiles[0.8] and row['order_count'] <= success_order_quantiles[0.8]) or (row['total_price'] <= total_price_quantiles[0.8] and row['order_count'] > success_order_quantiles[0.8]):
                    return 3
                else:
                    return 4
            X['loyal']= X.apply(classify_customer, axis=1)

            # Loại bỏ ngoại lai
            z_scores = np.abs(stats.zscore(X))
            X = X[(z_scores < 3).all(axis=1)]

            X_scaled = X            
                # Xác định các biến độc lập (features) và biến phụ thuộc (target)
            X = X_scaled[['total_price', 'avg_date', 'recency']]
            y = X_scaled['loyal']
            # Chia dữ liệu thành tâp huấn luyện và tập kiểm tra
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Ma trận nhầm lân
            def conf_matrix(y_testt, y_pred):
                conf_matrix = confusion_matrix(y_testt, y_pred)
                fig = plt.figure(figsize=(10, 6))
                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.title('Confusion Matrix')
                st.pyplot(fig)

            # Biểu đồ ROC
            def ROC(y_testt, y_pred):
                # Tính toán các giá trị fpr, tpr và ngưỡng cho từng lớp để vẽ biểu đồ ROC
                fpr = {}
                tpr = {}
                roc_auc = {}
                for i in range(len(np.unique(y))):
                    fpr[i], tpr[i], _ = roc_curve(y_testt == i, y_pred == i)
                    roc_auc[i] = auc(fpr[i], tpr[i])
                # Vẽ biểu đồ ROC
                fig = plt.figure(figsize=(10,6))
                for i in range(len(np.unique(y))):
                    plt.plot(fpr[i], tpr[i], label=f'ROC curve of class {i} (area = {roc_auc[i]:.2f})')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic (ROC)')
                plt.legend(loc="lower right")
                st.pyplot(fig)

            ############################################# Mô hình 1: Mô hình hồi quy logistic ban đầu
            st.header("Mô hình 1: Mô hình hồi quy logistic ban đầu")
            # Tạo mô hình hồi quy logistic
            model1 = LogisticRegression()
            # Huấn luyện mô hình
            model1.fit(X_train, y_train)
            # Dự đoán trên tập kiểm tra
            y_pred1 = model1.predict(X_test)
            # Đánh giá mô hình
            st.markdown(f'<div class="center"> <h5 style="text-align:center;">Accuracy: {round(accuracy_score(y_test, y_pred1), 6)}</h5> </div>', unsafe_allow_html=True)
            col1 ,col2 = st.columns(2)
            with col1:
                st.write("Confusion Matrix")
                conf_matrix(y_test, y_pred1)

            with col2:
                st.write("Classification Report")
                report1 = classification_report(y_test, y_pred1, output_dict=True)
                report1 = pd.DataFrame(report1).transpose()
                st.write(report1)
            st.write("Đường cong ROC")
            ROC(y_test, y_pred1)

            ########################################### Mô hình 2: Mô hình có regularizer L1 (Lasso)
            st.header("Mô hình 2: Mô hình có regularizer L1 (Lasso)")

                    # Tạo mô hình hồi quy logistic
            model2 = LogisticRegression(penalty='l1', solver='liblinear')
            # Huấn luyện mô hình
            model2.fit(X_train, y_train)

            y_pred2 = model2.predict(X_test)
                # Đánh giá mô hình
            st.markdown(f'<div class="center"> <h5 style="text-align:center;">Accuracy: {round(accuracy_score(y_test, y_pred2), 6)}</h5> </div>', unsafe_allow_html=True)
            col1 ,col2 = st.columns(2)
            with col1:
                st.write("Confusion Matrix")
                conf_matrix(y_test, y_pred2)

            with col2:
                st.write("Classification Report")
                report2 = classification_report(y_test, y_pred2, output_dict=True)
                report2 = pd.DataFrame(report2).transpose()
                st.write(report2)
            st.write("Đường cong ROC")
            ROC(y_test, y_pred2)


                ############################################## Mô hình 3: Mô hình có regularizer L2 và C= 0.1 (Ridge)
            st.header("Mô hình 3: Mô hình có regularizer L2 và C= 0.1 (Ridge)")

            # Tạo mô hình hồi quy logistic
            model3 = LogisticRegression(penalty='l2', solver='lbfgs', C= 0.1)
            # Huấn luyện mô hình
            model3.fit(X_train, y_train)
            # Dự đoán trên tập kiểm tra
            y_pred3 = model3.predict(X_test)
                # Đánh giá mô hình
            st.markdown(f'<div class="center"> <h5 style="text-align:center;">Accuracy: {round(accuracy_score(y_test, y_pred3), 6)}</h5> </div>', unsafe_allow_html=True)
            col1 ,col2 = st.columns(2)
            with col1:
                st.write("Confusion Matrix")
                conf_matrix(y_test, y_pred3)

            with col2:
                st.write("Classification Report")
                report3 = classification_report(y_test, y_pred3, output_dict=True)
                repor3 = pd.DataFrame(report3).transpose()
                st.write(report3)
            st.write("Đường cong ROC")
            ROC(y_test, y_pred3)

            ############################################ Mô hình 4: Mô hình với tỷ lệ thay đổi weight class (class weight balancing)
            st.header("Mô hình 4: Mô hình với tỷ lệ thay đổi weight class (class weight balancing)")

            # Tạo mô hình hồi quy logistic
            model4 = LogisticRegression(class_weight='balanced')
            # Huấn luyện mô hình
            model4.fit(X_train, y_train)
            # Dự đoán trên tập kiểm tra
            y_pred4 = model4.predict(X_test)
                # Đánh giá mô hình
            st.markdown(f'<div class="center"> <h5 style="text-align:center;">Accuracy: {round(accuracy_score(y_test, y_pred4), 6)}</h5> </div>', unsafe_allow_html=True)
            col1 ,col2 = st.columns(2)
            with col1:
                st.write("Confusion Matrix")
                conf_matrix(y_test, y_pred4)

            with col2:
                st.write("Classification Report")
                report4 = classification_report(y_test, y_pred4, output_dict=True)
                report4 = pd.DataFrame(report4).transpose()
                st.write(report4)
            st.write("Đường cong ROC")
            ROC(y_test, y_pred4)


        # ####################################Mô hình 5: Mô hình với giới hạn số lần lặp (maximum iterations)
            st.header("Mô hình 5: Mô hình với giới hạn số lần lặp (maximum iterations)")
            # Tạo mô hình hồi quy logistic
            model5 = LogisticRegression(max_iter=1000)
            # Huấn luyện mô hình
            model5.fit(X_train, y_train)
            # Dự đoán trên tập kiểm tra
            y_pred5 = model5.predict(X_test)

                # Đánh giá mô hình
            st.markdown(f'<div class="center"> <h5 style="text-align:center;">Accuracy: {round(accuracy_score(y_test, y_pred5), 6)}</h5> </div>', unsafe_allow_html=True)
            col1 ,col2 = st.columns(2)
            with col1:
                st.write("Confusion Matrix")
                conf_matrix(y_test, y_pred5)

            with col2:
                st.write("Classification Report")
                report5 = classification_report(y_test, y_pred5, output_dict=True)
                report5 = pd.DataFrame(report5).transpose()
                st.write(report5)
            st.write("Đường cong ROC")
            ROC(y_test, y_pred5)



##########################################################################################################################################################################

        #################################### Agglomerative clustering ################################
        elif prediction == 'Agglomerative clustering':

            #X = data[['success_order', 'total_price', 'recency']]
            # X = data[['total_price', 'success_order']]
            X = data[['total_price', 'order_count']]
            # Loại bỏ ngoại lai
            z_scores = np.abs(stats.zscore(X))
            X = X[(z_scores < 3).all(axis=1)]
            # Chuẩn hóa dữ liệu
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

                    
             # Hàm đánh giá và hiển thị kết quả phân cụm
            def evaluate_clustering(X, labels):
                silhouette_avg = silhouette_score(X, labels)
                return silhouette_avg
            # Vẽ biểu đồ phân cụm
            def plot_clusters(X, labels, title):
                fig, ax = plt.subplots(figsize=(12, 10))
                scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow', alpha=0.6, edgecolors='w', s=50)
                
                # Vẽ biên giới của từng cụm
                unique_labels = np.unique(labels)
                for label in unique_labels:
                    points = X[labels == label]
                    if len(points) >= 3:  
                        hull = ConvexHull(points)
                        for simplex in hull.simplices:
                            ax.plot(points[simplex, 0], points[simplex, 1], 'k-', alpha=0.5)
                
                legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
                ax.add_artist(legend1)
                ax.set_title(title)
                ax.set_xlabel('Total Price')
                ax.set_ylabel('Success Order')
                st.pyplot(fig)

            # Hiển thị dữ liệu sau khi phân cụm
            def data_after(df, lable):
                df = data
                label = pd.DataFrame(lable, columns=['label'])
                df = pd.concat([data, label], axis = 1)
                df = df[["Customer ID", "order_count",	"total_price	avg_date",	"recency", "label"]]
                st.write("Bảng dữ liệu sau khi phân cụm")
                st.dataframe(df)
                ########################################## Mô hình ban đầu
            st.header("Model 1: Mô hình mặc định")
            model1 = AgglomerativeClustering(n_clusters=5)
            labels1 = model1.fit_predict(X_scaled)
            silhouette_avg1= silhouette_score(X_scaled, labels1)
            st.markdown(f'<div class="center"> <h5 style="text-align:center;"> Silhouette Score: {silhouette_avg1}</h5> </div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1: 
                data_after(X, labels1)
            with col2:
                plot_clusters(X_scaled, labels1, "Model 1: Mô hình mặc định")


        
            # Mô hình 2: Thay đổi khoảng cách sang 'manhattan'
            st.header("\nModel 2: Tính khoảng cách các điểm bằng Manhattan")
            model2 = AgglomerativeClustering(n_clusters=5, metric='manhattan', linkage='average')
            labels2 = model2.fit_predict(X_scaled)
            silhouette_avg2= silhouette_score(X_scaled, labels2)
            st.markdown(f'<div class="center"> <h5 style="text-align:center;"> Silhouette Score: {silhouette_avg2}</h5> </div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1: 
                data_after(X, labels2)
            with col2:
                plot_clusters(X_scaled, labels2, "Model 2: Tính khoảng cách các điểm bằng Manhattan")
            
            ######################## Mô hình 3: Thay đổi phương pháp hợp nhất cụm sang 'complete'
            st.header("\nModel 3: Hợp nhất cụm bằng Complete")
            model3 = AgglomerativeClustering(n_clusters=5, linkage='complete')
            labels3 = model3.fit_predict(X_scaled)
            silhouette_avg3 = silhouette_score(X_scaled, labels3)
            st.markdown(f'<div class="center"> <h5 style="text-align:center;"> Silhouette Score: {silhouette_avg3}</h5> </div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1: 
                data_after(X, labels3)
            with col2:
                plot_clusters(X_scaled, labels3, "Model 3: Hợp nhất cụm bằng Complete")
   
            #################### Mô hình 4: Thay đổi phương pháp hợp nhất cụm sang 'average'
            st.header("\nModel 4: Hợp nhất cụm bằng Average")
            model4 = AgglomerativeClustering(n_clusters=5, linkage='average')
            labels4 = model4.fit_predict(X_scaled)
            silhouette_avg4 = silhouette_score(X_scaled, labels4)
            st.markdown(f'<div class="center"> <h5 style="text-align:center;"> Silhouette Score: {silhouette_avg4}</h5> </div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1: 
                data_after(X, labels4)
            with col2:
                plot_clusters(X_scaled, labels4, "Model 4: Hợp nhất cụm bằng Average")            
         
            # Mô hình 5: Thay đổi khoảng cách sang 'cosine'
            st.header("\nModel 5: Tính khoảng cách các điểm bằng Cosine")
            model5 = AgglomerativeClustering(n_clusters=5, metric='cosine', linkage='average')
            labels5 = model5.fit_predict(X_scaled)
            silhouette_avg5 = silhouette_score(X_scaled, labels5)
            st.markdown(f'<div class="center"> <h5 style="text-align:center;"> Silhouette Score: {silhouette_avg5}</h5> </div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1: 
                data_after(X, labels5)
            with col2:
                plot_clusters(X_scaled, labels5, "Model 5: Tính khoảng cách các điểm bằng Cosine")

            # So sánh các mô hình
            silhouette_scores = [silhouette_avg1, silhouette_avg2, silhouette_avg3, silhouette_avg4, silhouette_avg5]
            models = ['Default', 'Manhattan', 'Complete', 'Average', 'Cosine']

            st.header("So sánh Silhouette Score của các Models") 
            # Tạo biểu đồ
            fig, ax = plt.subplots(figsize=(10, 7))
            sns.barplot(x=models, y=silhouette_scores, ax=ax)
            ax.set_xlabel('Models')
            ax.set_ylabel('Silhouette Score')
            ax.set_title('Silhouette Scores')
                
            # Hiển thị biểu đồ trong Streamlit
            st.pyplot(fig)

            # st.header("Nhận xét")
            # nhan_xet = {
            #     " " :['Mô tả mô hình', 'Nhận xét'],
            #     "Mô hình 1" : ["Mô hình 1 sử dụng các tham số mặc định của thuật toán gom cụm Agglomerative Clustering, bao gồm phương pháp liên kết ward và độ đo khoảng cách Euclidean", 
            #                 'Silhouette Score = 0.60, tương đối cao cho thấy sự phân chia cụm khá tốt tuy nhiên vẫn có thể cải tiến mô hình.'],
            #     "Mô hình 2" : ['Mô hình 2 thay đổi phương thức tính khoảng cách thành khoảng cách Manhattan, một phương pháp tính khoảng cách dựa trên độ lớn của các vectơ giữa các điểm.', 
            #                 'Silhouette Score = 0.70, cho thấy việc sử dụng phương thức tính khoảng cách Manhattan đã cải thiện hiệu suất cho thuật toán.'],
            #     "Mô hình 3" : ['Mô hình 3 sử dụng phương pháp liên kết complete cho việc gom cụm, trong đó các cụm được hình thành dựa trên khoảng cách tối đa giữa các điểm trong các cụm.', 
            #                 'Silhouette Score = 0.71 không có sự khác biệt nhiều so với mô hình 2.'],
            #     "Mô hình 4" : ['Mô hình 4 sử dụng phương pháp liên kết average cho việc gom cụm, trong đó các cụm được hình thành dựa trên trung bình của các khoảng cách giữa các điểm trong các cụm.', 
            #                 'Silhouette Score = 0.73, mô hình này có điểm Silhouette Score cao nhất trong số các mô hình đã thử nghiệm, cho thấy phương pháp liên kết trung bình tạo ra các cụm có độ tách biệt tốt nhất.'],
            #     "Mô hình 5" : ['Mô hình 5 thay đổi phương thức tính khoảng cách thành khoảng cách Cosine, một phương pháp đo độ tương đồng giữa các vectơ bằng cách tính góc giữa chúng.', 
            #                 'Mô hình này có điểm Silhouette Score = 0.21 tương đối thấp, cho thấy việc sử dụng phương thức tính khoảng cách Cosine không phù hợp với bộ dữ liệu.'],
            # }
            # nhan_xet = pd.DataFrame(nhan_xet).set_index(" ")
            # # Hiển thị bảng
            # st.table(data = nhan_xet)


