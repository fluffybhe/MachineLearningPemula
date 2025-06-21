# 🚀 Submission Akhir - Clustering BMLP
**Nama:** Febhe Maulita May Pramasta  
**Proyek:** Clustering pada Dataset untuk Segmentasi Data  
**Program:** Laskar AI 2025

---

# **1. Perkenalan Dataset**


# Customer Personality Clustering and Classification

Dataset: https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis

Workflow
1. Cluster unlabeled dataset using Agglomerative Clustering 
2. Add label based on the clustering result
3. Export the dataset (clustered.csv)
4. Train and test clustering model based on labeled dataset using Decission Tree and Naive Bayes

Features: 
- ID: Customer's unique identifier
- Year_Birth: Customer's birth year
- Education: Customer's education level
- Marital_Status: Customer's marital status
- Income: Customer's yearly household income
- Kidhome: Number of children in customer's household
- Teenhome: Number of teenagers in customer's household
- Dt_Customer: Date of customer's enrollment with the company
- Recency: Number of days since customer's last purchase
- Complain: 1 if the customer complained in the last 2 years, 0 otherwise
- MntWines: Amount spent on wine in last 2 years
- MntFruits: Amount spent on fruits in last 2 years
- MntMeatProducts: Amount spent on meat in last 2 years
- MntFishProducts: Amount spent on fish in last 2 years
- MntSweetProducts: Amount spent on sweets in last 2 years
- MntGoldProds: Amount spent on gold in last 2 years
- NumDealsPurchases: Number of purchases made with a discount
- AcceptedCmp1: 1 if customer accepted the offer in the 1st campaign, 0 otherwise
- AcceptedCmp2: 1 if customer accepted the offer in the 2nd campaign, 0 otherwise
- AcceptedCmp3: 1 if customer accepted the offer in the 3rd campaign, 0 otherwise
- AcceptedCmp4: 1 if customer accepted the offer in the 4th campaign, 0 otherwise
- AcceptedCmp5: 1 if customer accepted the offer in the 5th campaign, 0 otherwise
- Response: 1 if customer accepted the offer in the last campaign, 0 otherwise
- NumWebPurchases: Number of purchases made through the company’s website
- NumCatalogPurchases: Number of purchases made using a catalogue
- NumStorePurchases: Number of purchases made directly in stores
- NumWebVisitsMonth: Number of visits to company’s website in the last month


# **2. Import Library**

Pada tahap ini,kita perlu mengimpor beberapa pustaka (library) Python yang dibutuhkan untuk analisis data dan pembangunan model machine learning.

    Requirement already satisfied: yellowbrick in c:\users\asus\anaconda3\lib\site-packages (1.5)
    Requirement already satisfied: matplotlib!=3.0.0,>=2.0.2 in c:\users\asus\anaconda3\lib\site-packages (from yellowbrick) (3.7.1)
    Requirement already satisfied: scipy>=1.0.0 in c:\users\asus\anaconda3\lib\site-packages (from yellowbrick) (1.10.1)
    Requirement already satisfied: scikit-learn>=1.0.0 in c:\users\asus\anaconda3\lib\site-packages (from yellowbrick) (1.5.2)
    Requirement already satisfied: numpy>=1.16.0 in c:\users\asus\anaconda3\lib\site-packages (from yellowbrick) (1.26.4)
    Requirement already satisfied: cycler>=0.10.0 in c:\users\asus\anaconda3\lib\site-packages (from yellowbrick) (0.11.0)
    Requirement already satisfied: contourpy>=1.0.1 in c:\users\asus\anaconda3\lib\site-packages (from matplotlib!=3.0.0,>=2.0.2->yellowbrick) (1.0.5)
    Requirement already satisfied: fonttools>=4.22.0 in c:\users\asus\anaconda3\lib\site-packages (from matplotlib!=3.0.0,>=2.0.2->yellowbrick) (4.25.0)
    Requirement already satisfied: kiwisolver>=1.0.1 in c:\users\asus\anaconda3\lib\site-packages (from matplotlib!=3.0.0,>=2.0.2->yellowbrick) (1.4.4)
    Requirement already satisfied: packaging>=20.0 in c:\users\asus\anaconda3\lib\site-packages (from matplotlib!=3.0.0,>=2.0.2->yellowbrick) (23.0)
    Requirement already satisfied: pillow>=6.2.0 in c:\users\asus\anaconda3\lib\site-packages (from matplotlib!=3.0.0,>=2.0.2->yellowbrick) (9.4.0)
    Requirement already satisfied: pyparsing>=2.3.1 in c:\users\asus\anaconda3\lib\site-packages (from matplotlib!=3.0.0,>=2.0.2->yellowbrick) (3.0.9)
    Requirement already satisfied: python-dateutil>=2.7 in c:\users\asus\anaconda3\lib\site-packages (from matplotlib!=3.0.0,>=2.0.2->yellowbrick) (2.8.2)
    Requirement already satisfied: joblib>=1.2.0 in c:\users\asus\anaconda3\lib\site-packages (from scikit-learn>=1.0.0->yellowbrick) (1.2.0)
    Requirement already satisfied: threadpoolctl>=3.1.0 in c:\users\asus\anaconda3\lib\site-packages (from scikit-learn>=1.0.0->yellowbrick) (3.5.0)
    Requirement already satisfied: six>=1.5 in c:\users\asus\anaconda3\lib\site-packages (from python-dateutil>=2.7->matplotlib!=3.0.0,>=2.0.2->yellowbrick) (1.16.0)


# **3. Memuat Dataset**

Pada tahap ini, Anda perlu memuat dataset ke dalam notebook. Jika dataset dalam format CSV, Anda bisa menggunakan pustaka pandas untuk membacanya. Pastikan untuk mengecek beberapa baris awal dataset untuk memahami strukturnya dan memastikan data telah dimuat dengan benar.

Jika dataset berada di Google Drive, pastikan Anda menghubungkan Google Drive ke Colab terlebih dahulu. Setelah dataset berhasil dimuat, langkah berikutnya adalah memeriksa kesesuaian data dan siap untuk dianalisis lebih lanjut.




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Year_Birth</th>
      <th>Education</th>
      <th>Marital_Status</th>
      <th>Income</th>
      <th>Kidhome</th>
      <th>Teenhome</th>
      <th>Dt_Customer</th>
      <th>Recency</th>
      <th>MntWines</th>
      <th>...</th>
      <th>NumWebVisitsMonth</th>
      <th>AcceptedCmp3</th>
      <th>AcceptedCmp4</th>
      <th>AcceptedCmp5</th>
      <th>AcceptedCmp1</th>
      <th>AcceptedCmp2</th>
      <th>Complain</th>
      <th>Z_CostContact</th>
      <th>Z_Revenue</th>
      <th>Response</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5524</td>
      <td>1957</td>
      <td>Graduation</td>
      <td>Single</td>
      <td>58138.0</td>
      <td>0</td>
      <td>0</td>
      <td>04-09-2012</td>
      <td>58</td>
      <td>635</td>
      <td>...</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>11</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2174</td>
      <td>1954</td>
      <td>Graduation</td>
      <td>Single</td>
      <td>46344.0</td>
      <td>1</td>
      <td>1</td>
      <td>08-03-2014</td>
      <td>38</td>
      <td>11</td>
      <td>...</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>11</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4141</td>
      <td>1965</td>
      <td>Graduation</td>
      <td>Together</td>
      <td>71613.0</td>
      <td>0</td>
      <td>0</td>
      <td>21-08-2013</td>
      <td>26</td>
      <td>426</td>
      <td>...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>11</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6182</td>
      <td>1984</td>
      <td>Graduation</td>
      <td>Together</td>
      <td>26646.0</td>
      <td>1</td>
      <td>0</td>
      <td>10-02-2014</td>
      <td>26</td>
      <td>11</td>
      <td>...</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>11</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5324</td>
      <td>1981</td>
      <td>PhD</td>
      <td>Married</td>
      <td>58293.0</td>
      <td>1</td>
      <td>0</td>
      <td>19-01-2014</td>
      <td>94</td>
      <td>173</td>
      <td>...</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>11</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 29 columns</p>
</div>



# **4. Exploratory Data Analysis (EDA)**
Pada langkah ini, kita akan melakukan eksplorasi data guna memahami lebih dalam karakteristik dan struktur dari dataset yang telah diimpor.

## 1. Memahami Struktur Data

   - Tinjau jumlah baris dan kolom dalam dataset.  
   - Tinjau jenis data di setiap kolom (numerikal atau kategorikal).

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2240 entries, 0 to 2239
    Data columns (total 29 columns):
     #   Column               Non-Null Count  Dtype  
    ---  ------               --------------  -----  
     0   ID                   2240 non-null   int64  
     1   Year_Birth           2240 non-null   int64  
     2   Education            2240 non-null   object 
     3   Marital_Status       2240 non-null   object 
     4   Income               2216 non-null   float64
     5   Kidhome              2240 non-null   int64  
     6   Teenhome             2240 non-null   int64  
     7   Dt_Customer          2240 non-null   object 
     8   Recency              2240 non-null   int64  
     9   MntWines             2240 non-null   int64  
     10  MntFruits            2240 non-null   int64  
     11  MntMeatProducts      2240 non-null   int64  
     12  MntFishProducts      2240 non-null   int64  
     13  MntSweetProducts     2240 non-null   int64  
     14  MntGoldProds         2240 non-null   int64  
     15  NumDealsPurchases    2240 non-null   int64  
     16  NumWebPurchases      2240 non-null   int64  
     17  NumCatalogPurchases  2240 non-null   int64  
     18  NumStorePurchases    2240 non-null   int64  
     19  NumWebVisitsMonth    2240 non-null   int64  
     20  AcceptedCmp3         2240 non-null   int64  
     21  AcceptedCmp4         2240 non-null   int64  
     22  AcceptedCmp5         2240 non-null   int64  
     23  AcceptedCmp1         2240 non-null   int64  
     24  AcceptedCmp2         2240 non-null   int64  
     25  Complain             2240 non-null   int64  
     26  Z_CostContact        2240 non-null   int64  
     27  Z_Revenue            2240 non-null   int64  
     28  Response             2240 non-null   int64  
    dtypes: float64(1), int64(25), object(3)
    memory usage: 507.6+ KB





    ID                      0
    Year_Birth              0
    Education               0
    Marital_Status          0
    Income                 24
    Kidhome                 0
    Teenhome                0
    Dt_Customer             0
    Recency                 0
    MntWines                0
    MntFruits               0
    MntMeatProducts         0
    MntFishProducts         0
    MntSweetProducts        0
    MntGoldProds            0
    NumDealsPurchases       0
    NumWebPurchases         0
    NumCatalogPurchases     0
    NumStorePurchases       0
    NumWebVisitsMonth       0
    AcceptedCmp3            0
    AcceptedCmp4            0
    AcceptedCmp5            0
    AcceptedCmp1            0
    AcceptedCmp2            0
    Complain                0
    Z_CostContact           0
    Z_Revenue               0
    Response                0
    dtype: int64






<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Year_Birth</th>
      <th>Income</th>
      <th>Kidhome</th>
      <th>Teenhome</th>
      <th>Recency</th>
      <th>MntWines</th>
      <th>MntFruits</th>
      <th>MntMeatProducts</th>
      <th>MntFishProducts</th>
      <th>...</th>
      <th>NumWebVisitsMonth</th>
      <th>AcceptedCmp3</th>
      <th>AcceptedCmp4</th>
      <th>AcceptedCmp5</th>
      <th>AcceptedCmp1</th>
      <th>AcceptedCmp2</th>
      <th>Complain</th>
      <th>Z_CostContact</th>
      <th>Z_Revenue</th>
      <th>Response</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2240.000000</td>
      <td>2240.000000</td>
      <td>2216.000000</td>
      <td>2240.000000</td>
      <td>2240.000000</td>
      <td>2240.000000</td>
      <td>2240.000000</td>
      <td>2240.000000</td>
      <td>2240.000000</td>
      <td>2240.000000</td>
      <td>...</td>
      <td>2240.000000</td>
      <td>2240.000000</td>
      <td>2240.000000</td>
      <td>2240.000000</td>
      <td>2240.000000</td>
      <td>2240.000000</td>
      <td>2240.000000</td>
      <td>2240.0</td>
      <td>2240.0</td>
      <td>2240.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5592.159821</td>
      <td>1968.805804</td>
      <td>52247.251354</td>
      <td>0.444196</td>
      <td>0.506250</td>
      <td>49.109375</td>
      <td>303.935714</td>
      <td>26.302232</td>
      <td>166.950000</td>
      <td>37.525446</td>
      <td>...</td>
      <td>5.316518</td>
      <td>0.072768</td>
      <td>0.074554</td>
      <td>0.072768</td>
      <td>0.064286</td>
      <td>0.013393</td>
      <td>0.009375</td>
      <td>3.0</td>
      <td>11.0</td>
      <td>0.149107</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3246.662198</td>
      <td>11.984069</td>
      <td>25173.076661</td>
      <td>0.538398</td>
      <td>0.544538</td>
      <td>28.962453</td>
      <td>336.597393</td>
      <td>39.773434</td>
      <td>225.715373</td>
      <td>54.628979</td>
      <td>...</td>
      <td>2.426645</td>
      <td>0.259813</td>
      <td>0.262728</td>
      <td>0.259813</td>
      <td>0.245316</td>
      <td>0.114976</td>
      <td>0.096391</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.356274</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>1893.000000</td>
      <td>1730.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.0</td>
      <td>11.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2828.250000</td>
      <td>1959.000000</td>
      <td>35303.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>24.000000</td>
      <td>23.750000</td>
      <td>1.000000</td>
      <td>16.000000</td>
      <td>3.000000</td>
      <td>...</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.0</td>
      <td>11.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>5458.500000</td>
      <td>1970.000000</td>
      <td>51381.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>49.000000</td>
      <td>173.500000</td>
      <td>8.000000</td>
      <td>67.000000</td>
      <td>12.000000</td>
      <td>...</td>
      <td>6.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.0</td>
      <td>11.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>8427.750000</td>
      <td>1977.000000</td>
      <td>68522.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>74.000000</td>
      <td>504.250000</td>
      <td>33.000000</td>
      <td>232.000000</td>
      <td>50.000000</td>
      <td>...</td>
      <td>7.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.0</td>
      <td>11.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>11191.000000</td>
      <td>1996.000000</td>
      <td>666666.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>99.000000</td>
      <td>1493.000000</td>
      <td>199.000000</td>
      <td>1725.000000</td>
      <td>259.000000</td>
      <td>...</td>
      <td>20.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>3.0</td>
      <td>11.0</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 26 columns</p>
</div>



    Jumlah baris duplikat:  0


## 2. Menangani Data yang Hilang

    <class 'pandas.core.frame.DataFrame'>
    Index: 2216 entries, 0 to 2239
    Data columns (total 29 columns):
     #   Column               Non-Null Count  Dtype  
    ---  ------               --------------  -----  
     0   ID                   2216 non-null   int64  
     1   Year_Birth           2216 non-null   int64  
     2   Education            2216 non-null   object 
     3   Marital_Status       2216 non-null   object 
     4   Income               2216 non-null   float64
     5   Kidhome              2216 non-null   int64  
     6   Teenhome             2216 non-null   int64  
     7   Dt_Customer          2216 non-null   object 
     8   Recency              2216 non-null   int64  
     9   MntWines             2216 non-null   int64  
     10  MntFruits            2216 non-null   int64  
     11  MntMeatProducts      2216 non-null   int64  
     12  MntFishProducts      2216 non-null   int64  
     13  MntSweetProducts     2216 non-null   int64  
     14  MntGoldProds         2216 non-null   int64  
     15  NumDealsPurchases    2216 non-null   int64  
     16  NumWebPurchases      2216 non-null   int64  
     17  NumCatalogPurchases  2216 non-null   int64  
     18  NumStorePurchases    2216 non-null   int64  
     19  NumWebVisitsMonth    2216 non-null   int64  
     20  AcceptedCmp3         2216 non-null   int64  
     21  AcceptedCmp4         2216 non-null   int64  
     22  AcceptedCmp5         2216 non-null   int64  
     23  AcceptedCmp1         2216 non-null   int64  
     24  AcceptedCmp2         2216 non-null   int64  
     25  Complain             2216 non-null   int64  
     26  Z_CostContact        2216 non-null   int64  
     27  Z_Revenue            2216 non-null   int64  
     28  Response             2216 non-null   int64  
    dtypes: float64(1), int64(25), object(3)
    memory usage: 519.4+ KB





    Index(['ID', 'Year_Birth', 'Education', 'Marital_Status', 'Income', 'Kidhome',
           'Teenhome', 'Dt_Customer', 'Recency', 'MntWines', 'MntFruits',
           'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
           'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
           'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth',
           'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1',
           'AcceptedCmp2', 'Complain', 'Z_CostContact', 'Z_Revenue', 'Response'],
          dtype='object')



### 1. Menghapus kolom ID karena bukan merupakan fitur




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year_Birth</th>
      <th>Education</th>
      <th>Marital_Status</th>
      <th>Income</th>
      <th>Kidhome</th>
      <th>Teenhome</th>
      <th>Dt_Customer</th>
      <th>Recency</th>
      <th>MntWines</th>
      <th>MntFruits</th>
      <th>...</th>
      <th>NumWebVisitsMonth</th>
      <th>AcceptedCmp3</th>
      <th>AcceptedCmp4</th>
      <th>AcceptedCmp5</th>
      <th>AcceptedCmp1</th>
      <th>AcceptedCmp2</th>
      <th>Complain</th>
      <th>Z_CostContact</th>
      <th>Z_Revenue</th>
      <th>Response</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1957</td>
      <td>Graduation</td>
      <td>Single</td>
      <td>58138.0</td>
      <td>0</td>
      <td>0</td>
      <td>04-09-2012</td>
      <td>58</td>
      <td>635</td>
      <td>88</td>
      <td>...</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>11</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1954</td>
      <td>Graduation</td>
      <td>Single</td>
      <td>46344.0</td>
      <td>1</td>
      <td>1</td>
      <td>08-03-2014</td>
      <td>38</td>
      <td>11</td>
      <td>1</td>
      <td>...</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>11</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1965</td>
      <td>Graduation</td>
      <td>Together</td>
      <td>71613.0</td>
      <td>0</td>
      <td>0</td>
      <td>21-08-2013</td>
      <td>26</td>
      <td>426</td>
      <td>49</td>
      <td>...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>11</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1984</td>
      <td>Graduation</td>
      <td>Together</td>
      <td>26646.0</td>
      <td>1</td>
      <td>0</td>
      <td>10-02-2014</td>
      <td>26</td>
      <td>11</td>
      <td>4</td>
      <td>...</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>11</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1981</td>
      <td>PhD</td>
      <td>Married</td>
      <td>58293.0</td>
      <td>1</td>
      <td>0</td>
      <td>19-01-2014</td>
      <td>94</td>
      <td>173</td>
      <td>43</td>
      <td>...</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>11</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 28 columns</p>
</div>



### 2.  Mendapatkan data usia pelanggan

### 3. Membuat kolom baru 'Spending' untuk melihat total uang yang dikeluarkan pelanggan di semua produk

### 4. Membuat kolom baru 'Accept_Offer' untuk melihat frekuensi total pelanggan menerima tawaran

### 5. Membuat kolom baru 'Num_Purchase' untuk melihat total frekuensi belanja

### 6. Mengganti nama kolom NumWebVisitsMonth

### 7. Menyeragamkan value untuk Education




    array(['Graduation', 'PhD', 'Master', 'Basic', '2n Cycle'], dtype=object)






    array(['Bachelor', 'PhD', 'Master', 'Basic'], dtype=object)



### 8. Menyeragamkan value untuk Marital_Status




    array(['Single', 'Together', 'Married', 'Divorced', 'Widow', 'Alone',
           'Absurd', 'YOLO'], dtype=object)






    array(['Single', 'Relationship', 'Married', 'Divorced', 'Widow'],
          dtype=object)



### 9. Membuat kolom baru untuk melihat pelanggan sudah berapa lama berlangganan

    <class 'pandas.core.frame.DataFrame'>
    Index: 2216 entries, 0 to 2239
    Data columns (total 31 columns):
     #   Column               Non-Null Count  Dtype         
    ---  ------               --------------  -----         
     0   Education            2216 non-null   object        
     1   Marital_Status       2216 non-null   object        
     2   Income               2216 non-null   float64       
     3   Kidhome              2216 non-null   int64         
     4   Teenhome             2216 non-null   int64         
     5   Dt_Customer          2216 non-null   datetime64[ns]
     6   Recency              2216 non-null   int64         
     7   MntWines             2216 non-null   int64         
     8   MntFruits            2216 non-null   int64         
     9   MntMeatProducts      2216 non-null   int64         
     10  MntFishProducts      2216 non-null   int64         
     11  MntSweetProducts     2216 non-null   int64         
     12  MntGoldProds         2216 non-null   int64         
     13  NumDealsPurchases    2216 non-null   int64         
     14  NumWebPurchases      2216 non-null   int64         
     15  NumCatalogPurchases  2216 non-null   int64         
     16  NumStorePurchases    2216 non-null   int64         
     17  Web_Visit            2216 non-null   int64         
     18  AcceptedCmp3         2216 non-null   int64         
     19  AcceptedCmp4         2216 non-null   int64         
     20  AcceptedCmp5         2216 non-null   int64         
     21  AcceptedCmp1         2216 non-null   int64         
     22  AcceptedCmp2         2216 non-null   int64         
     23  Complain             2216 non-null   int64         
     24  Z_CostContact        2216 non-null   int64         
     25  Z_Revenue            2216 non-null   int64         
     26  Response             2216 non-null   int64         
     27  Age                  2216 non-null   int64         
     28  Spending             2216 non-null   int64         
     29  Accept_Offer         2216 non-null   int64         
     30  Num_Purchase         2216 non-null   int64         
    dtypes: datetime64[ns](1), float64(1), int64(27), object(2)
    memory usage: 554.0+ KB





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Education</th>
      <th>Marital_Status</th>
      <th>Income</th>
      <th>Kidhome</th>
      <th>Teenhome</th>
      <th>Recency</th>
      <th>MntWines</th>
      <th>MntFruits</th>
      <th>MntMeatProducts</th>
      <th>MntFishProducts</th>
      <th>...</th>
      <th>AcceptedCmp2</th>
      <th>Complain</th>
      <th>Z_CostContact</th>
      <th>Z_Revenue</th>
      <th>Response</th>
      <th>Age</th>
      <th>Spending</th>
      <th>Accept_Offer</th>
      <th>Num_Purchase</th>
      <th>Enrolled_Days</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bachelor</td>
      <td>Single</td>
      <td>58138.0</td>
      <td>0</td>
      <td>0</td>
      <td>58</td>
      <td>635</td>
      <td>88</td>
      <td>546</td>
      <td>172</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>11</td>
      <td>1</td>
      <td>67</td>
      <td>1617</td>
      <td>1</td>
      <td>25</td>
      <td>663</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bachelor</td>
      <td>Single</td>
      <td>46344.0</td>
      <td>1</td>
      <td>1</td>
      <td>38</td>
      <td>11</td>
      <td>1</td>
      <td>6</td>
      <td>2</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>11</td>
      <td>0</td>
      <td>70</td>
      <td>27</td>
      <td>0</td>
      <td>6</td>
      <td>113</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Bachelor</td>
      <td>Relationship</td>
      <td>71613.0</td>
      <td>0</td>
      <td>0</td>
      <td>26</td>
      <td>426</td>
      <td>49</td>
      <td>127</td>
      <td>111</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>11</td>
      <td>0</td>
      <td>59</td>
      <td>776</td>
      <td>0</td>
      <td>21</td>
      <td>312</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Bachelor</td>
      <td>Relationship</td>
      <td>26646.0</td>
      <td>1</td>
      <td>0</td>
      <td>26</td>
      <td>11</td>
      <td>4</td>
      <td>20</td>
      <td>10</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>11</td>
      <td>0</td>
      <td>40</td>
      <td>53</td>
      <td>0</td>
      <td>8</td>
      <td>139</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PhD</td>
      <td>Married</td>
      <td>58293.0</td>
      <td>1</td>
      <td>0</td>
      <td>94</td>
      <td>173</td>
      <td>43</td>
      <td>118</td>
      <td>46</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>11</td>
      <td>0</td>
      <td>43</td>
      <td>422</td>
      <td>0</td>
      <td>19</td>
      <td>161</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 31 columns</p>
</div>



### 10. Membuat kolom baru untuk melihat total anak dari pelanggan




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Education</th>
      <th>Marital_Status</th>
      <th>Income</th>
      <th>Recency</th>
      <th>MntWines</th>
      <th>MntFruits</th>
      <th>MntMeatProducts</th>
      <th>MntFishProducts</th>
      <th>MntSweetProducts</th>
      <th>MntGoldProds</th>
      <th>...</th>
      <th>Complain</th>
      <th>Z_CostContact</th>
      <th>Z_Revenue</th>
      <th>Response</th>
      <th>Age</th>
      <th>Spending</th>
      <th>Accept_Offer</th>
      <th>Num_Purchase</th>
      <th>Enrolled_Days</th>
      <th>Children</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bachelor</td>
      <td>Single</td>
      <td>58138.0</td>
      <td>58</td>
      <td>635</td>
      <td>88</td>
      <td>546</td>
      <td>172</td>
      <td>88</td>
      <td>88</td>
      <td>...</td>
      <td>0</td>
      <td>3</td>
      <td>11</td>
      <td>1</td>
      <td>67</td>
      <td>1617</td>
      <td>1</td>
      <td>25</td>
      <td>663</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bachelor</td>
      <td>Single</td>
      <td>46344.0</td>
      <td>38</td>
      <td>11</td>
      <td>1</td>
      <td>6</td>
      <td>2</td>
      <td>1</td>
      <td>6</td>
      <td>...</td>
      <td>0</td>
      <td>3</td>
      <td>11</td>
      <td>0</td>
      <td>70</td>
      <td>27</td>
      <td>0</td>
      <td>6</td>
      <td>113</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Bachelor</td>
      <td>Relationship</td>
      <td>71613.0</td>
      <td>26</td>
      <td>426</td>
      <td>49</td>
      <td>127</td>
      <td>111</td>
      <td>21</td>
      <td>42</td>
      <td>...</td>
      <td>0</td>
      <td>3</td>
      <td>11</td>
      <td>0</td>
      <td>59</td>
      <td>776</td>
      <td>0</td>
      <td>21</td>
      <td>312</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Bachelor</td>
      <td>Relationship</td>
      <td>26646.0</td>
      <td>26</td>
      <td>11</td>
      <td>4</td>
      <td>20</td>
      <td>10</td>
      <td>3</td>
      <td>5</td>
      <td>...</td>
      <td>0</td>
      <td>3</td>
      <td>11</td>
      <td>0</td>
      <td>40</td>
      <td>53</td>
      <td>0</td>
      <td>8</td>
      <td>139</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PhD</td>
      <td>Married</td>
      <td>58293.0</td>
      <td>94</td>
      <td>173</td>
      <td>43</td>
      <td>118</td>
      <td>46</td>
      <td>27</td>
      <td>15</td>
      <td>...</td>
      <td>0</td>
      <td>3</td>
      <td>11</td>
      <td>0</td>
      <td>43</td>
      <td>422</td>
      <td>0</td>
      <td>19</td>
      <td>161</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 30 columns</p>
</div>



### 11. Drop baris yang memiliki data Income dan umur tidak wajar 




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Income</th>
      <th>Recency</th>
      <th>MntWines</th>
      <th>MntFruits</th>
      <th>MntMeatProducts</th>
      <th>MntFishProducts</th>
      <th>MntSweetProducts</th>
      <th>MntGoldProds</th>
      <th>NumDealsPurchases</th>
      <th>NumWebPurchases</th>
      <th>...</th>
      <th>Complain</th>
      <th>Z_CostContact</th>
      <th>Z_Revenue</th>
      <th>Response</th>
      <th>Age</th>
      <th>Spending</th>
      <th>Accept_Offer</th>
      <th>Num_Purchase</th>
      <th>Enrolled_Days</th>
      <th>Children</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2216.000000</td>
      <td>2216.000000</td>
      <td>2216.000000</td>
      <td>2216.000000</td>
      <td>2216.000000</td>
      <td>2216.000000</td>
      <td>2216.000000</td>
      <td>2216.000000</td>
      <td>2216.000000</td>
      <td>2216.000000</td>
      <td>...</td>
      <td>2216.000000</td>
      <td>2216.0</td>
      <td>2216.0</td>
      <td>2216.000000</td>
      <td>2216.000000</td>
      <td>2216.000000</td>
      <td>2216.000000</td>
      <td>2216.000000</td>
      <td>2216.000000</td>
      <td>2216.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>52247.251354</td>
      <td>49.012635</td>
      <td>305.091606</td>
      <td>26.356047</td>
      <td>166.995939</td>
      <td>37.637635</td>
      <td>27.028881</td>
      <td>43.965253</td>
      <td>2.323556</td>
      <td>4.085289</td>
      <td>...</td>
      <td>0.009477</td>
      <td>3.0</td>
      <td>11.0</td>
      <td>0.150271</td>
      <td>55.179603</td>
      <td>607.075361</td>
      <td>0.448556</td>
      <td>14.880866</td>
      <td>353.521209</td>
      <td>0.947202</td>
    </tr>
    <tr>
      <th>std</th>
      <td>25173.076661</td>
      <td>28.948352</td>
      <td>337.327920</td>
      <td>39.793917</td>
      <td>224.283273</td>
      <td>54.752082</td>
      <td>41.072046</td>
      <td>51.815414</td>
      <td>1.923716</td>
      <td>2.740951</td>
      <td>...</td>
      <td>0.096907</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.357417</td>
      <td>11.985554</td>
      <td>602.900476</td>
      <td>0.892440</td>
      <td>7.670957</td>
      <td>202.434667</td>
      <td>0.749062</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1730.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>3.0</td>
      <td>11.0</td>
      <td>0.000000</td>
      <td>28.000000</td>
      <td>5.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>35303.000000</td>
      <td>24.000000</td>
      <td>24.000000</td>
      <td>2.000000</td>
      <td>16.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>9.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>3.0</td>
      <td>11.0</td>
      <td>0.000000</td>
      <td>47.000000</td>
      <td>69.000000</td>
      <td>0.000000</td>
      <td>8.000000</td>
      <td>180.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>51381.500000</td>
      <td>49.000000</td>
      <td>174.500000</td>
      <td>8.000000</td>
      <td>68.000000</td>
      <td>12.000000</td>
      <td>8.000000</td>
      <td>24.500000</td>
      <td>2.000000</td>
      <td>4.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>3.0</td>
      <td>11.0</td>
      <td>0.000000</td>
      <td>54.000000</td>
      <td>396.500000</td>
      <td>0.000000</td>
      <td>15.000000</td>
      <td>355.500000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>68522.000000</td>
      <td>74.000000</td>
      <td>505.000000</td>
      <td>33.000000</td>
      <td>232.250000</td>
      <td>50.000000</td>
      <td>33.000000</td>
      <td>56.000000</td>
      <td>3.000000</td>
      <td>6.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>3.0</td>
      <td>11.0</td>
      <td>0.000000</td>
      <td>65.000000</td>
      <td>1048.000000</td>
      <td>1.000000</td>
      <td>21.000000</td>
      <td>529.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>666666.000000</td>
      <td>99.000000</td>
      <td>1493.000000</td>
      <td>199.000000</td>
      <td>1725.000000</td>
      <td>259.000000</td>
      <td>262.000000</td>
      <td>321.000000</td>
      <td>15.000000</td>
      <td>27.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>3.0</td>
      <td>11.0</td>
      <td>1.000000</td>
      <td>131.000000</td>
      <td>2525.000000</td>
      <td>5.000000</td>
      <td>44.000000</td>
      <td>699.000000</td>
      <td>3.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 28 columns</p>
</div>




    
![png](output_43_0.png)
    



    
![png](output_43_1.png)
    



    
![png](output_43_2.png)
    



    
![png](output_43_3.png)
    



    
![png](output_43_4.png)
    



    
![png](output_43_5.png)
    



    
![png](output_43_6.png)
    



    
![png](output_43_7.png)
    



    
![png](output_43_8.png)
    



    
![png](output_43_9.png)
    



    
![png](output_43_10.png)
    



    
![png](output_43_11.png)
    



    
![png](output_43_12.png)
    



    
![png](output_43_13.png)
    



    
![png](output_43_14.png)
    



    
![png](output_43_15.png)
    



    
![png](output_43_16.png)
    



    
![png](output_43_17.png)
    



    
![png](output_43_18.png)
    



    
![png](output_43_19.png)
    



    
![png](output_43_20.png)
    



    
![png](output_43_21.png)
    



    
![png](output_43_22.png)
    



    
![png](output_43_23.png)
    



    
![png](output_43_24.png)
    



    
![png](output_43_25.png)
    



    
![png](output_43_26.png)
    



    
![png](output_43_27.png)
    





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Income</th>
      <th>Recency</th>
      <th>MntWines</th>
      <th>MntFruits</th>
      <th>MntMeatProducts</th>
      <th>MntFishProducts</th>
      <th>MntSweetProducts</th>
      <th>MntGoldProds</th>
      <th>NumDealsPurchases</th>
      <th>NumWebPurchases</th>
      <th>...</th>
      <th>Complain</th>
      <th>Z_CostContact</th>
      <th>Z_Revenue</th>
      <th>Response</th>
      <th>Age</th>
      <th>Spending</th>
      <th>Accept_Offer</th>
      <th>Num_Purchase</th>
      <th>Enrolled_Days</th>
      <th>Children</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2205.000000</td>
      <td>2205.000000</td>
      <td>2205.000000</td>
      <td>2205.000000</td>
      <td>2205.000000</td>
      <td>2205.000000</td>
      <td>2205.000000</td>
      <td>2205.000000</td>
      <td>2205.000000</td>
      <td>2205.000000</td>
      <td>...</td>
      <td>2205.000000</td>
      <td>2205.0</td>
      <td>2205.0</td>
      <td>2205.00000</td>
      <td>2205.000000</td>
      <td>2205.000000</td>
      <td>2205.000000</td>
      <td>2205.000000</td>
      <td>2205.000000</td>
      <td>2205.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>51622.094785</td>
      <td>49.009070</td>
      <td>306.164626</td>
      <td>26.403175</td>
      <td>165.312018</td>
      <td>37.756463</td>
      <td>27.128345</td>
      <td>44.057143</td>
      <td>2.318367</td>
      <td>4.100680</td>
      <td>...</td>
      <td>0.009070</td>
      <td>3.0</td>
      <td>11.0</td>
      <td>0.15102</td>
      <td>55.095692</td>
      <td>606.821769</td>
      <td>0.450340</td>
      <td>14.887982</td>
      <td>353.718367</td>
      <td>0.948753</td>
    </tr>
    <tr>
      <th>std</th>
      <td>20713.063826</td>
      <td>28.932111</td>
      <td>337.493839</td>
      <td>39.784484</td>
      <td>217.784507</td>
      <td>54.824635</td>
      <td>41.130468</td>
      <td>51.736211</td>
      <td>1.886107</td>
      <td>2.737424</td>
      <td>...</td>
      <td>0.094827</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.35815</td>
      <td>11.705801</td>
      <td>601.675284</td>
      <td>0.894075</td>
      <td>7.615277</td>
      <td>202.563647</td>
      <td>0.749231</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1730.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>3.0</td>
      <td>11.0</td>
      <td>0.00000</td>
      <td>28.000000</td>
      <td>5.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>35196.000000</td>
      <td>24.000000</td>
      <td>24.000000</td>
      <td>2.000000</td>
      <td>16.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>9.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>3.0</td>
      <td>11.0</td>
      <td>0.00000</td>
      <td>47.000000</td>
      <td>69.000000</td>
      <td>0.000000</td>
      <td>8.000000</td>
      <td>180.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>51287.000000</td>
      <td>49.000000</td>
      <td>178.000000</td>
      <td>8.000000</td>
      <td>68.000000</td>
      <td>12.000000</td>
      <td>8.000000</td>
      <td>25.000000</td>
      <td>2.000000</td>
      <td>4.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>3.0</td>
      <td>11.0</td>
      <td>0.00000</td>
      <td>54.000000</td>
      <td>397.000000</td>
      <td>0.000000</td>
      <td>15.000000</td>
      <td>356.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>68281.000000</td>
      <td>74.000000</td>
      <td>507.000000</td>
      <td>33.000000</td>
      <td>232.000000</td>
      <td>50.000000</td>
      <td>34.000000</td>
      <td>56.000000</td>
      <td>3.000000</td>
      <td>6.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>3.0</td>
      <td>11.0</td>
      <td>0.00000</td>
      <td>65.000000</td>
      <td>1047.000000</td>
      <td>1.000000</td>
      <td>21.000000</td>
      <td>529.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>113734.000000</td>
      <td>99.000000</td>
      <td>1493.000000</td>
      <td>199.000000</td>
      <td>1725.000000</td>
      <td>259.000000</td>
      <td>262.000000</td>
      <td>321.000000</td>
      <td>15.000000</td>
      <td>27.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>3.0</td>
      <td>11.0</td>
      <td>1.00000</td>
      <td>84.000000</td>
      <td>2525.000000</td>
      <td>5.000000</td>
      <td>43.000000</td>
      <td>699.000000</td>
      <td>3.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 28 columns</p>
</div>



### 12. Dataset final setelah cleaning dan preprocessing




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Education</th>
      <th>Marital_Status</th>
      <th>Income</th>
      <th>Recency</th>
      <th>MntWines</th>
      <th>MntFruits</th>
      <th>MntMeatProducts</th>
      <th>MntFishProducts</th>
      <th>MntSweetProducts</th>
      <th>MntGoldProds</th>
      <th>...</th>
      <th>Complain</th>
      <th>Z_CostContact</th>
      <th>Z_Revenue</th>
      <th>Response</th>
      <th>Age</th>
      <th>Spending</th>
      <th>Accept_Offer</th>
      <th>Num_Purchase</th>
      <th>Enrolled_Days</th>
      <th>Children</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bachelor</td>
      <td>Single</td>
      <td>58138.0</td>
      <td>58</td>
      <td>635</td>
      <td>88</td>
      <td>546</td>
      <td>172</td>
      <td>88</td>
      <td>88</td>
      <td>...</td>
      <td>0</td>
      <td>3</td>
      <td>11</td>
      <td>1</td>
      <td>67</td>
      <td>1617</td>
      <td>1</td>
      <td>25</td>
      <td>663</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bachelor</td>
      <td>Single</td>
      <td>46344.0</td>
      <td>38</td>
      <td>11</td>
      <td>1</td>
      <td>6</td>
      <td>2</td>
      <td>1</td>
      <td>6</td>
      <td>...</td>
      <td>0</td>
      <td>3</td>
      <td>11</td>
      <td>0</td>
      <td>70</td>
      <td>27</td>
      <td>0</td>
      <td>6</td>
      <td>113</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Bachelor</td>
      <td>Relationship</td>
      <td>71613.0</td>
      <td>26</td>
      <td>426</td>
      <td>49</td>
      <td>127</td>
      <td>111</td>
      <td>21</td>
      <td>42</td>
      <td>...</td>
      <td>0</td>
      <td>3</td>
      <td>11</td>
      <td>0</td>
      <td>59</td>
      <td>776</td>
      <td>0</td>
      <td>21</td>
      <td>312</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Bachelor</td>
      <td>Relationship</td>
      <td>26646.0</td>
      <td>26</td>
      <td>11</td>
      <td>4</td>
      <td>20</td>
      <td>10</td>
      <td>3</td>
      <td>5</td>
      <td>...</td>
      <td>0</td>
      <td>3</td>
      <td>11</td>
      <td>0</td>
      <td>40</td>
      <td>53</td>
      <td>0</td>
      <td>8</td>
      <td>139</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PhD</td>
      <td>Married</td>
      <td>58293.0</td>
      <td>94</td>
      <td>173</td>
      <td>43</td>
      <td>118</td>
      <td>46</td>
      <td>27</td>
      <td>15</td>
      <td>...</td>
      <td>0</td>
      <td>3</td>
      <td>11</td>
      <td>0</td>
      <td>43</td>
      <td>422</td>
      <td>0</td>
      <td>19</td>
      <td>161</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 30 columns</p>
</div>



## 3.Visualisasi Data


    
![png](output_49_0.png)
    





    Wines     675093
    Meat      364513
    Gold       97146
    Fish       83253
    Sweet      59818
    Fruits     58219
    dtype: int64




    
![png](output_51_0.png)
    



    
![png](output_52_0.png)
    



    
![png](output_53_0.png)
    


# **5. Data Preprocessing**
Pada tahap ini, data preprocessing adalah langkah penting untuk memastikan kualitas data sebelum digunakan dalam model machine learning. Data mentah sering kali mengandung nilai kosong, duplikasi, atau rentang nilai yang tidak konsisten, yang dapat memengaruhi kinerja model. Oleh karena itu, proses ini bertujuan untuk membersihkan dan mempersiapkan data agar analisis berjalan optimal.




    Index(['Education', 'Marital_Status', 'Income', 'Recency', 'MntWines',
           'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
           'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
           'NumCatalogPurchases', 'NumStorePurchases', 'Web_Visit', 'AcceptedCmp3',
           'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2',
           'Complain', 'Z_CostContact', 'Z_Revenue', 'Response', 'Age', 'Spending',
           'Accept_Offer', 'Num_Purchase', 'Enrolled_Days', 'Children'],
          dtype='object')






<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Education</th>
      <th>Marital_Status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bachelor</td>
      <td>Single</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bachelor</td>
      <td>Single</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Bachelor</td>
      <td>Relationship</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Bachelor</td>
      <td>Relationship</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PhD</td>
      <td>Married</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2235</th>
      <td>Bachelor</td>
      <td>Married</td>
    </tr>
    <tr>
      <th>2236</th>
      <td>PhD</td>
      <td>Relationship</td>
    </tr>
    <tr>
      <th>2237</th>
      <td>Bachelor</td>
      <td>Divorced</td>
    </tr>
    <tr>
      <th>2238</th>
      <td>Master</td>
      <td>Relationship</td>
    </tr>
    <tr>
      <th>2239</th>
      <td>PhD</td>
      <td>Married</td>
    </tr>
  </tbody>
</table>
<p>2205 rows × 2 columns</p>
</div>






<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Income</th>
      <th>Recency</th>
      <th>MntWines</th>
      <th>MntFruits</th>
      <th>MntMeatProducts</th>
      <th>MntFishProducts</th>
      <th>MntSweetProducts</th>
      <th>MntGoldProds</th>
      <th>NumDealsPurchases</th>
      <th>NumWebPurchases</th>
      <th>...</th>
      <th>Children</th>
      <th>Education_Bachelor</th>
      <th>Education_Basic</th>
      <th>Education_Master</th>
      <th>Education_PhD</th>
      <th>Marital_Status_Divorced</th>
      <th>Marital_Status_Married</th>
      <th>Marital_Status_Relationship</th>
      <th>Marital_Status_Single</th>
      <th>Marital_Status_Widow</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>58138.0</td>
      <td>58</td>
      <td>635</td>
      <td>88</td>
      <td>546</td>
      <td>172</td>
      <td>88</td>
      <td>88</td>
      <td>3</td>
      <td>8</td>
      <td>...</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>46344.0</td>
      <td>38</td>
      <td>11</td>
      <td>1</td>
      <td>6</td>
      <td>2</td>
      <td>1</td>
      <td>6</td>
      <td>2</td>
      <td>1</td>
      <td>...</td>
      <td>2</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>71613.0</td>
      <td>26</td>
      <td>426</td>
      <td>49</td>
      <td>127</td>
      <td>111</td>
      <td>21</td>
      <td>42</td>
      <td>1</td>
      <td>8</td>
      <td>...</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>26646.0</td>
      <td>26</td>
      <td>11</td>
      <td>4</td>
      <td>20</td>
      <td>10</td>
      <td>3</td>
      <td>5</td>
      <td>2</td>
      <td>2</td>
      <td>...</td>
      <td>1</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>58293.0</td>
      <td>94</td>
      <td>173</td>
      <td>43</td>
      <td>118</td>
      <td>46</td>
      <td>27</td>
      <td>15</td>
      <td>5</td>
      <td>5</td>
      <td>...</td>
      <td>1</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2235</th>
      <td>61223.0</td>
      <td>46</td>
      <td>709</td>
      <td>43</td>
      <td>182</td>
      <td>42</td>
      <td>118</td>
      <td>247</td>
      <td>2</td>
      <td>9</td>
      <td>...</td>
      <td>1</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2236</th>
      <td>64014.0</td>
      <td>56</td>
      <td>406</td>
      <td>0</td>
      <td>30</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>7</td>
      <td>8</td>
      <td>...</td>
      <td>3</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2237</th>
      <td>56981.0</td>
      <td>91</td>
      <td>908</td>
      <td>48</td>
      <td>217</td>
      <td>32</td>
      <td>12</td>
      <td>24</td>
      <td>1</td>
      <td>2</td>
      <td>...</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2238</th>
      <td>69245.0</td>
      <td>8</td>
      <td>428</td>
      <td>30</td>
      <td>214</td>
      <td>80</td>
      <td>30</td>
      <td>61</td>
      <td>2</td>
      <td>6</td>
      <td>...</td>
      <td>1</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2239</th>
      <td>52869.0</td>
      <td>40</td>
      <td>84</td>
      <td>3</td>
      <td>61</td>
      <td>2</td>
      <td>1</td>
      <td>21</td>
      <td>3</td>
      <td>3</td>
      <td>...</td>
      <td>2</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>2205 rows × 37 columns</p>
</div>






<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Income</th>
      <th>Recency</th>
      <th>MntWines</th>
      <th>MntFruits</th>
      <th>MntMeatProducts</th>
      <th>MntFishProducts</th>
      <th>MntSweetProducts</th>
      <th>MntGoldProds</th>
      <th>NumDealsPurchases</th>
      <th>NumWebPurchases</th>
      <th>...</th>
      <th>Children</th>
      <th>Education_Bachelor</th>
      <th>Education_Basic</th>
      <th>Education_Master</th>
      <th>Education_PhD</th>
      <th>Marital_Status_Divorced</th>
      <th>Marital_Status_Married</th>
      <th>Marital_Status_Relationship</th>
      <th>Marital_Status_Single</th>
      <th>Marital_Status_Widow</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.314651</td>
      <td>0.310830</td>
      <td>0.974566</td>
      <td>1.548614</td>
      <td>1.748400</td>
      <td>2.449154</td>
      <td>1.480301</td>
      <td>0.849556</td>
      <td>0.361479</td>
      <td>1.424772</td>
      <td>...</td>
      <td>-1.266589</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.254877</td>
      <td>-0.380600</td>
      <td>-0.874776</td>
      <td>-0.638664</td>
      <td>-0.731678</td>
      <td>-0.652345</td>
      <td>-0.635399</td>
      <td>-0.735767</td>
      <td>-0.168834</td>
      <td>-1.132957</td>
      <td>...</td>
      <td>1.403420</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.965354</td>
      <td>-0.795458</td>
      <td>0.355155</td>
      <td>0.568110</td>
      <td>-0.175957</td>
      <td>1.336263</td>
      <td>-0.149031</td>
      <td>-0.039771</td>
      <td>-0.699147</td>
      <td>1.424772</td>
      <td>...</td>
      <td>-1.266589</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.206087</td>
      <td>-0.795458</td>
      <td>-0.874776</td>
      <td>-0.563241</td>
      <td>-0.667380</td>
      <td>-0.506392</td>
      <td>-0.586763</td>
      <td>-0.755100</td>
      <td>-0.168834</td>
      <td>-0.767567</td>
      <td>...</td>
      <td>0.068415</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.322136</td>
      <td>1.555404</td>
      <td>-0.394659</td>
      <td>0.417263</td>
      <td>-0.217292</td>
      <td>0.150396</td>
      <td>-0.003121</td>
      <td>-0.561768</td>
      <td>1.422105</td>
      <td>0.328602</td>
      <td>...</td>
      <td>0.068415</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 37 columns</p>
</div>




    
![png](output_59_0.png)
    



    
![png](output_61_0.png)
    


    <class 'pandas.core.frame.DataFrame'>
    Index: 2205 entries, 0 to 2239
    Data columns (total 18 columns):
     #   Column                       Non-Null Count  Dtype  
    ---  ------                       --------------  -----  
     0   Income                       2205 non-null   float64
     1   Web_Visit                    2205 non-null   float64
     2   Complain                     2205 non-null   float64
     3   Age                          2205 non-null   float64
     4   Spending                     2205 non-null   float64
     5   Accept_Offer                 2205 non-null   float64
     6   Num_Purchase                 2205 non-null   float64
     7   Enrolled_Days                2205 non-null   float64
     8   Children                     2205 non-null   float64
     9   Education_Bachelor           2205 non-null   bool   
     10  Education_Basic              2205 non-null   bool   
     11  Education_Master             2205 non-null   bool   
     12  Education_PhD                2205 non-null   bool   
     13  Marital_Status_Divorced      2205 non-null   bool   
     14  Marital_Status_Married       2205 non-null   bool   
     15  Marital_Status_Relationship  2205 non-null   bool   
     16  Marital_Status_Single        2205 non-null   bool   
     17  Marital_Status_Widow         2205 non-null   bool   
    dtypes: bool(9), float64(9)
    memory usage: 191.6 KB


# **6. Pembangunan Model Clustering**

## **a. Pembangunan Model Clustering**
Pada tahap ini, membangun model clustering dengan memilih algoritma yang sesuai yakni  model KMeans  untuk mengelompokkan data berdasarkan kesamaan.




    Index(['Income', 'Web_Visit', 'Complain', 'Age', 'Spending', 'Accept_Offer',
           'Num_Purchase', 'Enrolled_Days', 'Children', 'Education_Bachelor',
           'Education_Basic', 'Education_Master', 'Education_PhD',
           'Marital_Status_Divorced', 'Marital_Status_Married',
           'Marital_Status_Relationship', 'Marital_Status_Single',
           'Marital_Status_Widow'],
          dtype='object')



    k=3, Silhouette Score: 0.4409596049207569


    k=3, Silhouette Score: 0.5679359240208132





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Education</th>
      <th>Marital_Status</th>
      <th>Income</th>
      <th>Recency</th>
      <th>MntWines</th>
      <th>MntFruits</th>
      <th>MntMeatProducts</th>
      <th>MntFishProducts</th>
      <th>MntSweetProducts</th>
      <th>MntGoldProds</th>
      <th>...</th>
      <th>Z_CostContact</th>
      <th>Z_Revenue</th>
      <th>Response</th>
      <th>Age</th>
      <th>Spending</th>
      <th>Accept_Offer</th>
      <th>Num_Purchase</th>
      <th>Enrolled_Days</th>
      <th>Children</th>
      <th>Cluster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bachelor</td>
      <td>Single</td>
      <td>58138.0</td>
      <td>58</td>
      <td>635</td>
      <td>88</td>
      <td>546</td>
      <td>172</td>
      <td>88</td>
      <td>88</td>
      <td>...</td>
      <td>3</td>
      <td>11</td>
      <td>1</td>
      <td>67</td>
      <td>1617</td>
      <td>1</td>
      <td>25</td>
      <td>663</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bachelor</td>
      <td>Single</td>
      <td>46344.0</td>
      <td>38</td>
      <td>11</td>
      <td>1</td>
      <td>6</td>
      <td>2</td>
      <td>1</td>
      <td>6</td>
      <td>...</td>
      <td>3</td>
      <td>11</td>
      <td>0</td>
      <td>70</td>
      <td>27</td>
      <td>0</td>
      <td>6</td>
      <td>113</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Bachelor</td>
      <td>Relationship</td>
      <td>71613.0</td>
      <td>26</td>
      <td>426</td>
      <td>49</td>
      <td>127</td>
      <td>111</td>
      <td>21</td>
      <td>42</td>
      <td>...</td>
      <td>3</td>
      <td>11</td>
      <td>0</td>
      <td>59</td>
      <td>776</td>
      <td>0</td>
      <td>21</td>
      <td>312</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Bachelor</td>
      <td>Relationship</td>
      <td>26646.0</td>
      <td>26</td>
      <td>11</td>
      <td>4</td>
      <td>20</td>
      <td>10</td>
      <td>3</td>
      <td>5</td>
      <td>...</td>
      <td>3</td>
      <td>11</td>
      <td>0</td>
      <td>40</td>
      <td>53</td>
      <td>0</td>
      <td>8</td>
      <td>139</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PhD</td>
      <td>Married</td>
      <td>58293.0</td>
      <td>94</td>
      <td>173</td>
      <td>43</td>
      <td>118</td>
      <td>46</td>
      <td>27</td>
      <td>15</td>
      <td>...</td>
      <td>3</td>
      <td>11</td>
      <td>0</td>
      <td>43</td>
      <td>422</td>
      <td>0</td>
      <td>19</td>
      <td>161</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2235</th>
      <td>Bachelor</td>
      <td>Married</td>
      <td>61223.0</td>
      <td>46</td>
      <td>709</td>
      <td>43</td>
      <td>182</td>
      <td>42</td>
      <td>118</td>
      <td>247</td>
      <td>...</td>
      <td>3</td>
      <td>11</td>
      <td>0</td>
      <td>57</td>
      <td>1341</td>
      <td>0</td>
      <td>18</td>
      <td>381</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2236</th>
      <td>PhD</td>
      <td>Relationship</td>
      <td>64014.0</td>
      <td>56</td>
      <td>406</td>
      <td>0</td>
      <td>30</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>...</td>
      <td>3</td>
      <td>11</td>
      <td>0</td>
      <td>78</td>
      <td>444</td>
      <td>1</td>
      <td>22</td>
      <td>19</td>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2237</th>
      <td>Bachelor</td>
      <td>Divorced</td>
      <td>56981.0</td>
      <td>91</td>
      <td>908</td>
      <td>48</td>
      <td>217</td>
      <td>32</td>
      <td>12</td>
      <td>24</td>
      <td>...</td>
      <td>3</td>
      <td>11</td>
      <td>0</td>
      <td>43</td>
      <td>1241</td>
      <td>1</td>
      <td>19</td>
      <td>155</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2238</th>
      <td>Master</td>
      <td>Relationship</td>
      <td>69245.0</td>
      <td>8</td>
      <td>428</td>
      <td>30</td>
      <td>214</td>
      <td>80</td>
      <td>30</td>
      <td>61</td>
      <td>...</td>
      <td>3</td>
      <td>11</td>
      <td>0</td>
      <td>68</td>
      <td>843</td>
      <td>0</td>
      <td>23</td>
      <td>156</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2239</th>
      <td>PhD</td>
      <td>Married</td>
      <td>52869.0</td>
      <td>40</td>
      <td>84</td>
      <td>3</td>
      <td>61</td>
      <td>2</td>
      <td>1</td>
      <td>21</td>
      <td>...</td>
      <td>3</td>
      <td>11</td>
      <td>1</td>
      <td>70</td>
      <td>172</td>
      <td>1</td>
      <td>11</td>
      <td>622</td>
      <td>2</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>2205 rows × 31 columns</p>
</div>



## **b. Evaluasi Model Clustering**
Untuk menentukan jumlah cluster yang optimal dalam model clustering, Anda dapat menggunakan metode Elbow atau Silhouette Score.

Metode ini membantu kita menemukan jumlah cluster yang memberikan pemisahan terbaik antar kelompok data, sehingga model yang dibangun dapat lebih efektif. Berikut adalah **rekomendasi** tahapannya.
1. Gunakan Silhouette Score dan Elbow Method untuk menentukan jumlah cluster optimal.
2. Hitung Silhouette Score sebagai ukuran kualitas cluster.


    
![png](output_73_0.png)
    





    <Axes: title={'center': 'Distortion Score Elbow for AgglomerativeClustering Clustering'}, xlabel='k', ylabel='distortion score'>



## **d. Visualisasi Hasil Clustering**


<div>                            <div id="cd62a634-25f6-485e-aaa3-ae20fa188d75" class="plotly-graph-div" style="height:800px; width:800px;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("cd62a634-25f6-485e-aaa3-ae20fa188d75")) {                    Plotly.newPlot(                        "cd62a634-25f6-485e-aaa3-ae20fa188d75",                        [{"hovertemplate":"Income=%{x}<br>Num_Purchase=%{y}<br>Spending=%{z}<br>Cluster=%{marker.color}<extra></extra>","legendgroup":"","marker":{"color":[2,1,2,1,2,2,2,1,1,1,1,2,1,1,3,1,1,2,1,1,2,2,2,1,1,1,1,2,1,1,1,1,2,1,2,1,1,2,2,1,1,1,2,1,1,2,2,2,1,3,2,3,2,1,2,3,2,2,2,2,1,1,3,2,2,2,2,2,1,1,2,3,1,2,1,1,1,1,2,1,1,1,2,1,1,1,1,1,1,2,1,1,1,2,2,2,1,1,2,1,2,2,3,2,3,2,1,3,1,1,1,2,1,1,1,3,2,2,1,2,2,2,2,1,2,1,1,1,1,2,2,2,2,1,2,1,1,1,1,1,2,1,1,2,3,1,1,1,2,1,2,1,2,1,1,1,2,1,1,1,1,1,1,2,3,1,1,2,1,1,2,1,1,1,1,2,2,1,1,2,1,1,1,2,2,2,1,2,2,2,3,1,1,1,1,1,3,1,2,1,2,2,1,1,3,1,2,1,2,2,1,2,1,1,2,2,1,1,2,1,1,2,1,1,2,1,2,2,1,3,2,1,3,2,2,2,1,1,3,1,2,1,2,1,1,1,1,2,1,1,1,1,2,1,2,1,2,1,1,1,1,2,2,2,2,2,1,1,1,2,1,1,2,2,2,1,1,1,2,1,1,2,1,1,2,2,1,2,1,1,1,2,1,2,2,1,1,1,2,1,1,1,2,1,2,1,1,2,2,2,1,1,1,1,1,1,2,1,1,2,3,1,3,2,1,2,2,1,2,1,2,1,1,2,3,2,2,2,1,1,2,2,1,2,2,1,1,2,2,2,1,2,2,1,1,1,2,1,1,1,1,2,1,1,1,1,1,1,1,1,2,1,2,2,1,2,1,2,2,1,1,1,1,1,2,1,1,2,1,1,1,1,2,1,2,2,1,2,2,1,3,3,2,1,1,1,3,2,1,3,2,1,2,3,2,2,2,1,1,2,1,1,1,1,1,1,1,1,1,1,3,1,1,2,2,1,2,2,1,3,1,1,2,2,2,1,2,1,2,3,1,2,2,2,1,2,1,1,2,1,1,2,1,1,1,1,1,2,2,2,2,1,1,2,1,2,2,2,2,1,2,2,2,1,1,1,2,1,2,2,2,1,2,1,2,1,2,1,2,1,1,2,3,1,2,1,3,1,1,2,1,2,1,1,2,2,1,1,1,1,2,1,1,1,1,1,3,2,1,2,1,1,1,1,1,2,1,2,1,2,2,1,3,1,2,2,2,1,1,2,1,1,1,1,1,1,1,1,3,1,1,1,1,1,1,1,1,1,1,3,2,1,1,1,3,2,1,1,1,1,1,1,1,1,2,2,2,1,1,1,1,1,2,1,1,1,1,2,1,1,1,1,1,1,2,1,2,1,2,2,1,1,2,2,2,1,2,1,2,1,2,2,2,2,2,1,3,1,3,1,2,2,2,1,2,1,1,1,1,2,1,2,1,2,1,1,1,1,1,1,1,2,2,2,2,2,1,3,2,1,2,2,2,1,1,2,2,2,3,2,1,2,1,1,1,1,1,1,2,2,2,2,2,3,1,3,1,2,2,1,1,2,1,2,1,2,3,1,2,1,2,2,1,2,1,1,2,2,2,2,1,2,2,1,3,3,2,1,1,2,2,1,1,1,2,2,1,2,1,3,2,1,2,3,2,2,2,2,1,1,1,2,2,1,2,1,2,2,1,2,2,2,2,1,1,1,1,2,1,2,2,1,1,1,1,1,1,2,1,1,2,3,1,1,1,1,2,2,2,1,2,1,1,2,3,2,1,1,2,2,1,1,2,2,2,2,1,1,2,1,3,1,1,1,3,2,3,1,2,1,1,2,2,1,1,2,1,2,1,2,1,1,1,1,2,3,2,2,1,1,1,2,3,1,1,2,1,2,1,2,1,1,1,1,1,1,1,1,1,1,3,2,1,1,2,3,2,1,2,1,1,1,1,1,2,2,1,1,2,2,1,1,2,1,2,2,2,2,1,1,2,1,2,1,1,2,3,1,1,1,2,2,2,1,3,2,2,3,1,2,1,2,1,1,2,1,2,2,2,2,2,1,2,1,2,1,2,2,2,2,2,2,2,3,1,2,2,1,1,2,1,1,1,1,1,1,1,2,1,1,2,2,1,1,1,2,3,1,1,2,2,1,1,2,2,2,2,2,1,2,1,1,1,2,3,2,2,3,2,1,3,1,1,2,1,1,2,1,2,1,3,3,3,1,1,1,2,2,1,1,3,1,1,1,1,1,2,1,1,1,1,1,2,1,1,2,1,1,1,2,3,2,2,1,2,1,1,1,1,2,2,1,1,2,1,1,1,2,1,2,2,1,2,1,1,2,1,1,2,2,2,2,1,1,2,1,3,2,1,2,1,2,2,1,1,2,3,1,1,1,2,1,2,1,3,2,1,3,1,2,2,1,2,1,1,1,2,2,1,2,3,1,1,1,1,2,1,1,2,2,2,2,1,2,1,2,1,1,1,2,2,1,1,1,1,1,2,1,1,2,2,1,1,2,3,1,1,2,1,1,2,1,1,1,2,1,1,2,1,1,2,2,1,3,1,1,2,2,2,2,1,2,1,2,1,2,1,1,2,2,1,1,2,1,1,1,1,1,1,2,1,1,2,1,1,1,1,2,1,1,2,3,1,1,1,2,1,2,2,2,1,2,1,1,2,2,2,1,1,1,1,2,2,2,1,1,2,1,2,1,1,1,2,1,1,2,2,1,1,1,1,1,1,1,1,2,1,3,1,1,1,1,2,1,1,1,1,1,1,2,3,2,3,2,2,1,1,2,1,3,1,3,3,1,1,3,2,1,1,2,2,2,1,2,1,2,1,1,2,1,3,3,1,1,1,1,1,1,3,2,1,1,1,1,1,1,1,2,3,1,3,1,1,2,2,1,3,3,2,2,2,1,2,1,1,1,1,1,1,2,1,2,1,1,2,1,1,1,2,1,1,3,2,3,1,2,1,1,1,1,1,3,1,1,1,1,1,2,2,2,2,1,1,3,2,1,2,2,1,2,1,2,2,1,2,1,1,1,1,1,1,1,1,2,1,2,1,1,1,1,1,3,1,1,2,1,1,1,1,1,2,1,1,2,1,1,2,1,1,2,1,1,2,2,1,2,3,1,3,1,2,1,1,1,1,1,2,2,1,1,1,1,1,1,1,1,1,3,1,2,1,1,1,1,1,1,1,1,3,3,1,2,2,2,2,1,2,2,1,1,2,1,1,3,1,2,2,1,1,1,2,2,2,1,2,1,1,1,3,2,1,2,1,1,2,2,2,1,1,2,2,2,2,2,1,2,1,3,2,1,2,1,2,2,2,2,1,1,2,2,2,2,2,2,2,2,1,2,2,1,1,2,1,1,1,2,2,1,1,1,2,2,1,3,1,2,1,2,1,1,1,1,2,2,2,1,2,2,1,2,1,1,1,2,1,1,2,2,2,2,1,1,1,1,2,1,2,1,2,1,1,2,2,2,2,3,1,3,1,1,1,1,3,1,2,1,2,2,1,2,1,1,1,1,2,1,2,1,1,1,1,1,1,2,1,2,2,2,1,1,1,2,2,2,2,2,1,3,1,1,2,1,1,1,2,1,1,2,2,1,1,2,1,1,1,2,1,1,1,1,2,2,2,1,1,1,3,2,1,2,1,1,2,2,1,1,2,1,1,1,1,2,2,3,2,1,2,1,1,1,2,1,2,3,1,2,3,2,2,1,1,1,1,1,2,1,1,1,1,1,2,3,2,2,2,2,1,1,1,2,1,2,1,2,2,1,1,2,1,1,2,1,2,1,3,3,1,2,1,1,2,1,1,2,1,3,3,2,1,1,1,1,2,2,1,1,1,2,2,2,2,2,1,1,1,1,1,2,2,2,1,2,2,3,2,1,1,1,1,1,1,1,1,2,2,1,1,2,1,2,1,2,1,1,1,1,2,2,1,1,1,1,1,2,1,1,2,2,1,1,2,1,2,2,1,1,2,1,1,1,1,1,2,2,2,2,1,1,1,2,1,1,2,2,1,2,2,2,2,1,2,2,2,1,2,1,1,2,2,1,1,2,2,1,1,1,2,1,1,1,2,1,2,2,1,3,1,2,2,3,1,1,2,1,3,1,2,2,2,3,1,1,2,2,2,1,1,2,3,2,1,1,1,3,1,2,1,2,1,1,1,2,2,2,2,1,1,2,1,1,1,3,2,1,2,3,1,2,2,1,1,2,2,2,1,1,2,1,1,2,2,1,1,1,2,2,2,3,2,3,1,1,1,1,1,2,3,2,2,1,2,2,2,2,1,2,1,1,2,1,1,2,1,1,3,1,1,2,1,2,2,2,1,1,1,2,2,2,2,2,1,1,3,1,2,3,1,1,1,2,2,2,3,3,2,1,1,1,1,1,1,1,1,1,1,1,3,1,2,1,1,1,1,1,3,1,3,3,1,3,2,2,1,1,1,1,1,1,1,1,2,1,1,2,2,1,3,2,2,1,1,1,1,1,1,1,1,2,2,1,1,1,2,1,2,2,1,2,1,1,2,2,1,2,2,2,1,1,1,1,1,3,2,2,1,2,2,1,1,1,2,2,2,1,2,1,2,2,1,2,1,1,2,2,1,1,2,2,1,1,1,1,2,2,2,2,1,1,1,1,1,3,2,1,2,1,3,2,1,2,1,3,1,1,1,2,1,3,2,2,1,1,1,2,1,2,2,1,1,1,1,2,1,3,2,3,2,1,1,3,1,2,1,1,1,1,1,1,1,1,1,2,3,1,1,1,1,1,2,1,1,1,1,1,1,2,1,2,1,1,2,1,1,2,3,3,1,2,3,2,2,2,3,2,1,1,1,1,1,1,1,1,2,2,2,2,1,2,1,1,3,3,1,1,1,1,1,1,2,1,2,1,1,2,1,1,1,2,2,1,2,1,1,1,2,1,1,2,3,1,1,2,2,2,2,1,1,2,1,1,2,2,2,2,1],"coloraxis":"coloraxis","symbol":"circle"},"mode":"markers","name":"","scene":"scene","showlegend":false,"x":[58138.0,46344.0,71613.0,26646.0,58293.0,62513.0,55635.0,33454.0,30351.0,5648.0,7500.0,63033.0,59354.0,17323.0,82800.0,41850.0,37760.0,76995.0,33812.0,37040.0,2447.0,58607.0,65324.0,40689.0,18589.0,53359.0,38360.0,84618.0,10979.0,38620.0,40548.0,46610.0,68657.0,49389.0,67353.0,23718.0,42429.0,48948.0,80011.0,20559.0,21994.0,7500.0,79941.0,7500.0,41728.0,72550.0,65486.0,79143.0,35790.0,82582.0,66373.0,82384.0,70287.0,27938.0,55954.0,75777.0,66653.0,61823.0,67680.0,70666.0,25721.0,32474.0,88194.0,69096.0,74854.0,66991.0,65031.0,60631.0,28332.0,40246.0,75251.0,75825.0,26326.0,56046.0,29760.0,26304.0,23559.0,38620.0,81361.0,29440.0,36138.0,50388.0,79593.0,54178.0,42394.0,23626.0,30096.0,47916.0,51813.0,78497.0,50150.0,47823.0,34554.0,85693.0,65846.0,87195.0,24594.0,49096.0,52413.0,38557.0,89058.0,77298.0,68126.0,57288.0,86037.0,43974.0,50785.0,90765.0,36550.0,30753.0,21918.0,56129.0,32557.0,19510.0,30992.0,101970.0,71488.0,79607.0,54348.0,77376.0,62998.0,61331.0,73448.0,41551.0,62981.0,9548.0,33762.0,35860.0,36921.0,92859.0,65104.0,86111.0,68352.0,41883.0,59809.0,23957.0,38547.0,35688.0,49605.0,59354.0,65747.0,46344.0,34176.0,61010.0,69372.0,49967.0,60199.0,55375.0,80317.0,30523.0,70356.0,23228.0,74165.0,43482.0,62551.0,52332.0,66951.0,26091.0,33456.0,28718.0,50447.0,53537.0,52074.0,80427.0,83837.0,38853.0,38285.0,78497.0,51650.0,16248.0,66835.0,30477.0,28249.0,25271.0,32303.0,61286.0,74068.0,45759.0,24882.0,66973.0,38872.0,51148.0,31353.0,69661.0,80067.0,86718.0,46854.0,69142.0,75922.0,63693.0,102160.0,40637.0,18890.0,29604.0,48721.0,44794.0,64497.0,46097.0,77972.0,44377.0,46014.0,70951.0,41443.0,52195.0,83790.0,44551.0,69508.0,45204.0,72460.0,77622.0,30732.0,63887.0,42011.0,51369.0,51537.0,79930.0,34320.0,37070.0,81975.0,38590.0,15033.0,62745.0,22212.0,23661.0,79761.0,7500.0,73455.0,64961.0,22804.0,73687.0,61074.0,31686.0,80134.0,75027.0,67546.0,65176.0,31160.0,29938.0,102692.0,26490.0,75702.0,30899.0,63342.0,45989.0,18701.0,40737.0,15287.0,69674.0,44159.0,37717.0,43776.0,38179.0,80124.0,38097.0,72940.0,22070.0,69267.0,31788.0,61905.0,29315.0,33378.0,66313.0,60714.0,77882.0,69867.0,63841.0,24480.0,51369.0,37760.0,65640.0,44319.0,30631.0,75278.0,50898.0,79946.0,35416.0,32414.0,38361.0,82497.0,16626.0,29672.0,55951.0,35388.0,42386.0,68627.0,57912.0,35246.0,58821.0,46377.0,39747.0,23976.0,80950.0,27038.0,77457.0,64100.0,42670.0,12571.0,22574.0,70893.0,54198.0,28839.0,40321.0,66503.0,30833.0,64795.0,34421.0,47025.0,64325.0,40464.0,62187.0,14849.0,27255.0,54432.0,29999.0,24072.0,33996.0,66334.0,35178.0,22010.0,62204.0,75693.0,30675.0,83003.0,68655.0,41411.0,55212.0,59292.0,27190.0,82623.0,44300.0,84835.0,30372.0,33181.0,71113.0,71952.0,69759.0,72099.0,60000.0,38643.0,50737.0,68462.0,65073.0,46681.0,78618.0,62187.0,28442.0,37717.0,51479.0,54803.0,79530.0,31615.0,72025.0,52614.0,35684.0,48178.0,29548.0,63810.0,38578.0,46098.0,22585.0,30279.0,66426.0,30822.0,33581.0,19986.0,27421.0,35688.0,36143.0,10245.0,43795.0,63381.0,38823.0,83664.0,90300.0,62499.0,74293.0,51012.0,70777.0,68682.0,43824.0,15345.0,23442.0,14515.0,31395.0,75276.0,42373.0,30507.0,55521.0,48006.0,27213.0,65808.0,30351.0,50437.0,23616.0,53858.0,66465.0,46923.0,75072.0,75865.0,19789.0,80134.0,91065.0,49505.0,37401.0,30096.0,18492.0,82584.0,93027.0,48686.0,92910.0,75433.0,10404.0,61314.0,84865.0,42387.0,67309.0,75236.0,30015.0,50943.0,67272.0,51529.0,32011.0,7500.0,28691.0,56223.0,18100.0,30279.0,20130.0,23295.0,42618.0,81246.0,24027.0,55707.0,57959.0,56796.0,36230.0,70829.0,65991.0,38988.0,89572.0,42207.0,50300.0,66664.0,60597.0,70165.0,50520.0,80124.0,33183.0,66582.0,75261.0,31880.0,53790.0,49269.0,61456.0,37406.0,56937.0,38415.0,20518.0,62503.0,41644.0,55842.0,62010.0,41124.0,38961.0,37760.0,32233.0,43057.0,83151.0,78825.0,65104.0,60093.0,14045.0,28457.0,78952.0,46310.0,76005.0,58308.0,55614.0,59432.0,55563.0,78642.0,67911.0,65275.0,27203.0,48330.0,24279.0,64355.0,50943.0,53653.0,65665.0,81217.0,34935.0,61250.0,39665.0,60152.0,48920.0,89120.0,44124.0,81169.0,36443.0,26095.0,71367.0,80184.0,30630.0,73454.0,42691.0,70503.0,25545.0,32880.0,77863.0,50353.0,61839.0,49154.0,47682.0,72679.0,57954.0,65316.0,28567.0,47352.0,44931.0,76982.0,57247.0,22944.0,25315.0,43638.0,42710.0,84169.0,54058.0,24683.0,85620.0,47850.0,19514.0,27159.0,39548.0,21474.0,60504.0,22419.0,81698.0,43462.0,54880.0,79908.0,15315.0,87771.0,33039.0,81741.0,71499.0,62466.0,48799.0,52157.0,66565.0,29298.0,47691.0,38200.0,44989.0,38443.0,38593.0,64413.0,36959.0,61996.0,51287.0,13260.0,47472.0,54603.0,45207.0,40689.0,47821.0,27450.0,39453.0,26850.0,79800.0,61794.0,53863.0,24221.0,39684.0,92163.0,69882.0,33178.0,59973.0,17459.0,23910.0,42169.0,26224.0,31089.0,30081.0,62807.0,72906.0,61467.0,49618.0,21888.0,42429.0,26150.0,30801.0,81168.0,26877.0,45006.0,18978.0,22574.0,48240.0,45837.0,35791.0,54162.0,30522.0,54456.0,31632.0,72298.0,36975.0,72635.0,13624.0,84196.0,70971.0,34487.0,28769.0,69084.0,65488.0,62466.0,32218.0,83917.0,46102.0,84574.0,56181.0,63120.0,73691.0,63381.0,76140.0,62859.0,45906.0,77632.0,46463.0,105471.0,55282.0,78710.0,66886.0,98777.0,29103.0,67445.0,50616.0,49431.0,61278.0,26490.0,73059.0,46734.0,56253.0,19986.0,58330.0,25965.0,14661.0,18690.0,45068.0,21063.0,29187.0,54690.0,59304.0,59247.0,66731.0,77353.0,52614.0,26751.0,81300.0,70337.0,36145.0,65295.0,68118.0,68743.0,41039.0,38946.0,65777.0,66476.0,86857.0,77845.0,69476.0,50611.0,61209.0,42315.0,13084.0,47570.0,61923.0,34824.0,26518.0,45938.0,78468.0,78901.0,71427.0,71022.0,90247.0,41335.0,71952.0,35682.0,43185.0,66375.0,35178.0,25252.0,55250.0,33249.0,58398.0,50272.0,76618.0,87305.0,25851.0,58710.0,45160.0,74806.0,59111.0,18988.0,72190.0,7500.0,44794.0,80395.0,75012.0,56962.0,89891.0,35946.0,53593.0,66373.0,45072.0,89694.0,72025.0,67432.0,70545.0,17487.0,62882.0,64108.0,34941.0,48767.0,38702.0,82224.0,83844.0,17003.0,71163.0,33697.0,63564.0,83443.0,51518.0,58330.0,80952.0,75507.0,63855.0,62220.0,58512.0,40662.0,38829.0,35523.0,79146.0,78285.0,31626.0,75127.0,48726.0,74985.0,67430.0,46891.0,62058.0,72063.0,78939.0,42720.0,33622.0,6835.0,41452.0,40760.0,74250.0,51124.0,72258.0,71466.0,36283.0,20587.0,30467.0,31590.0,20425.0,17144.0,42564.0,43783.0,40780.0,62847.0,82017.0,16813.0,51267.0,46524.0,45183.0,70421.0,60161.0,73926.0,19329.0,61872.0,46984.0,34838.0,82716.0,48192.0,49681.0,56850.0,55267.0,59666.0,72504.0,26872.0,21359.0,73170.0,52750.0,91820.0,65968.0,30772.0,22507.0,65685.0,25804.0,76412.0,22063.0,57091.0,22419.0,87771.0,78353.0,93404.0,37859.0,80995.0,16529.0,55412.0,48789.0,56575.0,25130.0,35441.0,71391.0,49494.0,81702.0,45889.0,56628.0,34026.0,40049.0,34176.0,19419.0,82504.0,81205.0,61618.0,55284.0,49980.0,15072.0,49166.0,65324.0,82347.0,30843.0,46374.0,60474.0,38576.0,55357.0,37758.0,85710.0,23228.0,44602.0,7500.0,38683.0,49514.0,57906.0,43456.0,19485.0,53172.0,30545.0,70123.0,62450.0,21675.0,42395.0,61346.0,80812.0,42835.0,39922.0,86424.0,17117.0,24762.0,35797.0,36627.0,51111.0,86857.0,82072.0,46231.0,42243.0,51195.0,68092.0,31814.0,51390.0,76630.0,26868.0,48948.0,55260.0,64090.0,78331.0,37087.0,21846.0,81320.0,54137.0,66825.0,57100.0,58917.0,85072.0,86429.0,45684.0,47889.0,45921.0,78420.0,75114.0,52278.0,35641.0,95529.0,62820.0,73113.0,84169.0,42607.0,74637.0,46015.0,72354.0,39858.0,34469.0,83033.0,24401.0,77583.0,74116.0,74293.0,68397.0,79632.0,46107.0,64950.0,25443.0,75127.0,32892.0,71796.0,67536.0,55239.0,60554.0,64831.0,56067.0,82025.0,94384.0,14906.0,51563.0,57937.0,68274.0,39771.0,67893.0,27922.0,52190.0,44051.0,42767.0,46106.0,16927.0,59754.0,53700.0,59041.0,54237.0,70647.0,52597.0,41021.0,40233.0,50183.0,54753.0,92955.0,33471.0,34596.0,44010.0,84219.0,40706.0,15716.0,59052.0,80573.0,83715.0,82576.0,56962.0,35704.0,53103.0,46779.0,4861.0,33462.0,63693.0,80763.0,65352.0,82170.0,75759.0,79689.0,35340.0,85683.0,24884.0,42021.0,64449.0,64587.0,34824.0,75437.0,26091.0,52845.0,46086.0,78028.0,95169.0,56337.0,22434.0,36930.0,36130.0,65569.0,83844.0,19514.0,36736.0,77568.0,49187.0,30168.0,34053.0,38196.0,59412.0,70924.0,54165.0,32300.0,20180.0,34961.0,28440.0,64504.0,33564.0,17345.0,56320.0,28647.0,15038.0,32173.0,68316.0,74538.0,91700.0,68695.0,31056.0,79593.0,28071.0,37334.0,46423.0,37126.0,47703.0,61180.0,38998.0,8028.0,76081.0,34728.0,33168.0,33585.0,77037.0,35196.0,44529.0,70924.0,28764.0,69098.0,25959.0,27100.0,70596.0,42557.0,53312.0,72228.0,67605.0,62845.0,65196.0,42000.0,35860.0,65526.0,16860.0,83528.0,64176.0,22304.0,67023.0,32892.0,70713.0,59925.0,39722.0,46610.0,88347.0,87171.0,26907.0,50014.0,41014.0,66294.0,36715.0,79456.0,40479.0,75345.0,54233.0,24163.0,84460.0,43776.0,71691.0,85844.0,39190.0,71367.0,38578.0,57236.0,61825.0,79803.0,80910.0,27590.0,56775.0,83829.0,54210.0,38508.0,53187.0,30023.0,76045.0,50870.0,15315.0,65463.0,66480.0,76773.0,81698.0,54466.0,98777.0,16269.0,71819.0,33569.0,36262.0,22634.0,47025.0,70566.0,31605.0,52034.0,48526.0,46734.0,39552.0,86358.0,46931.0,16581.0,63998.0,67381.0,25930.0,42693.0,85606.0,72903.0,49669.0,36778.0,85696.0,10979.0,49678.0,56129.0,37155.0,21282.0,33419.0,63285.0,21255.0,42162.0,54450.0,57744.0,26576.0,57513.0,68142.0,7500.0,83145.0,54197.0,23091.0,46049.0,56715.0,79410.0,57304.0,44375.0,54450.0,59594.0,80685.0,40344.0,62710.0,48985.0,35322.0,77142.0,81657.0,14421.0,20130.0,74214.0,66726.0,23724.0,47353.0,33444.0,54386.0,28510.0,90638.0,48070.0,43140.0,54959.0,15056.0,26954.0,22327.0,44393.0,62000.0,31497.0,45894.0,78579.0,67369.0,58401.0,62307.0,43641.0,63841.0,46891.0,70091.0,78075.0,59184.0,54809.0,58113.0,51412.0,15287.0,66636.0,50965.0,84618.0,18351.0,40451.0,36317.0,42213.0,65748.0,77044.0,74918.0,56721.0,42160.0,61559.0,33629.0,68682.0,34377.0,8940.0,26228.0,77297.0,40211.0,33438.0,75032.0,61284.0,22518.0,54730.0,38452.0,44421.0,38197.0,41986.0,28427.0,37395.0,64722.0,55249.0,84906.0,28691.0,44213.0,25707.0,59062.0,76624.0,66000.0,27683.0,1730.0,7500.0,40521.0,20427.0,65106.0,69969.0,67433.0,77766.0,74716.0,68118.0,55158.0,62972.0,74190.0,39356.0,76653.0,35860.0,90687.0,73450.0,31454.0,47139.0,83829.0,53378.0,19656.0,45579.0,85485.0,55956.0,64191.0,38808.0,57183.0,23748.0,66303.0,37368.0,40800.0,71847.0,46149.0,78687.0,49118.0,37633.0,39767.0,26997.0,33986.0,57091.0,46831.0,83151.0,52531.0,15759.0,22804.0,43050.0,42997.0,48918.0,60033.0,34043.0,57811.0,78569.0,7500.0,94384.0,23148.0,44267.0,71626.0,60894.0,50200.0,81051.0,65169.0,59868.0,65695.0,64857.0,45143.0,74805.0,59060.0,27238.0,47009.0,46094.0,40321.0,37235.0,81843.0,46692.0,77382.0,37774.0,18393.0,72828.0,24711.0,45503.0,6560.0,71604.0,27244.0,48752.0,71434.0,90842.0,88097.0,51948.0,71853.0,35876.0,40049.0,39660.0,50127.0,43263.0,62845.0,18929.0,24367.0,33249.0,26887.0,50150.0,62061.0,85696.0,76542.0,70515.0,18227.0,69139.0,69109.0,69627.0,38136.0,62159.0,80695.0,33316.0,58554.0,17256.0,53034.0,52203.0,59601.0,75154.0,47025.0,37971.0,41335.0,67267.0,57338.0,50523.0,35791.0,50611.0,56242.0,48904.0,56243.0,21355.0,57420.0,46390.0,54342.0,20895.0,92344.0,26907.0,44964.0,75507.0,53761.0,22682.0,38887.0,41658.0,29791.0,63915.0,39996.0,26759.0,63841.0,51039.0,60544.0,65685.0,37716.0,36864.0,44511.0,36947.0,47352.0,67087.0,57045.0,36957.0,69389.0,80134.0,43142.0,80589.0,34412.0,57537.0,22634.0,51315.0,36026.0,24639.0,34578.0,65704.0,63810.0,54132.0,18690.0,28164.0,34596.0,43269.0,38741.0,31907.0,27100.0,31163.0,92533.0,34853.0,70844.0,31086.0,60544.0,20491.0,42523.0,39922.0,33402.0,36408.0,21645.0,78427.0,82657.0,51876.0,78041.0,52852.0,70038.0,69401.0,46053.0,77343.0,73892.0,40304.0,32727.0,68695.0,43300.0,26290.0,93790.0,38410.0,64866.0,57957.0,46015.0,16531.0,28072.0,49476.0,50725.0,83844.0,41145.0,67419.0,23162.0,34380.0,34704.0,94871.0,65148.0,39898.0,64857.0,59892.0,41020.0,57072.0,60474.0,62807.0,19414.0,19107.0,75484.0,70379.0,79419.0,64014.0,76998.0,49854.0,60585.0,42873.0,87679.0,57867.0,35765.0,65492.0,32952.0,53374.0,71706.0,68487.0,53253.0,31163.0,42014.0,54108.0,49667.0,63206.0,57136.0,46772.0,78931.0,53977.0,84219.0,46098.0,73538.0,79529.0,20981.0,51766.0,55759.0,33039.0,37787.0,27242.0,87188.0,69930.0,37697.0,37401.0,3502.0,58597.0,82032.0,28087.0,74004.0,19740.0,57036.0,53083.0,69283.0,46098.0,23331.0,23331.0,9255.0,67786.0,71969.0,59235.0,31928.0,74881.0,65819.0,51411.0,51983.0,42386.0,30390.0,30983.0,66033.0,37284.0,57530.0,76800.0,63943.0,76081.0,67445.0,37054.0,47175.0,31859.0,27215.0,70179.0,39922.0,49681.0,24645.0,79865.0,44322.0,47958.0,63972.0,75315.0,55517.0,75283.0,82800.0,38998.0,90638.0,27161.0,42014.0,38201.0,45203.0,81574.0,34935.0,60482.0,34633.0,78093.0,82460.0,45903.0,81361.0,35860.0,40442.0,61482.0,34968.0,75794.0,31497.0,74268.0,13724.0,45143.0,52569.0,48432.0,17144.0,36108.0,76445.0,36663.0,53843.0,90226.0,70638.0,44512.0,27116.0,54072.0,71855.0,51250.0,60432.0,65526.0,68655.0,12393.0,64509.0,33955.0,31353.0,55434.0,28359.0,57100.0,69139.0,52973.0,51717.0,18793.0,66664.0,50664.0,54414.0,54549.0,47111.0,41003.0,19444.0,36301.0,73059.0,42731.0,52854.0,22775.0,46681.0,59821.0,50002.0,69755.0,44078.0,30560.0,35924.0,64140.0,56386.0,24594.0,75774.0,39228.0,58494.0,58684.0,57136.0,56551.0,22448.0,82014.0,34213.0,25358.0,35544.0,36634.0,62670.0,50334.0,72066.0,50729.0,34916.0,64892.0,43602.0,33996.0,41473.0,63246.0,36732.0,69084.0,77766.0,37929.0,86610.0,80141.0,72635.0,69016.0,20193.0,27573.0,15862.0,49544.0,33228.0,70440.0,38232.0,22554.0,23536.0,49413.0,42231.0,78789.0,56534.0,58350.0,81217.0,49090.0,61787.0,18169.0,24336.0,18222.0,62335.0,42033.0,86580.0,41437.0,73705.0,61064.0,38452.0,18358.0,55012.0,9722.0,38175.0,58656.0,52117.0,64813.0,54222.0,83512.0,77520.0,41154.0,80398.0,18746.0,35196.0,60230.0,22108.0,44392.0,55424.0,17688.0,92491.0,90273.0,82571.0,38513.0,16653.0,42586.0,23529.0,74881.0,71107.0,46910.0,18690.0,37244.0,82427.0,75342.0,70044.0,79146.0,77437.0,54984.0,42403.0,55761.0,37292.0,45576.0,70321.0,58086.0,81795.0,28389.0,66835.0,69901.0,80360.0,63342.0,44989.0,31859.0,51569.0,30372.0,16014.0,41120.0,39763.0,38725.0,77981.0,62905.0,13533.0,59481.0,72117.0,21955.0,67131.0,36802.0,71853.0,28249.0,47808.0,25509.0,51012.0,70596.0,85431.0,42664.0,42586.0,29760.0,28973.0,39435.0,65370.0,20194.0,42473.0,64590.0,71232.0,34600.0,46904.0,49094.0,36075.0,60839.0,77298.0,34026.0,48918.0,82122.0,37697.0,34074.0,28520.0,62535.0,36273.0,63404.0,75774.0,78416.0,75702.0,59385.0,37070.0,44689.0,53977.0,7144.0,18701.0,90369.0,63159.0,37758.0,46757.0,79734.0,63207.0,72071.0,21840.0,58582.0,72282.0,50387.0,32583.0,62568.0,44635.0,33316.0,63967.0,52513.0,25293.0,54111.0,78394.0,80739.0,22669.0,29236.0,44911.0,54693.0,48186.0,54809.0,41580.0,80336.0,47743.0,62972.0,57333.0,32313.0,84953.0,27071.0,68148.0,65735.0,86836.0,4023.0,30093.0,57705.0,25008.0,83257.0,22280.0,72159.0,64260.0,82733.0,74290.0,58217.0,21024.0,70116.0,54006.0,69063.0,11448.0,61825.0,70886.0,69109.0,60208.0,32889.0,34738.0,30538.0,82326.0,26642.0,69932.0,44503.0,60200.0,49638.0,23830.0,33051.0,72905.0,69702.0,70300.0,70643.0,30396.0,50616.0,62772.0,30298.0,52413.0,44155.0,86979.0,76532.0,42081.0,67546.0,88420.0,31158.0,61671.0,69719.0,49160.0,24882.0,90933.0,113734.0,59292.0,45759.0,22148.0,73395.0,29819.0,45688.0,62820.0,60491.0,33235.0,35701.0,31535.0,92556.0,67384.0,80573.0,77870.0,52278.0,57107.0,18929.0,36038.0,20180.0,34230.0,31878.0,70932.0,91249.0,77598.0,80982.0,22701.0,55212.0,70617.0,64849.0,62694.0,61917.0,67472.0,21059.0,29543.0,75903.0,34984.0,54998.0,54356.0,34242.0,25410.0,73356.0,28320.0,23763.0,72570.0,34529.0,70792.0,63211.0,83891.0,36065.0,39898.0,51141.0,56939.0,80872.0,72335.0,61798.0,76842.0,29478.0,46998.0,96843.0,56551.0,70053.0,71670.0,44359.0,40887.0,48877.0,74485.0,64474.0,77226.0,72643.0,85738.0,55686.0,39062.0,34600.0,40794.0,54690.0,4428.0,32632.0,38236.0,36781.0,31385.0,24570.0,25509.0,94642.0,58646.0,53201.0,21994.0,57113.0,51373.0,23477.0,58116.0,72968.0,39791.0,91712.0,94472.0,13672.0,96547.0,79205.0,56559.0,32644.0,67506.0,24206.0,28420.0,22979.0,33279.0,46423.0,30368.0,63684.0,41638.0,68805.0,65814.0,71964.0,39146.0,96876.0,34445.0,68281.0,37284.0,36790.0,27943.0,22263.0,30261.0,28587.0,43815.0,48195.0,77027.0,71322.0,32765.0,29672.0,23272.0,49912.0,7500.0,68117.0,67225.0,17649.0,55914.0,40059.0,60905.0,75330.0,60689.0,38680.0,61416.0,49767.0,83273.0,29009.0,45057.0,37150.0,43020.0,53154.0,65308.0,76467.0,81044.0,32871.0,55801.0,76320.0,36927.0,48794.0,23478.0,71128.0,70179.0,59462.0,38361.0,70038.0,42554.0,67445.0,67046.0,42192.0,77610.0,31761.0,34350.0,54591.0,71866.0,25293.0,42000.0,69520.0,64713.0,27889.0,22123.0,27803.0,51651.0,68487.0,56796.0,87000.0,79823.0,30828.0,34109.0,53367.0,22390.0,71322.0,79244.0,49912.0,48699.0,80144.0,54252.0,81929.0,76068.0,53204.0,67911.0,14796.0,78128.0,14188.0,20425.0,58646.0,72967.0,48150.0,82332.0,58275.0,44953.0,34587.0,43018.0,45736.0,66476.0,27733.0,69805.0,72217.0,25818.0,37509.0,25176.0,43586.0,53230.0,50116.0,84906.0,61286.0,82333.0,65220.0,33590.0,30992.0,72309.0,16005.0,65487.0,58692.0,40590.0,20587.0,62637.0,29435.0,19346.0,35788.0,36997.0,23539.0,65333.0,78499.0,29732.0,41275.0,63516.0,42769.0,49572.0,69209.0,17148.0,40590.0,30560.0,32146.0,41713.0,34176.0,58482.0,59686.0,60896.0,62994.0,47320.0,74859.0,15253.0,31560.0,72071.0,91172.0,90000.0,41967.0,69263.0,65706.0,60934.0,71965.0,65210.0,79174.0,81380.0,48432.0,52914.0,38946.0,26067.0,44325.0,42523.0,26487.0,53233.0,67716.0,76234.0,84117.0,65487.0,25224.0,89616.0,40851.0,27469.0,82347.0,73803.0,8820.0,43322.0,55593.0,50501.0,37085.0,16185.0,57731.0,48432.0,73807.0,18929.0,35893.0,61014.0,32144.0,14918.0,45146.0,41769.0,88325.0,38054.0,80617.0,5305.0,36807.0,28427.0,82032.0,22775.0,40101.0,58025.0,75777.0,7500.0,33562.0,57642.0,58554.0,63777.0,57967.0,24434.0,11012.0,44802.0,26816.0,34421.0,61223.0,64014.0,56981.0,69245.0,52869.0],"y":[25,6,21,8,19,22,21,10,6,2,6,16,15,5,26,9,13,26,8,12,43,17,20,20,8,14,9,26,8,11,9,17,18,6,28,9,7,24,21,7,4,8,16,10,5,28,17,29,7,21,23,27,21,6,19,21,22,24,19,16,5,4,32,18,19,21,30,24,8,4,19,22,4,21,12,4,9,11,27,7,13,17,19,10,5,10,6,22,7,25,16,7,7,27,16,20,5,13,26,9,14,24,17,20,25,21,8,15,20,7,7,22,9,6,5,27,16,16,8,23,19,23,19,16,23,6,9,6,7,22,17,23,25,12,15,6,4,16,8,16,19,17,11,26,21,15,4,9,18,4,27,11,20,9,18,12,15,9,6,7,8,8,7,28,25,7,4,25,10,9,24,6,6,5,7,18,24,7,5,28,8,16,6,23,17,17,15,19,18,29,26,6,4,5,17,6,29,15,20,19,26,17,20,4,25,12,28,7,16,22,13,32,12,16,18,17,4,23,14,4,5,29,8,5,19,11,22,16,5,22,31,6,16,21,20,25,7,6,29,11,26,6,19,16,10,4,6,25,15,4,17,11,19,11,13,8,14,10,12,10,9,23,23,12,17,22,10,16,13,15,7,7,23,33,22,16,5,10,19,9,4,23,5,7,12,28,11,21,7,13,10,23,7,18,18,11,11,7,23,16,7,8,15,6,21,4,4,23,24,18,9,5,5,12,4,6,19,5,7,22,22,5,22,21,8,20,18,4,16,6,15,8,4,26,22,18,18,24,7,7,15,14,16,22,21,10,4,22,19,26,7,16,16,5,16,5,23,9,11,5,5,20,5,4,4,14,16,4,8,17,26,7,19,19,12,16,10,23,24,12,6,5,11,10,23,13,11,24,5,9,14,6,19,8,17,35,16,19,19,9,21,26,29,7,6,4,16,22,4,27,19,7,24,16,21,39,25,7,5,23,17,10,7,5,10,4,5,5,4,11,17,7,16,27,27,7,16,27,19,24,14,13,19,16,22,9,19,9,20,17,4,20,25,33,4,30,6,5,26,6,17,19,21,6,6,12,15,19,29,17,21,5,10,21,14,22,15,20,22,6,18,27,21,6,5,9,21,5,28,24,22,8,25,12,23,16,19,17,22,11,5,17,20,7,13,13,16,10,5,20,15,17,19,19,17,23,12,7,16,7,16,12,9,9,11,14,20,20,8,14,8,10,6,7,8,18,8,25,16,21,19,8,20,7,19,26,26,15,12,19,4,4,4,16,6,11,12,7,21,11,10,8,16,15,20,5,7,7,11,22,20,18,6,7,21,20,5,15,9,6,6,9,10,5,23,17,20,11,12,7,5,6,18,12,12,5,7,25,18,6,5,17,17,6,14,5,21,7,20,34,9,12,25,14,26,7,21,4,32,14,26,18,26,21,23,15,13,17,30,11,28,17,19,6,32,7,15,10,11,26,12,22,4,24,6,5,5,4,7,5,12,28,21,17,20,16,4,21,17,12,20,23,30,15,12,17,20,22,21,15,22,13,12,5,9,8,4,8,20,28,20,20,21,15,12,22,6,20,14,5,4,26,8,20,13,11,17,7,31,4,16,21,6,16,12,6,24,23,21,26,4,23,23,16,14,26,19,13,4,20,30,8,9,9,21,20,9,24,4,28,22,17,24,25,21,24,23,20,8,8,8,19,21,6,19,12,17,29,8,18,21,20,25,9,1,4,8,18,6,22,19,6,5,4,5,7,12,24,22,17,14,17,6,15,9,13,23,26,24,10,21,5,10,28,25,21,6,17,20,22,5,7,16,27,22,18,4,19,21,7,18,7,18,8,20,24,17,6,22,5,6,18,20,7,6,28,14,27,5,20,9,14,11,11,17,19,19,23,11,11,13,20,25,6,18,31,5,22,6,26,7,17,4,17,13,17,17,4,16,7,17,16,10,12,23,18,23,9,28,11,9,7,4,7,22,19,15,7,24,20,7,18,25,4,24,25,20,25,17,11,18,11,16,13,16,18,18,4,6,10,22,19,31,10,18,14,15,20,14,25,5,19,7,9,26,16,19,19,16,21,20,7,25,5,19,8,17,30,26,23,22,20,12,18,9,19,20,13,6,16,6,9,12,7,5,12,16,22,8,16,26,20,7,13,17,21,23,10,4,24,18,11,10,18,24,24,19,21,9,28,4,0,6,29,23,16,19,26,27,7,17,4,22,24,8,4,23,9,25,16,20,12,22,7,16,21,23,20,10,4,21,8,18,7,9,15,20,11,5,8,10,6,25,6,7,21,7,7,5,32,22,20,16,5,19,13,11,9,6,21,18,8,16,14,5,10,7,27,18,25,20,4,25,7,5,21,15,12,22,17,28,26,20,6,17,8,23,33,4,24,8,25,23,7,14,21,25,6,15,4,13,19,16,4,20,26,6,22,4,17,20,9,17,9,10,15,13,15,4,25,17,14,20,13,6,21,6,8,19,19,16,16,4,19,9,14,5,12,6,23,22,9,16,8,14,20,18,7,4,25,17,4,5,24,21,14,9,20,8,14,22,7,7,8,25,11,9,31,16,4,27,24,4,26,12,10,26,15,11,25,12,31,17,21,12,22,6,9,17,16,4,5,16,17,7,13,4,17,9,18,19,11,21,12,4,5,7,26,10,5,16,25,6,9,10,22,8,19,14,27,12,22,11,6,17,22,26,6,5,9,15,18,31,21,15,11,17,13,24,8,10,4,22,7,11,20,22,7,18,9,14,5,10,7,5,25,8,24,5,12,5,7,23,10,16,15,11,4,4,27,13,28,34,17,23,18,16,19,4,23,6,17,27,8,7,17,29,9,10,19,16,24,14,25,9,22,4,8,22,18,19,28,6,17,13,5,18,7,23,21,4,5,8,4,7,10,5,25,15,10,18,6,16,19,26,19,23,28,32,21,23,15,26,14,9,17,5,8,5,27,15,26,16,8,28,7,9,2,17,8,17,21,27,20,13,17,8,5,14,13,16,28,6,4,8,5,16,24,20,23,17,4,11,20,23,8,19,19,10,22,7,22,27,15,16,17,5,12,11,12,8,6,22,28,17,20,7,15,10,12,6,20,6,5,21,17,10,5,6,7,23,4,8,22,14,15,21,10,14,26,11,16,22,18,11,22,16,4,21,11,19,6,10,7,9,4,23,23,5,5,9,4,4,7,14,5,5,23,7,17,6,15,4,7,9,9,4,9,23,23,13,18,26,17,17,12,17,23,6,17,16,11,10,25,18,21,25,5,9,5,22,21,20,4,25,8,12,5,18,21,9,23,7,10,27,31,23,5,6,18,25,15,22,29,18,30,15,29,25,8,26,6,21,25,30,19,5,20,32,22,14,26,30,24,27,18,11,26,22,7,12,26,7,7,7,19,22,7,7,0,32,17,9,15,7,26,17,27,11,8,17,5,16,19,28,9,27,20,4,24,7,6,8,20,4,7,16,17,14,32,6,17,4,11,26,9,21,4,21,8,16,23,23,23,25,26,8,18,5,20,15,8,18,8,17,4,17,15,8,27,6,12,12,17,25,10,13,5,15,7,19,12,15,21,6,20,24,20,6,7,5,25,24,19,17,21,6,19,15,6,24,4,14,11,31,8,9,19,24,13,12,21,17,10,7,26,13,9,4,16,26,19,22,7,5,5,18,25,5,21,4,16,21,20,10,9,22,9,6,6,15,22,23,17,24,13,24,9,6,6,22,9,25,34,13,18,22,21,34,7,14,5,16,6,23,11,10,4,13,5,16,25,27,22,21,19,4,4,6,21,4,11,5,25,28,9,10,25,11,11,32,13,20,12,22,23,6,26,8,18,17,7,14,25,4,25,19,24,17,9,16,5,27,28,7,5,5,21,18,26,19,23,17,5,10,19,8,25,23,23,4,24,23,15,19,16,5,19,8,10,9,10,7,21,20,7,16,26,8,26,5,17,6,10,9,10,21,17,5,16,12,13,7,19,7,10,24,25,17,14,20,7,29,24,9,7,26,7,14,5,14,8,15,21,26,29,10,23,6,27,25,10,17,17,6,21,14,12,20,5,22,22,19,4,16,7,10,23,25,5,15,16,24,10,9,12,31,12,12,9,25,14,16,33,11,18,7,15,19,23,15,6,18,6,28,4,21,14,20,32,7,7,23,18,14,4,15,14,20,29,9,5,17,22,7,26,7,25,5,5,21,20,26,26,18,6,10,23,6,17,5,16,25,15,20,20,7,28,11,12,5,15,27,18,7,5,15,9,14,14,28,7,5,5,21,22,24,18,31,23,6,8,8,4,6,22,22,26,30,5,20,14,19,24,7,15,9,8,23,6,20,24,4,6,22,5,5,23,9,19,20,22,9,9,9,18,19,23,16,17,5,18,28,10,26,15,9,10,8,26,22,28,21,21,16,5,17,16,12,25,18,11,6,4,11,9,21,8,22,4,16,12,10,16,19,10,27,16,6,21,16,18,18,12,5,4,10,4,9,7,23,17,12,23,17,11,24,27,25,4,6,8,6,4,9,6,17,19,28,8,4,5,26,10,25,23,9,29,12,15,29,22,6,24,18,29,7,6,21,12,9,28,14,22,5,21,15,8,5,7,27,26,25,10,17,7,32,15,7,17,9,5,19,20,5,20,21,19,5,7,9,17,30,27,18,23,8,4,21,4,13,22,26,5,19,15,20,10,5,27,6,20,7,7,8,19,13,21,23,24,6,7,19,26,7,27,22,4,9,4,10,23,7,24,18,18,23,4,5,20,8,17,15,15,5,10,11,5,7,12,7,28,19,5,11,16,8,7,16,11,15,5,8,13,5,22,12,26,13,20,23,6,8,20,14,18,5,18,27,23,17,22,19,17,19,15,12,9,21,7,9,5,28,23,24,17,7,25,5,4,25,22,10,8,17,20,10,8,26,19,23,6,15,20,10,5,6,22,18,7,19,1,4,7,17,4,17,18,21,11,9,22,22,26,19,7,9,27,4,4,18,22,19,23,11],"z":[1617,27,776,53,422,716,590,169,46,49,61,1102,310,46,1315,96,317,1782,133,316,1730,972,544,444,75,257,131,1672,30,318,120,302,1196,65,913,81,67,902,1395,53,22,31,984,122,55,1319,507,1693,72,1617,606,1957,1093,29,518,1438,612,884,606,1076,34,11,1274,653,1562,1253,863,661,65,13,1890,2209,18,692,165,16,79,318,778,56,151,372,1366,194,32,43,45,606,63,978,410,72,55,1169,1120,1097,29,187,910,145,850,969,1820,608,730,551,114,1724,577,81,33,660,80,30,43,1135,559,1923,90,661,463,632,725,279,1482,29,106,49,38,2077,1053,1385,871,312,877,68,22,211,127,459,460,429,89,1021,1381,306,18,162,1231,13,1706,121,1293,88,421,259,1117,89,36,57,148,93,119,1835,1318,45,10,978,191,57,1033,66,43,50,39,608,1581,68,80,1282,91,315,31,834,1600,1270,263,535,1295,1150,1240,38,35,53,293,65,1804,241,1101,264,541,1198,482,17,1615,285,1135,114,1047,714,185,1245,189,576,1103,1753,17,433,1588,22,19,1117,69,23,605,57,2088,1009,26,1627,1102,17,1690,1390,2126,1531,64,26,1112,210,1148,33,1101,322,44,17,31,845,275,25,275,90,1495,184,981,67,1635,60,231,48,99,770,957,818,925,908,102,576,317,461,37,70,930,859,1478,390,52,137,1919,68,25,1374,32,100,825,916,62,728,90,273,66,1213,107,1157,546,241,53,37,1508,235,32,102,401,28,680,30,20,1307,630,1283,65,37,38,160,23,71,1161,59,33,946,1442,62,1674,1538,139,1182,507,18,940,42,1190,44,16,835,1443,1029,1188,1277,49,99,1670,900,425,2153,636,57,25,494,693,1348,51,1932,990,30,254,29,1371,73,120,81,37,1377,51,17,22,324,211,19,41,492,1005,114,1173,1738,222,1127,209,787,1598,186,47,27,96,64,1040,163,343,446,41,70,272,46,490,79,915,1485,369,1073,1242,78,1378,1474,751,48,45,10,1435,2119,20,1795,1188,43,1008,1688,684,1082,1068,57,46,733,467,122,36,34,156,14,37,34,20,192,1379,48,428,868,1001,59,431,971,348,2059,265,270,1289,893,926,135,1495,120,1288,1743,15,542,890,1388,17,906,44,25,973,31,518,530,506,70,54,296,266,747,895,1053,735,20,205,2092,309,1722,900,682,663,43,1816,976,746,64,32,263,1445,46,1210,945,1157,137,730,216,943,347,1924,397,2089,138,42,777,1902,52,1651,358,1727,210,42,1291,442,1895,411,521,1168,493,260,46,319,96,1062,232,76,54,201,406,1919,662,41,1600,134,69,21,54,91,526,162,1008,240,795,1772,64,1957,78,1572,795,889,331,265,1065,11,21,17,369,38,177,279,62,1230,182,49,75,227,215,444,47,143,55,63,1867,611,473,55,81,2086,1478,49,382,69,67,30,63,79,42,819,1151,594,129,213,67,28,45,1193,268,266,37,37,882,373,35,42,326,404,95,939,33,981,36,1237,1825,103,191,1586,963,889,93,1587,16,1375,393,1477,1179,1005,1348,1307,486,2252,235,1724,235,1092,599,2008,48,1174,67,365,165,210,1095,223,597,22,1064,74,26,60,20,59,34,198,1048,574,1029,670,990,11,1241,576,172,664,928,2074,467,257,1146,689,2114,1346,1044,493,957,270,29,121,126,23,57,636,1004,734,614,1453,1621,187,1443,57,637,948,59,18,903,112,596,233,1511,1152,48,1185,21,965,869,16,1947,71,65,1380,1027,411,836,11,480,606,284,1472,1677,1335,334,18,1338,1099,97,63,109,902,1574,38,960,28,1215,1497,449,1064,1757,1440,1006,1226,1171,92,99,66,564,1427,86,833,242,1869,1364,183,450,758,1507,654,81,137,16,93,1158,43,1525,1230,29,25,24,38,57,47,813,629,377,1612,729,50,270,177,315,767,1071,1580,60,573,41,170,1167,1526,458,63,405,1027,1478,72,48,726,860,1497,495,20,570,769,55,1808,55,577,162,1957,1576,1616,36,1482,22,63,680,542,50,39,1043,385,1633,46,761,76,290,89,70,1066,2009,1345,764,195,53,368,544,1853,35,446,1180,41,727,40,1062,40,292,15,341,369,401,393,10,486,69,1727,995,45,182,989,1677,595,156,1226,128,86,68,16,55,2114,2018,263,55,564,1281,83,353,1003,14,902,1139,797,1638,401,84,928,203,858,409,282,1423,1449,28,67,199,1453,1253,1198,178,1990,976,1001,1919,350,1650,46,1228,95,62,1761,467,1149,2052,1127,1416,1113,125,966,32,833,46,960,1829,779,811,1120,725,1187,2302,66,1250,683,305,50,995,41,77,171,131,54,53,475,694,96,361,1079,578,64,149,231,728,1778,93,23,662,1198,81,92,491,1564,926,1910,411,94,1085,23,6,54,1150,1415,405,1908,2486,1315,45,1789,19,573,805,108,23,1493,89,936,343,528,1901,724,58,297,488,859,1574,69,17,2194,116,279,103,44,406,1363,236,39,137,79,63,1314,71,58,702,94,80,22,1211,1461,1968,1091,55,1366,313,62,92,72,655,704,54,178,1033,36,122,122,1123,497,691,1363,12,2440,56,37,968,270,278,1250,938,1178,1155,653,49,506,38,1650,1826,18,906,46,1477,766,47,467,2116,1298,22,396,20,967,357,1401,15,2087,981,17,523,23,1192,1958,133,777,73,121,424,868,756,12,937,1862,367,443,267,61,1323,63,64,562,312,1156,1104,16,2008,44,1574,63,426,96,635,757,125,332,135,330,384,1722,78,24,1168,957,15,57,1956,2013,326,77,1313,30,315,660,74,46,86,1103,61,145,684,411,27,1060,1459,18,1833,237,48,515,1020,1370,1026,301,684,499,1004,218,606,75,106,990,1097,16,34,1711,438,75,225,18,470,91,2525,504,235,1232,88,17,24,60,1020,155,51,1812,1461,96,160,139,908,183,1161,988,1072,174,758,195,31,1084,685,1672,43,49,133,363,545,1065,2047,296,140,530,152,1598,114,101,25,789,87,90,1215,811,61,405,72,189,40,63,52,39,805,130,1631,34,152,21,71,899,711,351,8,119,21,10,973,1536,992,1529,860,928,418,587,641,23,2279,49,1779,1804,48,83,1862,793,84,183,1383,999,825,246,610,76,1286,30,99,1192,362,2130,1229,38,485,434,44,496,78,2346,794,17,26,76,15,62,198,49,802,1736,57,2302,37,310,1305,832,401,1501,1189,1194,656,1540,311,820,274,57,448,44,102,61,1643,397,1121,473,50,1678,58,170,373,1196,101,316,1400,1424,844,311,1149,129,32,299,320,441,1178,85,24,112,30,410,800,1313,1483,1258,28,227,1555,861,103,634,1918,131,586,62,447,939,797,1143,415,30,187,251,237,91,35,493,971,371,396,68,322,222,177,45,1899,22,41,1440,325,95,42,72,20,789,15,59,908,417,289,769,188,354,727,146,319,794,454,152,797,1690,20,1428,209,545,96,222,61,44,8,817,1371,42,60,78,23,19,61,449,37,38,1131,34,294,48,289,30,125,156,70,22,65,1930,2283,310,1319,879,1244,690,209,1134,1507,57,529,1091,180,77,2349,534,556,637,46,44,47,795,586,1574,13,1471,65,175,40,1078,1115,134,1540,87,217,1034,1180,819,32,46,1159,860,1631,444,1910,380,736,269,2524,612,75,1045,55,1064,1443,1366,436,38,436,747,1396,928,642,1048,943,907,1198,120,1658,1638,73,275,854,78,106,106,1923,1662,84,48,5,733,1234,112,1658,55,1513,271,904,120,103,269,22,1156,1086,1128,72,964,1383,18,793,100,80,167,677,23,92,641,1106,1033,1174,25,240,20,102,1034,156,458,17,792,92,407,1269,841,633,1019,1315,54,2525,52,436,268,147,1798,137,507,15,1105,974,66,778,49,169,299,221,1682,155,892,58,311,95,424,47,339,1410,35,562,2352,1334,32,41,39,1141,710,401,506,1538,57,1722,270,31,882,16,311,227,1152,122,77,1289,561,211,233,1092,601,54,78,1095,244,116,15,425,734,638,894,41,31,34,1750,612,29,823,20,396,813,1179,138,55,1511,117,32,67,335,825,694,1792,496,231,902,68,71,76,772,47,1586,1529,265,1603,2091,981,1533,82,244,36,404,71,1572,122,41,10,304,37,1754,815,1001,1157,688,1319,21,31,67,708,25,746,32,1695,1080,72,57,660,103,153,1220,279,745,264,2157,1526,69,1597,64,497,1067,31,211,877,11,2034,1991,1686,324,69,252,54,964,1327,68,60,55,1456,563,1504,564,1125,358,42,184,460,145,1376,792,1382,28,1033,882,2231,1101,369,48,467,44,38,81,167,45,664,622,48,461,1073,55,976,20,1149,43,224,101,209,968,1165,24,252,165,269,102,461,47,144,1338,917,318,253,502,119,1250,969,76,62,1382,84,255,43,398,76,902,823,1179,1921,185,433,34,907,416,44,1588,1797,40,1143,914,1502,1338,27,763,1331,525,9,953,68,131,882,1149,37,388,1428,1633,173,99,215,929,219,174,92,1052,275,587,1662,159,1024,35,1042,395,557,9,69,486,34,1365,13,1401,1574,1323,1217,103,84,1536,581,928,14,424,960,1555,656,55,20,367,2257,42,1049,48,685,39,43,835,1515,1039,1596,1828,34,149,946,28,450,46,1702,820,382,2126,2069,72,732,801,236,80,1685,277,507,68,49,397,39,306,976,1028,70,46,51,1263,1490,1564,1815,1198,416,85,47,137,21,70,909,1676,1735,1191,24,1182,1280,1336,749,91,414,44,63,901,38,455,775,15,38,1512,43,42,1089,162,1272,1173,816,74,134,157,561,1336,2092,463,1130,20,377,1544,138,1012,1766,45,161,144,1734,823,1138,1429,1371,982,44,318,434,198,359,576,88,63,27,193,101,2211,140,415,22,263,393,147,443,1958,174,2217,1691,25,809,895,484,413,331,25,13,115,16,92,52,1464,441,283,747,1870,165,1941,839,1655,23,66,37,21,22,78,79,265,1175,1305,46,25,46,874,100,930,868,45,1260,414,392,1109,639,79,1665,692,1392,44,50,467,223,158,1211,1575,1208,43,527,932,71,54,129,1615,1034,1088,137,1244,41,1174,993,100,1680,98,48,519,715,37,653,1455,602,26,102,176,433,1366,1001,961,1518,56,22,400,26,350,1216,874,40,1367,264,2053,1376,45,976,51,1264,75,57,140,497,266,2006,615,1021,22,133,437,1013,57,1038,1573,28,123,13,132,732,66,1631,608,1822,1282,8,43,1931,103,1169,395,258,25,185,180,22,44,94,81,798,1334,38,61,300,88,54,1782,68,258,31,66,174,28,731,274,1701,312,414,1565,29,94,1338,1092,1479,54,568,1199,873,1213,731,2043,1893,424,575,257,56,542,125,69,40,1016,907,1566,1169,46,1644,59,17,1853,1528,35,141,546,500,61,106,704,424,849,85,199,1147,54,43,53,521,1623,45,1435,32,16,52,1234,15,415,470,1438,53,51,679,586,653,468,50,84,1049,22,30,1341,444,1241,843,172],"type":"scatter3d"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"scene":{"domain":{"x":[0.0,1.0],"y":[0.0,1.0]},"xaxis":{"title":{"text":"Income"}},"yaxis":{"title":{"text":"Num_Purchase"}},"zaxis":{"title":{"text":"Spending"}}},"coloraxis":{"colorbar":{"title":{"text":"Cluster"}},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"legend":{"tracegroupgap":0},"margin":{"t":60},"height":800,"width":800},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('cd62a634-25f6-485e-aaa3-ae20fa188d75');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>



<div>                            <div id="446bd847-0b26-4204-8e88-863ab442633a" class="plotly-graph-div" style="height:800px; width:800px;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("446bd847-0b26-4204-8e88-863ab442633a")) {                    Plotly.newPlot(                        "446bd847-0b26-4204-8e88-863ab442633a",                        [{"hovertemplate":"x1=%{x}<br>x2=%{y}<br>Cluster=%{marker.color}<extra></extra>","legendgroup":"","marker":{"color":[2,1,2,1,2,2,2,1,1,1,1,2,1,1,3,1,1,2,1,1,2,2,2,1,1,1,1,2,1,1,1,1,2,1,2,1,1,2,2,1,1,1,2,1,1,2,2,2,1,3,2,3,2,1,2,3,2,2,2,2,1,1,3,2,2,2,2,2,1,1,2,3,1,2,1,1,1,1,2,1,1,1,2,1,1,1,1,1,1,2,1,1,1,2,2,2,1,1,2,1,2,2,3,2,3,2,1,3,1,1,1,2,1,1,1,3,2,2,1,2,2,2,2,1,2,1,1,1,1,2,2,2,2,1,2,1,1,1,1,1,2,1,1,2,3,1,1,1,2,1,2,1,2,1,1,1,2,1,1,1,1,1,1,2,3,1,1,2,1,1,2,1,1,1,1,2,2,1,1,2,1,1,1,2,2,2,1,2,2,2,3,1,1,1,1,1,3,1,2,1,2,2,1,1,3,1,2,1,2,2,1,2,1,1,2,2,1,1,2,1,1,2,1,1,2,1,2,2,1,3,2,1,3,2,2,2,1,1,3,1,2,1,2,1,1,1,1,2,1,1,1,1,2,1,2,1,2,1,1,1,1,2,2,2,2,2,1,1,1,2,1,1,2,2,2,1,1,1,2,1,1,2,1,1,2,2,1,2,1,1,1,2,1,2,2,1,1,1,2,1,1,1,2,1,2,1,1,2,2,2,1,1,1,1,1,1,2,1,1,2,3,1,3,2,1,2,2,1,2,1,2,1,1,2,3,2,2,2,1,1,2,2,1,2,2,1,1,2,2,2,1,2,2,1,1,1,2,1,1,1,1,2,1,1,1,1,1,1,1,1,2,1,2,2,1,2,1,2,2,1,1,1,1,1,2,1,1,2,1,1,1,1,2,1,2,2,1,2,2,1,3,3,2,1,1,1,3,2,1,3,2,1,2,3,2,2,2,1,1,2,1,1,1,1,1,1,1,1,1,1,3,1,1,2,2,1,2,2,1,3,1,1,2,2,2,1,2,1,2,3,1,2,2,2,1,2,1,1,2,1,1,2,1,1,1,1,1,2,2,2,2,1,1,2,1,2,2,2,2,1,2,2,2,1,1,1,2,1,2,2,2,1,2,1,2,1,2,1,2,1,1,2,3,1,2,1,3,1,1,2,1,2,1,1,2,2,1,1,1,1,2,1,1,1,1,1,3,2,1,2,1,1,1,1,1,2,1,2,1,2,2,1,3,1,2,2,2,1,1,2,1,1,1,1,1,1,1,1,3,1,1,1,1,1,1,1,1,1,1,3,2,1,1,1,3,2,1,1,1,1,1,1,1,1,2,2,2,1,1,1,1,1,2,1,1,1,1,2,1,1,1,1,1,1,2,1,2,1,2,2,1,1,2,2,2,1,2,1,2,1,2,2,2,2,2,1,3,1,3,1,2,2,2,1,2,1,1,1,1,2,1,2,1,2,1,1,1,1,1,1,1,2,2,2,2,2,1,3,2,1,2,2,2,1,1,2,2,2,3,2,1,2,1,1,1,1,1,1,2,2,2,2,2,3,1,3,1,2,2,1,1,2,1,2,1,2,3,1,2,1,2,2,1,2,1,1,2,2,2,2,1,2,2,1,3,3,2,1,1,2,2,1,1,1,2,2,1,2,1,3,2,1,2,3,2,2,2,2,1,1,1,2,2,1,2,1,2,2,1,2,2,2,2,1,1,1,1,2,1,2,2,1,1,1,1,1,1,2,1,1,2,3,1,1,1,1,2,2,2,1,2,1,1,2,3,2,1,1,2,2,1,1,2,2,2,2,1,1,2,1,3,1,1,1,3,2,3,1,2,1,1,2,2,1,1,2,1,2,1,2,1,1,1,1,2,3,2,2,1,1,1,2,3,1,1,2,1,2,1,2,1,1,1,1,1,1,1,1,1,1,3,2,1,1,2,3,2,1,2,1,1,1,1,1,2,2,1,1,2,2,1,1,2,1,2,2,2,2,1,1,2,1,2,1,1,2,3,1,1,1,2,2,2,1,3,2,2,3,1,2,1,2,1,1,2,1,2,2,2,2,2,1,2,1,2,1,2,2,2,2,2,2,2,3,1,2,2,1,1,2,1,1,1,1,1,1,1,2,1,1,2,2,1,1,1,2,3,1,1,2,2,1,1,2,2,2,2,2,1,2,1,1,1,2,3,2,2,3,2,1,3,1,1,2,1,1,2,1,2,1,3,3,3,1,1,1,2,2,1,1,3,1,1,1,1,1,2,1,1,1,1,1,2,1,1,2,1,1,1,2,3,2,2,1,2,1,1,1,1,2,2,1,1,2,1,1,1,2,1,2,2,1,2,1,1,2,1,1,2,2,2,2,1,1,2,1,3,2,1,2,1,2,2,1,1,2,3,1,1,1,2,1,2,1,3,2,1,3,1,2,2,1,2,1,1,1,2,2,1,2,3,1,1,1,1,2,1,1,2,2,2,2,1,2,1,2,1,1,1,2,2,1,1,1,1,1,2,1,1,2,2,1,1,2,3,1,1,2,1,1,2,1,1,1,2,1,1,2,1,1,2,2,1,3,1,1,2,2,2,2,1,2,1,2,1,2,1,1,2,2,1,1,2,1,1,1,1,1,1,2,1,1,2,1,1,1,1,2,1,1,2,3,1,1,1,2,1,2,2,2,1,2,1,1,2,2,2,1,1,1,1,2,2,2,1,1,2,1,2,1,1,1,2,1,1,2,2,1,1,1,1,1,1,1,1,2,1,3,1,1,1,1,2,1,1,1,1,1,1,2,3,2,3,2,2,1,1,2,1,3,1,3,3,1,1,3,2,1,1,2,2,2,1,2,1,2,1,1,2,1,3,3,1,1,1,1,1,1,3,2,1,1,1,1,1,1,1,2,3,1,3,1,1,2,2,1,3,3,2,2,2,1,2,1,1,1,1,1,1,2,1,2,1,1,2,1,1,1,2,1,1,3,2,3,1,2,1,1,1,1,1,3,1,1,1,1,1,2,2,2,2,1,1,3,2,1,2,2,1,2,1,2,2,1,2,1,1,1,1,1,1,1,1,2,1,2,1,1,1,1,1,3,1,1,2,1,1,1,1,1,2,1,1,2,1,1,2,1,1,2,1,1,2,2,1,2,3,1,3,1,2,1,1,1,1,1,2,2,1,1,1,1,1,1,1,1,1,3,1,2,1,1,1,1,1,1,1,1,3,3,1,2,2,2,2,1,2,2,1,1,2,1,1,3,1,2,2,1,1,1,2,2,2,1,2,1,1,1,3,2,1,2,1,1,2,2,2,1,1,2,2,2,2,2,1,2,1,3,2,1,2,1,2,2,2,2,1,1,2,2,2,2,2,2,2,2,1,2,2,1,1,2,1,1,1,2,2,1,1,1,2,2,1,3,1,2,1,2,1,1,1,1,2,2,2,1,2,2,1,2,1,1,1,2,1,1,2,2,2,2,1,1,1,1,2,1,2,1,2,1,1,2,2,2,2,3,1,3,1,1,1,1,3,1,2,1,2,2,1,2,1,1,1,1,2,1,2,1,1,1,1,1,1,2,1,2,2,2,1,1,1,2,2,2,2,2,1,3,1,1,2,1,1,1,2,1,1,2,2,1,1,2,1,1,1,2,1,1,1,1,2,2,2,1,1,1,3,2,1,2,1,1,2,2,1,1,2,1,1,1,1,2,2,3,2,1,2,1,1,1,2,1,2,3,1,2,3,2,2,1,1,1,1,1,2,1,1,1,1,1,2,3,2,2,2,2,1,1,1,2,1,2,1,2,2,1,1,2,1,1,2,1,2,1,3,3,1,2,1,1,2,1,1,2,1,3,3,2,1,1,1,1,2,2,1,1,1,2,2,2,2,2,1,1,1,1,1,2,2,2,1,2,2,3,2,1,1,1,1,1,1,1,1,2,2,1,1,2,1,2,1,2,1,1,1,1,2,2,1,1,1,1,1,2,1,1,2,2,1,1,2,1,2,2,1,1,2,1,1,1,1,1,2,2,2,2,1,1,1,2,1,1,2,2,1,2,2,2,2,1,2,2,2,1,2,1,1,2,2,1,1,2,2,1,1,1,2,1,1,1,2,1,2,2,1,3,1,2,2,3,1,1,2,1,3,1,2,2,2,3,1,1,2,2,2,1,1,2,3,2,1,1,1,3,1,2,1,2,1,1,1,2,2,2,2,1,1,2,1,1,1,3,2,1,2,3,1,2,2,1,1,2,2,2,1,1,2,1,1,2,2,1,1,1,2,2,2,3,2,3,1,1,1,1,1,2,3,2,2,1,2,2,2,2,1,2,1,1,2,1,1,2,1,1,3,1,1,2,1,2,2,2,1,1,1,2,2,2,2,2,1,1,3,1,2,3,1,1,1,2,2,2,3,3,2,1,1,1,1,1,1,1,1,1,1,1,3,1,2,1,1,1,1,1,3,1,3,3,1,3,2,2,1,1,1,1,1,1,1,1,2,1,1,2,2,1,3,2,2,1,1,1,1,1,1,1,1,2,2,1,1,1,2,1,2,2,1,2,1,1,2,2,1,2,2,2,1,1,1,1,1,3,2,2,1,2,2,1,1,1,2,2,2,1,2,1,2,2,1,2,1,1,2,2,1,1,2,2,1,1,1,1,2,2,2,2,1,1,1,1,1,3,2,1,2,1,3,2,1,2,1,3,1,1,1,2,1,3,2,2,1,1,1,2,1,2,2,1,1,1,1,2,1,3,2,3,2,1,1,3,1,2,1,1,1,1,1,1,1,1,1,2,3,1,1,1,1,1,2,1,1,1,1,1,1,2,1,2,1,1,2,1,1,2,3,3,1,2,3,2,2,2,3,2,1,1,1,1,1,1,1,1,2,2,2,2,1,2,1,1,3,3,1,1,1,1,1,1,2,1,2,1,1,2,1,1,1,2,2,1,2,1,1,1,2,1,1,2,3,1,1,2,2,2,2,1,1,2,1,1,2,2,2,2,1],"coloraxis":"coloraxis","symbol":"circle"},"mode":"markers","name":"","showlegend":false,"x":[2.017777994828221,-1.4583420382484722,0.9291394972202536,-1.8172857597771017,0.10873535024165497,0.7004533028147516,0.3326737764468466,-1.393151821210109,-1.4828768867960036,-2.401208548764697,-2.4492582316554397,0.6728321134444833,-0.23966077867011826,-2.272359909780444,3.207182470252155,-1.308610773477655,-0.9370500554578959,2.737876957403134,-1.5530051498892357,-1.0245988218750215,1.4966016843474481,0.8798622630105166,0.4767668367242597,-0.2664786406249307,-2.0087045058683697,-0.5152443624118171,-1.367472401398572,2.8347590992144474,-2.251624941613202,-1.0487398425101735,-0.9415101736898261,-0.06884222859639073,1.0450067045791174,-1.3422596985420998,1.4198197853225163,-1.8003149624295867,-1.4560329359166806,1.4115869825729337,1.7348939898088198,-2.0452395804869457,-2.2396481509218287,-2.342306173746781,1.0066960352134344,-1.742357072339634,-1.6211115098838151,1.9401149014672354,0.2431652092187855,2.5345975234389164,-1.6261607961906652,3.14843776461831,0.7658720787535349,3.4914418933271194,1.5723147040730492,-1.9412078478958357,0.13779522218066473,2.800169977017534,0.7112708225711974,0.9762224927959953,0.5297266356878451,0.8493065032384686,-1.6837907304907622,-1.9740271048572922,3.716374362662127,0.5437639991740353,1.6215048340917892,1.2578856583248612,1.446730034162056,1.1129554536249633,-1.76154926801773,-1.7674489439104017,2.3204466017698673,3.218476156187162,-2.1293348280936524,1.197277475533117,-1.3589336152326974,-2.1318030199021263,-1.8063912917570233,-1.0487398425101735,1.5936105187028884,-1.808509549943692,-0.7578115942008837,0.8542168499705277,1.9399397477062295,-0.8237388095510926,-1.6252922238569791,-1.7709829192686692,-1.8692628348690334,0.5908580907657758,-1.2126662499804681,1.9505021745709032,-0.3199833680717826,-1.3092504402367857,-1.67476769851446,2.076958457136068,0.763916493823581,1.9537524801021908,-2.0969199079157863,-0.7612673144898502,0.8882314632023374,-1.349062577620165,0.9849761164887185,1.464057283530012,2.310131593150992,0.7042912831849194,2.672323707325315,-0.01127056683293347,-1.1239345862973142,2.6804243556375758,-0.24988247208573464,-1.7503195848583277,-2.0283357384078697,0.47943344159808077,-1.189742961725648,-2.162229755805676,-1.9151957720564345,3.988435478131015,0.38270495805405247,2.2634087887648384,-1.0527620683999392,1.1075970901447907,0.27136990279469975,0.6576362605773092,0.7940197066647064,-0.6701685327000785,1.8825503287880334,-2.4255410520836396,-1.5121786566284137,-0.9562287651684102,-1.6284832177665498,2.7849856704815723,1.1274679650588615,2.0213881746452986,1.203540077324434,-0.9008274094688156,0.3077923936698368,-1.6304975840256992,-1.8036955936123824,-0.8887996338195422,-1.1427348865319282,-0.031304554882319355,0.7196621900473372,-0.33463577785708687,-1.3820460872373839,1.2194761256519824,2.5776526521945122,-0.49066160637225625,-1.2372294107226094,-0.8900761218622015,1.3851471453610404,-2.0235213606759275,2.1801696113401166,-1.6401605216907202,1.416959404664962,-1.2731842959479276,0.5310167061741304,-0.2969615631839021,0.7225435501784531,-1.730262675262623,-1.7892708098459034,-1.8265803066044177,-1.1007271059437829,-1.0712880147456574,-0.7741807525310425,2.6348756192887834,3.16968497482412,-1.5709898376947284,-1.8219285245907082,1.9505021745709032,-0.8931513389412676,-2.019716030274682,1.2489367084503737,-1.8393963053483473,-1.919795632313328,-2.059257694412523,-1.749162031170923,0.29557620938692114,1.9569570720769396,-1.3673870996807787,-1.6624452748302097,1.7582922510174877,-1.459405532759787,-0.3834164574323799,-1.8493789947600525,1.0677886256844458,2.0381271709877042,1.5229181768958113,0.14419455686185725,0.8799048823084243,1.3298380919242294,1.614889876144763,4.02495866438888,-1.5982576800939556,-1.551669036568774,-1.5635816145535462,-0.40047050980451915,-1.4632771576635222,3.0111479963392105,-0.6539703501646486,1.335900333932806,-0.4069809902623082,0.371222032244227,1.0396701378749111,-0.2107338414728955,-1.4489733109266352,3.0702057216646605,-0.8560594752063851,1.6862304230124328,-1.338562001280186,0.8691673174723051,1.0964867810751011,-1.2468045455159802,1.9126420968963602,-1.0136164490068575,-0.13110986748678305,0.5062927953920421,1.8002852423139764,-1.9197430784555636,0.20947166376516618,1.8741193300691343,-1.802563112493292,-2.358169682226178,1.5587577007312714,-1.918952735983905,-2.1271584631144984,0.8469567651799608,-2.114826295218084,2.6630609672876355,0.6357810032443189,-2.1468959008899415,2.612532328102736,1.6358668870953372,-1.8538303303264132,3.1934221037425226,1.5989095699156024,2.029314385417845,1.7431928401945247,-1.7556551737247783,-1.8913674817687527,4.4997391043929005,-1.4701991398209908,1.7263536654507452,-1.8594471055722093,1.2616771655441072,-0.512677114929657,-1.899747121127945,-1.7507400240089546,-1.5150541845021241,1.2138031103142197,-0.6729017021493706,-1.8227219385921174,-0.16897920403713043,-1.275675604649204,2.075751067405058,-0.4316110647324491,0.6165534313686281,-1.9245813402177478,1.58381805152239,-1.5399675464606586,-0.45000830228081623,-1.6164310763764087,-1.5289027164129745,0.91917206737432,1.3270393689652848,0.5251320459188509,1.5107535785283506,0.9167516885824911,-1.6927722230253537,-0.13110986748678305,-0.9370500554578959,0.06849526509741753,-1.4345881831744605,-1.763920976575145,1.306383841425422,1.2736605963355734,1.8792084890267247,-0.7269171690049047,-1.4905196816913993,-1.294137815603892,2.5386701902471644,-1.9993724253837446,-2.0346012549428996,1.974134239792609,-1.8098076359581077,-1.426000499782695,0.2879959776356442,1.5527331922895515,-1.3793643117727845,0.5469087439441677,-1.3303343641880774,-0.9262721160519286,-1.740044046760714,2.1017544575326674,-1.4448808287268047,1.6186646968321776,0.6898612922205275,-1.0147940634111465,-1.9850500992036768,-2.00728125142154,1.736755351693185,-0.37864062284049577,-1.8470033432066222,-1.4108552469724307,0.03456041187517915,-1.8659072895657192,0.6589138204553753,-1.904805976049363,-1.5823012386238833,1.373952767130462,0.17381999414502367,0.9567698558875957,-2.0490060403451515,-2.0192826309279366,-1.3025838354605743,-1.3573610843117279,-2.18397595099061,-1.7419952538148438,1.3971402679313258,-1.032388428566477,-2.02591275554842,0.9095252701590486,3.248102834717631,-1.9056010910071344,2.902272076692054,1.5708616020513944,-1.3472056844612734,1.5704225103803884,0.14767715496232095,-1.3491284981565765,1.0357780299163306,-1.4980085250819204,1.6412166034060258,-1.727654476856596,-1.57195937092106,1.3098997087623228,2.393070008314244,1.295042136460095,1.1281027361914946,1.2993563934417778,-1.5727429937114787,-0.828280842941924,1.284586871814896,0.4005082545636536,-0.397179752456488,2.8604231549633785,0.5486743007399201,-1.6309235538837659,-1.8227219385921174,0.20019848241773827,0.27275010025182855,2.0160491082022975,-1.7559490318934354,1.6934972396719319,0.2926574642377405,-1.8039007348393536,-0.5192445423068588,-1.5877218680512586,1.7995560623511255,-1.0377800083061368,-1.0387829783877174,-1.3432707643366946,-1.9396402378086373,1.2924680202794288,-1.912117891537083,-1.9392059516417945,-2.292532385506337,-1.135093282217269,-0.8887996338195422,-1.8698423634963428,-2.2605678251973504,0.41517981555886957,1.2668103449126498,-1.5066169319991842,1.486163512640764,2.573241159030822,-0.4428637836624422,0.992993931822729,-0.8929551525617412,1.0527940639405418,1.831161882060787,-0.9687010589093079,-1.8771420691564895,-2.1291486500631516,-1.5145169061032988,-1.5465403314260966,1.4102142253952534,-0.9609948149870238,-1.2388002734375463,1.1540560802209237,-1.4689907465433516,-1.3399305760545297,-0.17321203857132075,-1.861602567033213,-0.03394769916919564,-1.4938063160507637,0.32223294942037795,2.7888436156963174,-0.4436921742523655,1.1654388397932014,1.345926050563875,-1.5278996266487357,3.2369814791511664,3.57501411372336,0.8644115990526491,-1.6063976537459594,-1.8692628348690334,-2.343212217385118,2.638401695385857,3.2078003682545777,-1.5385558633027336,3.994397004839968,2.0410027889837496,-2.322133396302914,1.079921614315402,3.694858260986083,0.07253703928577736,2.322324024189243,1.5708875155463164,-1.7924215156402208,-1.3869176952728877,0.9094865470263391,-0.16219274989830013,-1.4755421779250075,-2.4052261170202245,-1.9842962032000881,-0.8057669888813953,-2.349758665580345,-1.9396402378086373,-2.209765292979957,-2.207272795740609,-1.0624387556257984,2.239193396851745,-1.958625618553338,-0.15663089023257992,1.0622726510875309,1.157247047775879,-1.6268497059124936,0.24446692320007493,1.371081906912439,-0.46958096465103144,3.9528785025998956,-0.04394553187973577,-0.651173354306668,1.1479878251485407,1.168749292414428,1.1003044691738442,-1.043439709900198,2.075751067405058,-1.5142061665340598,1.212525718992294,2.8040519885894235,-1.9858935806352234,0.17110974624205036,0.7188989285864407,2.051307283708756,-1.4597421253459473,2.031620408117524,-1.651111634912237,-2.2080461490251335,1.2134661348458682,-1.5783472925144468,-0.00043829694901347096,0.3086233997638919,-0.12882797637146093,-1.6121775811471115,-1.6589183050815957,-0.41263614444507196,-0.7104243458095455,1.0703419995280548,2.1513237480836,0.7487422848216522,0.5870198556764592,-2.383246019763138,-1.4907585704077073,2.365244178197184,-0.6517839011726434,2.3845724672471547,0.2899819045164747,0.7300884958124652,0.5692569679020976,-1.200433090758914,1.8935020429251002,1.4263704628779004,0.7338853046630038,-1.9275116738843219,-1.468957155882058,-1.6136607937331116,1.3697850869745771,-1.3869176952728877,1.3394902887039046,1.1350162412948226,1.6095328743218655,-1.5196513912141782,0.8833371433835029,-0.6711782630240047,0.920290943822504,-0.033148575405024076,2.71782057651688,-0.42332393354757875,2.488441523893143,-1.276065449004462,-2.0451113272136134,0.6530374414481577,2.908066167075892,-1.7809463591197616,1.64155847087514,-0.7684633928370727,2.2172637923427496,-1.1840036078665628,-1.8664163413292065,1.5124639980429282,-0.352058378330702,1.4579303157508026,-0.14234469847143352,-0.07722936671237288,1.0568483338428831,0.8161522587749072,-0.33278622756427095,-1.8409454646573657,-0.47961327142724125,-1.3627509853223057,1.7598794243213844,-0.571740586493646,-1.825421579732774,-1.7837537090635749,-1.0270757271229591,-0.6549903050383085,3.407798530948058,0.29149497248807593,-1.8803175368850336,1.9814495546726556,-1.1823451782295706,-1.0972741806738662,-1.9692793233458141,-1.5441862595105968,-1.1601613004234552,0.19754075766486476,-1.825672621382383,2.063137918974964,-0.6566702306706212,0.5063897584790703,2.3316587516788516,-2.105319397291547,4.296001812136604,-1.6929469024507817,1.812330084856874,1.2822900447874706,1.1331627940674884,-0.4978131681813964,-0.6746298296166395,0.9338368041494574,-2.0576726870487256,-1.5638165582311738,-1.8175564100352983,-0.4946274878412297,-1.656040553937786,-1.182610127205389,-0.3386248938588563,-1.6048170261326427,2.6195152091337506,-0.8435692089822114,-2.038323530688662,-1.2480195718522558,-0.3755293623601665,-0.3232385726452892,-0.2664786406249307,-1.468196700440994,-1.7787576680881034,-1.5457438612487695,-1.5995434430358597,3.757634308538231,0.44707226381605125,0.35131120103680546,-2.014547445541945,-1.1363802098630282,3.8436899415802626,1.4788714327531083,-1.851957243782827,-0.15536213481163552,-1.5977638241285204,-2.011405483894756,-1.5654648795853687,-1.7513140620542018,-1.5404335116321164,-1.940132961150939,0.8731105311767802,1.0467721263383791,0.4224054856144269,-0.9375780243563566,-1.1422002013679493,-1.4560329359166806,-2.0568842848987456,-1.8506954118699903,1.7503985426355253,-1.3375902062653142,-0.8620196819665109,-2.2372720779677655,-2.00728125142154,1.0629689408208194,-0.3332325401083092,-1.7287188428341758,-1.305917197690009,-0.8486084736288676,-0.14460168369992546,-1.78158998422874,1.3850740456600439,-1.767066790200691,1.5283816001553596,-2.2439397362455584,1.6282576264685633,3.1609682886858006,-1.4959176725208767,-1.3604791772825071,1.8980584669358365,0.4709346744118065,1.1331627940674884,-0.9429521588412585,2.3978142494187797,-1.610387643256659,2.580241748408252,-0.31248476213527165,2.0844148839868626,1.540257031718064,1.2668103449126498,1.5885579550206215,1.335343062000537,-0.04889909221952501,3.0766241078715755,-0.5147136399927409,4.461087085098763,-0.6883010660963299,1.8879725509256242,0.36692069925060644,3.0514836467290105,-1.892582073393228,1.9392965336428425,-1.2404137981958785,-0.4490590207227986,-0.6641350925755223,-1.4701991398209908,1.606692835890688,-0.4783928943761777,0.4232025408938772,-2.292532385506337,1.0542186121915973,-1.9506726810869317,-2.361356220721442,-2.223136064854689,-1.63384229792761,-2.0262994997089274,-1.9712331651752295,-0.6711930166092054,1.3353279651989276,0.41269192753987205,0.7689268155897403,0.9125651660181302,0.2926574642377405,-2.1247524407771836,2.3808571665713885,0.4360878903162152,-0.42671123693505536,0.589330042217497,1.115923783847914,2.6881501492315105,-0.5737492702819765,-1.0301200323348494,0.8542953343416904,0.6440435536349556,3.0405805388906293,2.389024725841669,0.7201033204327275,0.933845134916743,0.28493138565272036,-0.9291143648777658,-2.400056133049099,-0.7556289818334035,-0.8192629424997088,-1.9008029976775451,-1.438153620399307,0.05308622114174179,2.1772182867634693,1.0137755379477202,0.7036075455582844,1.9316531456665387,3.3269609516556105,-1.0333088885335924,2.393070008314244,-1.7108129392851685,-0.018474515818354593,0.8588551983478642,-1.032388428566477,-2.157620519300705,0.9563381402687237,-1.5876649715883981,0.34346672069943673,-0.6868532668533093,1.0786637302532307,2.1843910325187816,-1.9105873496877288,1.6519912511642134,-1.6304749236362488,0.8535132829715346,0.6877055992049836,-2.189199029902791,1.7120086782354853,-1.2765115542639058,-1.4632771576635222,2.31249281962754,1.3909842294445933,0.19857660420524506,2.184121651461274,-1.8825858386832814,0.3102948350308967,0.7658720787535349,-0.5727148076261518,2.7253148137036933,3.643999149813364,1.2116564705388269,-0.05754438973437161,-2.3621255399922925,1.162299240805652,1.6452975731880164,-1.5572690278693657,-1.157604251178418,-1.379241837488321,1.706317966591556,2.3159726761454986,-2.0177752059713856,1.2939816753958178,-1.5470369003090363,2.362687335903151,1.989251611662337,-0.17948150014503084,1.0542186121915973,3.8870169116105204,1.6587707912126486,1.1449545645587684,1.620743840087833,0.8894941858525927,-1.4113183226494543,-1.4529828824235864,-1.5712171619955795,0.7920396027915771,1.7196572502450933,-1.0327961672627923,0.9402334184960928,-0.7867122911459445,1.779599319720283,1.9154101528538368,-1.1613269606986527,0.16669434974812722,0.9239919980409821,1.7447909120832663,0.7022687227780446,-1.539475590255796,-2.7332079371857607,-1.7328536247397088,-1.4077929277368249,1.1564213859588923,-1.3173420137273568,1.7211177969412514,1.2187379012209518,-1.721427500481631,-2.206228911880546,-2.014607914056741,-1.9041683406295693,-2.044991141967629,-1.8026364971853734,1.1594022041031866,0.5037292224159864,-0.530281921772757,1.0142893035939753,2.403095976416749,-1.8356465161904074,-0.4904218966462355,-1.1090170035247078,-0.3647153128052234,1.0245303451070524,1.2443357793380667,1.9522728592750083,-1.868097366524598,0.4808815579129082,-1.4959069256994113,-1.3557573398056082,2.064307009785882,2.8060698987132415,0.051204977595815035,-1.147649838858603,-0.12229817162656169,1.1626201327185257,1.683210245113422,-1.9963159138020072,-1.6501664412401666,0.5847167746788455,0.9175292836010123,2.209874735723307,0.3121687357107478,-2.010352764869907,-0.6939823766209252,0.7664043833836436,-1.9052144378642852,2.963392954609996,-2.0037402952251644,0.9132686844352275,-1.825672621382383,4.296001812136604,2.0650881752732686,2.783216618445052,-1.6733100106050713,1.9106133267061378,-2.315936699694176,-1.1855221144226087,0.034441775152568896,0.24445765128082328,-1.9276873781463315,-1.734159146680523,1.6489385495846662,-0.4961538759942521,2.4100460316292263,-1.5200237319213465,0.8314009039278341,-1.5335574922499438,-0.8346218566725893,-1.3820460872373839,-1.7886412424534572,1.2192791751150096,3.725815352717678,1.0679784028488113,0.6230374801212264,-0.8657142790336743,-1.9191818369049438,-0.5884888913990881,0.4767668367242597,3.2569669230445113,-1.480307501603016,0.8860279207363775,1.6937273566575812,-1.7173464896369346,0.5223756809329372,-1.672192458668778,2.287440078819351,-1.9872238294075446,-0.5098960577372225,-2.6279840383893647,-0.6195082204548912,-0.5783793039336065,-0.056573000766300036,-0.4446944617764453,-2.3170598045651896,-0.16861989966763943,-1.7671303302452375,2.2748977200954146,0.9351535734339217,-1.8204771750873046,-1.0101138692690563,0.9951790533228504,2.9555594659598,0.1355688757663742,-1.302724580530567,2.217682848212558,-1.7944937844292583,-1.0106460566949005,-1.6297540045268308,-1.8599285410097552,-1.238709794124168,3.0405805388906293,2.242246134429335,-0.6296646114542689,-1.4722642723589394,0.3941100407774967,1.2456835019881412,-1.341761808562991,-0.20587228376258537,1.546215530146779,-2.1188378643667143,1.4115869825729337,1.1118359854689281,1.0619239747941753,2.1907029355401244,-0.6048782197976152,-1.711500421103609,1.125412322013929,-0.7486771938095576,0.542269636699763,-0.340812907833405,-0.20997100691302284,1.6917016426432772,2.509446177524274,-1.610063716022091,-1.37987607099539,-1.0364795640271363,1.8154087918863302,1.3365354418861326,1.873595914016491,-0.9483281016333465,3.2600265511121598,0.41294526010261645,0.7752813221194302,3.407798530948058,-0.710588912557122,2.483473283799327,-0.7592539384002904,1.2402361703048599,-1.1185761317994833,-1.5351117830930494,2.8770661558694797,-0.9442985526263923,1.6820698988565244,2.443545779549565,0.992993931822729,1.44885096064366,1.3909520696666808,-1.3043915920282585,1.5823852513340144,-2.071726815710566,0.9402334184960928,-1.6593970330600234,0.837159547908638,2.803711447005257,0.8389438982393846,0.8062186564351854,1.1430360746570025,0.403902204053214,1.7401770837544421,4.281972436450872,-2.04656045301119,1.4708963528159336,0.4134875242082693,-0.1447425464099111,-1.6097326259140985,0.6997789334574092,-1.9302965390362814,-0.6755063264607976,-0.5981628033127797,-1.386690055942895,-1.5067535213301797,-1.8026852277297807,0.3730660963500358,0.44757073543955694,-0.9234972572063304,-0.2586201683814144,1.5280582832301883,0.17368807092865032,-1.1172226024328686,-1.0305769832861977,-0.4205184205339466,0.4397707627241575,3.708959903424987,-1.4644778449417817,-1.5280821010485368,0.2974307071358433,1.4567483054728805,-1.217622109212177,-1.9330315914772525,0.12624606720100048,2.1122231079596587,1.592451506404431,2.532251272206663,0.19857660420524506,-1.472365346060795,1.5856818163359143,-1.5859469098001555,-2.976553899596475,-1.7721137434502297,1.614889876144763,3.045047911223655,0.07566625552249369,2.140944080399799,4.506353318199087,2.0567136378458355,-1.663510911447407,3.8794271740750665,-2.166368059306592,0.4044379131164459,0.9707759282501536,-0.7661008324309215,-1.9008029976775451,2.2209894442923637,-1.730262675262623,1.2352469049675716,-0.4902902281490109,1.93241594329104,3.1393687540351456,1.6815295821560114,-1.9911361791809978,-0.7748717020725923,0.480098507892778,0.9836283477771468,2.3159726761454986,-1.4759998609110754,-1.8561134416248004,3.5612990549911854,-1.1641319622313224,-0.8346761871505419,-1.6426316316586285,-1.453953686289535,-0.1474716685343621,1.7764350128807944,-0.7167748469905619,-1.8845248423475576,-1.908250435683512,-1.4384575373735424,-1.895877480564721,1.8992872701248935,-1.7533727385461721,-2.125164002786856,0.45648630441573834,-1.0360563738342874,-2.165146320162525,-1.9039242665069003,1.9971783434777013,2.478175941855352,2.894964567099416,0.8115626006304534,-1.9021775215799674,1.9399397477062295,-1.1960045877638792,-1.3243731355713635,-1.1919502890285292,-0.9011654111468047,0.18515595820419475,0.7621717694452157,-1.4910295826938713,-1.64843961512018,1.1947531617322649,-1.4446866583327824,-1.4450705347904078,-1.637013802911251,1.805545368318191,-0.11765189413368494,0.7848544230712047,1.7764350128807944,-1.6920664506724972,2.7049374645711324,-1.521462166812396,-2.023364830310705,1.0836781568422398,-0.7198151651876042,-0.6319337714076144,1.4606200500592739,0.706005532450574,1.930083034903217,1.8349959778702758,0.344152354628918,-1.3349544454056195,0.24327428859533043,-2.089183264544617,3.3398010673573792,2.5365867458609808,-2.2352613178839373,1.1339503093973708,-1.6593970330600234,1.8380223995318783,0.7471552135361964,-1.5462144015171917,-0.49466902386721956,3.014069208936451,2.859878257833519,-1.9749717980071777,-0.404428549300294,-1.7406115643646602,0.4282977764968471,-0.14221924021715118,1.7664596090051647,-1.7594236935873209,2.955334950926495,1.0032161766528105,-2.0519618526268397,2.9898272534716823,-1.6650362307450155,1.4322186794770946,2.7312925264222936,-1.3437240992592803,0.6530374414481577,-1.4165056885433462,-0.8121415407741374,-0.06692199623982802,0.690586453523537,0.7492532186394898,-2.1017114991379002,0.9609691230931092,3.520813696395438,-0.3889487134516544,-0.3248635278879242,-0.5779722730020657,-1.856075156183884,1.941171856289962,-1.3051437247228264,-2.105319397291547,0.80851047336226,0.22047143045220294,1.085696473472453,1.923748072778666,-1.3901068972084376,3.0514836467290105,-2.031440046715091,1.9721472919925187,-1.070986726108988,-0.1837544111148248,-2.0176238511590827,0.2836956002286322,0.9512633620355924,-1.5510436327681054,-0.3440274246825781,-1.163597130134747,-0.6207848905328668,0.025638645967223074,2.0079437672766574,-1.3270764906738937,-2.3803203052123605,1.7480797153388563,1.0967752133667446,-2.1425973633931044,-1.593807743765983,2.9937031939801564,3.6462288530162104,-0.5472640356900281,-1.460134309196236,2.118262210848152,-2.251624941613202,-0.5574153105091509,0.47943344159808077,-1.5883223219882696,-2.032808835647477,-1.6077418978821552,1.2891904748666716,-1.7487864552325134,-0.4966671605336183,1.0666565026760013,-0.11903754365443445,-1.356799752179179,1.2318496065002316,1.6856696171109415,-2.6251508640936305,3.3267377374679876,-0.6473457784788852,-1.7803511341722158,0.7263153222091393,0.36135440139906894,1.0190368007839758,1.0589521119750591,-0.8455844838530107,1.0666565026760013,0.4591595079150842,1.3833879230539516,-1.030132493423845,0.6017585890361451,-1.3434558393883949,-1.4710932950986164,1.0062876644161705,1.1586061647022,-2.066037180087085,-2.209765292979957,1.5424379232122856,0.5893854757162549,-1.941107091265787,-0.7712854305557022,-1.941869697728846,-0.08411542310857162,-1.6646652452574027,3.25773714384355,-0.08306535421284017,-1.008082130003253,1.6786207795454722,-1.8189076246784273,-2.113739727832799,-2.1613473231446063,-1.410918259244854,1.623330816966575,-1.4579143605605953,-1.5151700909107326,2.131507140045345,2.8710190177459736,-1.0756365910076322,-0.70939841362054,-1.1531908860238318,0.9167516885824911,-1.1613269606986527,1.496087513615588,0.8260453868605913,1.287191046706432,-0.6907243353199214,0.6242359540248327,-0.8280000240909372,-1.5150541845021241,0.8183663511747836,0.7457658262613612,2.8347590992144474,-2.18047698386585,-1.281684233798043,-1.419389640076657,-0.2623209307353592,0.35359421785980805,2.40024830462802,2.5952297129655584,-0.3222269746835167,-0.744883670604807,0.1614617155325462,-1.2016718394666057,1.831161882060787,-1.5560683119261065,-2.1029900235425543,-2.1253050915509903,1.5374823683869578,-1.1168343833701475,-1.400538232221492,1.3661308527491642,1.1365282784937227,-1.2286393557130868,-0.06879911679708788,-1.420768513021895,-0.8148610319313624,-1.7282725169789699,-1.2685519895489796,-1.8389662638954944,-1.750338998120431,1.0456077669508683,-0.9912569811630041,3.425791599935384,-1.9842962032000881,-0.9905653639820576,-2.0751622141257795,-1.014195871262676,1.6912826799779341,-0.024135141813626238,-0.9674106788942718,-2.0424970377695693,-2.0562740264395774,-1.7526512006469732,-2.2922505670260427,1.7283883283763743,2.1986211724929885,1.875259330297639,3.439112458663757,0.8196237737348112,1.115923783847914,0.33347578970545255,0.5635896632644256,0.734232709881014,-1.402719074842232,3.374032302589007,-1.3349544454056195,2.8655954171822136,3.1116572991759512,-1.6953805257602064,-0.9381508050731544,3.520813696395438,1.3848040304589229,-1.904461748732594,-1.0605970023500204,1.7324449718971333,0.76790015340868,0.9828688701738382,-0.15140765103463025,0.662898527104336,-1.8042468164828014,1.717298458342698,-1.4484658339991325,-1.4010731083369001,1.7745367177950755,-0.33540377355337514,3.0163193681122125,2.3741735137131115,-1.677373337809027,-0.4549667953504827,-1.1100189157096525,-1.8353990906115658,0.836772978450404,-1.3297101676950347,4.365894221835222,0.4435802938205601,-2.4085798703495183,-2.1468959008899415,-1.363536378295193,-1.693107706194994,-1.2898555911744074,-0.6657594542356498,-1.8291759375499586,1.2394868539612989,2.370553803322174,-2.1824681956510843,4.281972436450872,-2.0598058457531914,-0.5693617304166393,1.6725068220853418,1.0379310796762045,-0.12424035114941413,2.755124883951712,3.5165807155743933,1.7586306543892276,0.6599515192797695,1.6080071325182876,-0.6129882287125342,1.3929691929041723,-0.349043781094083,-1.730274925651302,-0.2991783884601836,-1.5165134768918307,-1.4108552469724307,-1.7337762698522055,2.423203430548149,-0.4909749085106819,2.123826550981513,0.17102078447666172,-2.037476298627591,2.2864730476422,-1.9311673534096205,-1.1425175859339358,-2.4499322706298075,1.8124306264335568,-1.7562055824589322,0.0007926132432060991,2.2711768293395442,2.453386302092018,2.874754255270925,-0.5690503082898992,1.0171507244419433,-1.502423621900534,-1.6870519500027323,-0.45764165716041516,-0.22978436372046238,-0.4720885701285001,2.3088087151404264,-2.125589890543377,-2.1752622123463334,-1.5876649715883981,-2.035585302389115,-0.3199833680717826,1.2818874440629624,2.118262210848152,1.861921980819647,1.4635764722146305,-2.3331924157167365,-0.3309075727321324,2.288682610111286,1.0923917441588655,-1.0887310182152377,0.4107642874441775,2.111541258657085,-1.4326731699119166,0.4734151433993029,-1.7450047293708164,0.19676576279664285,0.9777299934163723,0.6054887111501611,1.4095058341225437,-0.32992191738987725,-1.7436685413658624,-1.0333088885335924,-0.3575446122020156,-0.5646219832448509,-1.1525558230266655,-1.7287188428341758,0.5551194546795337,1.1819666345544169,-0.3219883491667421,0.0978326945115711,-1.2526582794455752,0.0994622851611637,-1.0024066158640232,-0.700190432712915,-2.1115874575842035,3.2254881185508864,-1.9749717980071777,-1.5491072015264566,1.6587707912126486,-0.23751266211785507,-1.746736475888846,-1.7082113626692752,-1.5392585290231198,-1.8332634351482981,0.8739599296136789,-1.7721443535994312,-1.808643356151744,0.9167516885824911,-0.425243040196461,-0.22815224218868152,0.7664043833836436,-1.262961069362769,-0.47933773791638656,0.5072946729520005,-1.2552365853626208,-0.47961327142724125,0.8945802214508238,0.03844568640598163,-0.8705811888318291,0.9580406407732205,3.1934221037425226,-1.6845669173547828,2.538732920718509,-1.2625036376381233,0.2049848980869323,-2.0176238511590827,-0.8726980225728361,-1.6303336241717983,-1.8110012800456012,-1.9214477146282227,0.9475193716154082,1.7995560623511255,-1.3067073007963514,-2.223136064854689,-1.6860548563653983,-1.9068077812857462,-1.6821665389698452,-1.5588292930478238,-0.8988976020599652,-2.023364830310705,-1.9154141415098407,2.7080988497118876,-1.6867252242913884,0.18312224901438431,-1.4616305778267953,-0.22815224218868152,-2.2716771850942847,-1.398782576465947,-1.302724580530567,-1.5556579854533312,-1.8600299450945854,-1.4912956697512096,3.0911611232069673,3.5359358366659497,-0.571890947177032,1.4083111023680777,1.2492428511731017,1.0590666725398197,0.5190972966362365,-0.8882753951741159,1.1475737214239052,2.1935206143624972,-1.589084387368038,-0.5988244346346974,0.8115626006304534,-0.6770840952873451,-1.668712454740489,4.784208095276133,0.0019369683065041461,0.9224048738205811,0.708781755909574,-0.7592539384002904,-2.0245398129197016,-1.9883215753461019,0.4317077526896185,0.19958266898118485,2.3159726761454986,-1.7437721874903456,1.7456027298640302,-1.8977103700107125,-1.2278138225368758,-1.8202668553274202,2.7601376506470645,1.0790209732214617,-1.3241332745176915,1.6080071325182876,-0.9772260890766244,-1.1485570290588445,1.5744065938445133,1.6937273566575812,0.8731105311767802,-2.2305112033151486,-2.157733211290291,1.189865351831682,1.2465364047919338,1.536327597176755,0.8618360056593843,3.0617637722974282,-0.22082699414570112,1.588424722186595,-0.7124371372327105,5.0591001948307115,0.6828016607927623,-1.556344140717216,1.3601829241053893,-1.7846011048261365,1.4782192381992745,1.8320655036668194,2.0127788022641186,-0.010780491577084704,-1.9154141415098407,0.13958812868710382,1.184787887907118,1.761768686853168,0.7565061449096824,0.7595231251585648,1.5192850420127653,1.4825110520555458,0.9942308979502463,1.4567483054728805,-1.0387829783877174,2.1510005249880715,2.019328684954391,-1.6365119545822941,-0.6754835924501862,0.9234683761426721,-1.6929469024507817,-1.5414569573934946,-1.4404525190355888,2.6659935450365273,1.7891874220609543,-1.5646038782145721,-1.6063976537459594,-3.013289961745691,1.2897921693393715,1.3655059801363403,-1.6559734789866891,2.5553895958552135,-2.064920612426267,1.9581800655694632,0.07235955059280368,1.3945083285103321,-0.6600572981505081,-1.8573725814323512,-1.0918264998380567,-2.5075103662119624,1.2277335998197334,1.4747186106792582,1.4090620426072524,-1.5925896018811248,1.5986050540684813,1.2821479493525723,-1.468676947340468,0.6311290536117974,-1.426000499782695,-1.8284661243099805,-1.2166768838951134,0.6210436672483647,-1.8360145429574801,-1.0347115831564144,0.6000459788337897,0.7682180404975111,1.1947531617322649,1.9392965336428425,-1.7048994163762807,0.26621145803175766,-1.9817246956501058,-1.553099256064151,1.473235060335236,-1.302724580530567,0.051204977595815035,-2.1745513302509414,1.1615807879153797,-1.314925743675699,-0.3805467426709245,1.707494693070411,1.2233074644831297,0.8841843502373006,1.525850163582594,3.207182470252155,-1.4910295826938713,3.6364628240807595,-2.007592415849138,0.13958812868710382,-0.8364269190923229,-0.8610558401271116,2.711173767884964,-1.5196513912141782,0.11137601108089694,-1.9133884522432154,1.139938947557031,0.9959525446238584,-1.2978414872011583,1.9723361989400978,-0.9562287651684102,-0.695100989869576,-0.3969301387102669,-0.8306762936196422,2.544165452755092,-1.4579143605605953,0.567477824769264,-1.9770875683511573,-0.6129882287125342,-1.1625351258794783,0.22964292228798705,-1.8026364971853734,-0.4457723072418822,1.6551429387136078,-1.7057531792098275,0.19139342370148302,3.4893580793937784,1.7415153450749008,-1.5018690441162157,-1.8838820753936767,-1.31112068130477,1.5507834699910763,0.9121653924353981,0.14523748165371844,0.24327428859533043,1.5708616020513944,-2.324169980738663,2.25760493583501,-0.9463640625461408,-1.470653314522843,0.8060680850514831,-2.0776809571176815,-0.365721367727722,-0.3309075727321324,1.4697322832082862,-0.713107904767781,-1.9338011214484194,1.1479878251485407,0.37729204149950457,-0.5985429761397058,-0.641852801092115,0.582263645984323,-0.31286514126745923,-1.492009306304545,-1.607036358021167,1.606692835890688,-0.8750705452665161,-0.9999131254330871,-2.2256898734100985,-0.397179752456488,0.9171213649120458,0.09436573771172889,1.0592858675660017,-1.4371577790677643,-1.9379059539706998,-1.79380234426097,2.964139087761938,0.6437969041096662,-2.0969199079157863,1.0831131953697604,-1.787649035962236,-0.113450837474549,1.002299558374883,1.2395356017341952,-0.8141275740264584,-1.858316837827772,2.343563527314203,-1.1111867872748336,-2.0063235407455355,-1.7050034992552288,-0.8144224120755363,0.8075268418162854,0.42656306734095584,2.7661811879276317,0.3176184884890051,-1.0931694930835223,1.0740490863492134,-1.2889117121607865,-1.7419952538148438,-1.5403532657845846,0.7726440755667525,-1.489677543583303,1.8980584669358365,3.439112458663757,0.15446954496006743,1.9021980529724756,3.2207074674540563,1.5283816001553596,2.454991924567158,-2.027491486858891,-1.206641407698047,-2.3202818453784264,-0.3416097994113646,-1.7622218933372054,2.163991526666956,-1.2440592300068345,-1.801104719799122,-2.2103695484387735,-0.6424247607991997,-1.624863160241882,2.0822631587388796,1.975583247490597,1.1981743886844078,1.6095328743218655,0.2528499757404947,1.0478751397848427,-2.341330688412378,-2.169467912199507,-2.161209032857248,0.6205683258288294,-1.709052438359678,0.997296869924271,-1.6504965129492972,2.1226993481610985,1.4119012065921717,-1.420768513021895,-1.89650354469561,0.652940970570938,-1.6341383057032004,-1.2162842915196315,2.129990346359906,-0.594819919945336,0.6531314249604214,-0.6211887905619922,4.507995595054496,3.443190894607432,-1.5553654355054038,2.6527885715289483,-2.014957938696204,-0.4963775743708943,0.6335983468580064,-2.025220534931525,-0.7948481867654423,1.626176021097231,-2.3634425892031796,4.452513329719422,3.947638139220326,2.6587854097724533,-0.2613144454961152,-1.6189912609189157,-0.2896828639556691,-2.101358782393151,1.5986050540684813,1.9096660735074642,-1.3370734771674473,-2.223136064854689,-1.360479907274562,2.2348571842233875,0.6232682370424673,1.913543569355054,0.7920396027915771,1.9261269377717858,0.2045878053409832,-1.6156112786059627,-0.4127659363883375,-0.40847628655550355,-0.8531210077020788,2.111040531223164,0.7232760703471125,1.9048855001171248,-2.0655581568284025,1.2489367084503737,1.119440239266931,3.2639223978041585,1.6404028457813165,-0.4946274878412297,-1.8876398351235861,0.7315958822505754,-1.727654476856596,-1.9761803712774684,-1.3420024872106544,-0.8501562405729457,-1.5743609442817887,0.9910802095524265,0.48672072127195165,-2.2350036851518595,-0.02607100220164767,1.5611069868494905,-1.9389427659749967,1.3381858816800014,-1.7839001400621128,1.0171507244419433,-1.919795632313328,-0.5844466126032138,-1.3555322981055165,-0.8929551525617412,1.0836781568422398,1.3898616532830312,-1.6257364273551902,-0.2896828639556691,-1.3589336152326974,-1.21380204403729,-1.5018315258127388,0.3319519388723381,-2.0605188502055785,-1.1792302764712002,1.4778500460587387,1.322831981401215,-0.3700365759248254,-0.6890257798541051,0.009656616052819629,-1.5742684193805778,1.634163877152543,1.464057283530012,-1.5335574922499438,-1.2898555911744074,2.4951490055124657,-1.5646038782145721,-0.646312078565445,-1.9803002680190365,-0.1404189670524246,-1.5420206700179089,0.8048085486148451,1.0831131953697604,1.8271077941937615,3.038019623854468,-0.6951027699474908,0.20947166376516618,-1.1165929726877641,0.9942308979502463,-0.8381790552951499,-1.899747121127945,2.2981158805226984,1.778870172339754,-1.672192458668778,0.6211043923569658,1.1785788029170838,1.1633293627310373,1.4043078222782857,-2.171340155941828,1.0200355366507496,1.9172636218727228,-0.002210837562867067,-1.9730451797680715,0.8985968721785891,-1.3969896293984019,-1.4326731699119166,0.9631578448324325,1.0489327920173028,-2.07095537408272,-0.3040819331994442,1.7639885277022567,2.1817580206166394,-1.2946906419752957,-1.637989618628629,-0.9126856381640762,1.3044322386556666,-0.8226551499940674,-0.3119986550827121,-1.319499267162381,1.6900947807139177,-0.26742693790747324,0.18486398302721616,2.5802097127080517,-1.3650040899074403,2.826658106604308,-1.8907335786446586,0.683239306728154,0.2792350254933499,2.7734289483747676,-1.9811624322428982,-1.846676450813794,0.08604848056667606,-2.0136526274557056,3.4017217467003396,-2.2406153575285686,1.533764018700308,1.3943419657272718,2.0496702750308624,2.9176319177865433,-1.0062299162701513,-2.0037168479493874,1.7427346413324398,0.07834595358625883,0.5320349278006931,-2.524950861026635,-0.06692199623982802,0.6102673857172557,2.288682610111286,1.814028225068191,-1.583334620050455,-1.8382592337784602,-0.8094670365971177,4.192473648537695,-1.895421313041973,1.480895749571689,-1.4193539117045357,0.8131859202255107,-1.4278979204221556,-2.1038197203105424,0.34798573904789765,1.9721556523201278,1.4653943781039511,2.0071697942650237,2.0728925954241895,-1.1142987490822178,-0.960047999480143,1.3708521363093382,-1.8799974616288229,-0.15496569937390878,-1.565691691467929,2.6275786325481323,1.3708108946262747,-0.2478539471969485,2.029314385417845,3.661414855903466,-1.7481527158099097,1.8566907682797478,0.22644843590762856,-0.7809484814656616,-1.6624452748302097,2.2692919869513384,1.973070673369207,0.14767715496232095,-1.3673870996807787,-2.1424518192146635,0.21229586644456147,-1.6792987675100617,-0.29227286630266447,0.41294526010261645,1.3477018828016416,-1.6953400269446368,-1.7883427468351771,-1.514614094139139,1.9406291030986353,1.9384243590511563,2.1122231079596587,3.0084027056016094,1.4948702337792816,1.478578234623085,-2.125589890543377,-1.5755971625430207,-1.908250435683512,-1.9183358220469449,-1.7987209245545184,1.1044501175835009,4.25751090743078,2.7093716325407775,2.176588245471174,-2.1514973710855396,1.5704225103803884,0.905388052742035,1.5232986647518791,0.8716689763420931,-0.9201165636708765,0.0723578308248823,-1.9052869174024438,-1.731544222155537,1.295456971282122,-1.7471394420990483,0.4994881903183494,1.0553528437862743,-1.9236861293958762,-1.999287680103074,3.252661202967367,-1.985567622061318,-2.1065286753466186,1.3852221053668403,-1.4390924336892248,1.2406513582387415,1.393865132177034,1.3579199195853844,-1.4817456006527046,-1.3241332745176915,-1.0063079640968624,0.13670387197809217,1.5665670536120695,2.7049832508115035,0.03684007724200879,1.130601433820344,-1.9767906450904693,0.07984769607199112,3.549853492867468,-0.8141275740264584,1.4491400157865477,2.9746395290654264,-1.2906957800447123,-0.8262200594467908,-1.1458534009033399,2.2477151952035417,0.8531495924138437,1.8923307897998183,2.3304053360505677,2.620510638918102,0.7447345711091059,-1.3229879647812464,-0.3700365759248254,-0.5437247958038476,-0.6711930166092054,-0.9635400348082853,-0.4892981300719453,-1.2760631916109766,-1.2974767999941526,-1.2301462242325152,-1.536820392969389,-1.3555322981055165,3.6483020436860603,-0.8923470584357351,0.17094347760078324,-2.2396481509218287,-0.2754259775807205,-0.5743957541777245,-1.6766903891113836,-0.0790197393146256,2.703264052984406,-1.221533751220856,3.9826530581561026,2.814531745991191,-2.320705777459436,3.1318881638734286,1.6607126953721418,0.053977759040579096,-0.26419221199375736,-0.20805690913557676,-2.1109161404854584,-2.0789075884305195,-1.3413009462606202,-1.9481040476775513,-1.1919502890285292,-1.7878465929151506,1.505340352235039,-0.06851824105182953,-0.21917623336270425,0.884309016104865,1.7009803283435292,-1.179378590461417,4.412529763495688,0.4156024848110388,2.3207987294951713,-1.8360145429574801,-1.2944065947665158,-1.7982372317234872,-2.0982241503008705,-2.0219220715841137,-1.6749144025659723,-1.4758393756537316,-0.44076661102923925,1.313255151611468,1.8945518676008661,-1.6627418028768723,-2.0346012549428996,-2.1156824637927745,1.5458164698292127,-2.1418593640788943,2.0105213912819444,1.035741562134442,-1.9941509123914356,1.5138991981804697,-0.4738120720422525,-0.12137235065548871,1.8826508223865575,0.5796968936919644,-1.6110786906893146,2.081798815879965,0.0715318336022633,2.3591065634067023,-1.8311933950877461,-1.4705164585765926,-0.2703215670361433,-0.9549333391785585,-0.9523476542293856,2.404841097424202,1.716779311129848,1.653140576102773,-1.8657089808291978,0.6562747003287316,0.7945803353855309,-1.5295183794592047,-1.4359602830019136,-1.8965887994145778,2.114561977639387,1.473235060335236,1.1743391308697857,-1.294137815603892,1.0590666725398197,-1.477295016869951,1.9392965336428425,0.6079410057915231,-1.4311098332037082,1.670243360893976,-1.5724336652767334,-1.8220349405269676,0.10284259581442622,0.8105529223041442,-2.07095537408272,0.344152354628918,1.5152584194356162,0.8265335529801596,-2.012973424364929,-1.9577736917126443,-1.6030120700843995,-0.1910889726174959,2.0127788022641186,1.157247047775879,1.3061700940678058,2.3601123042373873,-1.7043122124572567,-1.9205781798106139,0.47223328978141516,-2.2254412241904653,-0.021970456369524705,2.749466561889157,1.5458164698292127,-1.451683756218757,1.5766700192874175,-0.4174729861566493,3.4753526175506306,1.6264951233703355,-1.3283146492568043,1.4263704628779004,-1.887823390274907,2.2513960339617625,-2.192254532001779,-2.044991141967629,-0.8923470584357351,1.3234818341916383,-0.711574975988842,3.1304961588417837,0.5610964364687507,1.0400284857386037,-1.7727054027835598,-1.3781907437560086,-0.2078096018243316,1.3558777801722506,-1.852522025262655,1.534804574436815,1.7653687780943308,-2.133269993041934,-1.397440124303772,-2.164344070996329,-1.1612501484088391,0.5387212282858284,-0.8758008944962847,3.425791599935384,0.29557620938692114,2.7538287708412965,1.752640070909097,-1.9474684435970944,-1.9151957720564345,2.72805145318648,-2.0503157600011304,0.868378673914344,-0.1768224488376017,-0.4042266091391724,-2.206228911880546,-0.60945559321999,-1.042243414268525,-2.241746018008638,-1.6526564298246071,-1.235386200878443,-1.9403130451634292,2.0154658553000786,2.2596324755109927,-1.9531020596823663,-1.2215243156001137,-0.07184915493634958,-1.359604313541662,-1.2801864749114373,1.8563998602071046,-1.8503408304673887,-0.4042266091391724,-1.9379059539706998,-1.6601564349995062,-0.967988777575528,-1.8455053671819805,0.9871813338084441,-0.4678407638077417,2.2373855877848436,-0.27718995310276867,-0.12017126031042243,2.2737629742077905,-2.275289778027554,-1.6491468222498724,1.4043078222782857,3.162823176424242,3.010552207143765,-1.6157614132351998,0.8466146483240808,3.093799894826824,0.8747788978940276,1.0805414943479505,1.5431008040280103,3.3257090008660857,1.9706883595881148,-0.14908275794922232,0.21971983086214095,-1.0301200323348494,-1.7620596750007729,0.3681997968612777,-1.398782576465947,-1.738721142897133,-1.3322728400802304,1.9053780306333412,1.688566471050776,2.586175084689923,0.868378673914344,-1.9289892874741046,2.8723054162045485,-1.6404312916315749,-2.100176291173924,3.6356926032817203,2.5220926416880305,-2.1684802704740758,-1.2949873337234488,0.39817248759907437,0.8022750293508675,-1.3995172832189158,-2.0427419670673426,0.8337457722128596,0.22964292228798705,1.1911467464596166,-2.125589890543377,-0.9623751935421405,0.9327233905556633,-1.157532324619656,-2.3385330164346163,-1.4653393117320432,-0.030032987673431228,1.9662534425232692,-1.592032917093644,1.653345928967389,-2.872664295959913,-1.8551879223717018,-1.8389662638954944,1.3655059801363403,-2.2256898734100985,-0.5122777143336702,0.07936598412374447,2.800169977017534,-2.1186038609457296,-1.5693875394258225,0.915950092371468,0.4734151433993029,0.9448139218835545,0.14359156902066014,-1.9460177702134718,-1.7533911102028004,0.8866946145919916,-2.1126522449624163,-1.904805976049363,0.9861559124546591,0.8618360056593843,1.2263637706969333,1.0653320521637022,-0.4326226725896678],"xaxis":"x","y":[-0.06823838410942984,0.0559413689850022,-0.9262491450845287,0.11989358997047257,-0.6866222950550114,-0.8922614571218116,-0.776324719312136,-0.04068850487874979,1.210537137586174,1.5954607327865906,0.36664109424323993,-0.6612602238132941,-0.4975495485902316,0.3350967015089232,1.773652337844145,-0.054264826294215564,-0.23087369784436548,-0.2854649355294546,0.05303198476713885,-0.17861611922253354,-1.4908473891636853,0.36985897310327587,-0.8033551702253644,-0.5928613159786098,0.18268526300397667,-0.3968362074510421,-0.029883228521201212,-0.33500881618296496,0.24926964166369767,-0.14530999175569417,0.9808471947083532,0.5410942565558244,-0.8098236845686564,0.02708716859681389,-1.2311445525448788,0.09418163593769809,0.03689223912854108,1.158795440074529,-1.062707303900423,0.21548897840103162,0.3462557676547307,0.2773231900443906,-0.7850856712232859,1.2018470809893151,0.13653101525912667,-1.3181275238302577,-0.6616191321421194,-1.4591902333072395,0.0900850255659384,1.9736017008039173,-0.9576602733106817,0.6324587601382687,0.07664618938053758,0.20472549160874773,-0.6783062698413195,2.0484919882092365,-0.914272495846536,-0.9978957810030735,-0.7829681936012717,-0.7201769437180818,1.2956648274383138,0.2626319583609859,1.4566113539305627,-0.7533122030878304,-0.9467956435961884,-0.941596478012011,-1.299452404654986,0.06364830254843468,0.10491720291948965,0.19949267226488085,0.04093438882453982,0.8892724292337878,0.31162863002511537,1.2635181038708727,-0.10297777485344,0.31202796587372855,0.09569005219713796,-0.14530999175569417,-1.283294397731749,0.1432611901391043,0.8278462336876784,2.5572193726753,0.06374748603847746,-0.21122462138782952,0.13368363374044398,0.052811674227667936,0.1854855685113865,0.2653027046895342,-0.038633250584987656,-0.1623695169134125,-0.4804187585473698,-0.00732785460542538,0.10197153612555339,-1.361615515936626,-0.6860239199859347,-0.014356396870669915,0.2781149853904203,-0.3082637328276642,-1.0172298320364672,-0.03302668499189924,-0.7514329366836815,-1.1325757967108818,1.2262208705952773,0.2818470890404888,1.858476832839207,-0.6776092675312968,-0.08227081823908776,1.1462030874895357,-0.5740662852292673,0.12986638643700063,0.2066995779915782,-0.8343853394216045,1.0499628619153032,0.2728435578958713,0.22477147324427746,2.619281532437215,-0.669642089582003,0.14097475741310722,-0.10846018026181296,-1.0528187836860514,-0.7292468410293672,-0.9197189354821844,-0.8428264105490579,-0.3963148328668744,0.00012503872932734584,0.3536013219379574,0.01010521860838772,2.1928338327872026,0.084690041428173,-1.2884774679166893,0.3083026246425123,-1.203619973998986,-1.0956313407266507,-0.2173800458297577,-0.5639532110487285,1.259866031618184,0.2122513294035395,-0.3413290468506799,-0.07415618458701805,-0.5603497989538155,0.2760570779712744,-0.49802732432683033,-0.08400224299574714,-1.0991052133152113,2.106648728608263,-0.42111483106560943,0.03741052231759386,-0.17105692519750362,-0.9080885701826745,0.2782049995237665,-1.2968568683140898,0.0010872786260098996,-0.9577799229566963,-0.06659170411598019,0.35256253031151114,0.7311195804130782,-0.6483193523680207,0.07408612656704912,0.15928032332613648,0.14899550718659269,-0.08329554772019475,-0.10222660106584468,0.980285773646258,-1.4389743186427595,1.811243691894692,0.06827525528588514,0.21569975656388649,-0.1623695169134125,-0.19042740137394296,0.15730967663504744,-1.054952347167958,0.18007822048325414,0.2006591507086026,0.27031137603670635,0.12196433862588447,-0.6851087289378517,-1.1741254200167874,0.009823681543405424,1.2973685200921934,-1.2688861922077472,0.016714811101997372,-0.4779893661712252,0.17685820076791,-1.0094990324645958,0.12666212876755917,-0.9179035933851709,1.6632960171453959,0.24027660659846473,-0.8795884778141739,-1.2740497106028783,2.652446753310076,0.10092546440374335,2.4243991669174982,1.2621283527204512,-0.5022262707899738,0.0642858874854881,0.7015506348087499,-0.38259518462686165,-0.9673608125240073,-0.5564880386127652,-0.9246089507313089,-0.7822977280020376,-0.6031687775258944,0.10231734151005563,0.7515441971519911,-0.23599210251655175,-1.2731474297124277,0.009228258601118996,-0.7314923177677642,-1.0143547905886563,-0.15937714956750312,-1.4250835636546435,-0.2048102932454033,-0.5086496791693355,-0.6609416670747377,-0.9163798281872152,0.24702400095185137,0.32592489325689755,0.25149802141229227,0.21190322387313396,0.3566220220256214,-1.2627248147917558,0.1540190542320102,0.28633177188652775,-0.880659039017955,0.13549247900980352,-0.10538308435043053,-0.6665808608115334,0.2929377430452554,0.9709600453279412,-1.3401744412546375,0.17571105418942476,3.2441619589436907,-1.0218063559214852,-0.9963404787678606,-1.1429277680873628,0.12845202754414728,0.18886638972319644,3.5505709033093464,-0.03516514165857235,-1.2320924124170425,0.1803123121861104,0.22462077327012459,-0.4369991217064209,0.09257128300292523,0.1950753221469281,2.3613732084947405,-1.103457502436352,-0.37066715018590585,0.21863872267108228,0.567023516806024,-0.11651905690560053,0.04517910777966883,1.9281997780396334,-0.5891227290160986,0.1553898473747782,0.3491763182631819,-0.015144113159576459,-0.3705076638997344,0.006203414645571078,0.013988205656079852,-0.9753158429277636,0.07655176409117825,-0.564781680803992,1.3111294610990616,-0.9242508442668712,0.03937169381606328,-0.5086496791693355,-0.23087369784436548,-0.5651409883727435,0.024910316915877438,0.13207082908182463,-1.0655906258533614,-1.3235512027002287,-1.1176805818550726,-0.358927640333225,1.239490678754671,-0.07687324721726833,-0.020933353285432884,0.15303279176269333,0.28376683876670444,1.0962094873220825,0.19040054876417664,0.033589961471646244,-0.49063236069870697,-0.1278204694470757,-0.08967772671538092,-0.8173821492551995,0.0023870876869750623,-0.24209223395701016,0.047434050655840686,-0.11558644173277466,1.1842915591848455,0.15047673657856142,0.32619542669665974,-0.16957911466304823,0.09488278354368525,0.20094647409416275,-1.094028971843385,-0.4938311586102628,0.15078139622354755,0.0037676698793124496,-0.5654903250726601,0.18139970205960773,-0.8604348832767648,0.2447683279888195,0.1438391069161938,-1.0186237238024878,-0.7968875893460076,-0.7670696829526066,0.1677503082106616,0.25568797851380004,0.035566570147769115,-0.10435950560258599,0.3293227478386092,0.1510371469704551,0.19376204439532102,2.2435669888701186,0.20595479406605963,-0.9152020192281232,3.0296378539384223,0.22523600382182993,0.8903435467916359,-0.9865933891085502,-0.009149254419262934,1.2623852898020165,-0.657793961993395,2.3590872045016185,-0.8019305641645766,0.07082925981030093,0.22605257230933268,0.09072540003805059,1.2835819009384208,-1.1603189349498941,1.005359350416936,0.22695478405841785,-0.8368033765772546,-1.02661047083995,0.0695328348612861,0.9933217804454805,-0.7217231916022852,-0.56279379363163,-0.4539948123539838,-0.15437018313369522,-0.8344547005007352,0.012275207409464714,0.21863872267108228,-0.7783788356642652,-0.6883464679548115,-1.2852054345690647,0.1262066150442623,-0.825867425099508,-0.5645242649459585,0.18822552278327984,-0.4471981201280076,1.2652365235939418,0.005692449552218818,1.0019943158874876,-0.1839456730126333,2.343079747622671,0.23120725470110134,-0.9044209074379331,0.2252627630322572,0.25300655878835415,0.3625114863769196,-0.194267097366444,-0.3413290468506799,0.23204471021346615,0.2539949277171771,1.5700922013333496,-1.1165297075947394,0.060885500217795784,-0.9750865254008398,-0.06408056514386251,-0.37432081845471826,-0.7551807417233722,-0.1872536048621842,-1.0133345547001402,-1.1324036788939649,-0.21915554084756345,1.3319072545478783,0.2876622107589904,1.101615148964863,-0.012405061210589522,-1.0777423788561267,-0.251183013786428,-0.08239682500194516,1.1560255253753549,0.08725625523272669,1.0943312701384174,-0.49927607132727875,0.18331060408896535,-0.6305462259169683,1.1687733606641844,-0.612616930654241,-0.584230063942727,-0.4497593282460195,-0.8944684159051189,-0.9195825161498627,1.1535471533580495,3.047083586791724,1.6891550068067922,-1.1150546252593547,0.07969803552812978,0.1854855685113865,0.37593354175725496,2.225309012138144,-0.26725691686967956,0.13039251886960232,1.5923923609026807,1.1443411514100026,0.2988047286422985,-1.0074917743508292,4.233310023963535,-0.6794738952376446,-1.7589831271930676,-1.1731522889527857,0.13849567293227047,0.06292674959657792,-0.9789865519268799,-0.5442058464607777,-0.023807700316444948,0.3230883188672396,0.24439472591077602,-0.22357640122670358,0.3786644982287906,0.23120725470110134,0.31370010837126344,0.33594478683528833,-0.16373788389080193,1.1687905999373756,0.18796695096167257,-0.5273964448463481,-1.1037996249274966,-1.1090967350731908,0.08796104522782285,-0.6501481078817718,-1.1802161200675227,-0.5221533951045317,1.7291666506903653,1.7470126827014134,-0.3271919544834386,-0.8502950587206446,1.4360325197720323,-0.9774377617709884,-0.12876670501824847,0.04517910777966883,0.01324385263989016,-0.8958388290818576,2.2042037332479043,0.266998201277682,-0.7097606979575092,-0.9432470802982038,-1.4675400023000886,1.2492678910343813,0.8157688969194417,0.11824990926645347,0.31155462217124513,-1.1058821128775493,0.09354764231669206,-0.5847631419976524,-0.7286598933544365,-0.6495593615397128,0.1109537190971462,0.1224462702736236,1.9169642613270796,-0.36075037498350315,-0.9238104170504513,-0.34111643545125836,-0.7189239088546961,-0.8284539202796365,0.36450973607515275,-0.004217640258984948,-1.1312347109021845,-0.34552330585871055,-0.08554048377388712,-0.5543461058582193,0.2872132105406069,-0.8614565544602403,-0.020460720395912547,-0.9592399837720892,-1.1963125251357103,-0.8716214788279844,0.20680404900346014,0.08562887958959083,0.0695076567212017,-0.9414953925401599,0.06292674959657792,-1.1530899601479188,-1.0357462833838154,-1.0924616608737399,0.04349829289997128,-1.0225401680673112,0.8384215531992841,-0.9445764636656115,0.5637341255724979,-0.07510278880703392,-0.47651559887001027,-1.1951687485792268,-0.10777495632259919,0.2645256439955835,-0.739095406565609,0.9805803028018679,0.13407004266143982,0.3598288886017161,-0.27532782172271275,1.2635835814059704,1.0460298352146657,0.20959782948858582,-0.9874957632268793,-0.43928368880037816,-0.7856322550885666,-0.6114209322319119,-0.6116723333812819,-0.7929681828675924,0.15022188182362628,-0.4013292499840127,0.1514347186775041,-0.4477014044555009,0.013429442047484771,1.2846937351067862,-0.33290954760397135,0.10100062384670812,0.08423983560123478,-0.17299082994219891,-0.32710950125128535,1.973665858225867,-0.7252044065955467,0.13711242427546264,0.22066259403423366,-0.060722945366402416,2.137677285020883,0.2119168099220415,0.061653339406157213,2.212012994793945,-0.6697074164388638,0.14205584687174697,-0.19160167040526255,-0.4074712010549333,-0.7928892566666116,0.016286635670292985,0.2104066505752645,3.9947555438917277,0.11169197800038062,-1.0036553735266418,-1.1590190823523943,-1.096290695735087,-0.4144247552110644,-0.2953539505642093,-0.8247152478482616,0.28834319474628073,0.1383369012612928,0.21561354844084898,-0.4341026748552261,0.11868694193187197,-0.1294943220642261,-0.3961207696200047,0.08172758914880558,3.210290772309889,-0.2328114011926043,0.13606568745367795,-0.051136412221611395,-0.49622488559929545,0.6547123832828976,-0.5928613159786098,0.08809021252716061,0.14974745277337817,0.06231179209168721,-0.021818715706698612,2.9493772173004427,-0.7821895188692565,0.41714391944650575,0.23194032181428667,1.0847922108167447,1.8441592430145128,-0.9435712624161748,0.2064110535017045,-0.5105251215412349,1.1734051748706356,0.23313060323352838,0.08940813432710766,0.07588548754693099,-0.011587110195631559,0.23225707087474443,-0.9523533649283227,-0.7929253558192136,-0.7776617973128634,-0.21343731269225347,0.9826667088678809,0.03689223912854108,0.26562904014855665,0.17977825690822755,0.11645218355249931,-0.09103213361264377,-0.23757380898238212,0.32269424537812136,0.20094647409416275,0.09319464087512075,-0.5340465447924115,0.14048800114971846,0.03730987837024748,-0.35854705984126073,-0.560932395603128,0.16752003152908443,1.428855359555867,0.17744240837663813,0.07002718762726301,0.27351161495554405,-1.0327910676194028,-0.6120000767501029,0.004567850473230347,-0.09783121618011563,-1.1806488659458352,-0.5731223301314884,-1.096290695735087,2.1711321668971695,-0.08834029841769786,0.15175370329310364,-1.6069350030939813,-0.4347256482571515,-0.13940179582837342,0.17853071586858327,-1.1165297075947394,-1.026170680086388,-1.0067557538588978,0.6190762553396467,2.313977829726843,-0.4775308577091015,1.359604271819665,-0.271015570244654,-1.3428854533084669,-0.6831296264294284,-0.16257269903089888,0.19319252460348943,-1.446033417690018,-0.029385434765164177,-0.423302097819922,-0.2654945987940528,-0.03516514165857235,-1.204833328879136,0.7804202993897232,-0.8284202652679911,0.3625114863769196,-0.9895294592873582,0.21572007336473195,0.358859215931104,0.32248153642021676,0.15968195628836698,0.21074515173152716,0.2403793690949366,-0.3084484109013819,-1.1809174403056801,-0.8037957011087138,-0.7294404265317254,-0.9146734952020699,-0.5645242649459585,0.3089623758147145,0.998345814200017,-0.7085229126499517,1.8990113449080979,-0.8163945220603692,-1.0074057719574454,-1.46346081816086,-0.36664785362763075,-0.18751965657648725,-0.7346596087072603,-0.8287207227389027,-0.21675473263351264,1.0147008331982657,-0.6606853428515285,1.2832117234302132,-0.49149985889880576,-0.21623134985033243,0.37129393085475987,0.9238901343730576,-0.17376565815965686,0.24228017428234702,1.1477138738920765,-0.6565930702101745,-0.3039654591019122,-0.9342848253265815,-0.8605051061745533,0.030873651410563533,3.2162432320549916,-0.1991165151436469,1.005359350416936,0.13893681940037547,-0.6344168384591177,0.44858277995156304,2.2435669888701186,0.32032317280780004,-1.0394223778593779,0.059912709691405795,-0.7530380123595416,-0.32287242723813236,-0.5848888270543083,1.1448503136817465,0.17320080009052094,-1.3302179993642116,0.15882655469050597,-0.7414136587364092,-0.8353269295163567,0.2786180445382383,-0.8288624439641071,2.142078681406865,0.0642858874854881,-0.17588482021305696,-1.0741671430685347,-0.767266854540701,-0.28521989470533793,0.23452446065010965,-0.840262244768058,-0.9576602733106817,-0.42537209733457615,2.2562937479519296,2.8480640434910343,-0.8616007983309867,-0.4981644265147057,0.38318455521706124,-0.8714164461292091,-1.3180860496402604,0.0478744269289609,-0.10661086063748314,-0.03021829301800858,0.0011384504619025326,-0.03999308163874082,0.15329932693383028,-1.0819144834961403,1.2780772225054464,0.820575741465058,-1.1480922022767375,-0.5421256781066497,-0.9895294592873582,2.8132645196805415,-1.0312230687152257,-1.0278411472098703,0.03460382118942174,-0.8175660583858884,0.0021132887918471543,0.016177975253663344,0.04659200689492601,-0.8711449958193722,-1.0522742753344556,2.2230172304392912,-0.8683654258027556,-0.2650341609649598,-0.8891793421823121,-1.3279746917196273,-0.05837964843539526,-0.6738808453001424,-0.9279009918354452,-1.0200999141904323,0.1631025057003126,0.014004027260129078,0.5952087295731288,0.18939767344159872,0.0012093143292783634,-0.8508982184047048,0.015475104010360527,-1.060641581809655,-0.882643098773532,0.1371687322777387,0.31099603422710614,0.2774415518652719,0.22048346609011868,0.2161313016901347,0.012207654008432786,1.2203219409620365,0.2962171065318368,-0.44723196882605026,-0.6235330435328734,3.288903540839431,1.3196912406323584,-0.4276567155479578,-0.1010631193612788,0.7364813418766076,-1.0082402112806326,-1.0977630365761886,-1.1728652445463619,0.08571744040603653,-0.8249357117460058,0.09552983318794424,-0.05200326334367659,-1.3836122614040143,3.1040250432958802,-0.7135227138179583,-0.03309195332257051,-0.567608445488466,0.21624723894984282,-1.0574340386625778,0.2549169278456586,1.236792218298921,-0.7017318339362582,-1.0607453415049912,-1.2159080168861875,-0.710511957390093,0.2754149019319984,-0.41328895448985375,-0.8774848310468889,0.17280696382433455,2.141377472814651,0.20309214496960767,1.4067340138073392,0.14205584687174697,3.9947555438917277,-1.2082614525338498,1.0441493967394437,0.12363593610810981,-1.1266152083908447,0.3441793347414654,-0.02145065674761657,-0.5919040196377449,-0.7323066026593498,0.1788164039882156,0.14287894756829997,-1.2782143878977015,-0.37970624916164064,-1.3806330653954202,0.10384129263539384,0.2702656002899472,0.011286542731520374,-0.2927357857616822,-0.08400224299574714,0.03756445352743495,-0.8612232462475812,3.0612500135624394,-0.8156398691526152,-0.8853671160834395,-0.22366864190422087,0.07463599443801079,-0.32885221494323513,-0.8033551702253644,0.736898965096702,1.2077709570759063,2.535210694226874,-1.3439453332265738,0.16359660759838454,-0.8355470148761523,0.12401110864132021,-0.27637229743088776,0.1953201788239922,-0.46877038149577094,0.4643659415401145,-0.426273516406369,-0.3317800587686262,-0.5885299397930487,-0.4706653492431793,0.3678947326481247,-0.513290234503191,0.13287765781497804,1.220341645652899,0.38252206397865646,0.0683847153904485,-0.20714463288030135,-0.9593308766702029,2.120248287105837,-0.765892130666468,-0.04529380610867587,-1.4201567344615986,0.0497844176244416,2.1396299361112567,0.09047082791314916,0.22845835214406948,-0.0320652840354431,-0.21675473263351264,-1.0556704584277443,-0.386113567396868,0.03972541000259019,-0.876459391501192,-0.9072886763269075,1.1482823830137063,-0.5767883776325722,-1.1772472358176247,0.3076833519358863,1.158795440074529,-1.0192908861221441,0.20587491134409902,-1.2612598648997138,-0.4199901947346574,0.016368082340361766,-0.88269117934656,-0.25820648304796984,-0.664967562257029,-0.397617057607804,-0.5372327473029912,-0.9678211585533025,1.0727702860588115,0.15381020079963334,0.03900914942906294,-0.14493335243424618,-1.1025614414558442,-0.9147196069797113,-0.25235938464076224,0.9678376603814072,0.9392572805148222,-0.5529616260319002,-0.6853720344311428,1.973665858225867,-0.3200810774226262,-0.20545604976555187,2.1572743294709484,-0.889610647405592,1.0818349503274263,0.009248893633393337,-0.3320224572611481,-0.2782735169776411,0.10402343061387714,0.03220269281227873,-0.7551807417233722,-0.9710093999003406,-0.9821267171612567,0.0007012480668379869,-0.05137267181268756,0.2709100720172484,-0.8683654258027556,0.07010356151611681,-0.7628114004308776,-0.3993617136560489,-1.0256167362369517,-0.9332293128518284,-0.9557163134802437,-0.7484371341527603,1.415313902015375,2.968466955420126,0.1671782483235957,1.3307219024870467,-0.758929734748289,-0.4765716847248972,0.10660873580845061,-0.6887681532859194,0.20352760717985915,0.8913562061311682,0.807892584353339,0.027076436720089546,0.1011996326497246,0.01330066658130787,0.4618686608053983,-0.8184824255520656,-0.14711596197476814,-0.5080847089460098,-1.1835371545091473,-0.7040850295624487,1.0758490578941924,-0.23231004462787486,-0.5072035631385278,-0.7844497469833548,1.779181433585439,-0.03241918489739668,1.2713524766384494,-0.8291339179032372,-0.936026740010305,-0.13598076937392659,0.11142663492258682,-0.6540811646476066,-1.224906000407603,-1.1797679156050804,-0.020577336953002786,-0.767266854540701,-0.004288786533906833,-0.10758370500969705,0.14549874135218374,0.671998334323394,0.15724063235926486,-1.2740497106028783,1.9180357097318617,-0.6029331184001571,-1.044295871260303,2.7283458337628907,-1.3291604494810807,0.09671466757226428,4.169197321317233,0.32319169083753213,0.3166759274521848,-1.0104157086751067,-0.19334089198772766,0.24228017428234702,-0.10192902279969851,0.07408612656704912,0.0499416129054422,-0.4401073541383871,2.177249367482561,2.257152461840339,2.2385308707302367,0.19975686981815208,-0.3608967496666305,1.4473136379486067,-0.979137780542481,-0.03999308163874082,1.1104507515236748,0.22746532742953648,1.950366027803097,-0.06955548061661236,-0.39680044740797116,0.10071772205621239,-0.018931832725397925,-0.50863838282728,0.0879408158961702,-0.2620835396039063,0.21462505942736085,0.16294706269374457,-0.042932798887023595,0.19690056106855766,-0.08625703884457969,0.15453439322941206,0.24095475456776314,-0.7942593006686217,2.199930499203381,0.257197432256354,0.21753368549349603,-1.457177432426761,0.9824333276257381,-0.14717451539898863,-0.7058800227467272,0.22292595061928575,0.06374748603847746,-0.15199414612945403,-0.1065810836336729,-0.09084297398959167,2.1800407518699942,-0.7193015408707677,0.33235662862431975,0.01978763481980502,-0.11375775736071818,0.3606055736602424,1.2225276266898664,-0.03317416772758661,0.10240466972890956,-1.286452650190514,0.5656073728935043,0.14436493311426024,0.0879408158961702,1.3197820955301354,-1.2752296948105923,1.198668080644251,0.2569427775187499,-0.939254637606315,-0.35714510694722823,-0.30614225676090007,-1.0299788575919469,-0.7264496656409907,-0.19673728986800354,-0.12058911652298598,0.40063295306038965,1.165607299289994,-0.6618323335678286,0.20077519739392588,1.8696565615785745,-1.538010241453582,0.3441886403342009,-1.0424258525413113,0.07010356151611681,-1.18177907242866,-0.9231594622795579,0.061019050036029886,-0.365429541324202,-0.18272001219605727,0.7592391900217338,0.21384625256742404,-0.43145090901663913,0.19250102187589366,-0.533771538462036,0.5224785995883754,0.1999395994710027,0.19738519067404556,0.9992900488053063,-1.0398174557341264,0.23661333105547397,4.087548977502875,0.16980946013840545,0.23960185821527147,-0.09866122272081423,-0.036823710150456136,-0.739095406565609,-0.025232217609721032,-0.2279055013528744,-0.5301638880194048,-0.6321821937896017,-0.7213911423949488,0.30205965256074035,-1.0092107646476756,3.148904712933717,-0.41589340271975705,-0.5750944619850055,-0.3502317448682834,0.18430665560785162,0.004590365577254011,0.015319001836139342,0.2104066505752645,0.26707319098851223,-0.7407320249910488,-0.7784654383005002,1.2418695212081237,0.08404312989052017,-0.16257269903089888,0.15857770134902202,1.3624908713175887,2.2561501413998735,1.8699672852713187,0.24025251996344285,-0.8042368880810674,-0.961989661603084,0.025465427696312458,-0.4870424596228889,-0.0663061057958164,-0.35127875941724834,0.45020683572126063,-1.0113066265860484,-0.0007703947529299841,0.3898553517538747,-0.06601054709223186,0.3004885190308003,0.31516629211285324,0.12849764417394768,-0.2817861324809676,3.0353797623045504,-0.37459651504741265,-0.011102828886770092,-0.026114693397153218,0.24926964166369767,-0.37345257994870745,-0.8343853394216045,0.07881346350024372,0.21041027191014647,0.061412538005589955,-1.080274856464413,0.02369671597794972,1.9922421857906198,-1.2403118115781597,-0.5420064319848029,2.363062268605628,-1.121427635073969,-1.1126562646146703,0.4640340885230873,0.6863329103528607,-0.30877142973925126,0.05658966630239556,0.10521030053600777,-0.5547241309594076,-0.5918943082980247,-1.0233382450578843,-0.2363371812426038,-1.2403118115781597,0.4141908958768249,-1.0249121458167,-0.19452304483203472,-0.8816883222220477,0.0292515647637104,-0.002523726215623503,-0.8094083530557431,-0.8114773075043933,1.4354530579246076,0.31370010837126344,-0.8191419196527021,0.3632016287071183,0.18743320347616751,-0.29835678717907954,0.2540050215268383,-0.5676664785561385,0.0542819312676955,-0.10755472747133299,-0.6129328782025607,-0.17272028303109963,1.2125843447955096,0.024575686360684933,0.30665528785804813,0.29702052113420524,0.02176704512060603,-0.07978258482243683,-0.02329700809378258,0.10324772688642521,0.16157546986304278,1.9287416874175556,-0.049298422036820534,-0.226953539545156,-0.1198386035047606,-0.9242508442668712,-0.05837964843539526,0.1633473356108272,-0.6777855466194274,-1.1362825899631825,-0.3067569485818669,-0.861287298574521,-0.2352613656144671,2.3613732084947405,-0.7347553281526868,0.23188079782550092,-0.33500881618296496,0.2807881865214639,1.1747591794956862,-0.01356540343290206,0.6625788272653774,-0.714261835967931,-0.4381398287778485,-0.06637322611106555,-0.47468550904819284,0.8729484972409037,-0.6323723986547825,-0.1791793645565878,-1.1324036788939649,0.05055977578166667,0.1652860310814411,0.3116476631089447,0.007206447654538598,1.0798621881658605,-0.07813839830903806,-0.9561705234642544,0.13440573929050015,2.2531980636896334,-0.609579391281685,-0.024101569778516164,-0.316956828272133,0.16677541122476555,-0.09803364522485179,0.15190437948663002,0.17337860181304207,-1.05894399120387,-0.12017891233912202,1.814284512240643,0.24439472591077602,-0.2185436807164722,0.26998966036587485,-0.09820230800812312,-0.04583145716297287,-0.36411869241801,-0.29201122398105006,0.002350781110276776,0.12863418332457252,0.1963814745870369,0.3602687928890103,-0.14604632437060816,2.455215726833256,-0.21330445335882586,0.3929605184789557,-0.7753884273543988,-1.0074057719574454,0.41274424812592003,0.42342823506479677,-0.8395413703411501,1.2328180039703125,0.8285079196555887,1.165607299289994,1.0481141283827797,0.7217082595564795,0.08152362274638587,1.0242191910726983,3.148904712933717,-0.12382781816937194,0.12673361232780786,-0.1403948163884263,-1.0130581010315487,0.43465162385795925,-1.0105394289395229,1.776630946332852,-0.9763417405918255,0.09449185997587158,0.041231143599822835,1.2481374894370787,0.00022178919717215942,0.0067478778100640326,-0.5353555393614471,1.013796620436934,1.963200894810797,0.12524427866741636,-0.45097795878087715,-0.15668434170020387,0.200422996082775,1.4156940452670765,3.915299219894054e-05,2.822945190124585,-0.7737623624611907,0.3972841579252316,0.2929377430452554,-0.015448828604352685,0.17700077845169923,-0.015086225691162783,-0.25906611255449796,0.19940846550633937,0.024562239976225867,1.2436081184173504,0.1818106961943647,2.968466955420126,0.24261788722168384,-0.42173129746719157,0.1349918329057876,-1.0772593978581417,-0.6187826249225361,0.8789645589072325,3.8921381601282676,-1.386906338488448,-0.8650659888467067,-1.048704435462361,-0.38261533620230215,-1.1885481726149176,-0.4448690248306636,0.06834037944537825,-0.5055125526064439,0.10240295510256443,0.0037676698793124496,0.1722402894137148,-1.3828807044394764,-0.4046683505957986,-0.2154796038847543,1.66725502402663,0.18703741839299023,-1.36008981093265,0.18132346766156635,-0.09202331317711542,0.5250109980148574,1.2670902275613722,0.10974284618216475,0.5222050964423688,1.0606275848319897,-1.4315066357806354,3.088007354062529,-0.3450686258892031,-0.7841795827183212,0.03676538999704756,0.1526675283637173,0.736644329412996,0.6959042129957214,-0.4280943531829561,0.830489243629205,0.27146305831623635,0.3268239643181365,0.059912709691405795,0.25944143792227153,-0.4804187585473698,0.036695913337492676,-0.026114693397153218,-1.1369948821571574,0.24182137332338693,0.37608772517968286,-0.38230965990842997,1.1086220478777213,-1.0122104633844977,1.0485722039352456,-0.7413703574182917,-1.0334612187430756,-0.03536785744145935,-0.8458311644876408,1.2684593148687668,-0.7857682725009256,-1.0650559114541884,0.4738065955457429,0.26330570293079725,-0.5019916970583642,0.16971116585218135,-0.1991165151436469,-0.3698097502558349,-0.33419932441375083,-0.07760559668297394,0.14048800114971846,0.25598518993300473,-1.147611527579465,-0.5123359216062696,-0.7134687239835262,2.261838780259086,0.5440062262296612,-0.15127433782277686,-0.3033082136291421,0.25997205654069905,0.8824711500112176,0.21384625256742404,0.11188269763954857,-1.0312230687152257,-0.5465672426594297,0.05470168597987842,0.16096829643869112,0.08889898106633416,0.14440191375802933,-0.9580046237740781,0.20129530628301837,0.11831509498442223,-0.9242508442668712,-0.3957535273388405,-0.5048601956380746,-0.8774848310468889,-0.0772931655506509,0.7531953123879669,-0.9330163351777139,-0.11274001833678847,-0.4477014044555009,-0.9379183494267186,-0.6337406875261989,0.9137418543518526,-0.9568859915366138,3.2441619589436907,0.17527384585955005,0.9834161939398484,-0.09918689635534354,-0.694108087799956,0.24025251996344285,-0.19114456427037654,0.08939128728320096,0.09081855508173071,0.24593092682050063,-0.9755847277600228,0.005692449552218818,0.03755274269378613,0.32248153642021676,0.05852099620629322,0.2441259431412409,0.17435633789557878,0.06741206600295054,-0.24441061825573202,0.2569427775187499,0.22394023496181922,0.8269408255782298,0.10187395948680895,-0.6814331361171888,1.2043657263147913,-0.5048601956380746,0.3575383288852791,0.029715439252258745,-0.04529380610867587,0.017001826695179544,0.22956755567184725,1.1399599769489084,0.8527521103045111,0.7794602023486781,-0.34437513384036783,-0.8993976186696744,0.009871934702240685,-0.7799949700167624,-0.7135559604025825,-0.23974456655036322,-0.8269644888407628,-0.09097015755025073,0.1015195226205164,-0.39885297510686774,-0.7058800227467272,0.8592949460527366,0.027484321437790525,2.6438491181341686,0.5354956544883904,0.179933462691864,-0.9855943172923691,2.1572743294709484,0.1564566862567842,0.2479677967126724,-0.7954595137044056,-0.7361334543369359,-0.03999308163874082,0.19221483803617176,-1.1544488636700585,0.14677082134265515,-0.1414850574018208,0.19505291396211907,2.072693955351279,-0.911411274286029,-0.042665925858311395,-1.048704435462361,-0.10669143705017213,-0.10724853554764129,-0.08775493653983868,-1.3439453332265738,-0.9523533649283227,0.31971770557107143,0.2743361525512618,-0.8609986552519382,-1.1108240791246473,-0.8061112488147067,0.15290177159815388,-0.4386029355753239,-0.5673404014206377,-0.22218493402185624,-0.3595926601494932,2.488689619440826,-0.9821002825131923,0.043637342300632063,-1.1380439673881082,0.16125870818707996,1.2439994455093322,-1.1860568806781475,-1.3830710639148862,-0.6473697361133075,0.22394023496181922,0.42452365127437985,-1.2908302888319534,1.1909660960190118,0.4764497014415474,-1.025819235851134,-0.144874817758037,-1.1429196519079339,-1.0758775229378177,-0.936026740010305,-0.1839456730126333,-1.270988812107094,-1.1320035953326744,1.2370868669669475,-0.29329479560417954,-1.0381227099373047,0.11169197800038062,0.07015735623607096,1.1827506994571249,-0.05935170869880057,-1.0569499314139672,0.07331953799822051,0.07969803552812978,0.6831107058520385,-1.3256222396979969,-0.8759859498441017,0.05538334711039979,2.316418684922404,0.2218979390889527,-0.09413114721905713,0.4921213788562444,-1.199455047790224,0.8432808604845753,0.14119921410437483,-0.29402727416552077,0.4030658377221435,0.32151515105144496,0.15644035438298853,-1.189208266148936,0.02871332511369503,-1.2514105909030873,-0.9001706586590552,0.108553578159524,-0.9081700746992255,0.033589961471646244,0.17923387960872225,1.099399623114402,-0.8238070141598723,0.2223652997521753,-0.08812300767193908,-0.7217159149353313,-0.7153877961678984,0.3606055736602424,-1.446033417690018,0.1313695898521645,1.5706051409782855,0.2666151179424469,-0.029087654197774183,-1.1747706758065344,-0.04529380610867587,-0.7135227138179583,0.32534774529307503,-0.9948229077700468,-0.027516158679871595,-0.46234161895711645,0.015663984784038302,-1.0560451856805824,0.15446408624447744,-1.168112510448217,1.773652337844145,0.01978763481980502,0.9196718060258755,0.2547896883090847,0.42452365127437985,-0.3216599718247273,0.9864942872039172,1.0734682614373114,0.04349829289997128,-0.6211093629758679,0.2447113518542827,-0.8298281844312991,-0.7580537850150979,-0.03743904904945664,-0.25606786423454064,2.1928338327872026,0.8373303978197268,-0.37460527865712456,-0.38292469699373716,-0.21836228269165148,-0.02329700809378258,-0.5900285502329341,1.3901314462852128,-0.38261533620230215,-0.048293197053120275,0.4202125062446758,0.012207654008432786,0.7146565412417168,-1.035498096394262,0.13342874481219422,-0.7124021117092761,-0.36299183655352807,0.09346403494516953,0.07021919531405203,0.16373434482103708,0.03837032435789075,-1.1538565697709744,0.13417171057420008,-0.7016155502041279,-0.6618323335678286,-0.9865933891085502,0.3274723937634512,1.1737063100570266,-0.2875078099112379,1.2040847342651186,-0.9459525402221014,0.2953917597113292,-0.4330947429028067,-0.38230965990842997,-1.2801238620388649,0.9365257888947798,0.1344943330746675,-0.8502950587206446,-0.8718288399575301,-0.353970306049839,-0.31117856711273534,-0.7628489410332483,-0.4738156189023931,1.1126767000304014,0.08528453055427505,-1.204833328879136,-0.26304122617372677,-0.1455598136150514,0.34070752347167094,-0.4539948123539838,-1.0577324186633497,-0.6433961087323378,-0.9705788505010019,0.026418856292268388,0.2295961315713437,0.18584013750560077,3.2743675305899425,-0.9701108804078329,0.2781149853904203,-0.9651334573594379,0.2069595446038963,-0.5464187749881334,0.20155096250372861,0.21991491070555463,-0.22424059972856264,0.10733895178173876,-0.11084587891443065,1.0324638973792981,0.22527997041604672,0.13894781856522895,-0.31638574270550013,-0.9055897733669893,-0.8375512656355872,2.224648517759994,-0.8651648972896057,-0.1983370401514494,-1.024731919403244,-0.06535080796328617,0.1510371469704551,0.08995417370545296,-0.9043900317447829,-0.0074119067537377435,-1.1806488659458352,0.3929605184789557,2.8551898859197986,-1.0001831838950235,0.8673852338904307,0.07002718762726301,-1.59109959150605,0.21524401065027296,-0.1866481961516452,0.3480303707886812,-0.4748491931778339,0.1572544736530452,-0.07021471809065838,-0.12048788272547677,0.06171126140013491,0.3350999534929519,-0.32377226684378774,0.13445010820329192,0.16629124459413325,2.0879152019354876,-1.1216771070324945,-1.0924616608737399,-0.7341803512830062,-0.8141319453609798,0.37733158657825466,0.3263006004127294,0.2791776789764618,-0.8436173035721813,0.18369864199131777,0.44631307941511883,0.1414310056613278,-1.2301153935335682,-1.1987052461349073,-0.024101569778516164,0.09391000202826578,-0.9642973426621982,1.1396424486824952,-0.12345558835336877,-0.3527441458011801,-0.3428969960635131,-0.8214525333885849,-0.3119604938287792,3.9144742135191595,2.959237314973591,0.0933109547188113,-0.2925495759128416,0.18263106743989108,-0.4616191606037044,-0.6810151991699064,0.2053826726205181,-0.3191556482175792,1.0628163527239791,0.3823316879557492,2.68920971013126,2.989831342136312,-0.22734958688384732,0.6042097486873802,1.1799301296963747,0.6255195586214954,0.2839712270674825,-1.2514105909030873,-1.3073306912467866,0.0005057869969714775,0.32248153642021676,1.2000576696479164,-0.06178712193841795,-0.7939209647377641,-1.2256680923515546,-0.8711449958193722,-0.07741267428026122,0.4671081387275517,0.132504597719958,0.8042929480274125,-0.5208126446494978,0.9836959094592712,-0.1402067268639604,-0.9111479387275403,-1.168348040302194,0.2938214833196812,-1.054952347167958,-1.016751595325342,2.2015799039897948,1.251847306767333,-0.4341026748552261,0.21719960593229756,1.4172869670664652,0.09072540003805059,0.11498753694859405,-0.04669586266963786,0.9356848967229596,0.06931147639965014,-0.9654119661919862,-0.7924003887134056,0.2729208913355022,-0.5615991599348141,-1.194773800328488,0.15764823934978575,-1.1436798355391433,0.18028095571616284,-0.7841795827183212,0.2006591507086026,0.8642515733038186,1.104696815872799,-0.1872536048621842,-0.939254637606315,-0.8958698583094059,0.13238279620733537,0.6255195586214954,-0.10297777485344,-0.15442908920744977,0.057258480085715956,-0.7482280781991405,0.21910753370480682,-0.11093617420336686,-1.070516390354256,-1.124034728714116,0.6365510579883297,-0.34413742981360007,-0.6673196289525574,0.08257878389222657,-1.2620069851911422,-1.1325757967108818,0.011286542731520374,-0.015086225691162783,-0.28272337948524084,0.07331953799822051,0.786732844038965,0.2447834935038646,-0.48671740034435573,0.039414222083034765,0.4314046392025342,-0.9651334573594379,-1.2574926860624571,-0.4293279911942197,-0.2523822134256109,0.32592489325689755,1.0957915939576974,-1.0758775229378177,-0.5497923159723292,0.09257128300292523,0.044589932132486305,0.2417487800624986,0.12401110864132021,-0.765624643304956,0.34419629753943237,0.5055834964086613,-0.9458057884291062,0.3006311656359559,0.1615893676363206,-0.012149511336845272,-0.6340130705763891,0.26197078666348034,0.3862127398777866,0.018922998198654582,-0.03536785744145935,-0.9687130321293916,-0.9981587862868768,0.27157130527322965,-0.4637331387558318,0.2055503193710262,-1.2338824690561447,1.0734052822412439,0.047519673259320015,-0.23116323733504548,-1.2693803423227195,-0.2581183966773876,0.7204695849153416,-0.051636576692997684,-1.212669341196882,0.6738634093104432,-0.6037982984324117,-0.43724505749304166,-0.07666360556796563,3.1861847885168664,0.1647623473403997,-0.6497133134515315,-0.7438821610942604,2.991008371537199,-0.016322766357871978,0.18285503080752194,-0.6426234681590077,0.2278921521793131,1.6717853866626615,0.34493602015474395,-0.9998053209869543,0.3964580518746758,-0.00323397043240747,0.5482495462395991,-0.09490139507674154,0.20829543354356658,-1.0908360806893214,-0.62318697593922,-0.5981920434878624,0.43251561423476687,-0.5301638880194048,-0.6164898643965205,1.1086220478777213,0.8632612254464443,0.02281407171282742,0.1969900211756239,-0.36321191204652115,2.8857871490449365,0.16746098346060576,-1.1744303579612023,0.0222039546690734,-1.0090621214880477,0.07426567137690739,0.2827512827504112,0.406628747221093,0.05508978886911609,-1.1714622214239816,-1.2379173604345886,0.131414861789873,2.2387267866661835,-0.1774107354509276,0.06110806589219176,0.18573078249604744,-0.5494817480978961,0.11787885053592946,1.132967936727089,-1.156210844988733,0.6615456945144413,-0.9963404787678606,1.9226593327290713,0.12758327712031034,0.8893288160217444,-0.4504995807547352,-0.26788389214476477,1.2973685200921934,0.12193060300186559,-1.489949835418777,-0.657793961993395,0.009823681543405424,0.2957041697894252,-0.620841884977576,0.04943707024576446,0.6870704676302273,-0.5529616260319002,-1.1883144185935117,0.11099020579866732,0.18631801690912894,1.2467172211066966,-1.1496635357776455,0.009913927317122487,-1.224906000407603,2.128799942984274,-1.2795859181379707,2.2200493789896525,0.27146305831623635,0.04452457178201837,0.16294706269374457,0.24731012323309773,0.16829398588462816,-0.9817664925463059,3.9050466048752455,-0.2851474778324876,-1.4648659620088198,0.2939928125674231,1.2623852898020165,-0.6497098361117029,0.18642573575055954,-0.9900135560969178,-0.12352724957840265,-0.574772872463411,0.11980036435734638,0.09533481507066317,-1.0674423867624867,0.14668919843588155,0.3173102362723337,0.10183700890823733,0.2478766835377368,0.22419529955452508,3.040813747678115,0.24640258899412246,0.2834042974119901,-1.0612562828177172,-0.002298602247927475,-0.8818326892097452,0.17139859122302692,-1.076388341309717,-0.004998900446973523,-0.042665925858311395,-0.1362275853070327,-0.6447186578570028,-0.970514632948657,-0.14307683747891753,-0.5805776165341362,-0.8224661839482972,0.2395722325694055,0.48333866869448033,0.5147731331629413,-0.22424059972856264,-1.1713170568561384,3.3505933541779394,-0.06893487793003615,0.927249286101765,-0.07014317743230085,-1.2870621723514901,-0.9199727793444303,-1.3359601776985057,1.0476322400954505,0.9480377878666787,0.4387179031996288,1.1865568860372393,0.6365510579883297,-0.4073322956493258,-0.3084484109013819,-0.521499791891109,-0.449601152533651,-0.11675926377563915,1.1566027171445625,2.324131117542378,-0.017741324522275594,1.104696815872799,0.7830368118727541,-0.14878542596457597,-0.7835804517203331,0.3462557676547307,-0.5205267702063591,-0.30316615830223015,0.04251366244429685,-0.5485577151116404,1.0791208956237561,-0.09254263384928206,0.5281835516636093,1.0735253185803455,0.3206580436182102,2.977153970946558,1.2851706400137566,-0.6331248156551454,0.595558915825959,-0.426912200338649,0.2816985013308903,0.2952297886038278,1.0773115092272054,0.2555620106509859,-0.09084297398959167,0.13619105775367754,-1.0308014973139736,0.5659691139880363,-0.43211857727543695,-0.9687319932156943,-0.8648335224743096,-0.13264370902667985,2.7103167022197643,-0.91023465563995,-0.1545542834468695,0.2223652997521753,1.1561980048304736,0.11116363847362938,0.2515522675235538,0.27933045556492236,0.05509660924439793,0.07066271249750618,-0.49487075482500753,-0.9215780769013178,-1.3066376301072804,0.07113168715243053,0.28376683876670444,0.2869367061512033,1.0614522602679517,0.17705413628364006,0.9541977208006155,-0.993539450252898,0.14729532479390037,-1.223242935466967,0.8133296359596741,-0.5191762832492608,-1.3637214571374918,-0.8689777454802925,0.11223298920987664,-0.05376679028263164,-0.6011488086532151,-1.4593286358059914,0.14807775298871914,0.06381604200093556,-0.6130738449269343,-0.21673963085339695,-0.1526343990888204,0.8068996994793791,0.29752594095444523,-1.0967026445641308,0.20956007111330496,0.2565268782857319,-0.7037016112257996,0.03467286819160372,0.07943898926065902,0.18345133662269328,-1.2930403687233236,-1.1747706758065344,-1.0476665810829975,-0.07687324721726833,-0.7799949700167624,0.03875636392803324,-1.446033417690018,-0.6353718313554314,0.03516048409719632,-0.8895232304192263,0.027189210367157067,0.19703377160113597,-0.6673827517475546,-0.8752314056822552,0.27157130527322965,0.40063295306038965,-0.9844147102994933,0.2687200921832818,0.25177224020544947,0.19740738572243585,0.05060293167665016,-0.5414324938501922,-1.3830710639148862,-1.1090967350731908,-0.9323238744571833,-0.14020122870961416,0.08570645025215347,0.24817905833236129,0.29027971478965914,0.3426074878946505,-0.5062244952518364,1.9986638739610758,1.0614522602679517,0.0817567070313256,-0.9680502732067339,-0.45115800970600173,1.9769769596228688,1.5352683554891076,0.044733492751555325,-1.1963125251357103,1.3359092009792664,1.0677986793961918,0.264631676451662,0.2161313016901347,-0.14878542596457597,1.2407414103935042,-0.3093442072737969,0.9053687621283241,-0.8930986559612231,0.15074683598647148,0.1516729857415224,0.024823236535131098,-0.5866266497843085,-1.142470151685219,0.15696955247611263,-1.2185036551136832,-1.0656193155065892,0.31463495584694595,-0.022109035831414527,0.3214915174558102,-0.11861902853854266,-0.8651992398835335,1.0019994551300315,1.814284512240643,-0.6851087289378517,1.0646689699155645,0.004122799184377638,0.2539292585423744,0.22477147324427746,1.0411242752328402,0.20050668191251994,-0.7348561267102623,-0.5015928446665849,0.6873326427647756,0.31099603422710614,-0.2787087060972036,0.9715387409148624,0.3213743747611834,0.09319851134642916,-0.1537108904321076,0.18826716077060143,0.8523824112205224,1.1033702373823542,0.23552486319461416,-0.13837474259686147,-0.5764549662036343,-0.014501411508649623,-0.019495726565566508,0.24074862375189632,0.05617051816399773,0.6873326427647756,0.2295961315713437,0.07393043424792985,-0.24705679306434383,0.20065473812450893,0.16593868089640745,-0.35730035934604815,-0.14617581258141144,-0.4346018874884869,-0.6432238967878359,-0.10521430924150865,0.3074166230783521,0.0755770892087972,-0.9458057884291062,4.3408164147634665,2.067769339407193,0.13470681382061098,0.28196488782411966,2.9057763957008538,-0.9431638899685493,-0.7921658072227814,1.1386988421013418,2.0467043939091565,-0.9436047446195254,-0.6070140272525326,0.5524981765225347,-0.18751965657648725,0.07793080121318048,0.34777131243159104,0.029715439252258745,0.07309267094318724,0.0450518122671801,-0.21825029761375914,-0.04355916233570955,-0.2265910743424484,-0.7348561267102623,0.1784978997971642,-0.32605450047435935,0.1431882782945376,0.30248611697063416,1.7641254985939105,0.9809721195054927,0.17355825475019646,-0.02484094717336164,0.44138187055933886,1.3759643366125804,-0.058136474891398206,0.19871764295426061,-1.037494340619882,0.4202125062446758,-1.0447221470627777,0.27146305831623635,-0.2953429904755237,-0.8351661188327957,1.0098641363990277,0.35489817779630123,0.06276369149074341,-0.7027584267654922,-1.0162792811708332,0.07474354176946525,-0.9794014357604792,0.6192096656695569,0.22700116620283736,0.15190437948663002,-0.8759859498441017,0.34070752347167094,-0.4459386111856373,-0.6434441381859414,2.0484919882092365,0.13593494969917316,0.017808286077479495,0.17849100091729703,-0.8458311644876408,-1.080798096000913,-0.689071582333643,0.184450856294313,1.2239374529139608,-1.0173092267948642,0.3072193753846139,0.2447683279888195,-0.7656814676854249,0.15290177159815388,0.26062963120983484,-1.0071268728959415,0.782714263700089],"yaxis":"y","type":"scattergl"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"xaxis":{"anchor":"y","domain":[0.0,1.0],"title":{"text":"x1"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"x2"}},"coloraxis":{"colorbar":{"title":{"text":"Cluster"}},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"legend":{"tracegroupgap":0},"margin":{"t":60},"height":800,"width":800},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('446bd847-0b26-4204-8e88-863ab442633a');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


## **e. Analisis dan Interpretasi Hasil Cluster**

### - Distribusi Kolom Numerik




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Income</th>
      <th>Recency</th>
      <th>MntWines</th>
      <th>MntFruits</th>
      <th>MntMeatProducts</th>
      <th>MntFishProducts</th>
      <th>MntSweetProducts</th>
      <th>MntGoldProds</th>
      <th>NumDealsPurchases</th>
      <th>NumWebPurchases</th>
      <th>...</th>
      <th>Z_CostContact</th>
      <th>Z_Revenue</th>
      <th>Response</th>
      <th>Age</th>
      <th>Spending</th>
      <th>Accept_Offer</th>
      <th>Num_Purchase</th>
      <th>Enrolled_Days</th>
      <th>Children</th>
      <th>Cluster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1207.000000</td>
      <td>1207.000000</td>
      <td>1207.000000</td>
      <td>1207.000000</td>
      <td>1207.000000</td>
      <td>1207.000000</td>
      <td>1207.000000</td>
      <td>1207.000000</td>
      <td>1207.000000</td>
      <td>1207.000000</td>
      <td>...</td>
      <td>1207.0</td>
      <td>1207.0</td>
      <td>1207.000000</td>
      <td>1207.000000</td>
      <td>1207.000000</td>
      <td>1207.000000</td>
      <td>1207.000000</td>
      <td>1207.000000</td>
      <td>1207.000000</td>
      <td>1207.0</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>36488.717481</td>
      <td>48.515327</td>
      <td>68.164043</td>
      <td>6.891466</td>
      <td>31.952775</td>
      <td>10.136703</td>
      <td>6.748136</td>
      <td>21.077051</td>
      <td>2.268434</td>
      <td>2.628003</td>
      <td>...</td>
      <td>3.0</td>
      <td>11.0</td>
      <td>0.104391</td>
      <td>53.425021</td>
      <td>144.970174</td>
      <td>0.207954</td>
      <td>9.274234</td>
      <td>334.732394</td>
      <td>1.213753</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>std</th>
      <td>12800.387061</td>
      <td>29.131791</td>
      <td>88.456891</td>
      <td>12.628405</td>
      <td>36.688080</td>
      <td>18.465567</td>
      <td>12.133770</td>
      <td>30.311154</td>
      <td>1.580097</td>
      <td>1.976221</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.305894</td>
      <td>11.142562</td>
      <td>143.603814</td>
      <td>0.499422</td>
      <td>4.444445</td>
      <td>201.121179</td>
      <td>0.695193</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1730.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>3.0</td>
      <td>11.0</td>
      <td>0.000000</td>
      <td>28.000000</td>
      <td>5.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>27208.000000</td>
      <td>24.000000</td>
      <td>10.000000</td>
      <td>1.000000</td>
      <td>9.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>5.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>3.0</td>
      <td>11.0</td>
      <td>0.000000</td>
      <td>46.000000</td>
      <td>43.000000</td>
      <td>0.000000</td>
      <td>6.000000</td>
      <td>154.000000</td>
      <td>1.000000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>36736.000000</td>
      <td>49.000000</td>
      <td>29.000000</td>
      <td>3.000000</td>
      <td>18.000000</td>
      <td>4.000000</td>
      <td>3.000000</td>
      <td>12.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>...</td>
      <td>3.0</td>
      <td>11.0</td>
      <td>0.000000</td>
      <td>52.000000</td>
      <td>78.000000</td>
      <td>0.000000</td>
      <td>8.000000</td>
      <td>326.000000</td>
      <td>1.000000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>45863.000000</td>
      <td>74.000000</td>
      <td>96.500000</td>
      <td>7.000000</td>
      <td>43.000000</td>
      <td>11.000000</td>
      <td>8.000000</td>
      <td>25.500000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>...</td>
      <td>3.0</td>
      <td>11.0</td>
      <td>0.000000</td>
      <td>61.000000</td>
      <td>218.500000</td>
      <td>0.000000</td>
      <td>12.000000</td>
      <td>502.000000</td>
      <td>2.000000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>max</th>
      <td>71322.000000</td>
      <td>99.000000</td>
      <td>691.000000</td>
      <td>151.000000</td>
      <td>270.000000</td>
      <td>179.000000</td>
      <td>157.000000</td>
      <td>321.000000</td>
      <td>15.000000</td>
      <td>25.000000</td>
      <td>...</td>
      <td>3.0</td>
      <td>11.0</td>
      <td>1.000000</td>
      <td>84.000000</td>
      <td>835.000000</td>
      <td>3.000000</td>
      <td>25.000000</td>
      <td>699.000000</td>
      <td>3.000000</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 29 columns</p>
</div>






<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Income</th>
      <th>Recency</th>
      <th>MntWines</th>
      <th>MntFruits</th>
      <th>MntMeatProducts</th>
      <th>MntFishProducts</th>
      <th>MntSweetProducts</th>
      <th>MntGoldProds</th>
      <th>NumDealsPurchases</th>
      <th>NumWebPurchases</th>
      <th>...</th>
      <th>Z_CostContact</th>
      <th>Z_Revenue</th>
      <th>Response</th>
      <th>Age</th>
      <th>Spending</th>
      <th>Accept_Offer</th>
      <th>Num_Purchase</th>
      <th>Enrolled_Days</th>
      <th>Children</th>
      <th>Cluster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>841.000000</td>
      <td>841.000000</td>
      <td>841.000000</td>
      <td>841.000000</td>
      <td>841.000000</td>
      <td>841.000000</td>
      <td>841.000000</td>
      <td>841.000000</td>
      <td>841.000000</td>
      <td>841.000000</td>
      <td>...</td>
      <td>841.0</td>
      <td>841.0</td>
      <td>841.000000</td>
      <td>841.000000</td>
      <td>841.000000</td>
      <td>841.000000</td>
      <td>841.000000</td>
      <td>841.000000</td>
      <td>841.000000</td>
      <td>841.0</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>67879.845422</td>
      <td>50.380499</td>
      <td>539.640904</td>
      <td>48.843044</td>
      <td>300.523187</td>
      <td>69.116528</td>
      <td>49.894174</td>
      <td>70.598098</td>
      <td>2.582640</td>
      <td>5.895363</td>
      <td>...</td>
      <td>3.0</td>
      <td>11.0</td>
      <td>0.102259</td>
      <td>57.694411</td>
      <td>1078.615933</td>
      <td>0.330559</td>
      <td>21.717004</td>
      <td>375.865636</td>
      <td>0.699168</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>std</th>
      <td>10848.238322</td>
      <td>28.414874</td>
      <td>262.520610</td>
      <td>47.081228</td>
      <td>221.360576</td>
      <td>64.222887</td>
      <td>48.823682</td>
      <td>57.327486</td>
      <td>2.273163</td>
      <td>2.499712</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.303169</td>
      <td>11.593331</td>
      <td>403.173147</td>
      <td>0.550005</td>
      <td>4.503788</td>
      <td>203.175076</td>
      <td>0.685643</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2447.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>3.0</td>
      <td>11.0</td>
      <td>0.000000</td>
      <td>29.000000</td>
      <td>277.000000</td>
      <td>0.000000</td>
      <td>10.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>60597.000000</td>
      <td>27.000000</td>
      <td>347.000000</td>
      <td>12.000000</td>
      <td>132.000000</td>
      <td>17.000000</td>
      <td>12.000000</td>
      <td>28.000000</td>
      <td>1.000000</td>
      <td>4.000000</td>
      <td>...</td>
      <td>3.0</td>
      <td>11.0</td>
      <td>0.000000</td>
      <td>49.000000</td>
      <td>777.000000</td>
      <td>0.000000</td>
      <td>19.000000</td>
      <td>209.000000</td>
      <td>0.000000</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>67546.000000</td>
      <td>52.000000</td>
      <td>499.000000</td>
      <td>33.000000</td>
      <td>235.000000</td>
      <td>50.000000</td>
      <td>33.000000</td>
      <td>53.000000</td>
      <td>2.000000</td>
      <td>6.000000</td>
      <td>...</td>
      <td>3.0</td>
      <td>11.0</td>
      <td>0.000000</td>
      <td>58.000000</td>
      <td>1029.000000</td>
      <td>0.000000</td>
      <td>22.000000</td>
      <td>395.000000</td>
      <td>1.000000</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>75702.000000</td>
      <td>73.000000</td>
      <td>708.000000</td>
      <td>74.000000</td>
      <td>419.000000</td>
      <td>104.000000</td>
      <td>76.000000</td>
      <td>100.000000</td>
      <td>3.000000</td>
      <td>8.000000</td>
      <td>...</td>
      <td>3.0</td>
      <td>11.0</td>
      <td>0.000000</td>
      <td>67.000000</td>
      <td>1336.000000</td>
      <td>1.000000</td>
      <td>25.000000</td>
      <td>551.000000</td>
      <td>1.000000</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>max</th>
      <td>113734.000000</td>
      <td>99.000000</td>
      <td>1449.000000</td>
      <td>199.000000</td>
      <td>1725.000000</td>
      <td>259.000000</td>
      <td>262.000000</td>
      <td>249.000000</td>
      <td>15.000000</td>
      <td>27.000000</td>
      <td>...</td>
      <td>3.0</td>
      <td>11.0</td>
      <td>1.000000</td>
      <td>83.000000</td>
      <td>2525.000000</td>
      <td>2.000000</td>
      <td>43.000000</td>
      <td>698.000000</td>
      <td>3.000000</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 29 columns</p>
</div>






<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Income</th>
      <th>Recency</th>
      <th>MntWines</th>
      <th>MntFruits</th>
      <th>MntMeatProducts</th>
      <th>MntFishProducts</th>
      <th>MntSweetProducts</th>
      <th>MntGoldProds</th>
      <th>NumDealsPurchases</th>
      <th>NumWebPurchases</th>
      <th>...</th>
      <th>Z_CostContact</th>
      <th>Z_Revenue</th>
      <th>Response</th>
      <th>Age</th>
      <th>Spending</th>
      <th>Accept_Offer</th>
      <th>Num_Purchase</th>
      <th>Enrolled_Days</th>
      <th>Children</th>
      <th>Cluster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>157.000000</td>
      <td>157.000000</td>
      <td>157.000000</td>
      <td>157.000000</td>
      <td>157.000000</td>
      <td>157.000000</td>
      <td>157.000000</td>
      <td>157.00000</td>
      <td>157.000000</td>
      <td>157.000000</td>
      <td>...</td>
      <td>157.0</td>
      <td>157.0</td>
      <td>157.000000</td>
      <td>157.000000</td>
      <td>157.000000</td>
      <td>157.000000</td>
      <td>157.000000</td>
      <td>157.000000</td>
      <td>157.000000</td>
      <td>157.0</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>80878.261146</td>
      <td>45.458599</td>
      <td>885.229299</td>
      <td>56.203822</td>
      <td>466.280255</td>
      <td>82.108280</td>
      <td>61.859873</td>
      <td>78.55414</td>
      <td>1.286624</td>
      <td>5.808917</td>
      <td>...</td>
      <td>3.0</td>
      <td>11.0</td>
      <td>0.770701</td>
      <td>54.019108</td>
      <td>1630.235669</td>
      <td>2.955414</td>
      <td>21.464968</td>
      <td>381.044586</td>
      <td>0.248408</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>std</th>
      <td>10137.546690</td>
      <td>29.885999</td>
      <td>328.704155</td>
      <td>51.376531</td>
      <td>261.768121</td>
      <td>66.607786</td>
      <td>52.013245</td>
      <td>62.73718</td>
      <td>1.291247</td>
      <td>2.224957</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.421727</td>
      <td>13.900042</td>
      <td>418.854836</td>
      <td>0.942881</td>
      <td>4.315340</td>
      <td>196.313765</td>
      <td>0.538945</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>min</th>
      <td>48192.000000</td>
      <td>0.000000</td>
      <td>152.000000</td>
      <td>0.000000</td>
      <td>45.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>...</td>
      <td>3.0</td>
      <td>11.0</td>
      <td>0.000000</td>
      <td>29.000000</td>
      <td>416.000000</td>
      <td>2.000000</td>
      <td>12.000000</td>
      <td>6.000000</td>
      <td>0.000000</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>75261.000000</td>
      <td>19.000000</td>
      <td>693.000000</td>
      <td>21.000000</td>
      <td>265.000000</td>
      <td>31.000000</td>
      <td>24.000000</td>
      <td>33.00000</td>
      <td>1.000000</td>
      <td>4.000000</td>
      <td>...</td>
      <td>3.0</td>
      <td>11.0</td>
      <td>1.000000</td>
      <td>42.000000</td>
      <td>1379.000000</td>
      <td>2.000000</td>
      <td>18.000000</td>
      <td>218.000000</td>
      <td>0.000000</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>81574.000000</td>
      <td>40.000000</td>
      <td>934.000000</td>
      <td>35.000000</td>
      <td>449.000000</td>
      <td>59.000000</td>
      <td>43.000000</td>
      <td>56.00000</td>
      <td>1.000000</td>
      <td>5.000000</td>
      <td>...</td>
      <td>3.0</td>
      <td>11.0</td>
      <td>1.000000</td>
      <td>53.000000</td>
      <td>1676.000000</td>
      <td>3.000000</td>
      <td>21.000000</td>
      <td>403.000000</td>
      <td>0.000000</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>87679.000000</td>
      <td>71.000000</td>
      <td>1092.000000</td>
      <td>80.000000</td>
      <td>687.000000</td>
      <td>120.000000</td>
      <td>95.000000</td>
      <td>119.00000</td>
      <td>1.000000</td>
      <td>7.000000</td>
      <td>...</td>
      <td>3.0</td>
      <td>11.0</td>
      <td>1.000000</td>
      <td>66.000000</td>
      <td>1919.000000</td>
      <td>4.000000</td>
      <td>25.000000</td>
      <td>557.000000</td>
      <td>0.000000</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>max</th>
      <td>105471.000000</td>
      <td>99.000000</td>
      <td>1493.000000</td>
      <td>190.000000</td>
      <td>974.000000</td>
      <td>250.000000</td>
      <td>194.000000</td>
      <td>245.00000</td>
      <td>11.000000</td>
      <td>11.000000</td>
      <td>...</td>
      <td>3.0</td>
      <td>11.0</td>
      <td>1.000000</td>
      <td>80.000000</td>
      <td>2525.000000</td>
      <td>5.000000</td>
      <td>34.000000</td>
      <td>697.000000</td>
      <td>3.000000</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 29 columns</p>
</div>



### - Sebaran Income tiap Cluster


    
![png](output_83_0.png)
    


### - Sebaran Spending tiap Cluster


    
![png](output_85_0.png)
    


### - Sebaran Jumlah Pembelian tiap Cluster


    
![png](output_87_0.png)
    


### - Sebaran Jumlah Anak tiap Cluster


    
![png](output_89_0.png)
    


### - Jumlah Website Diakses tiap Cluster


    
![png](output_91_0.png)
    


### - Perbandingan Education tiap Cluster


    
![png](output_94_0.png)
    


### - Perbandingan Marital Status tiap Cluster


    
![png](output_96_0.png)
    


### - Perbandingan Tempat Belanja tiap Cluster


    
![png](output_98_0.png)
    



    
![png](output_99_0.png)
    


### - Perbandingan Produk yang Sering Dibeli tiap Cluster


    
![png](output_101_0.png)
    



    
![png](output_102_0.png)
    


### Hasil Interpretasi



1. **Cluster 1**:
   - Memiliki pendapatan rendah, berkisar antara `$1,730 - $71,322` dengan rata-rata `$36,488`.
   - Melakukan pengeluaran rendah, yaitu `$5 - $835` dengan rata-rata sekitar `$144.97`.
   - Frekuensi pembelian rendah, antara 0 - 25 kali dengan rata-rata 9 kali per pelanggan.
   - Jumlah anak bervariasi antara 0 - 3, dengan mayoritas memiliki 1 - 2 anak.
   - Status pernikahan sebagian besar adalah Married (Menikah) dan Relationship (Berkomitmen).

2. **Cluster 2**:
   - Memiliki pendapatan menengah, yaitu `$2,447 - $113,734` dengan rata-rata `$67,879`.
   - Melakukan pengeluaran menengah, sekitar `$277 - $2,525` dengan rata-rata `$1,078`.
   - Frekuensi pembelian cukup sering, antara 10 - 43 kali dengan rata-rata 21 kali per pelanggan.
   - Jumlah anak berkisar 0 - 3, dengan mayoritas memiliki 0 - 1 anak.
   - Status pernikahan sebagian besar adalah Married (Menikah) dan Relationship (Berkomitmen).

3. **Cluster 3**:
   - Memiliki pendapatan tinggi, yaitu `$48,192 - $105,471` dengan rata-rata `$80,878`.
   - Melakukan pengeluaran tinggi, sekitar `$416 - $2,525` dengan rata-rata `$1,630`.
   - Frekuensi pembelian sering, antara 12 - 34 kali dengan rata-rata 21 kali per pelanggan.
   - Mayoritas pelanggan tidak memiliki anak.
   - Status pernikahan sebagian besar adalah Married (Menikah) dan Single (Lajang).

**Tambahan**:
- Pelanggan pada **Cluster 1** merupakan yang paling aktif mengakses website.
- Pelanggan pada **Cluster 3** lebih memilih berbelanja melalui Catalog daripada Website. 



# **7. Mengeksport Data**

Simpan hasilnya ke dalam file CSV.
---

> 📘 *Dokumen ini dihasilkan dari Jupyter Notebook menggunakan konversi otomatis ke format Markdown. Untuk eksplorasi lebih lanjut, silakan akses notebook aslinya.*

