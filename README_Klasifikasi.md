# 🎯 Submission Akhir - Klasifikasi BMLP
**Nama:** Febhe Maulita May Pramasta  
**Proyek:** Klasifikasi Data Menggunakan Machine Learning  
**Program:** Laskar AI 2025

---

# **1. Import Library**
Pada tahap ini,kita perlu mengimpor beberapa pustaka (library) Python yang dibutuhkan untuk analisis data dan pembangunan model machine learning.

    C:\Users\ASUS\anaconda3\Lib\site-packages\pandas\core\arrays\masked.py:60: UserWarning: Pandas requires version '1.3.6' or newer of 'bottleneck' (version '1.3.5' currently installed).
      from pandas.core import (


# **2. Memuat Dataset dari Hasil Clustering**

Memuat dataset hasil clustering dari file CSV ke dalam variabel DataFrame.




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
  </tbody>
</table>
<p>5 rows × 31 columns</p>
</div>



    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2205 entries, 0 to 2204
    Data columns (total 31 columns):
     #   Column               Non-Null Count  Dtype  
    ---  ------               --------------  -----  
     0   Education            2205 non-null   object 
     1   Marital_Status       2205 non-null   object 
     2   Income               2205 non-null   float64
     3   Recency              2205 non-null   int64  
     4   MntWines             2205 non-null   int64  
     5   MntFruits            2205 non-null   int64  
     6   MntMeatProducts      2205 non-null   int64  
     7   MntFishProducts      2205 non-null   int64  
     8   MntSweetProducts     2205 non-null   int64  
     9   MntGoldProds         2205 non-null   int64  
     10  NumDealsPurchases    2205 non-null   int64  
     11  NumWebPurchases      2205 non-null   int64  
     12  NumCatalogPurchases  2205 non-null   int64  
     13  NumStorePurchases    2205 non-null   int64  
     14  Web_Visit            2205 non-null   int64  
     15  AcceptedCmp3         2205 non-null   int64  
     16  AcceptedCmp4         2205 non-null   int64  
     17  AcceptedCmp5         2205 non-null   int64  
     18  AcceptedCmp1         2205 non-null   int64  
     19  AcceptedCmp2         2205 non-null   int64  
     20  Complain             2205 non-null   int64  
     21  Z_CostContact        2205 non-null   int64  
     22  Z_Revenue            2205 non-null   int64  
     23  Response             2205 non-null   int64  
     24  Age                  2205 non-null   int64  
     25  Spending             2205 non-null   int64  
     26  Accept_Offer         2205 non-null   int64  
     27  Num_Purchase         2205 non-null   int64  
     28  Enrolled_Days        2205 non-null   int64  
     29  Children             2205 non-null   int64  
     30  Cluster              2205 non-null   int64  
    dtypes: float64(1), int64(28), object(2)
    memory usage: 534.2+ KB





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
      <th>2200</th>
      <td>Bachelor</td>
      <td>Married</td>
    </tr>
    <tr>
      <th>2201</th>
      <td>PhD</td>
      <td>Relationship</td>
    </tr>
    <tr>
      <th>2202</th>
      <td>Bachelor</td>
      <td>Divorced</td>
    </tr>
    <tr>
      <th>2203</th>
      <td>Master</td>
      <td>Relationship</td>
    </tr>
    <tr>
      <th>2204</th>
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
      <th>Cluster</th>
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
      <td>1</td>
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
      <td>2</td>
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
      <th>2200</th>
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
      <td>2</td>
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
      <th>2201</th>
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
      <td>2</td>
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
      <th>2202</th>
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
      <td>2</td>
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
      <th>2203</th>
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
      <td>2</td>
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
      <th>2204</th>
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
  </tbody>
</table>
<p>2205 rows × 38 columns</p>
</div>



# **3. Data Splitting**
Tahap Data Splitting bertujuan untuk memisahkan dataset menjadi dua bagian: data latih (training set) dan data uji (test set).

    Training set shape: X_train=(1323, 37), y_train=(1323,)
    Test set shape: X_test=(882, 37), y_test=(882,)





    0       1
    1       0
    2       1
    3       0
    4       1
           ..
    2200    1
    2201    1
    2202    1
    2203    1
    2204    0
    Name: Cluster, Length: 2205, dtype: int64



# **4. Membangun Model Klasifikasi**
Setelah memilih algoritma klasifikasi yang sesuai, langkah selanjutnya adalah melatih model menggunakan data latih.

## **a. Membangun Model Klasifikasi**

    Model training selesai.


## **b. Evaluasi Model Klasifikasi**

    ==== Decision Tree Classifier ====
    Confusion Matrix:
    [[475   9   0]
     [ 19 314   1]
     [  1   5  58]]
    Accuracy: 0.9603
    
    ----------------------------------------
    



    
![png](output_14_1.png)
    


    ==== Decision Tree Classifier ====
    Confusion Matrix:
    [[469  12   3]
     [ 18 257  59]
     [  0   0  64]]
    Accuracy: 0.8957
    
    ----------------------------------------
    



    
![png](output_15_1.png)
    


                    Model  Accuracy  Precision    Recall  F1-Score
    0  Decision Tree (DT)  0.960317   0.960435  0.960317  0.960166
    1    Naive Bayes (NB)  0.895692   0.927119  0.895692  0.901780


## **e. Analisis Hasil Evaluasi Model Klasifikasi**

1. Model Decision Tree menunjukkan performa yang sangat memuaskan dengan mencapai F1-Score sebesar 96%. Hal ini mengindikasikan bahwa model tersebut cukup efektif dan layak digunakan untuk keperluan klasifikasi.

2. Model Naive Bayes memberikan hasil yang sedikit lebih rendah dibandingkan dengan Decision Tree, dengan F1-Score sebesar 90%. Meskipun demikian, performanya masih tergolong baik.

3. Rekomendasi: Untuk meningkatkan kualitas analisis, disarankan untuk menambah volume data serta melakukan segmentasi pelanggan yang lebih variatif. Hal ini akan memberikan insight bisnis yang lebih mendetail dan akurat.
---

> 📘 *Dokumen ini dihasilkan dari Jupyter Notebook menggunakan konversi otomatis ke format Markdown. Untuk eksplorasi lebih lanjut, silakan akses notebook aslinya.*

