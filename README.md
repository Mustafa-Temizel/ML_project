# Machine Learning with used car Data Set

# Projenin Amacı:
Bu proje, günümüzde bilindik 2. el alım-satım yapan internet sitelerinin de kullandığı ortalama araç fiyat tahminini sağlar. Aracınızın şu ana kadar satılmış olanlarından yola çıkarak size ortalama fiyatını söyleyebilir.

# Veri Seti:
Veri setimiz 852122 satırdan ve 8 satırdan oluşmaktadır. Parametrelerimiz şu şekildedir.
  * **Price:** Aracın satılan fiyatını söyler.
  * **Year:** Aracın üretim tarihini söyler.
  * **Mileage:** Aracın satılmadan önce ne kadar sürüldüğü hakkında bilgi verir.
  * **City:** Aracın hangi şehirde satıldığını söyler.
  * **State:** Aracın hangi eyalette satıldığını söyler.
  * **Vin:** Aracın seri numarasıdır.
  * **Make:** Aracın markasını söyler.
  * **Model:** Aracın hangi model olduğunu söyler.

# Algoritmalar:
Bu projede kullanılan algoritmalar şunlardır:

```python
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.decomposition import PCA
```

# Model Seçimi
Projede gözetimli ve gözetimsiz deneylerde farklı taktikler kullanılarak modelleme yapılmıştır.Gözetimli modelde farklı Algoritmalar kullanılarak **r2_score** algoritması ile en iyisi seçilmiştir ve bu da **KneighborsRegressor**  olmuştur. Gözetimsiz modelde ise **KMeans** algoritması kullanılarak dirsek methodu ile alabileceği en iyi sayı seçilmiştir.

# Gözetimli ve Gözetimsiz tercihi
Projemizde belli gruplandırma parametreleri bulunmadığı ve bağımlı değişkenimiz oldukça geniş bir aralıkta yüksek değerler aldığı için gözetimsiz öğrenme modeline pek uygun değildi. Her ne kadar string ifadelerden kurtulup veri setimizi optimize etsek de değer aralıklarından dolayı gruplandırmaya değil tahmin edilmeye daha uygun bir veri setiydi.

# Kaggle Notebook Linki:
https://www.kaggle.com/code/mustafatemiizel/ml-project-aygaz
