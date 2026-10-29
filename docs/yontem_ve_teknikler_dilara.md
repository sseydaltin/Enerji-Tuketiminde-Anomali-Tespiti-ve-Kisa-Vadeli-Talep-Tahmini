# Yöntem ve Teknikler – Dilara Yıldırım

## Araştırma yöntemi

Çalışma, tek bir haneye ait elektrik tüketimi zaman serisinde 24 saatlik çok
adımlı tahmin yöntemlerini karşılaştıran deneysel bir araştırma olarak
yürütülmektedir. Karşılaştırma; Persistence ve Seasonal Naive referansları,
klasik makine öğrenmesi yöntemleri Ridge ve MLPRegressor ile tekrarlayan sinir
ağları LSTM ve GRU'yu kapsar. Tüm yöntemlerde aynı kronolojik veri bölmesi,
168 saatlik geçmiş ve 24 saatlik tahmin ufku kullanılır. Temel bağımlı değişken,
gelecek 24 saatin `Global_active_power` (kW) değerleridir. Geçmiş tüketim ve
seçilen geçmiş/takvim öznitelikleri bağımsız değişkenler; seed, pencere
boyutları ve veri bölme tarihleri kontrol edilen koşullardır.

Çalışmanın deney protokolünde eğitim kümesi parametre öğrenimi için, doğrulama
kümesi erken durdurma ve model karşılaştırması için kullanılır. Test kümesi
yalnızca proje planında final değerlendirme aşamasına ayrılmıştır; bu görev
kapsamındaki eğitimler test üzerinde çalıştırılmaz ve test metrikleri
raporlanmaz.

## Veri seti ve ön işleme

Veri kaynağı, Fransa'nın Sceaux kentindeki tek hanenin 2006-12-16 ile
2010-11-26 arasındaki dakikalık ölçümlerini içeren UCI *Individual Household
Electric Power Consumption* veri kümesidir. Ham tablodaki tarih ve saat
birleştirilerek zaman indeksi kurulur; `?` değerleri eksik olarak okunur ve
ölçüm sütunları sayısal tipe dönüştürülür. Dakikalık veri saatliğe
indirgenirken aktif/reaktif güç, gerilim ve akımın saatlik ortalamaları;
alt sayaç ve türetilmiş `other_energy` değerlerinin saatlik toplamları
hesaplanır. Eksik saatler için proje yapılandırmasındaki kısa boşluk
interpolasyonu ve uzun boşluk haftalık gecikme doldurması uygulanır. Doldurulan
saatler `is_imputed` bayrağı ile saklanır. Eğitim pencerelerinde bu hedefler
kullanılabilir; doğrulama metrikleri hesaplanırken hedef ufkunda doldurulmuş
saat bulunan pencere bütünüyle dışarıda bırakılır.

Saatlik tablo, `configs/config.yaml` içinde tanımlı tarihlerle kronolojik olarak
train (2006-12-16–2009-09-30), val (2009-10-01–2010-03-31) ve test
(2010-04-01–2010-11-26) kümelerine ayrılır. Eğitim ve doğrulama pencerelerinin
hedefleri kendi kümeleri içinde kalır. Val penceresinin girdisi, tahmin anında
bilinen geçmiş olduğu için train döneminin son 168 saatini içerebilir.

## Zaman serisi özellikleri

Tahmin örneği, ardışık 168 saatlik geçmişten `X[t-168:t]` ve izleyen 24 saatlik
hedeften `y[t:t+24]` oluşturulur; pencere adımı bir saattir. Bu tasarım bir
haftalık bağlamı günlük tahmin ufkuna taşır ve bütün modeller için ortak
karşılaştırma zemini sağlar. Varsayılan B öznitelik seti `Global_active_power`
ile saat, haftanın günü ve ay için sinüs/kosinüs kodlamaları; hafta sonu ve
resmî tatil göstergelerinden oluşur. C setindeki alt sayaçlar, gerilim ve
reaktif güç daha sonraki ablasyonlarda kullanılabilir; bu metin bu ek
deneylerin yapılmış olduğunu ileri sürmez.

## Ridge ve MLP baseline modelleri

Ridge regresyon, yalnız geçmiş aktif güç serisinden config'te tanımlanmış
gecikmeleri (1–24, 48 ve 168 saat) öznitelik olarak alır ve 24 hedefi tek
regresyonda üretir. Düzenlileştirme katsayısı `alpha=1.0` yapılandırmadan
okunur. MLP baseline, aynı 168 saatlik tek değişkenli pencereyi 168 uzunluklu
vektöre dönüştürür; `MLPRegressor` iki gizli katmanlı `[128, 64]` ağ ile 24
çıktıyı eşzamanlı tahmin eder. Tahmin girdisi ve hedefi için kullanılan
`StandardScaler` nesneleri yalnız train pencereleri/hedefleri üzerinde fit
edilir. MLPRegressor'ın kendi rastgele validation ayırması kapalıdır; model
train verisinde fit edilir ve val kümesi yalnız son değerlendirme için
kullanılır.

## LSTM ve GRU mimarileri

LSTM ve GRU, varsayılan B öznitelik setinde `[batch, 168, 9]` biçimindeki
girdiyi alır. Her iki ağda da gizil boyut 64, katman sayısı 2 ve katmanlar arası
dropout 0.2'dir. Son recurrent katmanın son gizil durumu doğrusal katmana
aktarılır; bu katman izleyen 24 saat için `[batch, 24]` tahmin üretir.
Çıkışta aktivasyon kullanılmaz; model regresyon değerlerini verir. Mimari
parametreleri `configs/config.yaml` dosyasından okunur.

## Eğitim ve doğrulama stratejisi

Özellik ölçekleyicisi ve hedef ölçekleyicisi sadece train verisi üzerinde fit
edilir. Eğitim hedefleri train saatleri içinde kalır; train'in sonundaki geçmiş
val girdisi için kullanılabilir, fakat val hedefleri train/val sınırını
geçemez. Tekrarlanabilirlik için config seed'i tüm rastgelelik kaynaklarına
uygulanır. LSTM ve GRU, PyTorch ile açık epoch döngüsünde Adam optimizer ve
config'teki öğrenme oranı/ağırlık çürümesiyle eğitilir. Kayıp, ölçeklenmiş
hedeflerde MSE'dir. Her epoch için train MSE, val MSE ve kW ölçeğinde val MAE
kaydedilir. ReduceLROnPlateau val kaybını izler; gradyan normu 1.0 ile
kırpılır; erken durdurma sabrı 8 epoch ve model seçme ölçütü val MAE'dir. En
iyi ağırlıklar checkpoint olarak saklanır; train/val kayıp eğrileri PNG olarak
üretilir.

F3/F4 baseline'ları ile F5/F6 recurrent modellerinin sonuçları
`results/experiments.csv` ve `results/tables/` altındaki CSV dosyalarında
deney kimliğiyle tutulur. Her epoch geçmişi ayrı CSV, öğrenme eğrisi model
başına PNG olarak kaydedilir. Hiperparametre ayarı ve final karşılaştırma
yalnız val üzerinde yapılır; test sütunları final değerlendirme yapılana kadar
boş bırakılır.

## Performans metrikleri

Başarım, inverse-transform sonrası özgün kW ölçeğinde MAE ve RMSE ile; yüzde
hata ölçüleri MAPE* ve sMAPE ile; açıklanan varyans uyumu R² ile raporlanır.
MAPE*, yalnız gerçek hedefin `0.1 kW` üstünde olduğu gözlemlerde hesaplanır;
uygun değer yoksa sonuç tanımsız (`NaN`) bırakılır. sMAPE paydasının sıfır
olduğu gerçek/tahmin çiftinin katkısı sıfır kabul edilir. Ayrıca 24 saatlik
ufkun her adımı için bu metrikler ayrı ayrı hesaplanır. Bu sayede genel
ortalamanın gizleyebileceği ufukla artan hata incelenebilir.

## Veri sızıntısını önleme

Veri bölmesi zaman sıralıdır; rastgele örnek bölmesi uygulanmaz. Ölçekleyiciler
yalnız train üzerinde fit edilir; val üzerinde yalnız `transform` ve
değerlendirme yapılır. Hiperparametre, erken durdurma ve checkpoint seçimi
test sonuçlarına bakmaz. Her hedef penceresi tek bir veri kümesinin içinde
kalır; val'in geçmiş girdisi train'in sonundan gelebilir, çünkü bu yalnızca
tahmin anında bilinen geçmiş gözlemlerdir. Doldurulmuş hedefli val pencereleri
metrik hesabı öncesi çıkarılır. Bu uygulama test değerlendirmesinin
ertelenmesini ve doldurulmuş hedeflerle yapay biçimde kolaylaşabilecek
performansın önlenmesini sağlar.

## İş paketleri ve teknik uygulama süreci

Dilara'nın sorumluluğundaki tahmin hattı, proje planındaki İP2 (veri hazırlama
ve EDA), İP3 (tahmin modelleri ve referanslar), İP5 (değerlendirme ve hata
analizi) ve İP6'nın raporlama/demoya aktarım bileşenleriyle ilişkilidir. Teknik
uygulama; ham verinin doğrulanması ve saatliğe indirgenmesi, kronolojik
bölme/pencereleme testleri, ortak metriklerin ve referansların kurulması,
LSTM/GRU eğitimi, val karşılaştırması ve deney çıktılarının tablo/grafik
olarak kaydedilmesi sırasıyla yürütülür. Kod, yapılandırma, birim testleri ve
deney günlükleri aynı protokolü izler.

### Uygulama durumu ve yorumlama sınırı

Bu metindeki yöntem açıklaması uygulanmış kod ile sonraki deney planını
birbirinden ayırır. F3 Ridge, F4 `MLPRegressor`, F5 LSTM ve F6 GRU
validation deneyleri tamamlanmıştır. Her model 4.213 doldurulmamış val
penceresinde MAE, RMSE, MAPE*, sMAPE ve R² ile ölçülmüş; 24 adımın ayrı
metrikleri `results/tables/forecast_horizon_metrics.csv` dosyasına yazılmıştır.
Özet sonuçlar `results/tables/baseline_results.csv`, deney ayarları ve
MAE/RMSE ise `results/experiments.csv` içindedir. MLPRegressor 60 iterasyonda
convergence warning vermiştir; uyarı gizlenmemiştir. GRU koşusu MPS paylaşımlı
bellek ayırma hatası sonrasında CPU'da tamamlanmıştır; kullanılan seed ve
hiperparametreler değişmemiştir. Test kümesi bu deneylerde kullanılmamıştır.
Bu validation sonuçları, tek başına genellenebilir üstünlük iddiası değil,
proje protokolündeki sonraki final değerlendirme öncesi gözlemlerdir.
