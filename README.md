# imageClustering

Computer Vision 
Homework -I Unsupervised Learning Based Segmentation
Miraç BEKTAŞ- 401886

K-means Segmentation
Algoritma işlem adımları
1-	Cluster (Sınıf ) Sayısı : k seçimi. 
2-	Centroid olarak adlandırılan k sınıflarına rastgele değerler atama
3-	Tüm noktaları en yakın sınıfa atama
4-	Yeni oluşan kümelerin ağırlık merkezini yeniden hesapla
5-	3 ve 4. Adımları ağırlık merkezi değişmeyinceye kadar tekrarla.

Çoğu segmentasyon algoritması başlangıçta sınıf sayısı=k değerini kullanıcıdan ister. Bu sayı deneme yanılma yoluyla en optimum hale getirilebilir. Ayrıca optimum sınıf sayısını bulan özel algoritmalar da vardır.Dosyadan okunan resim dosyası bir diziye atanır. Segmentasyon yapılacak future space seçilerek veri seti ona göre oluşturulur.
   for (int i = 0; i < bmpImage.Height; i++)
            {
                for (int j = 0; j < bmpImage.Width; j++)
                {
                    Color c = bmpImage.GetPixel(i, j);
                    double[] pixelArray = new double[] { c.R, c.G, c.B };
                    dataSetList.Add(pixelArray);
                }
            }
Burada dataSetList değişkeni her bir satırda 3 future bulundurmaktadır. Her bir pikselin R,G ve B değerlerinden oluşan bu veri ile 3D Color Space Segmentation işlemi yapılıyor.
for (int i = 0; i < bmpImage.Height; i++)
            {
                for (int j = 0; j < bmpImage.Width; j++)
                {
                    Color oc = bmpImage.GetPixel(i, j);
                    int grayScale = (int)((oc.R * 0.3) + (oc.G * 0.59) + (oc.B * 0.11));
                    Color nc = Color.FromArgb(oc.A, grayScale, grayScale, grayScale);
                  
                    double[] pixelArray = new double[] { i, j, nc.R };
                    dataSetList.Add(pixelArray);
                }
            }
Yukarıdaki kod parçacığında ise resmin parlaklık değeri ve pikselin konum bilgisiyle beraber Intencity Location Based Segmentation işlemi için veri seti oluşturulur.
for (int i = 0; i < bmpImage.Height; i++)
            {
                for (int j = 0; j < bmpImage.Width; j++)
                {
                    Color oc = bmpImage.GetPixel(i, j);
                    int grayScale = (int)((oc.R * 0.3) + (oc.G * 0.59) + (oc.B * 0.11));
                    Color nc = Color.FromArgb(oc.A, grayScale, grayScale, grayScale);
                   
                    double[] pixelArray = new double[] { nc.R };
                    dataSetList.Add(pixelArray);
                }
            }
Yukarıdaki kod parçasında 1D intensity space segmentation işlemi için her bir pikselin parlaklık değeri ile veri seti oluşturulmuştur.



for (int i = 0; i < bmpImage.Height; i++)
            {
                for (int j = 0; j < bmpImage.Width; j++)
                {
                    Color c = bmpImage.GetPixel(i, j);
                    double[] pixelArray = new double[] {i,j, c.R, c.G, c.B };
                    dataSetList.Add(pixelArray);
                }
            }
Yukarıdaki kod parçasında ise Multi Dimension Space Segmentation için resmin R,G,B ve konum bilgileriyle veri seti oluşturulmuştur.
Başlangıçta verilen K küme sayısına göre rastgele kümeler oluşturuluyor ve bunlara başlangıç değerleri atanıyor. K= Sınıf Sayısı. Her bir sınıf için segmentasyon işleminde farklı renkler kullanılmıştır.

List<Centroid> centroidList = new List<Centroid>();
 for (int i = 0; i < k; i++)
  {
    Centroid centroid = new Centroid(dataSetList.ToArray(), Misc.centroidColors[i]);
    centroidList.Add(centroid);
  }
 Pixeller rastgele oluşturulan kümelere göre ilgili kümeye atanıp o renge boyanıyor.

 	updateImage(centroidList.ToArray(), bmpImage, pictureBox1);
Döngü içerisinde bütün pikselleri en yakın sınıfa atama işlemi yapılır. Yakınlık hesabı için kullanıcıya Minimum Distance ve Mahalonobis Distance seçenekleri sunulur. Bu seçimlerine göre noktanın uzaklığı hesaplanarak en yakın sınıfa atanır. Daha sonra centroid ağırlıkları hesaplanır ve bir önceki ağırlıklar ile farkı var mı bakılır (hasChanged).
   while (true)
            {
                foreach (Centroid centroid in centroidList)
                    centroid.Reset();

                for (int i = 0; i < dataSet.GetLength(0); i++)
                {
                    double[] point = dataSet[i];
                    int closestIndex = -1;
                    double minDistance = Double.MaxValue;
                    for (int k = 0; k < centroidList.Count; k++)
                    {
                        double distance = 0;
                        if (minDist.Checked)
                        {
                            distance = calcDistance(centroidList[k].Array, point);
                        }
                        else
                        {
                            distance = Utils.MahalanobisDist2(dataSet, covarianceMatris, centroidList[k].Array, point);
                        }                     
                        if (distance < minDistance)
                        {
                            closestIndex = k;
                            minDistance = distance;
                        }
                    }
                    centroidList[closestIndex].addPoint(point);
                }

                foreach (Centroid centroid in centroidList)
                    centroid.MoveCentroid();

                updateImage(centroidList.ToArray(),bmpImage,pictureBox1);

                bool hasChanged = false;
                foreach (Centroid centroid in centroidList)
                    if (centroid.HasChanged())
                    {
                        hasChanged = true;
                        break;
                    }
                if (!hasChanged)
                    break;
            }
İterasyonlar sonucunda 4 farklı özellik uzayında yapılan segmentasyon işlemleri genel olarak aşağıdaki gibi oluşmuştur.
 
Bu işlemde Sınıf sayısı: 3, Distance : Minimum Distance olarak alınmıştır.

 
Bu işlemde Sınıf sayısı: 4, Distance : Minimum Distance olarak alınmıştır.


 
Bu örnekte Sınıf sayısı:4, Distance: Mahalanobis Distance seçilmiştir.

Expectation Maximization:  N-Dimensional Probaility Density Function
K-means algoritması dağınık ve iç içe geçmiş görüntülerde etkili bir segmentasyon işlemi gerçekleştiremez. Bu algoritma ile her bir örneğin bütün sınıflarda olma olasılıkları hesaplanır. Expectation Maximization (Beklenti maksimizasyonu), expectation (E) adımı ve maksimizasyon (M) adımı olarak iki adımın art arda tekrarlanmasıyla gerçekleşir. E-adımı parametrelerin o anki tahminlerini kullanarak bir log-olabilirlik beklentisi fonksiyonu oluşturur. M adımı parametre değerlerini log-olabilirlik beklentisini maksimize edecek şekilde günceller. Yani bu iki adımın her biri diğerinin girdisini hesaplayarak birbirini besler. Beklenti maksimizasyon adımları tahmindeki hata miktarı belirli bir oranın altına düşene kadar yinelenir. İterasyonlar sonucunda bir ağırlık matrisi oluşur. Bu matris N = veri sayısı ve sınıfsayısı boyutundadır. Her bir veri setinin sınıflarda olma ağırlığını bulundurur. Örneğin 3 sınıflı bir ağırlık matrisinde olasılık değerleri (0.9207, 0.0573, 0.0220) şeklinde ise bu örnek k=0.sınıftadır. Bu olasılıkların toplamının da 1’e eşit olduğu unutulmamalıdır. 
Bunun için işlem adımları aşağıdaki şekildedir.
1-	Mk, Covariance Matris, popülasyon fonksiyonu gibi değişkenler başlangıç değerleriyle beraber oluşturulması.
2-	Her bir örneğin sınıflarda ağırlaklandırılması.
3-	Mevcut ağırlıklandırmaya göre bütün parametrelerin yeniden hesaplanması.
4-	Log-likelihood fonksiyonunun hesaplanması.
5-	Koveryans kriterlerinin uygulanması
6-	Log-likehood fonksiyonu aynı değerleri üretiyorsa iterasyonlar durur.  Değilse 2.adıma dönülür.

Uygulamada matris işlemlerini gerçekleştirebilmek için Utils adında bir sınıf oluşturulmuş ve burada static bazı metodlarımız vardır. Bunlar matris oluşturma, çarpma ters alma gibi işlemleri yerine getirir. Bu metodlar yardımıyla algoritmanın ihtiyaç duyduğu başlangıç değerleri aşağıdaki şekilde oluşturulur.

N= Veri seti boyutu. Uygulama için 256x256 = 65536
K= Sınıf sayısı. Başlangıçta 3 olarak alınmıştır.
D = Future Space. Özellik uzayı. Çalışmada 3 alınmıştır. R,G,B

double[][] w = Utils.MatrisOlustur(N, K);  // Ağırlık matrisi
double[] a = new double[K] { 1.0/K, 1.0/K, 1.0/K }; // popülasyon fonksiyon matrisi
double[][] u = Utils.MatrisOlustur(K, d);  // sınıf ortalamaları matrisi
double[][] V = Utils.MatrisOlustur(K, d, 10); // sınıf varyans matrisi
double[] Nk = new double[K];  // yardımcı matris (Ağırlık matrisinin kolon içeriklerinin toplamını tutar)
         
Ardından EM algoritması iterasyonları başlatılır. Multivariate Gaussian Distribution için Olasılık yoğunluk fonksiyonu aşağıdaki formüle göre hesaplanır. Burada M değeri dağılımın ortalamasını gösteren d boyutlu bir vektörü, Ʃ ise d x d boyutunda kovaryans matrisidir.
 
for (int iter = 0; iter < Convert.ToInt32(iteration.Text) ; ++iter)
{
      UpdateMembershipWts(w, x, u, V, a);  // E step uygulanıyor. PDF fonksyionu hesaplanıp ağırlıklar oluşturuluyor.
      UpdateNk(Nk, w);  // M step uygulanıyor. (Maksimizasyon)
      UpdateMixtureWts(a, Nk); // Sınıf ağırlıkları güncelleniyor
      UpdateMeans(u, w, x, Nk); // Sınıf ortalamaları güncelleniyor.
      UpdateVariances(V, u, w, x, Nk); // Varyanslar hesaplanıyor.
}

Ağırlıklarda değişme olmamaya başladığı durumda iterasyon sona eriyor. Ardından her bir veri setinin hangi sınıfa dahil olduğunu anlamak için ağırlık matrisine bakılıyor. Daha önce verilen örnekte olduğu gibi, Örneğin 3 sınıflı bir ağırlık matrisinde olasılık değerleri (0.9207, 0.0573, 0.0220) şeklinde ise bu örnek k=0.sınıftadır. Bu olasılıkların toplamının da 1’e eşit olduğu unutulmamalıdır. Buna göre renklendirme yapılarak görüntü üzerinde segmentasyon tamamlanır.

for (int i = 0; i < bmpImage.Height; i++)
            {
                for (int j = 0; j < bmpImage.Width; j++)
                {
                    int winner = 0;
                    double winnerValue = 0;
                    for (int k = 0; k < d; k++)
                    {
                        if (w[mm][k] > winnerValue)
                        {
                            winnerValue = w[mm][k];
                            winner = k;
                        }
                    }
                    mm++;

                    if (winner == 0)
                    {
                        bmp.SetPixel(i, j, Color.Red);
                    }
                    if (winner == 1)
                    {
                        bmp.SetPixel(i, j, Color.Green);
                    }
                    if (winner == 2)
                    {
                        bmp.SetPixel(i, j, Color.Blue);
                    }
                }
            }
Bu işlemler sonucunda aşağıdaki gibi segmentasyon işlemi gerçekleştirilmiş olur.

  

Farklı görüntüler ve çeşitli değişkenlerle yapılan bazı test sonuçları aşağıda verilmiştir.
      
Orijinal Görüntü		Sınıf sayısı:3 K-means		Sınıf sayısı:4 K-means
   
     Sınıf sayısı:5 K-means	              Sınıf sayısı:6 K-means



 
Orijinal Görüntü 
 
Sınıf sayısı 6 K-Means Algoritması ile

 
Sınıf Sayısı 3. E-M Algoritması.



