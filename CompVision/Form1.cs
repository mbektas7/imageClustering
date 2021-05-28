using CompVision.Helpers;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace CompVision
{
    public partial class Form1 : Form
    {
        const int N = 65536; const int K = 3; const int d = 3;
        private List<double[]> dataSetList;
        public static Bitmap bmpImage;
        private Bitmap grayImage;
        Image file;
        int k = 3;
        public Form1()
        {
            InitializeComponent();
            minDist.Checked = true;
            clusterCount.Text = k.ToString();
            dataSetList = new List<double[]>();
            Assembly asm = Assembly.GetExecutingAssembly();
            string path = System.IO.Path.GetDirectoryName(asm.Location);
            string imageFilePath = Path.Combine(path, "lena256.png");
            //string imageFilePath = Path.Combine(path, "sebze.png");
            //string imageFilePath = Path.Combine(path, "panda.png");
            //string imageFilePath = Path.Combine(path, "lenaGray.jpg");
            //string imageFilePath = Path.Combine(path, "cameraman.jpg");
            bmpImage = new Bitmap(imageFilePath);
            orginalImageBox.Image = bmpImage;

            iteration.Text = "5";
            //////Mahalanobis Dist test
            //double[][] table = new double[5][];
            //table[0] = new double[] { 64, 580, 29 };
            //table[1] = new double[] { 66, 570, 33 };
            //table[2] = new double[] { 68, 590, 37 };
            //table[3] = new double[] { 69, 660, 46 };
            //table[4] = new double[] { 73, 600, 55 };

            //double md = Utils.MahalanobisDist(table, 1, 3);
        }

        private double calcDistance(double[] array1, double[] array2)
        {
            double res = 0;
            for (int i = 0; i < array1.Length; i++)
            {
                res += Math.Pow(array1[i] - array2[i], 2);
            }
            return Math.Sqrt(res);
        }

        private void updateImage(Centroid[] centroids, Bitmap originalImage, PictureBox pictureBox)
        {
            Bitmap bmpNew = new Bitmap(bmpImage.Width, bmpImage.Height);
            for (int i = 0; i < bmpImage.Width; i++)
            {
                for (int j = 0; j < bmpImage.Width; j++)
                {
                    Color c = bmpImage.GetPixel(i, j);
                    double[] pixelArray = new double[] { c.R, c.G, c.B };

                    double minDistance = Double.MaxValue;
                    Centroid centroidWinner = null;
                    foreach (Centroid centroid in centroids)
                    {
                        double distance = calcDistance(pixelArray, centroid.Array);
                        if (distance < minDistance)
                        {
                            minDistance = distance;
                            centroidWinner = centroid;
                        }
                    }

                    bmpNew.SetPixel(i, j, centroidWinner.CentroidColor);
                }
            }

            pictureBox.Image = bmpNew;
        }

        private void updateMultiDimensionalImage(Centroid[] centroids, Bitmap originalImage, PictureBox pictureBox)
        {
            Bitmap bmpNew = new Bitmap(bmpImage.Width, bmpImage.Height);
            for (int i = 0; i < bmpImage.Width; i++)
            {
                for (int j = 0; j < bmpImage.Width; j++)
                {
                    Color c = bmpImage.GetPixel(i, j);
                    double[] pixelArray = new double[] {i,j, c.R, c.G, c.B };

                    double minDistance = Double.MaxValue;
                    Centroid centroidWinner = null;
                    foreach (Centroid centroid in centroids)
                    {
                        double distance = calcDistance(pixelArray, centroid.Array);
                        if (distance < minDistance)
                        {
                            minDistance = distance;
                            centroidWinner = centroid;
                        }
                    }

                    bmpNew.SetPixel(i, j, centroidWinner.CentroidColor);
                }
            }

            pictureBox.Image = bmpNew;
        }
        private void updateGrayImage(Centroid[] centroids, Bitmap originalImage, PictureBox pictureBox)
        {
            Bitmap bmpNew = new Bitmap(originalImage.Width, originalImage.Height);
            for (int i = 0; i < originalImage.Width; i++)
            {
                for (int j = 0; j < originalImage.Width; j++)
                {
                    Color oc = originalImage.GetPixel(i, j);
                    int grayScale = (int)((oc.R * 0.3) + (oc.G * 0.59) + (oc.B * 0.11));
                    Color nc = Color.FromArgb(oc.A, grayScale, grayScale, grayScale);
                    double[] pixelArray = new double[] {i,j, nc.R };

                    double minDistance = Double.MaxValue;
                    Centroid centroidWinner = null;
                    foreach (Centroid centroid in centroids)
                    {
                        double distance = calcDistance(pixelArray, centroid.Array);
                        if (distance < minDistance)
                        {
                            minDistance = distance;
                            centroidWinner = centroid;
                        }
                    }

                    bmpNew.SetPixel(i, j, centroidWinner.CentroidColor);
                }
            }

            pictureBox.Image = bmpNew;
        }

        private void update1dImage(Centroid[] centroids, Bitmap originalImage, PictureBox pictureBox)
        {
            Bitmap bmpNew = new Bitmap(originalImage.Width, originalImage.Height);
            for (int i = 0; i < originalImage.Width; i++)
            {
                for (int j = 0; j < originalImage.Width; j++)
                {
                    Color oc = originalImage.GetPixel(i, j);
                    int grayScale = (int)((oc.R * 0.3) + (oc.G * 0.59) + (oc.B * 0.11));
                    Color nc = Color.FromArgb(oc.A, grayScale, grayScale, grayScale);
                    
                    double[] pixelArray = new double[] { nc.R };

                    double minDistance = Double.MaxValue;
                    Centroid centroidWinner = null;
                    foreach (Centroid centroid in centroids)
                    {
                        double distance = calcDistance(pixelArray, centroid.Array);
                        if (distance < minDistance)
                        {
                            minDistance = distance;
                            centroidWinner = centroid;
                        }
                    }

                    bmpNew.SetPixel(i, j, centroidWinner.CentroidColor);
                }
            }

            pictureBox.Image = bmpNew;
        }

        private void button1_Click(object sender, EventArgs e)
        {
           
            dataSetList = new List<double[]>();
            k = Convert.ToInt32(clusterCount.Text);
            for (int i = 0; i < bmpImage.Height; i++)
            {
                for (int j = 0; j < bmpImage.Width; j++)
                {
                    Color c = bmpImage.GetPixel(i, j);
                    double[] pixelArray = new double[] { c.R, c.G, c.B };
                    dataSetList.Add(pixelArray);
                }
            }

            // Başlangıçta verilen K küme sayısına göre rastgele kümeler oluşturuluyor.
            List<Centroid> centroidList = new List<Centroid>();
            for (int i = 0; i < k; i++)
            {
                Centroid centroid = new Centroid(dataSetList.ToArray(), Misc.centroidColors[i]);
                centroidList.Add(centroid);
            }

            //pixeller rastgele oluşturulan kümelere göre ilgili kümeye atanıp o renge boyanıyor.
            updateImage(centroidList.ToArray(), bmpImage, pictureBox1);

            double[][] dataSet = dataSetList.ToArray();

            double[][] covarianceMatris = null;

            // distance hesabı olarak mahanolabis seçilmişse öncesinde covaryans matrisi hesaplanıyor.
            if (mahaDist.Checked)
            {
                covarianceMatris = Utils.CovMatrix(dataSet);
            }

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

        }

        private void button2_Click(object sender, EventArgs e)
        {
            dataSetList = new List<double[]>();
            k = Convert.ToInt32(clusterCount.Text);
            grayImage = bmpImage;
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


            // Başlangıçta verilen K küme sayısına göre rastgele kümeler oluşturuluyor.
            List<Centroid> centroidList = new List<Centroid>();
            for (int i = 0; i < k; i++)
            {
                Centroid centroid = new Centroid(dataSetList.ToArray(), Misc.centroidColors[i]);
                centroidList.Add(centroid);
            }

            //pixeller rastgele oluşturulan kümelere göre ilgili kümeye atanıp o renge boyanıyor.
            updateGrayImage(centroidList.ToArray(),grayImage,pictureBox2);

            double[][] dataSet = dataSetList.ToArray();
            double[][] covarianceMatris = null;
            // distance hesabı olarak mahanolabis seçilmişse öncesinde covaryans matrisi hesaplanıyor.
            if (mahaDist.Checked)
            {
                covarianceMatris = Utils.CovMatrix(dataSet);
            }
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

                updateGrayImage(centroidList.ToArray(),grayImage,pictureBox2);

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

        }

        private void button3_Click(object sender, EventArgs e)
        {
            dataSetList = new List<double[]>();
            k = Convert.ToInt32(clusterCount.Text);
            grayImage = bmpImage;
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


            // Başlangıçta verilen K küme sayısına göre rastgele kümeler oluşturuluyor.
            List<Centroid> centroidList = new List<Centroid>();
            for (int i = 0; i < k; i++)
            {
                Centroid centroid = new Centroid(dataSetList.ToArray(), Misc.centroidColors[i]);
                centroidList.Add(centroid);
            }

            //pixeller rastgele oluşturulan kümelere göre ilgili kümeye atanıp o renge boyanıyor.
            update1dImage(centroidList.ToArray(), grayImage, pictureBox3);

            double[][] dataSet = dataSetList.ToArray();
            double[][] covarianceMatris = null;
            // distance hesabı olarak mahanolabis seçilmişse öncesinde covaryans matrisi hesaplanıyor.
            if (mahaDist.Checked)
            {
                covarianceMatris = Utils.CovMatrix(dataSet);
            }
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

                update1dImage(centroidList.ToArray(), grayImage, pictureBox3);

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

        }

        private void button4_Click(object sender, EventArgs e)
        {
            dataSetList = new List<double[]>();
            k = Convert.ToInt32(clusterCount.Text);

            for (int i = 0; i < bmpImage.Height; i++)
            {
                for (int j = 0; j < bmpImage.Width; j++)
                {
                    Color c = bmpImage.GetPixel(i, j);
                    double[] pixelArray = new double[] {i,j, c.R, c.G, c.B };
                    dataSetList.Add(pixelArray);
                }
            }


            // Başlangıçta verilen K küme sayısına göre rastgele kümeler oluşturuluyor.
            List<Centroid> centroidList = new List<Centroid>();
            for (int i = 0; i < k; i++)
            {
                Centroid centroid = new Centroid(dataSetList.ToArray(), Misc.centroidColors[i]);
                centroidList.Add(centroid);
            }

            //pixeller rastgele oluşturulan kümelere göre ilgili kümeye atanıp o renge boyanıyor.
            updateMultiDimensionalImage(centroidList.ToArray(), bmpImage, pictureBox4);

            double[][] dataSet = dataSetList.ToArray();
            double[][] covarianceMatris = null;
            // distance hesabı olarak mahanolabis seçilmişse öncesinde covaryans matrisi hesaplanıyor.
            if (mahaDist.Checked)
            {
                covarianceMatris = Utils.CovMatrix(dataSet);
            }
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

                updateMultiDimensionalImage(centroidList.ToArray(), bmpImage, pictureBox4);

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
        }

        private void button5_Click(object sender, EventArgs e)
        {
            Image File;
            OpenFileDialog f = new OpenFileDialog();
            f.Filter = "Image files (*.jpg, *.png) | *.jpg; *.png";

            if (f.ShowDialog() == DialogResult.OK)
            {
                File = Image.FromFile(f.FileName);
                orginalImageBox.Image = File;
                bmpImage = (Bitmap)orginalImageBox.Image;
                file = File;
            }
        }

        private void button6_Click(object sender, EventArgs e)
        {
            bmpImage = (Bitmap)orginalImageBox.Image;
            double[][] x = new double[bmpImage.Width * bmpImage.Height][];
            int m = 0;
            for (int i = 0; i < bmpImage.Height; i++)
            {
                for (int j = 0; j < bmpImage.Width; j++)
                {

                    Color c = bmpImage.GetPixel(i, j);
                    x[m] = new double[] { Convert.ToDouble(c.R), Convert.ToDouble(c.G), Convert.ToDouble(c.B) };
                    m++;
                }
            }

            // Başlangıç değerleri ayarlanıyor.
            double[][] w = Utils.MatrisOlustur(N, K);  // Ağırlık matrisi
            double[] a = new double[K] { 1.0 / K, 1.0 / K, 1.0 / K }; // popülasyon fonksiyon matrisi
            double[][] u = Utils.MatrisOlustur(K, d);  // sınıf ortalamaları matrisi
            double[][] V = Utils.MatrisOlustur(K, d, 10); // sınıf varyans matrisi
            double[] Nk = new double[K];  // yardımcı matris (Ağırlık matrisinin kolon içeriklerinin toplamını tutar)
            u[0][0] = 50; u[0][1] = 50; u[0][2] = 150;  // 3.sınıf ağırlıkları
            u[1][0] = 61; u[1][1] = 61; u[1][2] = 61;  // 2.sınıf ağırlıkları 
            u[2][0] = 180; u[2][1] = 30; u[2][2] = 30; //1 .sınıf ağırlıkları

            // EM Algoritması iterasyonları Başlıyor.
            for (int iter = 0; iter < Convert.ToInt32(iteration.Text) ; ++iter)
            {
                UpdateMembershipWts(w, x, u, V, a);  // E step uygulanıyor
                UpdateNk(Nk, w);  // M step uygulanıyor.
                UpdateMixtureWts(a, Nk);
                UpdateMeans(u, w, x, Nk);
                UpdateVariances(V, u, w, x, Nk);
            }

            saveImage(w); // resim kaydediliyor.

        }
        static void UpdateMembershipWts(double[][] w, double[][] x, double[][] u, double[][] V, double[] a)
        {
            for (int i = 0; i < N; ++i)
            {
                double rowSum = 0.0;
                for (int k = 0; k < K; ++k)
                {
                    //pdf fonksiyonu hesaplanıyor.
                    double pdf = NaiveProb(x[i], u[k], V[k]);
                    w[i][k] = a[k] * pdf;
                    rowSum += w[i][k];
                }
                for (int k = 0; k < K; ++k)
                    w[i][k] = w[i][k] / rowSum;
            }
        }
        static void UpdateNk(double[] Nk, double[][] w)
        {
            for (int k = 0; k < K; ++k)
            {
                double sum = 0.0;
                for (int i = 0; i < N; ++i)
                    sum += w[i][k];
                Nk[k] = sum;
            }
        }
        static void UpdateMixtureWts(double[] a, double[] Nk)
        {
            for (int k = 0; k < K; ++k)
                a[k] = Nk[k] / N;
        }
        static void UpdateMeans(double[][] u, double[][] w,
        double[][] x, double[] Nk)
        {
            double[][] result = Utils.MatrisOlustur(K, d);
            for (int k = 0; k < K; ++k)
            {
                for (int i = 0; i < N; ++i)
                    for (int j = 0; j < d; ++j)
                        result[k][j] += w[i][k] * x[i][j];
                for (int j = 0; j < d; ++j)
                    result[k][j] = result[k][j] / Nk[k];
            }
            for (int k = 0; k < K; ++k)
                for (int j = 0; j < d; ++j)
                    u[k][j] = result[k][j];
        }
        static void UpdateVariances(double[][] V, double[][] u, double[][] w, double[][] x, double[] Nk)
        {
            double[][] result = Utils.MatrisOlustur(K, d);
            for (int k = 0; k < K; ++k)
            {
                for (int i = 0; i < N; ++i)
                    for (int j = 0; j < d; ++j)
                        result[k][j] += w[i][k] * (x[i][j] - u[k][j]) *
                        (x[i][j] - u[k][j]);
                for (int j = 0; j < d; ++j)
                    result[k][j] = result[k][j] / Nk[k];
            }
            for (int k = 0; k < K; ++k)
                for (int j = 0; j < d; ++j)
                    V[k][j] = result[k][j];
        }
        static double ProbDenFunc(double x, double u, double v)
        {
            // Univariate Gaussian fonksiyon kullanarak ağırlandırılma yapılmıştur.
            if (v == 0.0) throw new Exception("0 ağırlığı olamaz");
            double left = 1.0 / Math.Sqrt(2.0 * Math.PI * v);
            double pwr = -1 * ((x - u) * (x - u)) / (2 * v);
            var exp = Math.Exp(pwr);
            var carpim = left * exp;
            return carpim;
        }
        static double NaiveProb(double[] x, double[] u, double[] v)
        {
            // Gaus PDF
            double sum = 0.0;
            for (int j = 0; j < d; ++j)
                sum += ProbDenFunc(x[j], u[j], v[j]);
            return sum / d;
        }
        public void saveImage(double[][] w)
        {
            int width = Convert.ToInt32(bmpImage.Width);
            int height = Convert.ToInt32(bmpImage.Height);
            string fileName = DateTime.Now.ToFileTime().ToString();
            Bitmap bmp = new Bitmap(width, height);
            int mm = 0;
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
            pictureBox5.Image = bmp;
           // bmp.Save(fileName + ".png");
        }
    }
}
