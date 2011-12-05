using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Diagnostics;
using System.Reflection;
using Cudafy.Types;
using Cudafy.Host;
using Cudafy.Maths.BLAS;
using Cudafy.Maths.BLAS.Types;
using Cudafy.UnitTests;
using NUnit.Framework;

namespace Cudafy.Maths.UnitTests
{
    [TestFixture]
    public class BLAS2 : ICudafyUnitTest
    {
        private GPGPU _gpu;

        private GPGPUBLAS _blas;

        private float[] _hiMatrixS;
        private float[] _hiVectorXNS;
        private float[] _hiVectorXMS;
        private float[] _hiVectorYNS;
        private float[] _hiVectorYMS;

        private float[] _diMatrixS;
        private float[] _diVectorXNS;
        private float[] _diVectorXMS;
        private float[] _diVectorYNS;
        private float[] _diVectorYMS;

        private float[] _hoVectorXNS;
        private float[] _hoVectorYMS;
        private float[] _hoVectorYNS;
        private float[] _hoMatrixS;
        private float[] _hoMatrixSymmetricPackedS;
        private float[] _hoMatrixSymmetricS;

        private float[] _hiMatrixSymmetricS;
        private float[] _hiMatrixSymmetricPackedS;
        private float[] _hiMatrixBandedPackedS;
        private float[] _hiMatrixBandedSymmetricPackedS;
        private float[] _diMatrixSymmetricS;
        private float[] _diMatrixSymmetricPackedS;
        private float[] _diMatrixBandedPackedS;
        private float[] _diMatrixBandedSymmetricPackedS;

        private const int M = 128; // Row size of matrix A
        private const int N = 256; // Column size of matrix A
        private const int Ns = 256; // Symmetric size (Ns x Ns), Must be equal to N
        private const int kl = 5; // number of subdiagonals of matrix A
        private const int ku = 4; // number of superdiagonals of matrix A
        private const int k = 13; // number of superdiagonals and subdiagonals of symmetric matrix A

        [TestFixtureSetUp]
        public void SetUp()
        {
            _gpu = CudafyHost.CreateDevice(CudafyModes.Target);
            _blas = GPGPUBLAS.Create(_gpu);
            
            _hiMatrixS = new float[M * N];
            _hoMatrixS = new float[M * N];
            _hiVectorXMS = new float[M];
            _hiVectorXNS = new float[N];
            _hiVectorYMS = new float[M];
            _hiVectorYNS = new float[N];
            _hoVectorXNS = new float[N];
            _hoVectorYMS = new float[M];
            _hoVectorYNS = new float[N];

            _hiMatrixSymmetricS = new float[Ns * Ns];
            _hoMatrixSymmetricS = new float[Ns * Ns];
            _hiMatrixSymmetricPackedS = new float[Ns * (Ns + 1) / 2];
            _hiMatrixBandedPackedS = new float[(kl + ku + 1) * Ns];
            _hiMatrixBandedSymmetricPackedS = new float[(k + 1) * Ns];
            _hoMatrixSymmetricPackedS = new float[Ns * (Ns + 1) / 2];

            _diMatrixS = _gpu.Allocate<float>(_hiMatrixS);
            _diMatrixSymmetricS = _gpu.Allocate<float>(_hiMatrixSymmetricS);
            _diVectorXNS = _gpu.Allocate<float>(_hiVectorXNS);
            _diVectorYMS = _gpu.Allocate<float>(_hiVectorYMS);
            _diVectorXMS = _gpu.Allocate<float>(_hiVectorXMS);
            _diVectorYNS = _gpu.Allocate<float>(_hiVectorYNS);

            _diMatrixSymmetricPackedS = _gpu.Allocate<float>(_hiMatrixSymmetricPackedS);
            _diMatrixBandedPackedS = _gpu.Allocate<float>(_hiMatrixBandedPackedS);
            _diMatrixBandedSymmetricPackedS = _gpu.Allocate<float>(_hiMatrixBandedSymmetricPackedS);
        }

        [TestFixtureTearDown]
        public void TearDown()
        {
            _blas.Dispose();

            _gpu.Free(_diMatrixS);
            _gpu.Free(_diVectorXNS);
            _gpu.Free(_diVectorYMS);
            _gpu.Free(_diVectorXMS);
            _gpu.Free(_diVectorYNS);

            _gpu.Free(_diMatrixSymmetricS);
            _gpu.Free(_diMatrixSymmetricPackedS);
            _gpu.Free(_diMatrixBandedPackedS);
            _gpu.Free(_diMatrixBandedSymmetricPackedS);
        }
        
        /// <summary>
        /// Create Banded matrix by random numbers.
        /// </summary>
        /// <param name="buffer"></param>
        private void CreateBandedMatrix(float[] buffer, int m, int n, int kl, int ku)
        {
            Random rand = new Random(Environment.TickCount);
            for (int i = 0; i < m; i++)
            {
                int index = _blas.GetIndexColumnMajor(i, i, m);
                if (index < 0 || index >= m * n)
                {
                    continue;
                }
                buffer[index] = rand.Next(512);

                // Set superdiagonal
                for (int si = 1; si <= ku; si++)
                {
                    index = _blas.GetIndexColumnMajor(i, i + si, m);
                    if (index < 0 || index >= m * n)
                    {
                        continue;
                    }

                    buffer[index] = rand.Next(512);
                }

                // Set subdiagonal
                for (int si = 1; si <= kl; si++)
                {
                    index = _blas.GetIndexColumnMajor(i, i - si, m);
                    if (index < 0 || index >= m * n)
                    {
                        continue;
                    }

                    buffer[index] = rand.Next(512);
                }
            }
            System.Threading.Thread.Sleep(rand.Next(50));
        }

        private void CreateBandedSymmetricMatrix(float[] buffer, int n, int k)
        {
            Random rand = new Random(Environment.TickCount);
            for (int i = 0; i < n; i++)
            {
                int index = _blas.GetIndexColumnMajor(i, i, n);
                buffer[index] = rand.Next(512);

                // Set superdiagonal and subdiagonal to same value.
                for (int si = 1; si <= k; si++)
                {
                    float value = rand.Next(512);
                    index = _blas.GetIndexColumnMajor(i, i + si, n);
                    if (index >= 0 && index < n * n)
                    {
                        buffer[index] = value;
                    }
                }
            }

            // Copy symmetric
            for (int i = 0; i < n; i++)
            {
                for (int j = i + 1; j < n; j++)
                {
                    buffer[_blas.GetIndexColumnMajor(j, i, n)] = buffer[_blas.GetIndexColumnMajor(i, j, n)];
                }
            }
            System.Threading.Thread.Sleep(rand.Next(50));
        }

        private void CreateSymmetricMatrix(float[] buffer, int n)
        {
            Random rand = new Random(Environment.TickCount);

            for (int i = 0; i < n; i++)
            {
                for (int j = i; j < n; j++)
                {
                    int value = rand.Next(32);
                    buffer[_blas.GetIndexColumnMajor(i, j, n)] = value;
                    buffer[_blas.GetIndexColumnMajor(j, i, n)] = value;
                }
            }

            System.Threading.Thread.Sleep(rand.Next(50));
        }

        /// <summary>
        /// Fill buffer by random numbers.
        /// </summary>
        /// <param name="buffer"></param>
        private void FillBuffer(float[] buffer)
        {
            Random rand = new Random(Environment.TickCount);
            for (int i = 0; i < buffer.Length; i++)
            {
                buffer[i] = rand.Next(32);
            }
            System.Threading.Thread.Sleep(rand.Next(100));
        }

        private void ClearBuffer(float[] buffer)
        {
            for (int i = 0; i < buffer.Length; i++)
            {
                buffer[i] = 0.0f;
            }
        }

        [Test]
        public void TestGBMV()
        {
            ClearBuffer(_hiMatrixS);
            ClearBuffer(_hiVectorXNS);
            ClearBuffer(_hiVectorYMS);
            CreateBandedMatrix(_hiMatrixS, M, N, kl, ku);
            _blas.ConvertBandedMatrixCBC(_hiMatrixS, _hiMatrixBandedPackedS, M, N, kl, ku);
            FillBuffer(_hiVectorXNS);
            FillBuffer(_hiVectorYMS);

            _gpu.CopyToDevice(_hiMatrixBandedPackedS, _diMatrixBandedPackedS);
            _gpu.CopyToDevice(_hiVectorXNS, _diVectorXNS);
            _gpu.CopyToDevice(_hiVectorYMS, _diVectorYMS);

            float alpha = 5.0f;
            float beta = 3.0f;

            _blas.GBMV(M, N, kl, ku, alpha, _diMatrixBandedPackedS, _diVectorXNS, beta, _diVectorYMS);
            _gpu.CopyFromDevice(_diVectorYMS, _hoVectorYMS);

            for (int vi = 0; vi < M; vi++)
            {
                float cpuResult = 0.0f;
                for (int j = 0; j < N; j++)
                {
                    cpuResult += alpha * _hiMatrixS[_blas.GetIndexColumnMajor(vi, j, M)] * _hiVectorXNS[j];
                }

                // Expected result
                cpuResult += beta * _hiVectorYMS[vi];

                Assert.AreEqual(cpuResult, _hoVectorYMS[vi]);
            }

            // Transpose Test
            ClearBuffer(_hiVectorXNS);
            ClearBuffer(_hiVectorYMS);
            FillBuffer(_hiVectorXMS);
            FillBuffer(_hiVectorYNS);
            _gpu.CopyToDevice(_hiVectorXMS, _diVectorXMS);
            _gpu.CopyToDevice(_hiVectorYNS, _diVectorYNS);
            _blas.GBMV(Ns, Ns, kl, ku, alpha, _diMatrixBandedPackedS, _diVectorXMS, beta, _diVectorYNS, cublasOperation.T);
            _gpu.CopyFromDevice(_diVectorYNS, _hoVectorYNS);

            for (int vi = 0; vi < N; vi++)
            {
                float cpuResult = 0.0f;
                for (int j = 0; j < M; j++)
                {
                    cpuResult += alpha * _hiMatrixS[_blas.GetIndexColumnMajor(j, vi, M)] * _hiVectorXMS[j];
                }

                // Expected result
                cpuResult = cpuResult + beta * _hiVectorYNS[vi];

                Assert.AreEqual(cpuResult, _hoVectorYNS[vi]);
            }
        }

        [Test]
        public void TestGEMV()
        {
            ClearBuffer(_hiMatrixS);
            ClearBuffer(_hiVectorXNS);
            ClearBuffer(_hiVectorYMS);
            ClearBuffer(_hiVectorXMS);
            ClearBuffer(_hiVectorYNS);
            FillBuffer(_hiMatrixS);
            FillBuffer(_hiVectorXNS);
            FillBuffer(_hiVectorYMS);
            FillBuffer(_hiVectorXMS);
            FillBuffer(_hiVectorYNS);

            _gpu.CopyToDevice(_hiMatrixS, _diMatrixS);
            _gpu.CopyToDevice(_hiVectorXNS, _diVectorXNS);
            _gpu.CopyToDevice(_hiVectorYMS, _diVectorYMS);

            float alpha = 5.0f;
            float beta = 3.0f;

            _blas.GEMV(M, N, alpha, _diMatrixS, _diVectorXNS, beta, _diVectorYMS);

            _gpu.CopyFromDevice(_diVectorYMS, _hoVectorYMS);

            for (int i = 0; i < M; i++)
            {
                float cpuResult = 0.0f;

                for (int j = 0; j < N; j++)
                {
                    cpuResult += alpha * _hiMatrixS[_blas.GetIndexColumnMajor(i, j, M)] * _hiVectorXNS[j];
                }

                cpuResult = cpuResult + beta * _hiVectorYMS[i];
                Assert.AreEqual(cpuResult, _hoVectorYMS[i]);
            }


            // Test Transpose
            _gpu.CopyToDevice(_hiVectorXMS, _diVectorXMS);
            _gpu.CopyToDevice(_hiVectorYNS, _diVectorYNS);

            _blas.GEMV(M, N, alpha, _diMatrixS, _diVectorXMS, beta, _diVectorYNS, cublasOperation.T);

            _gpu.CopyFromDevice(_diVectorYNS, _hoVectorYNS);

            for (int i = 0; i < N; i++)
            {
                float cpuResult = 0.0f;

                for (int j = 0; j < M; j++)
                {
                    cpuResult += alpha * _hiMatrixS[_blas.GetIndexColumnMajor(j, i, M)] * _hiVectorXMS[j];
                }

                cpuResult = cpuResult + beta * _hiVectorYNS[i];

                Assert.AreEqual(cpuResult, _hoVectorYNS[i]);
            }
        }

        [Test]
        public void TestGER()
        {
            ClearBuffer(_hiMatrixS);
            ClearBuffer(_hiVectorXMS);
            ClearBuffer(_hiVectorYNS);
            FillBuffer(_hiMatrixS);
            FillBuffer(_hiVectorXMS);
            FillBuffer(_hiVectorYNS);

            float alpha = 3.0f;

            _gpu.CopyToDevice(_hiMatrixS, _diMatrixS);
            _gpu.CopyToDevice(_hiVectorXMS, _diVectorXMS);
            _gpu.CopyToDevice(_hiVectorYNS, _diVectorYNS);

            _blas.GER(M, N, alpha, _diVectorXMS, _diVectorYNS, _diMatrixS);

            _gpu.CopyFromDevice(_diMatrixS, _hoMatrixS);

            for (int i = 0; i < M; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    float cpuResult = alpha * _hiVectorXMS[i] * _hiVectorYNS[j] + _hiMatrixS[_blas.GetIndexColumnMajor(i, j, M)];
                    Assert.AreEqual(cpuResult, _hoMatrixS[_blas.GetIndexColumnMajor(i, j, M)]);
                }
            }


        }

        [Test]
        public void TestSBMV()
        {
            // Lower fill mode test.
            ClearBuffer(_hiMatrixSymmetricS);
            ClearBuffer(_hiMatrixBandedSymmetricPackedS);
            ClearBuffer(_hiVectorXNS);
            ClearBuffer(_hiVectorYNS);
            
            CreateBandedSymmetricMatrix(_hiMatrixSymmetricS, Ns, k);
            _blas.ConvertBandedSymmetricMatrixCBC(_hiMatrixSymmetricS, _hiMatrixBandedSymmetricPackedS, Ns, k, cublasFillMode.Lower);
            FillBuffer(_hiVectorXNS);
            FillBuffer(_hiVectorYNS);

            float alpha = 4.0f;
            float beta = 3.0f;

            _gpu.CopyToDevice(_hiMatrixBandedSymmetricPackedS, _diMatrixBandedSymmetricPackedS);
            _gpu.CopyToDevice(_hiVectorXNS, _diVectorXNS);
            _gpu.CopyToDevice(_hiVectorYNS, _diVectorYNS);

            _blas.SBMV(Ns, k, alpha, _diMatrixBandedSymmetricPackedS, _diVectorXNS, beta, _diVectorYNS);

            _gpu.CopyFromDevice(_diVectorYNS, _hoVectorYNS);

            for (int vi = 0; vi < Ns; vi++)
            {
                float cpuResult = 0.0f;
                for (int j = 0; j < Ns; j++)
                {
                    cpuResult += alpha * _hiMatrixSymmetricS[_blas.GetIndexColumnMajor(vi, j, Ns)] * _hiVectorXNS[j];
                }

                // Expected result
                cpuResult += beta * _hiVectorYNS[vi];

                Assert.AreEqual(cpuResult, _hoVectorYNS[vi]);
            }

            // Upper fill mode test.
            ClearBuffer(_hiMatrixBandedSymmetricPackedS);
            _blas.ConvertBandedSymmetricMatrixCBC(_hiMatrixSymmetricS, _hiMatrixBandedSymmetricPackedS, Ns, k, cublasFillMode.Upper);
            _gpu.CopyToDevice(_hiMatrixBandedSymmetricPackedS, _diMatrixBandedSymmetricPackedS);
            _gpu.CopyToDevice(_hiVectorYNS, _diVectorYNS);

            _blas.SBMV(Ns, k, alpha, _diMatrixBandedSymmetricPackedS, _diVectorXNS, beta, _diVectorYNS, cublasFillMode.Upper);

            _gpu.CopyFromDevice(_diVectorYNS, _hoVectorYNS);

            for (int vi = 0; vi < Ns; vi++)
            {
                float cpuResult = 0.0f;
                for (int j = 0; j < Ns; j++)
                {
                    cpuResult += alpha * _hiMatrixSymmetricS[_blas.GetIndexColumnMajor(vi, j, Ns)] * _hiVectorXNS[j];
                }

                // Expected result
                cpuResult += beta * _hiVectorYNS[vi];

                Assert.AreEqual(cpuResult, _hoVectorYNS[vi]);
            }
        }

        [Test]
        public void TestSPMV()
        {
            // Lower packed symmetric matrix test.
            ClearBuffer(_hiMatrixSymmetricS);
            ClearBuffer(_hiMatrixSymmetricPackedS);
            ClearBuffer(_hiVectorXNS);
            ClearBuffer(_hiVectorYNS);

            CreateSymmetricMatrix(_hiMatrixSymmetricS, Ns);
            FillBuffer(_hiVectorXNS);
            FillBuffer(_hiVectorYNS);
            _blas.PackSymmetricMatrix(_hiMatrixSymmetricS, _hiMatrixSymmetricPackedS, Ns, cublasFillMode.Lower);

            _gpu.CopyToDevice(_hiMatrixSymmetricPackedS, _diMatrixSymmetricPackedS);
            _gpu.CopyToDevice(_hiVectorXNS, _diVectorXNS);
            _gpu.CopyToDevice(_hiVectorYNS, _diVectorYNS);

            float alpha = 3.0f;
            float beta = 5.0f;

            _blas.SPMV(Ns, alpha, _diMatrixSymmetricPackedS, _diVectorXNS, beta, _diVectorYNS);

            _gpu.CopyFromDevice(_diVectorYNS, _hoVectorYNS);

            for (int vi = 0; vi < Ns; vi++)
            {
                float cpuResult = 0.0f;
                for (int j = 0; j < Ns; j++)
                {
                    cpuResult += alpha * _hiMatrixSymmetricS[_blas.GetIndexColumnMajor(vi, j, Ns)] * _hiVectorXNS[j];
                }

                // Expected result
                cpuResult += beta * _hiVectorYNS[vi];

                Assert.AreEqual(cpuResult, _hoVectorYNS[vi]);
            }

            // Upper packed symmetric matrix test.
            ClearBuffer(_hiMatrixSymmetricPackedS);
            _blas.PackSymmetricMatrix(_hiMatrixSymmetricS, _hiMatrixSymmetricPackedS, Ns, cublasFillMode.Upper);

            _gpu.CopyToDevice(_hiMatrixSymmetricPackedS, _diMatrixSymmetricPackedS);
            _gpu.CopyToDevice(_hiVectorYNS, _diVectorYNS);

            _blas.SPMV(Ns, alpha, _diMatrixSymmetricPackedS, _diVectorXNS, beta, _diVectorYNS, cublasFillMode.Upper);

            _gpu.CopyFromDevice(_diVectorYNS, _hoVectorYNS);

            for (int vi = 0; vi < Ns; vi++)
            {
                float cpuResult = 0.0f;
                for (int j = 0; j < Ns; j++)
                {
                    cpuResult += alpha * _hiMatrixSymmetricS[_blas.GetIndexColumnMajor(vi, j, Ns)] * _hiVectorXNS[j];
                }

                // Expected result
                cpuResult += beta * _hiVectorYNS[vi];

                Assert.AreEqual(cpuResult, _hoVectorYNS[vi]);
            }
        }

        [Test]
        public void TestSPR()
        {
            // Lower packed symmetric matrix test.
            ClearBuffer(_hiMatrixSymmetricS);
            ClearBuffer(_hiMatrixSymmetricPackedS);
            ClearBuffer(_hiVectorXNS);

            CreateSymmetricMatrix(_hiMatrixSymmetricS, Ns);
            FillBuffer(_hiVectorXNS);
            _blas.PackSymmetricMatrix(_hiMatrixSymmetricS, _hiMatrixSymmetricPackedS, Ns, cublasFillMode.Lower);

            _gpu.CopyToDevice(_hiMatrixSymmetricPackedS, _diMatrixSymmetricPackedS);
            _gpu.CopyToDevice(_hiVectorXNS, _diVectorXNS);

            float alpha = 3.0f;

            _blas.SPR(Ns, alpha, _diVectorXNS, _diMatrixSymmetricPackedS);

            _gpu.CopyFromDevice(_diMatrixSymmetricPackedS, _hoMatrixSymmetricPackedS);

            for (int j = 0; j < Ns; j++)
            {
                for (int i = j; i < Ns; i++)
                {
                    float cpuResult = alpha * _hiVectorXNS[i] * _hiVectorXNS[j] + _hiMatrixSymmetricS[_blas.GetIndexColumnMajor(i, j, Ns)];

                    Assert.AreEqual(cpuResult, _hoMatrixSymmetricPackedS[_blas.GetIndexPackedSymmetric(i, j, Ns, cublasFillMode.Lower)]);
                }
            }

            // Upper packed symmetric matrix test.

            _blas.PackSymmetricMatrix(_hiMatrixSymmetricS, _hiMatrixSymmetricPackedS, Ns, cublasFillMode.Upper);

            _gpu.CopyToDevice(_hiMatrixSymmetricPackedS, _diMatrixSymmetricPackedS);

            _blas.SPR(Ns, alpha, _diVectorXNS, _diMatrixSymmetricPackedS, cublasFillMode.Upper);

            _gpu.CopyFromDevice(_diMatrixSymmetricPackedS, _hoMatrixSymmetricPackedS);

            for (int j = 0; j < Ns; j++)
            {
                for (int i = 0; i < j; i++)
                {
                    float cpuResult = alpha * _hiVectorXNS[i] * _hiVectorXNS[j] + _hiMatrixSymmetricS[_blas.GetIndexColumnMajor(i, j, Ns)];

                    Assert.AreEqual(cpuResult, _hoMatrixSymmetricPackedS[_blas.GetIndexPackedSymmetric(i, j, Ns, cublasFillMode.Upper)]);
                }
            }
        }

        [Test]
        public void TestSPR2()
        {
            // Lower packed symmetric matrix test.
            ClearBuffer(_hiMatrixSymmetricS);
            ClearBuffer(_hiMatrixSymmetricPackedS);
            ClearBuffer(_hiVectorXNS);
            ClearBuffer(_hiVectorYNS);
            CreateSymmetricMatrix(_hiMatrixSymmetricS, Ns);
            FillBuffer(_hiVectorXNS);
            FillBuffer(_hiVectorYNS);

            _blas.PackSymmetricMatrix(_hiMatrixSymmetricS, _hiMatrixSymmetricPackedS, Ns, cublasFillMode.Lower);

            _gpu.CopyToDevice(_hiMatrixSymmetricPackedS, _diMatrixSymmetricPackedS);
            _gpu.CopyToDevice(_hiVectorXNS, _diVectorXNS);
            _gpu.CopyToDevice(_hiVectorYNS, _diVectorYNS);

            float alpha = 3.0f;

            _blas.SPR2(Ns, alpha, _diVectorXNS, _diVectorYNS, _diMatrixSymmetricPackedS);

            _gpu.CopyFromDevice(_diMatrixSymmetricPackedS, _hoMatrixSymmetricPackedS);

            for (int j = 0; j < Ns; j++)
            {
                for (int i = j; i < Ns; i++)
                {
                    float cpuResult = alpha * (_hiVectorXNS[i] * _hiVectorYNS[j] + _hiVectorYNS[i] * _hiVectorXNS[j]) + _hiMatrixSymmetricS[_blas.GetIndexColumnMajor(i, j, Ns)];

                    Assert.AreEqual(cpuResult, _hoMatrixSymmetricPackedS[_blas.GetIndexPackedSymmetric(i, j, Ns, cublasFillMode.Lower)]);
                }
            }

            // Upper packed symmetric matrix test.

            _blas.PackSymmetricMatrix(_hiMatrixSymmetricS, _hiMatrixSymmetricPackedS, Ns, cublasFillMode.Upper);

            _gpu.CopyToDevice(_hiMatrixSymmetricPackedS, _diMatrixSymmetricPackedS);

            _blas.SPR2(Ns, alpha, _diVectorXNS, _diVectorYNS, _diMatrixSymmetricPackedS, cublasFillMode.Upper);

            _gpu.CopyFromDevice(_diMatrixSymmetricPackedS, _hoMatrixSymmetricPackedS);

            for (int j = 0; j < Ns; j++)
            {
                for (int i = 0; i < j; i++)
                {
                    float cpuResult = alpha * (_hiVectorXNS[i] * _hiVectorYNS[j] + _hiVectorYNS[i] * _hiVectorXNS[j]) + _hiMatrixSymmetricS[_blas.GetIndexColumnMajor(i, j, Ns)];

                    Assert.AreEqual(cpuResult, _hoMatrixSymmetricPackedS[_blas.GetIndexPackedSymmetric(i, j, Ns, cublasFillMode.Upper)]);
                }
            }
        }

        [Test]
        public void TestSYMV()
        {
            // Lower symmetric matrix test.
            ClearBuffer(_hiMatrixSymmetricS);
            ClearBuffer(_hiVectorXNS);
            ClearBuffer(_hiVectorYNS);
            CreateSymmetricMatrix(_hiMatrixSymmetricS, Ns);
            FillBuffer(_hiVectorXNS);
            FillBuffer(_hiVectorYNS);

            float alpha = 3.0f;
            float beta = 5.0f;

            _gpu.CopyToDevice(_hiMatrixSymmetricS, _diMatrixSymmetricS);
            _gpu.CopyToDevice(_hiVectorXNS, _diVectorXNS);
            _gpu.CopyToDevice(_hiVectorYNS, _diVectorYNS);

            _blas.SYMV(Ns, alpha, _diMatrixSymmetricS, _diVectorXNS, beta, _diVectorYNS);

            _gpu.CopyFromDevice(_diVectorYNS, _hoVectorYNS);

            for (int i = 0; i < Ns; i++)
            {
                float cpuResult = 0.0f;

                for (int j = 0; j < Ns; j++)
                {
                    cpuResult += alpha * _hiMatrixSymmetricS[_blas.GetIndexColumnMajor(i, j, Ns)] * _hiVectorXNS[j];
                }

                cpuResult += beta * _hiVectorYNS[i];

                Assert.AreEqual(cpuResult, _hoVectorYNS[i]);
            }
            
            // Upper mode test
            _gpu.CopyToDevice(_hiVectorYNS, _diVectorYNS);

            _blas.SYMV(Ns, alpha, _diMatrixSymmetricS, _diVectorXNS, beta, _diVectorYNS, cublasFillMode.Upper);

            _gpu.CopyFromDevice(_diVectorYNS, _hoVectorYNS);

            for (int i = 0; i < Ns; i++)
            {
                float cpuResult = 0.0f;

                for (int j = 0; j < Ns; j++)
                {
                    cpuResult += alpha * _hiMatrixSymmetricS[_blas.GetIndexColumnMajor(i, j, Ns)] * _hiVectorXNS[j];
                }

                cpuResult += beta * _hiVectorYNS[i];

                Assert.AreEqual(cpuResult, _hoVectorYNS[i]);
            }
        }

        [Test]
        public void TestSYR()
        {
            // Lower symmetric matrix test.
            ClearBuffer(_hiMatrixSymmetricS);
            ClearBuffer(_hiVectorXNS);
            CreateSymmetricMatrix(_hiMatrixSymmetricS, Ns);
            FillBuffer(_hiVectorXNS);

            float alpha = 3.0f;

            _gpu.CopyToDevice(_hiMatrixSymmetricS, _diMatrixSymmetricS);
            _gpu.CopyToDevice(_hiVectorXNS, _diVectorXNS);

            _blas.SYR(Ns, alpha, _diVectorXNS, _diMatrixSymmetricS);

            _gpu.CopyFromDevice(_diMatrixSymmetricS, _hoMatrixSymmetricS);

            for (int j = 0; j < Ns; j++)
            {
                for (int i = j; i < Ns; i++)
                {
                    float cpuResult = alpha * _hiVectorXNS[i] * _hiVectorXNS[j] + _hiMatrixSymmetricS[_blas.GetIndexColumnMajor(i, j, Ns)];
                    Assert.AreEqual(cpuResult, _hoMatrixSymmetricS[_blas.GetIndexColumnMajor(i, j, Ns)]);
                }
            }

            // Upper symmetrici matrix test.
            _gpu.CopyToDevice(_hiMatrixSymmetricS, _diMatrixSymmetricS);

            _blas.SYR(Ns, alpha, _diVectorXNS, _diMatrixSymmetricS, cublasFillMode.Upper);

            _gpu.CopyFromDevice(_diMatrixSymmetricS, _hoMatrixSymmetricS);

            for (int j = 0; j < Ns; j++)
            {
                for (int i = 0; i <= j; i++)
                {
                    float cpuResult = alpha * _hiVectorXNS[i] * _hiVectorXNS[j] + _hiMatrixSymmetricS[_blas.GetIndexColumnMajor(i, j, Ns)];
                    Assert.AreEqual(cpuResult, _hoMatrixSymmetricS[_blas.GetIndexColumnMajor(i, j, Ns)]);
                }
            }
        }

        [Test]
        public void TestSYR2()
        {
            // Lower symmetric matrix test.
            ClearBuffer(_hiMatrixSymmetricS);
            ClearBuffer(_hiVectorXNS);
            ClearBuffer(_hiVectorYNS);
            CreateSymmetricMatrix(_hiMatrixSymmetricS, Ns);
            FillBuffer(_hiVectorXNS);
            FillBuffer(_hiVectorYNS);

            float alpha = 3.0f;

            _gpu.CopyToDevice(_hiMatrixSymmetricS, _diMatrixSymmetricS);
            _gpu.CopyToDevice(_hiVectorXNS, _diVectorXNS);
            _gpu.CopyToDevice(_hiVectorYNS, _diVectorYNS);

            _blas.SYR2(Ns, alpha, _diVectorXNS, _diVectorYNS, _diMatrixSymmetricS);

            _gpu.CopyFromDevice(_diMatrixSymmetricS, _hoMatrixSymmetricS);

            for (int j = 0; j < Ns; j++)
            {
                for (int i = j; i < Ns; i++)
                {
                    float cpuResult = alpha * (_hiVectorXNS[i] * _hiVectorYNS[j] + _hiVectorYNS[i] * _hiVectorXNS[j]) + _hiMatrixSymmetricS[_blas.GetIndexColumnMajor(i, j, Ns)];
                    Assert.AreEqual(cpuResult, _hoMatrixSymmetricS[_blas.GetIndexColumnMajor(i, j, Ns)]);
                }
            }

            // Upper symmetric matrix test.
            _gpu.CopyToDevice(_hiMatrixSymmetricS, _diMatrixSymmetricS);

            _blas.SYR2(Ns, alpha, _diVectorXNS, _diVectorYNS, _diMatrixSymmetricS, cublasFillMode.Upper);

            _gpu.CopyFromDevice(_diMatrixSymmetricS, _hoMatrixSymmetricS);

            for (int j = 0; j < Ns; j++)
            {
                for (int i = 0; i <= j; i++)
                {
                    float cpuResult = alpha * (_hiVectorXNS[i] * _hiVectorYNS[j] + _hiVectorYNS[i] * _hiVectorXNS[j]) + _hiMatrixSymmetricS[_blas.GetIndexColumnMajor(i, j, Ns)];
                    Assert.AreEqual(cpuResult, _hoMatrixSymmetricS[_blas.GetIndexColumnMajor(i, j, Ns)]);
                }
            }
        }

        [Test]
        public void TestTBMV()
        {
            ClearBuffer(_hiMatrixSymmetricS);
            ClearBuffer(_hiVectorXNS);
            ClearBuffer(_hiMatrixBandedSymmetricPackedS);

            // Create Lower triangular banded matrix
            CreateBandedMatrix(_hiMatrixSymmetricS, Ns, Ns, k, 0);
            _blas.ConvertBandedSymmetricMatrixCBC(_hiMatrixSymmetricS, _hiMatrixBandedSymmetricPackedS, Ns, k, cublasFillMode.Lower);

            FillBuffer(_hiVectorXNS);

            // No-transpose and Lower matrix test.
            _gpu.CopyToDevice(_hiMatrixBandedSymmetricPackedS, _diMatrixBandedSymmetricPackedS);
            _gpu.CopyToDevice(_hiVectorXNS, _diVectorXNS);

            _blas.TBMV(Ns, k, _diMatrixBandedSymmetricPackedS, _diVectorXNS);

            _gpu.CopyFromDevice(_diVectorXNS, _hoVectorXNS);

            for (int i = 0; i < Ns; i++)
            {
                float cpuResult = 0.0f;

                for (int j = 0; j < Ns; j++)
                {
                    cpuResult += _hiMatrixSymmetricS[_blas.GetIndexColumnMajor(i, j, Ns)] * _hiVectorXNS[j];
                }

                Assert.AreEqual(cpuResult, _hoVectorXNS[i]);
            }

            // No-transpose and Upper matrix test.
            ClearBuffer(_hiMatrixSymmetricS);
            ClearBuffer(_hiMatrixBandedSymmetricPackedS);
            CreateBandedMatrix(_hiMatrixSymmetricS, Ns, Ns, 0, k);

            _blas.ConvertBandedSymmetricMatrixCBC(_hiMatrixSymmetricS, _hiMatrixBandedSymmetricPackedS, Ns, k, cublasFillMode.Upper);
            
            _gpu.CopyToDevice(_hiMatrixBandedSymmetricPackedS, _diMatrixBandedSymmetricPackedS);
            _gpu.CopyToDevice(_hiVectorXNS, _diVectorXNS);

            _blas.TBMV(Ns, k, _diMatrixBandedSymmetricPackedS, _diVectorXNS, cublasOperation.N, cublasFillMode.Upper);

            _gpu.CopyFromDevice(_diVectorXNS, _hoVectorXNS);

            for (int i = 0; i < Ns; i++)
            {
                float cpuResult = 0.0f;

                for (int j = 0; j < Ns; j++)
                {
                    cpuResult += _hiMatrixSymmetricS[_blas.GetIndexColumnMajor(i, j, Ns)] * _hiVectorXNS[j];
                }

                Assert.AreEqual(cpuResult, _hoVectorXNS[i]);
            }

            // Transpose and Lower matrix test.
            ClearBuffer(_hiMatrixSymmetricS);
            ClearBuffer(_hiMatrixBandedSymmetricPackedS);
            CreateBandedMatrix(_hiMatrixSymmetricS, Ns, Ns, k, 0);

            _blas.ConvertBandedSymmetricMatrixCBC(_hiMatrixSymmetricS, _hiMatrixBandedSymmetricPackedS, Ns, k, cublasFillMode.Lower);

            _gpu.CopyToDevice(_hiMatrixBandedSymmetricPackedS, _diMatrixBandedSymmetricPackedS);
            _gpu.CopyToDevice(_hiVectorXNS, _diVectorXNS);

            _blas.TBMV(Ns, k, _diMatrixBandedSymmetricPackedS, _diVectorXNS, cublasOperation.T, cublasFillMode.Lower);

            _gpu.CopyFromDevice(_diVectorXNS, _hoVectorXNS);

            for (int i = 0; i < Ns; i++)
            {
                float cpuResult = 0.0f;

                for (int j = 0; j < Ns; j++)
                {
                    cpuResult += _hiMatrixSymmetricS[_blas.GetIndexColumnMajor(j, i, Ns)] * _hiVectorXNS[j];
                }

                Assert.AreEqual(cpuResult, _hoVectorXNS[i]);
            }

            // Transpose and Upper matrix test.
            ClearBuffer(_hiMatrixSymmetricS);
            ClearBuffer(_hiMatrixBandedSymmetricPackedS);
            CreateBandedMatrix(_hiMatrixSymmetricS, Ns, Ns, 0, k);

            _blas.ConvertBandedSymmetricMatrixCBC(_hiMatrixSymmetricS, _hiMatrixBandedSymmetricPackedS, Ns, k, cublasFillMode.Upper);

            _gpu.CopyToDevice(_hiMatrixBandedSymmetricPackedS, _diMatrixBandedSymmetricPackedS);
            _gpu.CopyToDevice(_hiVectorXNS, _diVectorXNS);

            _blas.TBMV(Ns, k, _diMatrixBandedSymmetricPackedS, _diVectorXNS, cublasOperation.T, cublasFillMode.Upper);

            _gpu.CopyFromDevice(_diVectorXNS, _hoVectorXNS);

            for (int i = 0; i < Ns; i++)
            {
                float cpuResult = 0.0f;

                for (int j = 0; j < Ns; j++)
                {
                    cpuResult += _hiMatrixSymmetricS[_blas.GetIndexColumnMajor(j, i, Ns)] * _hiVectorXNS[j];
                }

                Assert.AreEqual(cpuResult, _hoVectorXNS[i]);
            }
        }

        /* TBSV : Singularity test need. cublas dose not support singularity test.
        [Test]
        public void TestTBSV()
        {
            ClearBuffer(_hiMatrixSymmetricS);
            ClearBuffer(_hiVectorXNS);
            ClearBuffer(_hiMatrixBandedSymmetricPackedS);

            // Create Lower triangular banded matrix
            CreateBandedMatrix(_hiMatrixSymmetricS, Ns, Ns, k, 0);
            _blas.PackBandedSymmetricMatrix(_hiMatrixSymmetricS, _hiMatrixBandedSymmetricPackedS, Ns, k, cublasFillMode.Lower);

            FillBuffer(_hiVectorXNS);

            // No-transpose and Lower matrix test.
            _gpu.CopyToDevice(_hiMatrixBandedSymmetricPackedS, _diMatrixBandedSymmetricPackedS);
            _gpu.CopyToDevice(_hiVectorXNS, _diVectorXNS);

            _blas.TBSV(Ns, k, _diMatrixBandedSymmetricPackedS, _diVectorXNS);

            _blas.TBMV(Ns, k, _diMatrixBandedSymmetricPackedS, _diVectorXNS);
            
            _gpu.CopyFromDevice(_diVectorXNS, _hoVectorXNS);

            for (int i = 0; i < Ns; i++)
            {
                Assert.AreEqual(_hiVectorXNS[i], _hoVectorXNS[i], 0.1f);
            }


            // No-transpose and Upper matrix test.
            _blas.PackBandedSymmetricMatrix(_hiMatrixSymmetricS, _hiMatrixBandedSymmetricPackedS, Ns, k, cublasFillMode.Upper);

            _gpu.CopyToDevice(_hiMatrixBandedSymmetricPackedS, _diMatrixBandedSymmetricPackedS);
            _gpu.CopyToDevice(_hiVectorXNS, _diVectorXNS);

            _blas.TBSV(Ns, k, _diMatrixBandedSymmetricPackedS, _diVectorXNS, cublasOperation.N, cublasFillMode.Upper);

            _blas.TBMV(Ns, k, _diMatrixBandedSymmetricPackedS, _diVectorXNS, cublasOperation.N, cublasFillMode.Upper);

            _gpu.CopyFromDevice(_diVectorXNS, _hoVectorXNS);

            for (int i = 0; i < Ns; i++)
            {
                Assert.AreEqual(_hiVectorXNS[i], _hoVectorXNS[i], 0.1f);
            }
        }
        */

        [Test]
        public void TestTPMV()
        {
            // No Transpose, Lower matrix.
            ClearBuffer(_hiMatrixSymmetricS);
            ClearBuffer(_hiVectorXNS);
            ClearBuffer(_hiMatrixSymmetricPackedS);

            CreateBandedMatrix(_hiMatrixSymmetricS, Ns, Ns, Ns - 1, 0); // Create Lower triangular matrix.
            _blas.PackSymmetricMatrix(_hiMatrixSymmetricS, _hiMatrixSymmetricPackedS, Ns, cublasFillMode.Lower);

            _gpu.CopyToDevice(_hiMatrixSymmetricPackedS, _diMatrixSymmetricPackedS);
            _gpu.CopyToDevice(_hiVectorXNS, _diVectorXNS);

            _blas.TPMV(Ns, _diMatrixSymmetricPackedS, _diVectorXNS);

            _gpu.CopyFromDevice(_diVectorXNS, _hoVectorXNS);

            for (int i = 0; i < Ns; i++)
            {
                float cpuResult = 0.0f;

                for (int j = 0; j < Ns; j++)
                {
                    cpuResult += _hiMatrixSymmetricS[_blas.GetIndexColumnMajor(i, j, Ns)] * _hiVectorXNS[j];
                }

                Assert.AreEqual(cpuResult, _hoVectorXNS[i]);
            }

            // No Transpose, Upper matrix.
            ClearBuffer(_hiMatrixSymmetricS);
            CreateBandedMatrix(_hiMatrixSymmetricS, Ns, Ns, 0, Ns - 1); // Create Lower triangular matrix.
            _blas.PackSymmetricMatrix(_hiMatrixSymmetricS, _hiMatrixSymmetricPackedS, Ns, cublasFillMode.Upper);

            _gpu.CopyToDevice(_hiMatrixSymmetricPackedS, _diMatrixSymmetricPackedS);
            _gpu.CopyToDevice(_hiVectorXNS, _diVectorXNS);

            _blas.TPMV(Ns, _diMatrixSymmetricPackedS, _diVectorXNS, cublasOperation.N, cublasFillMode.Upper);

            _gpu.CopyFromDevice(_diVectorXNS, _hoVectorXNS);

            for (int i = 0; i < Ns; i++)
            {
                float cpuResult = 0.0f;

                for (int j = 0; j < Ns; j++)
                {
                    cpuResult += _hiMatrixSymmetricS[_blas.GetIndexColumnMajor(i, j, Ns)] * _hiVectorXNS[j];
                }

                Assert.AreEqual(cpuResult, _hoVectorXNS[i]);
            }

            // Transpose, Lower matrix.
            ClearBuffer(_hiMatrixSymmetricS);
            CreateBandedMatrix(_hiMatrixSymmetricS, Ns, Ns, Ns - 1, 0); // Create Lower triangular matrix.
            _blas.PackSymmetricMatrix(_hiMatrixSymmetricS, _hiMatrixSymmetricPackedS, Ns, cublasFillMode.Lower);

            _gpu.CopyToDevice(_hiMatrixSymmetricPackedS, _diMatrixSymmetricPackedS);
            _gpu.CopyToDevice(_hiVectorXNS, _diVectorXNS);

            _blas.TPMV(Ns, _diMatrixSymmetricPackedS, _diVectorXNS, cublasOperation.T, cublasFillMode.Lower);

            _gpu.CopyFromDevice(_diVectorXNS, _hoVectorXNS);

            for (int i = 0; i < Ns; i++)
            {
                float cpuResult = 0.0f;

                for (int j = 0; j < Ns; j++)
                {
                    cpuResult += _hiMatrixSymmetricS[_blas.GetIndexColumnMajor(j, i, Ns)] * _hiVectorXNS[j];
                }

                Assert.AreEqual(cpuResult, _hoVectorXNS[i]);
            }

            // Transpose, Lower matrix.
            ClearBuffer(_hiMatrixSymmetricS);
            CreateBandedMatrix(_hiMatrixSymmetricS, Ns, Ns, 0, Ns - 1); // Create Lower triangular matrix.
            _blas.PackSymmetricMatrix(_hiMatrixSymmetricS, _hiMatrixSymmetricPackedS, Ns, cublasFillMode.Upper);

            _gpu.CopyToDevice(_hiMatrixSymmetricPackedS, _diMatrixSymmetricPackedS);
            _gpu.CopyToDevice(_hiVectorXNS, _diVectorXNS);

            _blas.TPMV(Ns, _diMatrixSymmetricPackedS, _diVectorXNS, cublasOperation.T, cublasFillMode.Upper);

            _gpu.CopyFromDevice(_diVectorXNS, _hoVectorXNS);

            for (int i = 0; i < Ns; i++)
            {
                float cpuResult = 0.0f;

                for (int j = 0; j < Ns; j++)
                {
                    cpuResult += _hiMatrixSymmetricS[_blas.GetIndexColumnMajor(j, i, Ns)] * _hiVectorXNS[j];
                }

                Assert.AreEqual(cpuResult, _hoVectorXNS[i]);
            }
        }

        /* TPSV : Singularity test need. cublas does not support singularity test. */

        [Test]//[Ignore]
        public void TestTRMV()
        {
            // No Transpose, Lower matrix.
            ClearBuffer(_hiMatrixSymmetricS);
            ClearBuffer(_hiVectorXNS);

            CreateBandedMatrix(_hiMatrixSymmetricS, Ns, Ns, Ns - 1, 0); // Create Lower triangular matrix.

            _gpu.CopyToDevice(_hiMatrixSymmetricS, _diMatrixSymmetricS);
            _gpu.CopyToDevice(_hiVectorXNS, _diVectorXNS);

            _blas.TRMV(Ns, _diMatrixSymmetricPackedS, _diVectorXNS);

            _gpu.CopyFromDevice(_diVectorXNS, _hoVectorXNS);

            for (int i = 0; i < Ns; i++)
            {
                float cpuResult = 0.0f;

                for (int j = 0; j < Ns; j++)
                {
                    cpuResult += _hiMatrixSymmetricS[_blas.GetIndexColumnMajor(i, j, Ns)] * _hiVectorXNS[j];
                }

                Assert.AreEqual(cpuResult, _hoVectorXNS[i]);
            }

            // No Transpose, Upper matrix.
            CreateBandedMatrix(_hiMatrixSymmetricS, Ns, Ns, 0, Ns - 1); // Create Lower triangular matrix.

            _gpu.CopyToDevice(_hiMatrixSymmetricS, _diMatrixSymmetricS);
            _gpu.CopyToDevice(_hiVectorXNS, _diVectorXNS);

            _blas.TRMV(Ns, _diMatrixSymmetricPackedS, _diVectorXNS, cublasOperation.N, cublasFillMode.Upper);

            _gpu.CopyFromDevice(_diVectorXNS, _hoVectorXNS);

            for (int i = 0; i < Ns; i++)
            {
                float cpuResult = 0.0f;

                for (int j = 0; j < Ns; j++)
                {
                    cpuResult += _hiMatrixSymmetricS[_blas.GetIndexColumnMajor(i, j, Ns)] * _hiVectorXNS[j];
                }

                Assert.AreEqual(cpuResult, _hoVectorXNS[i]);
            }

            // Transpose, Lower matrix.
            CreateBandedMatrix(_hiMatrixSymmetricS, Ns, Ns, Ns - 1, 0); // Create Lower triangular matrix.

            _gpu.CopyToDevice(_hiMatrixSymmetricS, _diMatrixSymmetricS);
            _gpu.CopyToDevice(_hiVectorXNS, _diVectorXNS);

            _blas.TRMV(Ns, _diMatrixSymmetricPackedS, _diVectorXNS, cublasOperation.T);

            _gpu.CopyFromDevice(_diVectorXNS, _hoVectorXNS);

            for (int i = 0; i < Ns; i++)
            {
                float cpuResult = 0.0f;

                for (int j = 0; j < Ns; j++)
                {
                    cpuResult += _hiMatrixSymmetricS[_blas.GetIndexColumnMajor(j, i, Ns)] * _hiVectorXNS[j];
                }

                Assert.AreEqual(cpuResult, _hoVectorXNS[i]);
            }

            // Transpose, Upper matrix.
            CreateBandedMatrix(_hiMatrixSymmetricS, Ns, Ns, 0, Ns - 1); // Create Lower triangular matrix.

            _gpu.CopyToDevice(_hiMatrixSymmetricS, _diMatrixSymmetricS);
            _gpu.CopyToDevice(_hiVectorXNS, _diVectorXNS);

            _blas.TRMV(Ns, _diMatrixSymmetricPackedS, _diVectorXNS, cublasOperation.T, cublasFillMode.Upper);

            _gpu.CopyFromDevice(_diVectorXNS, _hoVectorXNS);

            for (int i = 0; i < Ns; i++)
            {
                float cpuResult = 0.0f;

                for (int j = 0; j < Ns; j++)
                {
                    cpuResult += _hiMatrixSymmetricS[_blas.GetIndexColumnMajor(j, i, Ns)] * _hiVectorXNS[j];
                }

                Assert.AreEqual(cpuResult, _hoVectorXNS[i]);
            }
        }

        /* TRSV : Singularity test need. cublas does not support singularity test. */

        public void TestSetUp()
        {

        }

        public void TestTearDown()
        {

        }
    }
}
