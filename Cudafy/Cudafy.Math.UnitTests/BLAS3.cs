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
    public class BLAS3 : ICudafyUnitTest
    {
        private GPGPU _gpu;

        private GPGPUBLAS _blas;

        private float[] _hiMatrixMN;
        private float[] _hiMatrixMK;
        private float[] _hiMatrixKN;

        private float[] _hiMatrixNK;
        private float[] _hiMatrixNK2;
        private float[] _hiMatrixNN;

        private float[] _diMatrixMN;
        private float[] _diMatrixMK;
        private float[] _diMatrixKN;

        private float[] _diMatrixNK;
        private float[] _diMatrixNK2;
        private float[] _diMatrixNN;

        private float[] _hoMatrixMN;
        private float[] _hoMatrixNN;

        private int M = 128;
        private int K = 128;
        private int N = 128;

        private float alpha = 6.0f;
        private float beta = 4.0f;

        [TestFixtureSetUp]
        public void SetUp()
        {
            _gpu = CudafyHost.CreateDevice(CudafyModes.Target);
            _blas = GPGPUBLAS.Create(_gpu);

            _hiMatrixMN = new float[M * N];
            _hiMatrixMK = new float[M * K];
            _hiMatrixKN = new float[K * N];

            _hiMatrixNK = new float[N * K];
            _hiMatrixNK2 = new float[N * K];
            _hiMatrixNN = new float[N * N];

            _hoMatrixMN = new float[M * N];
            _hoMatrixNN = new float[N * N];

            _diMatrixMN = _gpu.Allocate(_hiMatrixMN);
            _diMatrixMK = _gpu.Allocate(_hiMatrixMK);
            _diMatrixKN = _gpu.Allocate(_hiMatrixKN);

            _diMatrixNK = _gpu.Allocate(_hiMatrixNK);
            _diMatrixNK2 = _gpu.Allocate(_hiMatrixNK2);
            _diMatrixNN = _gpu.Allocate(_hiMatrixNN);
        }

        [TestFixtureTearDown]
        public void TearDown()
        {
            _blas.Dispose();

            _gpu.Free(_diMatrixMN);
            _gpu.Free(_diMatrixMK);
            _gpu.Free(_diMatrixKN);

            _gpu.Free(_diMatrixNK);
            _gpu.Free(_diMatrixNK2);
            _gpu.Free(_diMatrixNN);
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

        public void FillBuffer(float[] buffer)
        {
            Random rand = new Random(Environment.TickCount);
            for (int i = 0; i < buffer.Length; i++)
            {
                buffer[i] = rand.Next(32);
            }
            System.Threading.Thread.Sleep(rand.Next(100));
        }

        public void ClearBuffer(float[] buffer)
        {
            for (int i = 0; i < buffer.Length; i++)
            {
                buffer[i] = 0.0f;
            }
        }

        [Test]
        public void TestGEMM()
        {
            FillBuffer(_hiMatrixMK); // A
            FillBuffer(_hiMatrixKN); // B
            FillBuffer(_hiMatrixMN); // C

            _gpu.CopyToDevice(_hiMatrixMK, _diMatrixMK);
            _gpu.CopyToDevice(_hiMatrixKN, _diMatrixKN);
            _gpu.CopyToDevice(_hiMatrixMN, _diMatrixMN);

            _blas.GEMM(M, K, N, alpha, _diMatrixMK, _diMatrixKN, beta, _diMatrixMN);

            _gpu.CopyFromDevice(_diMatrixMN, _hoMatrixMN);

            for (int i = 0; i < M; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    float cpuResult = 0.0f;

                    for (int k = 0; k < K; k++)
                    {
                        cpuResult += alpha * _hiMatrixMK[_blas.GetIndexColumnMajor(i, k, M)] * _hiMatrixKN[_blas.GetIndexColumnMajor(k, j, K)];
                    }

                    cpuResult += beta * _hiMatrixMN[_blas.GetIndexColumnMajor(i, j, M)];

                    Assert.AreEqual(cpuResult, _hoMatrixMN[_blas.GetIndexColumnMajor(i, j, M)]);

                }
            }
        }

        [Test]
        public void TestSYMM()
        {
            ClearBuffer(_hiMatrixMK);

            CreateSymmetricMatrix(_hiMatrixMK, M); // A
            FillBuffer(_hiMatrixKN); // B
            FillBuffer(_hiMatrixMN); // C
            
            _gpu.CopyToDevice(_hiMatrixMK, _diMatrixMK);
            _gpu.CopyToDevice(_hiMatrixKN, _diMatrixKN);
            _gpu.CopyToDevice(_hiMatrixMN, _diMatrixMN);

            _blas.SYMM(M, N, alpha, _diMatrixMK, _diMatrixKN, beta, _diMatrixMN);

            _gpu.CopyFromDevice(_diMatrixMN, _hoMatrixMN);

            for (int i = 0; i < M; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    float cpuResult = 0.0f;

                    for (int k = 0; k < K; k++)
                    {
                        cpuResult += alpha * _hiMatrixMK[_blas.GetIndexColumnMajor(i, k, M)] * _hiMatrixKN[_blas.GetIndexColumnMajor(k, j, K)];
                    }

                    cpuResult += beta * _hiMatrixMN[_blas.GetIndexColumnMajor(i, j, M)];

                    Assert.AreEqual(cpuResult, _hoMatrixMN[_blas.GetIndexColumnMajor(i, j, M)]);

                }
            }
        }

        [Test]
        public void TestSYRK()
        {
            ClearBuffer(_hiMatrixNK);
            ClearBuffer(_hiMatrixNN);

            FillBuffer(_hiMatrixNK); // A
            CreateSymmetricMatrix(_hiMatrixNN, N); // C

            _gpu.CopyToDevice(_hiMatrixNK, _diMatrixNK);
            _gpu.CopyToDevice(_hiMatrixNN, _diMatrixNN);

            _blas.SYRK(M, N, alpha, _diMatrixNK, beta, _diMatrixNN);

            _gpu.CopyFromDevice(_diMatrixNN, _hoMatrixNN);

            for (int i = 0; i < M; i++)
            {
                for (int j = 0; j <= i; j++)
                {
                    float cpuResult = 0.0f;

                    for (int k = 0; k < K; k++)
                    {
                        cpuResult += alpha * _hiMatrixNK[_blas.GetIndexColumnMajor(i, k, N)] * _hiMatrixNK[_blas.GetIndexColumnMajor(j, k, N)];
                    }

                    cpuResult += beta * _hiMatrixNN[_blas.GetIndexColumnMajor(i, j, M)];

                    Assert.AreEqual(cpuResult, _hoMatrixNN[_blas.GetIndexColumnMajor(i, j, M)]);

                }
            }
        }

        [Test]
        public void TestSYR2K()
        {
            ClearBuffer(_hiMatrixNK);
            ClearBuffer(_hiMatrixNK2);
            ClearBuffer(_hiMatrixNN);

            FillBuffer(_hiMatrixNK); // A
            FillBuffer(_hiMatrixNK2); // B
            CreateSymmetricMatrix(_hiMatrixNN, N); // C

            _gpu.CopyToDevice(_hiMatrixNK, _diMatrixNK);
            _gpu.CopyToDevice(_hiMatrixNK2, _diMatrixNK2);
            _gpu.CopyToDevice(_hiMatrixNN, _diMatrixNN);

            _blas.SYR2K(M, N, alpha, _diMatrixNK, _diMatrixNK2, beta, _diMatrixNN);

            _gpu.CopyFromDevice(_diMatrixNN, _hoMatrixNN);

            for (int i = 0; i < M; i++)
            {
                for (int j = 0; j <= i; j++)
                {
                    float cpuResult = 0.0f;

                    for (int k = 0; k < K; k++)
                    {
                        cpuResult += alpha * (_hiMatrixNK[_blas.GetIndexColumnMajor(i, k, N)] * _hiMatrixNK2[_blas.GetIndexColumnMajor(j, k, N)] + _hiMatrixNK2[_blas.GetIndexColumnMajor(i, k, N)] * _hiMatrixNK[_blas.GetIndexColumnMajor(j, k, N)]);
                    }

                    cpuResult += beta * _hiMatrixNN[_blas.GetIndexColumnMajor(i, j, N)];

                    Assert.AreEqual(cpuResult, _hoMatrixNN[_blas.GetIndexColumnMajor(i, j, N)]);

                }
            }
        }

        /* TRSM : Singularity test need. cublas does not support singularity test. */

        public void TestSetUp()
        {

        }

        public void TestTearDown()
        {

        }
    }
}
