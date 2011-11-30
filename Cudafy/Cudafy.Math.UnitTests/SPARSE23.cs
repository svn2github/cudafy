/* Added by Kichang Kim (kkc0923@hotmail.com) */
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Diagnostics;
using System.Reflection;
using Cudafy.Types;
using Cudafy.Host;
using Cudafy.Maths.SPARSE;
using Cudafy.UnitTests;
using NUnit.Framework;

namespace Cudafy.Maths.UnitTests
{
    [TestFixture]
    public class SPARSE23 : ICudafyUnitTest
    {
        private GPGPU _gpu;

        private GPGPUSPARSE _sparse;

        int M = 512;
        int N = 128;
        int K = 256;

        private float[] _hiMatrixMN;
        private float[] _hiMatrixMK;
        private float[] _hiMatrixKN;
        private float[] _hiVectorN;
        private float[] _hiVectorM;

        private float[] _hoVectorM;
        private float[] _hoMatrixMN;

        private float[] _diMatrixMN;
        private float[] _diVectorN;
        private float[] _diVectorM;
        private float[] _diMatrixMK;
        private float[] _diMatrixKN;

        private float[] _hiCSRVals;
        private int[] _hiCSRRows;
        private int[] _hiCSRCols;
        private float[] _diCSRVals;
        private int[] _diCSRRows;
        private int[] _diCSRCols;
        private int[] _hinnzPerRow;
        private int[] _dinnzPerRow;

        float alpha = 3.0f;
        float beta = 4.0f;

        [TestFixtureSetUp]
        public void SetUp()
        {
            _gpu = CudafyHost.CreateDevice(CudafyModes.Target);
            _sparse = GPGPUSPARSE.Create(_gpu);

            _hiMatrixMN = new float[M * N];
            _hiVectorM = new float[M];
            _hiVectorN = new float[N];
            _hiMatrixMK = new float[M * K];
            _hiMatrixKN = new float[K * N];

            _hoMatrixMN = new float[M * N];
            _hoVectorM = new float[M];

            _diMatrixMN = _gpu.Allocate(_hiMatrixMN);
            _diVectorN = _gpu.Allocate(_hiVectorN);
            _diVectorM = _gpu.Allocate(_hiVectorM);
            _diMatrixMK = _gpu.Allocate(_hiMatrixMK);
            _diMatrixKN = _gpu.Allocate(_hiMatrixKN);

            _hinnzPerRow = new int[M + 1];

            _dinnzPerRow = _gpu.Allocate(_hinnzPerRow);
        }

        [TestFixtureTearDown]
        public void TearDown()
        {
            _sparse.Dispose();

            _gpu.Free(_diMatrixMN);
            _gpu.Free(_diMatrixMK);
            _gpu.Free(_diMatrixKN);
            _gpu.Free(_diVectorN);
            _gpu.Free(_diVectorM);

            _gpu.Free(_dinnzPerRow);
        }

        private void FillBuffer(float[] buffer)
        {
            Random rand = new Random(Environment.TickCount);
            for (int i = 0; i < buffer.Length; i++)
            {
                buffer[i] = (float)rand.Next(32);
            }

            System.Threading.Thread.Sleep(rand.Next(50));
        }

        private void FillBufferSparse(float[] buffer)
        {
            Random rand = new Random(Environment.TickCount);
            for (int i = 0; i < buffer.Length; i++)
            {
                if (rand.Next(5) < 2)
                {
                    buffer[i] = (float)rand.Next(32);
                }
                else
                {
                    buffer[i] = 0.0f;
                }
            }

            System.Threading.Thread.Sleep(rand.Next(50));
        }

        [Test]
        public void TestCSRMV()
        {
            FillBufferSparse(_hiMatrixMN);
            FillBuffer(_hiVectorM);
            FillBuffer(_hiVectorN);

            _gpu.CopyToDevice(_hiMatrixMN, _diMatrixMN);
            _gpu.CopyToDevice(_hiVectorM, _diVectorM);
            _gpu.CopyToDevice(_hiVectorN, _diVectorN);

            int nnz = _sparse.NNZ(M, N, _diMatrixMN, _dinnzPerRow);

            _hiCSRVals = new float[nnz];
            _hiCSRCols = new int[nnz];
            _hiCSRRows = new int[M + 1];
            _diCSRVals = _gpu.Allocate(_hiCSRVals);
            _diCSRCols = _gpu.Allocate(_hiCSRCols);
            _diCSRRows = _gpu.Allocate(_hiCSRRows);
            
            _sparse.Dense2CSR(M, N, _diMatrixMN, _dinnzPerRow, _diCSRVals, _diCSRRows, _diCSRCols);

            _sparse.CSRMV(M, N, alpha, _diCSRVals, _diCSRRows, _diCSRCols, _diVectorN, beta, _diVectorM);

            _gpu.CopyFromDevice(_diVectorM, _hoVectorM);

            _gpu.Free(_diCSRVals);
            _gpu.Free(_diCSRCols);
            _gpu.Free(_diCSRRows);

            for (int i = 0; i < M; i++)
            {
                float cpuResult = 0.0f;

                for (int j = 0; j < N; j++)
                {
                    cpuResult += alpha * _hiMatrixMN[_sparse.GetIndexColumnMajor(i, j, M)] * _hiVectorN[j];
                }

                cpuResult += beta * _hiVectorM[i];

                Assert.AreEqual(cpuResult, _hoVectorM[i]);
            }
        }

        [Test]
        public void TestCSRMM()
        {
            FillBufferSparse(_hiMatrixMK);
            FillBuffer(_hiMatrixKN);
            FillBuffer(_hiMatrixMN);

            _gpu.CopyToDevice(_hiMatrixMK, _diMatrixMK);
            _gpu.CopyToDevice(_hiMatrixKN, _diMatrixKN);
            _gpu.CopyToDevice(_hiMatrixMN, _diMatrixMN);

            int nnz = _sparse.NNZ(M, K, _diMatrixMK, _dinnzPerRow);

            _hiCSRVals = new float[nnz];
            _hiCSRCols = new int[nnz];
            _hiCSRRows = new int[M + 1];
            _diCSRVals = _gpu.Allocate(_hiCSRVals);
            _diCSRCols = _gpu.Allocate(_hiCSRCols);
            _diCSRRows = _gpu.Allocate(_hiCSRRows);

            _sparse.Dense2CSR(M, K, _diMatrixMK, _dinnzPerRow, _diCSRVals, _diCSRRows, _diCSRCols);

            _sparse.CSRMM(M, K, N, alpha, _diCSRVals, _diCSRRows, _diCSRCols, _diMatrixKN, beta, _diMatrixMN);

            _gpu.Free(_diCSRVals);
            _gpu.Free(_diCSRCols);
            _gpu.Free(_diCSRRows);

            _gpu.CopyFromDevice(_diMatrixMN, _hoMatrixMN);

            for (int i = 0; i < M; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    float cpuResult = 0.0f;

                    for (int k = 0; k < K; k++)
                    {
                        cpuResult += alpha * _hiMatrixMK[_sparse.GetIndexColumnMajor(i, k, M)] * _hiMatrixKN[_sparse.GetIndexColumnMajor(k, j, K)];
                    }

                    cpuResult += beta * _hiMatrixMN[_sparse.GetIndexColumnMajor(i, j, M)];

                    Assert.AreEqual(cpuResult, _hoMatrixMN[_sparse.GetIndexColumnMajor(i, j, M)]);
                }
            }
        }


        public void TestSetUp()
        {

        }

        public void TestTearDown()
        {

        }
    }
}
