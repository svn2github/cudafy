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
using Cudafy.Maths.SPARSE;
using Cudafy.Maths.SPARSE.Types;
using Cudafy.Maths.LA;
using Cudafy.UnitTests;
using NUnit.Framework;

namespace Cudafy.Maths.UnitTests
{
    [TestFixture]
    public class LASOLVER : ICudafyUnitTest
    {
        GPGPU _gpu;
        GPGPUSPARSE _sparse;
        GPGPUBLAS _blas;
        Solver _solver;

        float[] _hiMatrixMN;
        float[] _hiVectorN;
        float[] _hiVectorN2;

        float[] _hoVectorN;

        float[] _diMatrixMN;
        float[] _diVectorN;
        float[] _diVectorN2;

        float[] _diVectorP;
        float[] _diVectorAX;
        int[] _diPerRow;
        float[] _diCSRVals;
        int[] _diCSRRows;
        int[] _diCSRCols;

        int[] _hoPerRow;
        float[] _hoCSRVals;
        int[] _hoCSRRows;
        int[] _hoCSRCols;

        int N = 8000;

        [TestFixtureSetUp]
        public void SetUp()
        {
            _gpu = CudafyHost.CreateDevice(CudafyModes.Target);
            _sparse = GPGPUSPARSE.Create(_gpu);
            _blas = GPGPUBLAS.Create(_gpu);
            _solver = new Solver(_gpu, _blas, _sparse);
        }

        [TestFixtureTearDown]
        public void TearDown()
        {
            _blas.Dispose();
            _sparse.Dispose();
        }

        private void CreateDiagonalMatrix(float[] buffer, int n)
        {
            Random rand = new Random(Environment.TickCount);

            for (int i = 0; i < n; i++)
            {
                buffer[_blas.GetIndexColumnMajor(i, i, n)] = rand.Next(256);

                if (buffer[_blas.GetIndexColumnMajor(i, i, n)] == 0)
                {
                    buffer[_blas.GetIndexColumnMajor(i, i, n)] = 100;
                }
            }

            System.Threading.Thread.Sleep(rand.Next(50));
        }

        private void CreateDiagonalMatrix(float[] buffer, int n, float value)
        {
            for (int i = 0; i < n; i++)
            {
                buffer[_blas.GetIndexColumnMajor(i, i, n)] = value;
            }

        }

        private void FillBuffer(float[] buffer)
        {
            Random rand = new Random(Environment.TickCount);

            for (int i = 0; i < buffer.Length; i++)
            {
                buffer[i] = rand.Next(512);
            }

            System.Threading.Thread.Sleep(rand.Next(50));
        }

        private void FillBuffer(float[] buffer, float value)
        {
            for (int i = 0; i < buffer.Length; i++)
            {
                buffer[i] = value;
            }
        }

        private void Randomize(float[] buffer, int n)
        {
            Random rand = new Random(Environment.TickCount);

            for (int j = 0; j < n; j++)
            {
                for (int i = 0; i < j; i++)
                {
                    if (rand.Next(100) < 1)
                    {
                        float val = rand.Next(512);
                        buffer[_blas.GetIndexColumnMajor(i, j, n)] = val;
                        buffer[_blas.GetIndexColumnMajor(j, i, n)] = val;
                    }
                }
            }

            System.Threading.Thread.Sleep(rand.Next(50));
        }

        private void Print(float[] array)
        {
            Random rand = new Random(Environment.TickCount);

            for(int i = 0; i < array.Length; i++)
            {
                Console.Write("{0} ", array[i]);
            }
            Console.Write(Environment.NewLine);
        }

        public void TestSetUp()
        {
        }

        public void TestTearDown()
        {
        }

        [Test]
        public void TestCSRSV()
        {
            Stopwatch sw = new Stopwatch();

            _hiMatrixMN = new float[N * N];
            _hoVectorN = new float[N];
            CreateDiagonalMatrix(_hiMatrixMN, N, 6);

            //Randomize(_hiMatrixMN, N);

            _hiVectorN = new float[N];
            _hiVectorN2 = new float[N];
            FillBuffer(_hiVectorN2, 6);

            _diMatrixMN = _gpu.CopyToDevice(_hiMatrixMN);
            _diVectorN = _gpu.Allocate(_hiVectorN);
            _diVectorN2 = _gpu.CopyToDevice(_hiVectorN2);

            _diPerRow = _gpu.Allocate<int>(N);
            _diVectorP = _gpu.Allocate<float>(N);
            _diVectorAX = _gpu.Allocate<float>(N);

            int nnz = _sparse.NNZ(N, N, _diMatrixMN, _diPerRow);

            _diCSRVals = _gpu.Allocate<float>(nnz);
            _diCSRCols = _gpu.Allocate<int>(nnz);
            _diCSRRows = _gpu.Allocate<int>(N + 1);

            _sparse.Dense2CSR(N, N, _diMatrixMN, _diPerRow, _diCSRVals, _diCSRRows, _diCSRCols);

            cusparseMatDescr descr = cusparseMatDescr.DefaultTriangular();

            cusparseSolveAnalysisInfo info = new cusparseSolveAnalysisInfo();
            _sparse.CreateSolveAnalysisInfo(ref info);
            _sparse.CSRSV_ANALYSIS(N, _diCSRVals, _diCSRRows, _diCSRCols, cusparseOperation.Transpose, info, descr);
            _sparse.CSRSV_SOLVE(N, 1.0f, _diCSRVals, _diCSRRows, _diCSRCols, _diVectorN2, _diVectorN, cusparseOperation.Transpose, info, descr);

            _sparse.CSRMV(N, N, 1.0f, _diCSRVals, _diCSRRows, _diCSRCols, _diVectorN, 0.0f, _diVectorN2);

            _gpu.CopyFromDevice(_diVectorN2, _hoVectorN);

            float maxError = 0.0f;

            for (int i = 0; i < N; i++)
            {
                float error = Math.Abs(_hoVectorN[i] - _hiVectorN2[i]);

                if (error > maxError)
                {
                    maxError = error;
                }
            }

            Console.WriteLine("max error : {0}", maxError);

            _gpu.FreeAll();
        }

        [Test]
        public void TestCGSolver()
        {
            Stopwatch sw = new Stopwatch();

            _hiMatrixMN = new float[N * N];
            _hoVectorN = new float[N];
            CreateDiagonalMatrix(_hiMatrixMN, N, 6);

            _hiVectorN = new float[N];
            _hiVectorN2 = new float[N];
            FillBuffer(_hiVectorN2, 6);

            _diMatrixMN = _gpu.CopyToDevice(_hiMatrixMN);
            _diVectorN = _gpu.Allocate(_hiVectorN);
            _diVectorN2 = _gpu.CopyToDevice(_hiVectorN2);

            _diPerRow = _gpu.Allocate<int>(N);
            _diVectorP = _gpu.Allocate<float>(N);
            _diVectorAX = _gpu.Allocate<float>(N);

            int nnz = _sparse.NNZ(N, N, _diMatrixMN, _diPerRow);

            _diCSRVals = _gpu.Allocate<float>(nnz);
            _diCSRCols = _gpu.Allocate<int>(nnz);
            _diCSRRows = _gpu.Allocate<int>(N + 1);

            _sparse.Dense2CSR(N, N, _diMatrixMN, _diPerRow, _diCSRVals, _diCSRRows, _diCSRCols);

            sw.Start();
            SolveResult result = _solver.CG(N, _diCSRVals, _diCSRRows, _diCSRCols, _diVectorN, _diVectorN2, _diVectorP, _diVectorAX, 0.01f, 1000);
            long time = sw.ElapsedMilliseconds;

            _sparse.CSRMV(N, N, 1.0f, _diCSRVals, _diCSRRows, _diCSRCols, _diVectorN, 0.0f, _diVectorN2);

            _gpu.CopyFromDevice(_diVectorN2, _hoVectorN);

            float maxError = 0.0f;

            for (int i = 0; i < N; i++)
            {
                float error = Math.Abs(_hoVectorN[i] - _hiVectorN2[i]);

                if (error > maxError)
                {
                    maxError = error;
                }
            }

            Console.WriteLine("Time : {0} ms", time);
            Console.WriteLine("Iterate Count : {0}", result.IterateCount);
            Console.WriteLine("Residual : {0}", result.LastError);
            Console.WriteLine("max error : {0}", maxError);

            _gpu.FreeAll();
        }

        [Test]
        public void TestBiCGSTABSolver()
        {
            Stopwatch sw = new Stopwatch();

            _hiMatrixMN = new float[N * N];
            _hoVectorN = new float[N];
            CreateDiagonalMatrix(_hiMatrixMN, N, 6);


            _hiVectorN = new float[N];
            _hiVectorN2 = new float[N];
            FillBuffer(_hiVectorN2, 6);

            _diMatrixMN = _gpu.CopyToDevice(_hiMatrixMN);
            _diVectorN = _gpu.Allocate(_hiVectorN);
            _diVectorN2 = _gpu.CopyToDevice(_hiVectorN2);

            _diPerRow = _gpu.Allocate<int>(N);
            _diVectorP = _gpu.Allocate<float>(N);
            _diVectorAX = _gpu.Allocate<float>(N);

            int nnz = _sparse.NNZ(N, N, _diMatrixMN, _diPerRow);

            _diCSRVals = _gpu.Allocate<float>(nnz);
            _diCSRCols = _gpu.Allocate<int>(nnz);
            _diCSRRows = _gpu.Allocate<int>(N + 1);

            // For temporary memory.
            float[] r0 = _gpu.Allocate<float>(N);
            float[] r = _gpu.Allocate<float>(N);
            float[] v = _gpu.Allocate<float>(N);
            float[] s = _gpu.Allocate<float>(N);
            float[] t = _gpu.Allocate<float>(N);

            _sparse.Dense2CSR(N, N, _diMatrixMN, _diPerRow, _diCSRVals, _diCSRRows, _diCSRCols);

            sw.Start();
            SolveResult result = _solver.BiCGSTAB(N, _diCSRVals, _diCSRRows, _diCSRCols, _diVectorN, _diVectorN2, _diVectorAX, r0, r, v, _diVectorP, s, t, 0.00001f, 1000);
            long time = sw.ElapsedMilliseconds;

            _sparse.CSRMV(N, N, 1.0f, _diCSRVals, _diCSRRows, _diCSRCols, _diVectorN, 0.0f, _diVectorN2);

            _gpu.CopyFromDevice(_diVectorN2, _hoVectorN);

            float maxError = 0.0f;

            for (int i = 0; i < N; i++)
            {
                float error = Math.Abs(_hoVectorN[i] - _hiVectorN2[i]);

                if (error > maxError)
                {
                    maxError = error;
                }
            }

            Console.WriteLine("Time : {0} ms", time);
            Console.WriteLine("Iterate Count : {0}", result.IterateCount);
            Console.WriteLine("Residual : {0}", result.LastError);
            Console.WriteLine("max error : {0}", maxError);

            _gpu.FreeAll();
        }

        /* Now working ...
        [Test]
        public void TestCGPreconditionedSolver()
        {
            Stopwatch sw = new Stopwatch();

            _hiMatrixMN = new float[N * N];
            _hoVectorN = new float[N];
            CreateDiagonalMatrix(_hiMatrixMN, N, 6);

            _hiVectorN = new float[N];
            _hiVectorN2 = new float[N];
            FillBuffer(_hiVectorN2, 6);

            _diMatrixMN = _gpu.CopyToDevice(_hiMatrixMN);
            _diVectorN = _gpu.CopyToDevice(_hiVectorN);
            _diVectorN2 = _gpu.CopyToDevice(_hiVectorN2);

            _diPerRow = _gpu.Allocate<int>(N);
            _diVectorP = _gpu.Allocate<float>(N);
            float[] _diY = _gpu.Allocate<float>(N);
            float[] _diOmega = _gpu.Allocate<float>(N);
            float[] _diZM1 = _gpu.Allocate<float>(N);
            float[] _diZM2 = _gpu.Allocate<float>(N);
            float[] _diRM2 = _gpu.Allocate<float>(N);

            int nnz = _sparse.NNZ(N, N, _diMatrixMN, _diPerRow);

            _diCSRVals = _gpu.Allocate<float>(nnz);
            _diCSRCols = _gpu.Allocate<int>(nnz);
            _diCSRRows = _gpu.Allocate<int>(N + 1);

            int nzICP = 2 * N - 1;
            float[] _diICPVals = _gpu.Allocate<float>(nzICP);
            int[] _diICPRows = _gpu.Allocate<int>(N + 1);
            int[] _diICPCols = _gpu.Allocate<int>(nzICP);
            

            _sparse.Dense2CSR(N, N, _diMatrixMN, _diPerRow, _diCSRVals, _diCSRRows, _diCSRCols);

            _hoPerRow = new int[N];
            _hoCSRCols = new int[nnz];
            _hoCSRRows = new int[N + 1];
            _hoCSRVals = new float[nnz];

            _gpu.CopyFromDevice(_diPerRow, _hoPerRow);
            _gpu.CopyFromDevice(_diCSRVals, _hoCSRVals);
            _gpu.CopyFromDevice(_diCSRRows, _hoCSRRows);
            _gpu.CopyFromDevice(_diCSRCols, _hoCSRCols);

            sw.Start();
            SolveResult result = _solver.CGPreconditioned(N, _diCSRVals, _diCSRRows, _diCSRCols, _diVectorN, _diVectorN2, _diICPVals, _diICPRows, _diICPCols, _diY, _diVectorP, _diOmega, _diZM1, _diZM2, _diRM2,
                0.01f, 1000);
            long time = sw.ElapsedMilliseconds;

            _sparse.CSRMV(N, N, 1.0f, _diCSRVals, _diCSRRows, _diCSRCols, _diVectorN, 0.0f, _diVectorN2);

            _gpu.CopyFromDevice(_diVectorN2, _hoVectorN);

            float maxError = 0.0f;

            for (int i = 0; i < N; i++)
            {
                float error = Math.Abs(_hoVectorN[i] - _hiVectorN2[i]);

                if (error > maxError)
                {
                    maxError = error;
                }
            }

            Console.WriteLine("Time : {0} ms", time);
            Console.WriteLine("Iterate Count : {0}", result.IterateCount);
            Console.WriteLine("Residual : {0}", result.LastError);
            Console.WriteLine("max error : {0}", maxError);

            _gpu.FreeAll();
        }
         * */
    }
}
