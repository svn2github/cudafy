/*
Linear system solver. Conjugate Gradient, 
Working ..., not completed.

I referred NVIDIA conjugate gradient solver sample code.
*/

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Diagnostics;
using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;
using Cudafy.Maths.BLAS;
using Cudafy.Maths.SPARSE;
using Cudafy.Maths.SPARSE.Types;

namespace Cudafy.Maths.LA
{
    public class Solver
    {
        GPGPU gpu;
        GPGPUBLAS blas;
        GPGPUSPARSE sparse;

        public Solver(GPGPU gpu, GPGPUBLAS blas, GPGPUSPARSE sparse)
        {
            this.gpu = gpu;
            this.blas = blas;
            this.sparse = sparse;

            var km = CudafyModule.TryDeserialize();
            if (km == null || !km.TryVerifyChecksums())
            {
                km = CudafyTranslator.Cudafy();
                km.TrySerialize();
            }

            gpu.LoadModule(km);
        }

        [Cudafy]
        public static void DefineLower(GThread thread, int n, int[] rowsICP, int[] colsICP)
        {
            rowsICP[0] = 0;
            colsICP[0] = 0;

            int inz = 1;

            for (int k = 1; k < n; k++)
            {
                rowsICP[k] = inz;
                for (int j = k - 1; j <= k; j++)
                {
                    colsICP[inz] = j;
                    inz++;
                }
            }

            rowsICP[n] = inz;
        }

        [Cudafy]
        public static void CopyAIntoH(GThread thread, int n, float[] vals, int[] rows, float[] valsICP, int[] rowsICP)
        {
            int tid = thread.blockIdx.x;

            if (tid == 0)
            {
                valsICP[0] = vals[0];
            }
            else if (tid < n)
            {
                valsICP[rowsICP[tid]] = vals[rows[tid]];
                valsICP[rowsICP[tid] + 1] = vals[rows[tid] + 1];
            }
        }

        [Cudafy]
        public static void ConstructH(GThread thread, int n, float[] valsICP, int[] rowsICP)
        {
            int tid = thread.blockIdx.x;

            if (tid < n)
            {
                valsICP[rowsICP[tid + 1] - 1] = (float)Math.Sqrt(valsICP[rowsICP[tid + 1] - 1]);

                if (tid < n - 1)
                {
                    valsICP[rowsICP[tid + 1]] /= valsICP[rowsICP[tid + 1] - 1];
                    valsICP[rowsICP[tid + 1] + 1] -= valsICP[rowsICP[tid + 1]] * valsICP[rowsICP[tid + 1]];
                }
            }
        }

        /// <summary>
        /// Solves symmetric linear system with conjugate gradient solver.
        /// A * x = b
        /// </summary>
        /// <param name="n">number of rows and columns of matrix A.</param>
        /// <param name="csrValA">array of nnz elements, where nnz is the number of non-zero elements and can be obtained from csrRowA[m] - csrRowA[0].</param>
        /// <param name="csrRowA">array of m+1 index elements.</param>
        /// <param name="csrColA">array of nnz column indices.</param>
        /// <param name="dx">vector of n elements.</param>
        /// <param name="db">vector of n elements.</param>
        /// <param name="dp">vector of n elements. (temporary vector)</param>
        /// <param name="dAx">vector of n elements. (temporary vector)</param>
        /// <param name="tolerence">iterate tolerence of conjugate gradient solver.</param>
        /// <param name="maxIterate">max iterate count of conjugate gradient solver.</param>
        /// <returns>if A has singulrarity or failure in max iterate count, returns false. return true otherwise.</returns>
        public SolveResult CG(
            int n, float[] csrValA, int[] csrRowA, int[] csrColA,
            float[] dx, float[] db, float[] dp, float[] dAx, float tolerence = 0.00001f, int maxIterate = 300)
        {
            SolveResult result = new SolveResult();
            int k; // Iterate count.
            float a, b, r0, r1;

            sparse.CSRMV(n, n, 1.0f, csrValA, csrRowA, csrColA, dx, 0.0f, dAx);
            blas.AXPY(-1.0f, dAx, db);

            r1 = blas.DOT(db, db);

            k = 1;
            r0 = 0;

            while (true)
            {
                if (k > 1)
                {
                    b = r1 / r0;
                    blas.SCAL(b, dp);
                    blas.AXPY(1.0f, db, dp);
                }
                else
                {
                    blas.COPY(db, dp);
                }

                sparse.CSRMV(n, n, 1.0f, csrValA, csrRowA, csrColA, dp, 0.0f, dAx);
                a = r1 / blas.DOT(dp, dAx);
                blas.AXPY(a, dp, dx);
                blas.AXPY(-a, dAx, db);

                r0 = r1;
                r1 = blas.DOT(db, db);

                k++;

                if (r1 <= tolerence * tolerence)
                {
                    result.IsSuccess = true;
                    result.LastError = r1;
                    result.IterateCount = k;
                    break;
                }

                if (k > maxIterate)
                {
                    result.IsSuccess = false;
                    result.LastError = r1;
                    result.IterateCount = k;
                    break;
                }
            }

            return result;
        }

        /// <summary>
        /// Now working ...
        /// </summary>
        /// <param name="n"></param>
        /// <param name="csrValA"></param>
        /// <param name="csrRowA"></param>
        /// <param name="csrColA"></param>
        /// <param name="dx"></param>
        /// <param name="db"></param>
        /// <param name="csrValICP"></param>
        /// <param name="csrRowICP"></param>
        /// <param name="csrColICP"></param>
        /// <param name="dy"></param>
        /// <param name="dp"></param>
        /// <param name="domega"></param>
        /// <param name="zm1"></param>
        /// <param name="zm2"></param>
        /// <param name="rm2"></param>
        /// <param name="tolerence"></param>
        /// <param name="maxIterate"></param>
        /// <returns></returns>
        public SolveResult CGPreconditioned(
            int n, float[] csrValA, int[] csrRowA, int[] csrColA, float[] dx, float[] db, 
            float[] csrValICP, int[] csrRowICP, int[] csrColICP, 
            float[] dy,float[] dp, float[] domega, float[] zm1, float[] zm2, float[] rm2, float tolerence = 0.0001f, int maxIterate = 300)
        {
            SolveResult result = new SolveResult();

            // Make Incomplete Cholesky Preconditioner.
            gpu.Launch().DefineLower(n, csrRowICP, csrColICP);
            gpu.Launch(n, 1).CopyAIntoH(n, csrValA, csrRowA, csrValICP, csrRowICP);
            gpu.Launch(n, 1).ConstructH(n, csrValICP, csrRowICP);

            cusparseMatDescr descrM = new cusparseMatDescr();
            descrM.MatrixType = cusparseMatrixType.Triangular;
            descrM.FillMode = cusparseFillMode.Lower;
            descrM.IndexBase = cusparseIndexBase.Zero;
            descrM.DiagType = cusparseDiagType.NonUnit;

            cusparseSolveAnalysisInfo info = new cusparseSolveAnalysisInfo();
            sparse.CreateSolveAnalysisInfo(ref info);
            cusparseSolveAnalysisInfo infoTrans = new cusparseSolveAnalysisInfo();
            sparse.CreateSolveAnalysisInfo(ref infoTrans);

            sparse.CSRSV_ANALYSIS(n, csrValICP, csrRowICP, csrColICP, cusparseOperation.NonTranspose, info, descrM);
            sparse.CSRSV_ANALYSIS(n, csrValICP, csrRowICP, csrColICP, cusparseOperation.Transpose, infoTrans, descrM);

            int k = 0;
            float r1 = blas.DOT(db, db);
            float alpha, beta;

            while (true)
            {
                sparse.CSRSV_SOLVE(n, 1.0f, csrValICP, csrRowICP, csrColICP, db, dy, cusparseOperation.NonTranspose, info, descrM);
                sparse.CSRSV_SOLVE(n, 1.0f, csrValICP, csrRowICP, csrColICP, dy, zm1, cusparseOperation.Transpose, infoTrans, descrM);

                k++;

                if (k == 1)
                {
                    blas.COPY(zm1, dp);
                }
                else
                {
                    beta = blas.DOT(db, zm1) / blas.DOT(rm2, zm2);
                    blas.SCAL(beta, dp);
                    blas.AXPY(1.0f, zm1, dp);
                }

                sparse.CSRMV(n, n, 1.0f, csrValA, csrRowA, csrColA, dp, 0.0f, domega);
                alpha = blas.DOT(db, zm1) / blas.DOT(dp, domega);

                blas.AXPY(alpha, dp, dx);
                blas.COPY(db, rm2);
                blas.COPY(zm1, zm2);
                blas.AXPY(-alpha, domega, db);

                r1 = blas.DOT(db, db);

                if (r1 <= tolerence * tolerence)
                {
                    result.IsSuccess = true;
                    result.IterateCount = k;
                    result.LastError = r1;
                    break;
                }
                if (k > maxIterate)
                {
                    result.IsSuccess = false;
                    result.IterateCount = k;
                    result.LastError = r1;
                    break;
                }
            }

            return result;
        }


    }
}
