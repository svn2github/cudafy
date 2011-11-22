/*
 * Added by Kichang Kim
 * kkc0923@hotmail.com
 * */
namespace Cudafy.Maths.SPARSE
{
    using GASS.CUDA.Types;
    using Cudafy.Maths.SPARSE.Types;
    using System;
    using System.Runtime.InteropServices;

    public class CUSPARSEDriver64 : ICUSPARSEDriver
    {
        internal const string CUSPARSE_DLL_NAME = "cusparse64_40_17";

        [DllImport("kernel32.dll", CharSet = CharSet.Auto, SetLastError = true)]
        public static extern IntPtr LoadLibrary(string lpFileName);

        #region Native Functions : Help Functions
        [DllImport(CUSPARSE_DLL_NAME)]
        public static extern CUSPARSEStatus cusparseCreate(ref cusparseHandle handle);

        [DllImport(CUSPARSE_DLL_NAME)]
        public static extern CUSPARSEStatus cusparseDestroy(cusparseHandle handle);
        #endregion

        #region Native Functions : SPARSE Level 1
        #region AXPY
        [DllImport(CUSPARSE_DLL_NAME)]
        public static extern CUSPARSEStatus cusparseDaxpyi(cusparseHandle handle, int nnz, double alpha, IntPtr xVal, IntPtr xInd, IntPtr y, cusparseIndexBase idxBase);
        #endregion
        #endregion

        public string GetDllName()
        {
            return CUSPARSE_DLL_NAME;
        }

        #region Wrapper Functions : HelperFunctions
        public CUSPARSEStatus CusparseCreate(ref cusparseHandle handle)
        {
            return cusparseCreate(ref handle);
        }

        public CUSPARSEStatus CusparseDestroy(cusparseHandle handle)
        {
            return cusparseDestroy(handle);
        }

        public CUSPARSEStatus CusparseGetVersion(cusparseHandle handle, ref int version)
        {
            throw new NotImplementedException();
        }

        public CUSPARSEStatus CusparseSetKernelStream(cusparseHandle handle, GASS.CUDA.cudaStream streamId)
        {
            throw new NotImplementedException();
        }

        public CUSPARSEStatus CusparseCreateMatDescr(ref cusparseMatDescr descrA)
        {
            throw new NotImplementedException();
        }

        public CUSPARSEStatus CusparseDestroyMatDescr(cusparseMatDescr descrA)
        {
            throw new NotImplementedException();
        }

        public CUSPARSEStatus CusparseSetMatType(cusparseMatDescr descrA, cusparseMatrixType type)
        {
            throw new NotImplementedException();
        }

        public cusparseMatrixType CusparseGetMatType(cusparseMatDescr descrA)
        {
            throw new NotImplementedException();
        }

        public CUSPARSEStatus CusparseSetMatFillMode(cusparseMatDescr descrA, cusparseFillMode fillMode)
        {
            throw new NotImplementedException();
        }

        public cusparseFillMode CusparseGetMatFillMode(cusparseMatDescr descrA)
        {
            throw new NotImplementedException();
        }

        public CUSPARSEStatus CusparseSetMatDiagType(cusparseMatDescr descrA, cusparseDiagType diagType)
        {
            throw new NotImplementedException();
        }

        public cusparseDiagType CusparseGetMatDiagType(cusparseMatDescr descrA)
        {
            throw new NotImplementedException();
        }

        public CUSPARSEStatus CusparseSetMatIndexBase(cusparseMatDescr descrA, cusparseIndexBase ibase)
        {
            throw new NotImplementedException();
        }

        public cusparseIndexBase CusparseGetMatIndexBase(cusparseMatDescr descrA)
        {
            throw new NotImplementedException();
        }

        public CUSPARSEStatus CusparseCreateSolveAnalysisInfo(ref cusparseSolveAnalysisInfo info)
        {
            throw new NotImplementedException();
        }

        public CUSPARSEStatus CusparseDestroySolveAnalysisInfo(cusparseSolveAnalysisInfo info)
        {
            throw new NotImplementedException();
        }

        #endregion

        #region Wrapper Functions : SPARSE Level 1
        #region AXPY
        public CUSPARSEStatus CusparseDaxpyi(cusparseHandle handle, int nnz, double alpha, IntPtr xVal, IntPtr xInd, IntPtr y, cusparseIndexBase idxBase)
        {
            return cusparseDaxpyi(handle, nnz, alpha, xVal, xInd, y, idxBase);
        }
        #endregion
        #endregion
    }
}
