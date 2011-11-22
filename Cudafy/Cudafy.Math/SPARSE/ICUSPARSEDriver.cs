/*
 * Added by Kichang Kim
 * kkc0923@hotmail.com
 * */
namespace Cudafy.Maths.SPARSE
{
    using GASS.CUDA;
    using GASS.CUDA.Types;
    using Cudafy.Maths.SPARSE.Types;
    using System;
    using System.Runtime.InteropServices;    

    public interface ICUSPARSEDriver
    {
        string GetDllName();

        #region Helper Functions
        CUSPARSEStatus CusparseCreate(ref cusparseHandle handle);
        CUSPARSEStatus CusparseDestroy(cusparseHandle handle);
        CUSPARSEStatus CusparseGetVersion(cusparseHandle handle, ref int version);
        CUSPARSEStatus CusparseSetKernelStream(cusparseHandle handle, cudaStream streamId);
        CUSPARSEStatus CusparseCreateMatDescr(ref cusparseMatDescr descrA);
        CUSPARSEStatus CusparseDestroyMatDescr(cusparseMatDescr descrA);
        CUSPARSEStatus CusparseSetMatType(cusparseMatDescr descrA, cusparseMatrixType type);
        cusparseMatrixType CusparseGetMatType(cusparseMatDescr descrA);
        CUSPARSEStatus CusparseSetMatFillMode(cusparseMatDescr descrA, cusparseFillMode fillMode);
        cusparseFillMode CusparseGetMatFillMode(cusparseMatDescr descrA);
        CUSPARSEStatus CusparseSetMatDiagType(cusparseMatDescr descrA, cusparseDiagType diagType);
        cusparseDiagType CusparseGetMatDiagType(cusparseMatDescr descrA);
        CUSPARSEStatus CusparseSetMatIndexBase(cusparseMatDescr descrA, cusparseIndexBase ibase);
        cusparseIndexBase CusparseGetMatIndexBase(cusparseMatDescr descrA);
        CUSPARSEStatus CusparseCreateSolveAnalysisInfo(ref cusparseSolveAnalysisInfo info);
        CUSPARSEStatus CusparseDestroySolveAnalysisInfo(cusparseSolveAnalysisInfo info);
        #endregion

        #region Sparse Level 1 Functions
        
        #region AXPY
        CUSPARSEStatus CusparseDaxpyi(cusparseHandle handle, int nnz, double alpha, IntPtr xVal, IntPtr xInd, IntPtr y, cusparseIndexBase idxBase);
        #endregion

        #endregion
    }
}
