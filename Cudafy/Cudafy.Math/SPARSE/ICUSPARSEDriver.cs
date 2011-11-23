/* Added by Kichang Kim (kkc0923@hotmail.com) */
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
        CUSPARSEStatus CusparseSaxpyi(cusparseHandle handle, int nnz, float alpha, IntPtr xVal, IntPtr xInd, IntPtr y, cusparseIndexBase idxBase);
        CUSPARSEStatus CusparseDaxpyi(cusparseHandle handle, int nnz, double alpha, IntPtr xVal, IntPtr xInd, IntPtr y, cusparseIndexBase idxBase);
        #endregion

        #region DOT
        CUSPARSEStatus CusparseSdoti(cusparseHandle handle, int nnz, IntPtr xVal, IntPtr xInd, IntPtr y, ref float resultHost, cusparseIndexBase idxBase);
        CUSPARSEStatus CusparseDdoti(cusparseHandle handle, int nnz, IntPtr xVal, IntPtr xInd, IntPtr y, ref double resultHost, cusparseIndexBase idxBase);
        #endregion

        #region GTHR
        CUSPARSEStatus CusparseSgthr(cusparseHandle handle, int nnz, IntPtr y, IntPtr xVal, IntPtr xInd, cusparseIndexBase idxBase);
        CUSPARSEStatus CusparseDgthr(cusparseHandle handle, int nnz, IntPtr y, IntPtr xVal, IntPtr xInd, cusparseIndexBase idxBase);
        #endregion

        #region GTHRZ
        CUSPARSEStatus CusparseSgthrz(cusparseHandle handle, int nnz, IntPtr y, IntPtr xVal, IntPtr xInd, cusparseIndexBase idxBase);
        CUSPARSEStatus CusparseDgthrz(cusparseHandle handle, int nnz, IntPtr y, IntPtr xVal, IntPtr xInd, cusparseIndexBase idxBase);
        #endregion

        #region ROT
        CUSPARSEStatus CusparseSroti(cusparseHandle handle, int nnz, IntPtr xVal, IntPtr xInd, IntPtr y, float c, float s, cusparseIndexBase idxBase);
        CUSPARSEStatus CusparseDroti(cusparseHandle handle, int nnz, IntPtr xVal, IntPtr xInd, IntPtr y, double c, double s, cusparseIndexBase idxBase);
        #endregion

        #region SCTR
        CUSPARSEStatus CusparseSsctr(cusparseHandle handle, int nnz, IntPtr xVal, IntPtr xInd, IntPtr y, cusparseIndexBase idxBase);
        CUSPARSEStatus CusparseDsctr(cusparseHandle handle, int nnz, IntPtr xVal, IntPtr xInd, IntPtr y, cusparseIndexBase idxBase);
        #endregion
        #endregion
    }
}
