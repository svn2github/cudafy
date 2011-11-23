/* Added by Kichang Kim (kkc0923@hotmail.com) */
namespace Cudafy.Maths.SPARSE
{
    using GASS.CUDA;
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

        [DllImport(CUSPARSE_DLL_NAME)]
        public static extern CUSPARSEStatus cusparseGetVersion(cusparseHandle handle, ref int version);

        [DllImport(CUSPARSE_DLL_NAME)]
        public static extern CUSPARSEStatus cusparseSetKernelStream(cusparseHandle handle, cudaStream streamId);

        [DllImport(CUSPARSE_DLL_NAME)]
        public static extern CUSPARSEStatus cusparseCreateMatDescr(ref cusparseMatDescr descrA);

        [DllImport(CUSPARSE_DLL_NAME)]
        public static extern CUSPARSEStatus cusparseDestroyMatDescr(cusparseMatDescr descrA);

        [DllImport(CUSPARSE_DLL_NAME)]
        public static extern CUSPARSEStatus cusparseSetMatType(cusparseMatDescr descrA, cusparseMatrixType type);

        [DllImport(CUSPARSE_DLL_NAME)]
        public static extern cusparseMatrixType cusparseGetMatType(cusparseMatDescr descrA);

        [DllImport(CUSPARSE_DLL_NAME)]
        public static extern CUSPARSEStatus cusparseSetMatFillMode(cusparseMatDescr descrA, cusparseFillMode fillMode);

        [DllImport(CUSPARSE_DLL_NAME)]
        public static extern cusparseFillMode cusparseGetMatFillMode(cusparseMatDescr descrA);

        [DllImport(CUSPARSE_DLL_NAME)]
        public static extern CUSPARSEStatus cusparseSetMatDiagType(cusparseMatDescr descrA, cusparseDiagType diagType);

        [DllImport(CUSPARSE_DLL_NAME)]
        public static extern cusparseDiagType cusparseGetMatDiagType(cusparseMatDescr descrA);

        [DllImport(CUSPARSE_DLL_NAME)]
        public static extern CUSPARSEStatus cusparseSetMatIndexBase(cusparseMatDescr descrA, cusparseIndexBase ibase);

        [DllImport(CUSPARSE_DLL_NAME)]
        public static extern cusparseIndexBase cusparseGetMatIndexBase(cusparseMatDescr descrA);

        [DllImport(CUSPARSE_DLL_NAME)]
        public static extern CUSPARSEStatus cusparseCreateSolveAnalysisInfo(ref cusparseSolveAnalysisInfo info);

        [DllImport(CUSPARSE_DLL_NAME)]
        public static extern CUSPARSEStatus cusparseDestroySolveAnalysisInfo(cusparseSolveAnalysisInfo info);
        #endregion

        #region Native Functions : SPARSE Level 1
        #region AXPY
        [DllImport(CUSPARSE_DLL_NAME)]
        public static extern CUSPARSEStatus cusparseSaxpyi(cusparseHandle handle, int nnz, float alpha, IntPtr xVal, IntPtr xInd, IntPtr y, cusparseIndexBase idxBase);
        [DllImport(CUSPARSE_DLL_NAME)]
        public static extern CUSPARSEStatus cusparseDaxpyi(cusparseHandle handle, int nnz, double alpha, IntPtr xVal, IntPtr xInd, IntPtr y, cusparseIndexBase idxBase);
        #endregion

        #region DOT
        [DllImport(CUSPARSE_DLL_NAME)]
        public static extern CUSPARSEStatus cusparseSdoti(cusparseHandle handle, int nnz, IntPtr xVal, IntPtr xInd, IntPtr y, ref float resultHost, cusparseIndexBase idxBase);
        [DllImport(CUSPARSE_DLL_NAME)]
        public static extern CUSPARSEStatus cusparseDdoti(cusparseHandle handle, int nnz, IntPtr xVal, IntPtr xInd, IntPtr y, ref double resultHost, cusparseIndexBase idxBase);
        #endregion

        #region GTHR
        [DllImport(CUSPARSE_DLL_NAME)]
        public static extern CUSPARSEStatus cusparseSgthr(cusparseHandle handle, int nnz, IntPtr y, IntPtr xVal, IntPtr xInd, cusparseIndexBase ibase);
        [DllImport(CUSPARSE_DLL_NAME)]
        public static extern CUSPARSEStatus cusparseDgthr(cusparseHandle handle, int nnz, IntPtr y, IntPtr xVal, IntPtr xInd, cusparseIndexBase ibase);
        #endregion

        #region GTHRZ
        [DllImport(CUSPARSE_DLL_NAME)]
        public static extern CUSPARSEStatus cusparseSgthrz(cusparseHandle handle, int nnz, IntPtr y, IntPtr xVal, IntPtr xInd, cusparseIndexBase ibase);
        [DllImport(CUSPARSE_DLL_NAME)]
        public static extern CUSPARSEStatus cusparseDgthrz(cusparseHandle handle, int nnz, IntPtr y, IntPtr xVal, IntPtr xInd, cusparseIndexBase ibase);
        #endregion

        #region ROT
        [DllImport(CUSPARSE_DLL_NAME)]
        public static extern CUSPARSEStatus cusparseSroti(cusparseHandle handle, int nnz, IntPtr xVal, IntPtr xInd, IntPtr y, float c, float s, cusparseIndexBase idxBase);
        [DllImport(CUSPARSE_DLL_NAME)]
        public static extern CUSPARSEStatus cusparseDroti(cusparseHandle handle, int nnz, IntPtr xVal, IntPtr xInd, IntPtr y, double c, double s, cusparseIndexBase idxBase);
        #endregion

        #region SCTR
        [DllImport(CUSPARSE_DLL_NAME)]
        public static extern CUSPARSEStatus cusparseSsctr(cusparseHandle handle, int nnz, IntPtr xVal, IntPtr xInd, IntPtr y, cusparseIndexBase ibase);
        [DllImport(CUSPARSE_DLL_NAME)]
        public static extern CUSPARSEStatus cusparseDsctr(cusparseHandle handle, int nnz, IntPtr xVal, IntPtr xInd, IntPtr y, cusparseIndexBase ibase);
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
            return cusparseGetVersion(handle, ref version);
        }

        public CUSPARSEStatus CusparseSetKernelStream(cusparseHandle handle, GASS.CUDA.cudaStream streamId)
        {
            return cusparseSetKernelStream(handle, streamId);
        }

        public CUSPARSEStatus CusparseCreateMatDescr(ref cusparseMatDescr descrA)
        {
            return cusparseCreateMatDescr(ref descrA);
        }

        public CUSPARSEStatus CusparseDestroyMatDescr(cusparseMatDescr descrA)
        {
            return cusparseDestroyMatDescr(descrA);
        }

        public CUSPARSEStatus CusparseSetMatType(cusparseMatDescr descrA, cusparseMatrixType type)
        {
            return cusparseSetMatType(descrA, type);
        }

        public cusparseMatrixType CusparseGetMatType(cusparseMatDescr descrA)
        {
            return cusparseGetMatType(descrA);
        }

        public CUSPARSEStatus CusparseSetMatFillMode(cusparseMatDescr descrA, cusparseFillMode fillMode)
        {
            return cusparseSetMatFillMode(descrA, fillMode);
        }

        public cusparseFillMode CusparseGetMatFillMode(cusparseMatDescr descrA)
        {
            return cusparseGetMatFillMode(descrA);
        }

        public CUSPARSEStatus CusparseSetMatDiagType(cusparseMatDescr descrA, cusparseDiagType diagType)
        {
            return cusparseSetMatDiagType(descrA, diagType);
        }

        public cusparseDiagType CusparseGetMatDiagType(cusparseMatDescr descrA)
        {
            return cusparseGetMatDiagType(descrA);
        }

        public CUSPARSEStatus CusparseSetMatIndexBase(cusparseMatDescr descrA, cusparseIndexBase ibase)
        {
            return cusparseSetMatIndexBase(descrA, ibase);
        }

        public cusparseIndexBase CusparseGetMatIndexBase(cusparseMatDescr descrA)
        {
            return cusparseGetMatIndexBase(descrA);
        }

        public CUSPARSEStatus CusparseCreateSolveAnalysisInfo(ref cusparseSolveAnalysisInfo info)
        {
            return cusparseCreateSolveAnalysisInfo(ref info);
        }

        public CUSPARSEStatus CusparseDestroySolveAnalysisInfo(cusparseSolveAnalysisInfo info)
        {
            return cusparseDestroySolveAnalysisInfo(info);
        }

        #endregion

        #region Wrapper Functions : SPARSE Level 1
        #region AXPY
        public CUSPARSEStatus CusparseSaxpyi(cusparseHandle handle, int nnz, float alpha, IntPtr xVal, IntPtr xInd, IntPtr y, cusparseIndexBase idxBase)
        {
            return cusparseSaxpyi(handle, nnz, alpha, xVal, xInd, y, idxBase);
        }
        public CUSPARSEStatus CusparseDaxpyi(cusparseHandle handle, int nnz, double alpha, IntPtr xVal, IntPtr xInd, IntPtr y, cusparseIndexBase idxBase)
        {
            return cusparseDaxpyi(handle, nnz, alpha, xVal, xInd, y, idxBase);
        }
        #endregion

        #region DOT
        public CUSPARSEStatus CusparseSdoti(cusparseHandle handle, int nnz, IntPtr xVal, IntPtr xInd, IntPtr y, ref float resultHost, cusparseIndexBase idxBase)
        {
            return cusparseSdoti(handle, nnz, xVal, xInd, y, ref resultHost, idxBase);
        }
        public CUSPARSEStatus CusparseDdoti(cusparseHandle handle, int nnz, IntPtr xVal, IntPtr xInd, IntPtr y, ref double resultHost, cusparseIndexBase idxBase)
        {
            return cusparseDdoti(handle, nnz, xVal, xInd, y, ref resultHost, idxBase);
        }
        #endregion

        #region GTHR
        public CUSPARSEStatus CusparseSgthr(cusparseHandle handle, int nnz, IntPtr y, IntPtr xVal, IntPtr xInd, cusparseIndexBase ibase)
        {
            return cusparseSgthr(handle, nnz, y, xVal, xInd, ibase);
        }
        public CUSPARSEStatus CusparseDgthr(cusparseHandle handle, int nnz, IntPtr y, IntPtr xVal, IntPtr xInd, cusparseIndexBase ibase)
        {
            return cusparseDgthr(handle, nnz, y, xVal, xInd, ibase);
        }
        #endregion

        #region GTHRZ
        public CUSPARSEStatus CusparseSgthrz(cusparseHandle handle, int nnz, IntPtr y, IntPtr xVal, IntPtr xInd, cusparseIndexBase ibase)
        {
            return cusparseSgthrz(handle, nnz, y, xVal, xInd, ibase);
        }
        public CUSPARSEStatus CusparseDgthrz(cusparseHandle handle, int nnz, IntPtr y, IntPtr xVal, IntPtr xInd, cusparseIndexBase ibase)
        {
            return cusparseDgthrz(handle, nnz, y, xVal, xInd, ibase);
        }
        #endregion

        #region ROT
        public CUSPARSEStatus CusparseSroti(cusparseHandle handle, int nnz, IntPtr xVal, IntPtr xInd, IntPtr y, float c, float s, cusparseIndexBase idxBase)
        {
            return cusparseSroti(handle, nnz, xVal, xInd, y, c, s, idxBase);
        }
        public CUSPARSEStatus CusparseDroti(cusparseHandle handle, int nnz, IntPtr xVal, IntPtr xInd, IntPtr y, double c, double s, cusparseIndexBase idxBase)
        {
            return cusparseDroti(handle, nnz, xVal, xInd, y, c, s, idxBase);
        }
        #endregion

        #region SCTR
        public CUSPARSEStatus CusparseSsctr(cusparseHandle handle, int nnz, IntPtr xVal, IntPtr xInd, IntPtr y, cusparseIndexBase ibase)
        {
            return cusparseSsctr(handle, nnz, xVal, xInd, y, ibase);
        }
        public CUSPARSEStatus CusparseDsctr(cusparseHandle handle, int nnz, IntPtr xVal, IntPtr xInd, IntPtr y, cusparseIndexBase ibase)
        {
            return cusparseDsctr(handle, nnz, xVal, xInd, y, ibase);
        }
        #endregion

        #endregion



    }
}
