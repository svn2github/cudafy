/* Added by Kichang Kim (kkc0923@hotmail.com) */
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Runtime.InteropServices;
using System.Diagnostics;
using Cudafy.Types;
using Cudafy.Host;
using Cudafy.Maths.SPARSE.Types;

using GASS.CUDA.Types;
using GASS.CUDA;

namespace Cudafy.Maths.SPARSE
{
    internal class CudaSPARSE : GPGPUSPARSE
    {
        private GPGPU _gpu;
        private cusparseHandle _sparse;
        private ICUSPARSEDriver _driver;
        private CUSPARSEStatus _status;

        private CUSPARSEStatus LastStatus
        {
            get { return _status; }
            set
            {
                _status = value;
                if (_status != CUSPARSEStatus.Success)
                    throw new CudafyMathException("SPARSE Error : {0}", _status.ToString());
            }
        }

        internal CudaSPARSE(GPGPU gpu)
            : base()
        {
            if (IntPtr.Size == 8)
            {
                _driver = new CUSPARSEDriver64();
            }
            else
            {
                throw new NotImplementedException("64bit is only supported now.");
            }

            LastStatus = _driver.CusparseCreate(ref _sparse);
            _gpu = gpu;
        }

        protected override void Shutdown()
        {
            try
            {
                LastStatus = _driver.CusparseDestroy(_sparse);
            }
            catch(DllNotFoundException ex)
            {
                Debug.WriteLine(ex.Message);
            }
        }

        public override string GetVersionInfo()
        {
            int version = 0;
            _driver.CusparseGetVersion(_sparse, ref version);
            return string.Format("CUDA Version : {0}", version);
        }

        private static eDataType GetDataType<T>()
        {
            eDataType type;
            Type t = typeof(T);
            if (t == typeof(Double))
                type = eDataType.D;
            else if (t == typeof(Single))
                type = eDataType.S;
            else if (t == typeof(ComplexD))
                type = eDataType.Z;
            else if (t == typeof(ComplexF))
                type = eDataType.C;
            else
                throw new CudafyMathException(CudafyHostException.csX_NOT_SUPPORTED, typeof(T).Name);
            return type;
        }

        private CUdeviceptr GetDeviceMemory(object vector, ref int n)
        {
            CUDevicePtrEx ptrEx = _gpu.GetDeviceMemory(vector) as CUDevicePtrEx;
            n = (n == 0 ? ptrEx.TotalSize : n);

            return ptrEx.DevPtr;
        }

        private CUdeviceptr GetDeviceMemory(object vector)
        {
            CUDevicePtrEx ptrEx = _gpu.GetDeviceMemory(vector) as CUDevicePtrEx;
            return ptrEx.DevPtr;
        }

        #region SPARSE Level 1

        #region AXPY
        public override void AXPY(float alpha, float[] vectorx, int[] indexx, float[] vectory, int n = 0, cusparseIndexBase ibase = cusparseIndexBase.Zero)
        {
            CUdeviceptr ptrx = GetDeviceMemory(vectorx, ref n);
            CUdeviceptr ptry = GetDeviceMemory(vectory);
            CUdeviceptr ptrix = GetDeviceMemory(indexx);

            LastStatus = _driver.CusparseSaxpyi(_sparse, n, alpha, ptrx.Pointer, ptrix.Pointer, ptry.Pointer, ibase);
        }
        public override void AXPY(double alpha, double[] vectorx, int[] indexx, double[] vectory, int n = 0, cusparseIndexBase ibase = cusparseIndexBase.Zero)
        {
            CUdeviceptr ptrx = GetDeviceMemory(vectorx, ref n);
            CUdeviceptr ptry = GetDeviceMemory(vectory);
            CUdeviceptr ptrix = GetDeviceMemory(indexx);

            LastStatus = _driver.CusparseDaxpyi(_sparse, n, alpha, ptrx.Pointer, ptrix.Pointer, ptry.Pointer, ibase);
        }
        #endregion

        #region DOT
        public override float DOT(float[] vectorx, int[] indexx, float[] vectory, int n = 0, cusparseIndexBase ibase = cusparseIndexBase.Zero)
        {
            CUdeviceptr ptrx = GetDeviceMemory(vectorx, ref n);
            CUdeviceptr ptry = GetDeviceMemory(vectory);
            CUdeviceptr ptrix = GetDeviceMemory(indexx);

            float result = 0;

            LastStatus = _driver.CusparseSdoti(_sparse, n, ptrx.Pointer, ptrix.Pointer, ptry.Pointer, ref result, ibase);
            return result;
        }
        public override double DOT(double[] vectorx, int[] indexx, double[] vectory, int n = 0, cusparseIndexBase ibase = cusparseIndexBase.Zero)
        {
            CUdeviceptr ptrx = GetDeviceMemory(vectorx, ref n);
            CUdeviceptr ptry = GetDeviceMemory(vectory);
            CUdeviceptr ptrix = GetDeviceMemory(indexx);

            double result = 0;

            LastStatus = _driver.CusparseDdoti(_sparse, n, ptrx.Pointer, ptrix.Pointer, ptry.Pointer, ref result, ibase);
            return result;
        }
        #endregion

        #region GTHR
        public override void GTHR(float[] vectory, float[] vectorx, int[] indexx, int n = 0, cusparseIndexBase ibase = cusparseIndexBase.Zero)
        {
            CUdeviceptr ptrx = GetDeviceMemory(vectorx, ref n);
            CUdeviceptr ptry = GetDeviceMemory(vectory);
            CUdeviceptr ptrix = GetDeviceMemory(indexx);

            LastStatus = _driver.CusparseSgthr(_sparse, n, ptry.Pointer, ptrx.Pointer, ptrix.Pointer, ibase);
        }
        public override void GTHR(double[] vectory, double[] vectorx, int[] indexx, int n = 0, cusparseIndexBase ibase = cusparseIndexBase.Zero)
        {
            CUdeviceptr ptrx = GetDeviceMemory(vectorx, ref n);
            CUdeviceptr ptry = GetDeviceMemory(vectory);
            CUdeviceptr ptrix = GetDeviceMemory(indexx);

            LastStatus = _driver.CusparseDgthr(_sparse, n, ptry.Pointer, ptrx.Pointer, ptrix.Pointer, ibase);
        }
        #endregion

        #region GTHRZ
        public override void GTHRZ(float[] vectory, float[] vectorx, int[] indexx, int n = 0, cusparseIndexBase ibase = cusparseIndexBase.Zero)
        {
            CUdeviceptr ptrx = GetDeviceMemory(vectorx, ref n);
            CUdeviceptr ptry = GetDeviceMemory(vectory);
            CUdeviceptr ptrix = GetDeviceMemory(indexx);

            LastStatus = _driver.CusparseSgthrz(_sparse, n, ptry.Pointer, ptrx.Pointer, ptrix.Pointer, ibase);
        }
        public override void GTHRZ(double[] vectory, double[] vectorx, int[] indexx, int n = 0, cusparseIndexBase ibase = cusparseIndexBase.Zero)
        {
            CUdeviceptr ptrx = GetDeviceMemory(vectorx, ref n);
            CUdeviceptr ptry = GetDeviceMemory(vectory);
            CUdeviceptr ptrix = GetDeviceMemory(indexx);

            LastStatus = _driver.CusparseDgthrz(_sparse, n, ptry.Pointer, ptrx.Pointer, ptrix.Pointer, ibase);
        }
        #endregion

        #region ROT
        public override void ROT(float[] vectorx, int[] indexx, float[] vectory, float c, float s, int n = 0, cusparseIndexBase ibase = cusparseIndexBase.Zero)
        {
            CUdeviceptr ptrx = GetDeviceMemory(vectorx, ref n);
            CUdeviceptr ptry = GetDeviceMemory(vectory);
            CUdeviceptr ptrix = GetDeviceMemory(indexx);

            LastStatus = _driver.CusparseSroti(_sparse, n, ptrx.Pointer, ptrix.Pointer, ptry.Pointer, c, s, ibase);
        }
        public override void ROT(double[] vectorx, int[] indexx, double[] vectory, double c, double s, int n = 0, cusparseIndexBase ibase = cusparseIndexBase.Zero)
        {
            CUdeviceptr ptrx = GetDeviceMemory(vectorx, ref n);
            CUdeviceptr ptry = GetDeviceMemory(vectory);
            CUdeviceptr ptrix = GetDeviceMemory(indexx);

            LastStatus = _driver.CusparseDroti(_sparse, n, ptrx.Pointer, ptrix.Pointer, ptry.Pointer, c, s, ibase);
        }
        #endregion

        #region SCTR
        public override void SCTR(float[] vectorx, int[] indexx, float[] vectory, int n = 0, cusparseIndexBase ibase = cusparseIndexBase.Zero)
        {
            CUdeviceptr ptrx = GetDeviceMemory(vectorx, ref n);
            CUdeviceptr ptry = GetDeviceMemory(vectory);
            CUdeviceptr ptrix = GetDeviceMemory(indexx);

            LastStatus = _driver.CusparseSsctr(_sparse, n, ptrx.Pointer, ptrix.Pointer, ptry.Pointer, ibase);
        }
        public override void SCTR(double[] vectorx, int[] indexx, double[] vectory, int n = 0, cusparseIndexBase ibase = cusparseIndexBase.Zero)
        {
            CUdeviceptr ptrx = GetDeviceMemory(vectorx, ref n);
            CUdeviceptr ptry = GetDeviceMemory(vectory);
            CUdeviceptr ptrix = GetDeviceMemory(indexx);

            LastStatus = _driver.CusparseDsctr(_sparse, n, ptrx.Pointer, ptrix.Pointer, ptry.Pointer, ibase);
        }
        #endregion
        #endregion
    }
}
