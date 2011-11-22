using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
/*
 * Added by Kichang Kim
 * kkc0923@hotmail.com
 * */
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

        private CUdeviceptr SetupVector(object vector, ref int n)
        {
            CUDevicePtrEx ptrEx = _gpu.GetDeviceMemory(vector) as CUDevicePtrEx;
            n = (n == 0 ? ptrEx.TotalSize : n);

            return ptrEx.DevPtr;
        }

        private CUdeviceptr SetupVector(object vector)
        {
            CUDevicePtrEx ptrEx = _gpu.GetDeviceMemory(vector) as CUDevicePtrEx;
            return ptrEx.DevPtr;
        }

        public override void AXPY(double alpha, double[] vectorx, int[] indexx, double[] vectory, int n = 0, cusparseIndexBase ibase = cusparseIndexBase.Zero)
        {
            CUdeviceptr ptrx = SetupVector(vectorx, ref n);
            CUdeviceptr ptry = SetupVector(vectory);
            CUdeviceptr ptrix = SetupVector(indexx);

            LastStatus = _driver.CusparseDaxpyi(_sparse, n, alpha, ptrx.Pointer, ptrix.Pointer, ptry.Pointer, ibase);
        }
    }
}
