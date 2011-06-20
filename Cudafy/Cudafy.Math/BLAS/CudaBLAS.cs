/*
CUDAfy.NET - LGPL 2.1 License
Please consider purchasing a commerical license - it helps development, frees you from LGPL restrictions
and provides you with support.  Thank you!
Copyright (C) 2011 Hybrid DSP Systems
http://www.hybriddsp.com

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Runtime.InteropServices;
using System.Diagnostics;
using Cudafy.Types;
using Cudafy.Host;

//using CUBLAS4Wrapper;

using GASS.CUDA.BLAS;
using GASS.CUDA.BLAS.Types;
using GASS.CUDA.Types;
using GASS.CUDA;

namespace Cudafy.Maths.BLAS
{
    internal interface ICUBLASDriverv2Ex
    {
        CUBLASStatusv2 cublasCaxpy(cublasHandle handle, int n, ref ComplexF alpha, IntPtr x, int incx, IntPtr y, int incy);
        CUBLASStatusv2 cublasZaxpy(cublasHandle handle, int n, ref ComplexD alpha, IntPtr x, int incx, IntPtr y, int incy);
        CUBLASStatusv2 cublasCdotu(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref ComplexF result);
        CUBLASStatusv2 cublasCdotc(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref ComplexF result);
        CUBLASStatusv2 cublasZdotu(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref ComplexD result);
        CUBLASStatusv2 cublasZdotc(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref ComplexD result);
        CUBLASStatusv2 cublasCrot(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref float c, ref ComplexF s);
        CUBLASStatusv2 cublasZrot(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref double c, ref ComplexD s);
        CUBLASStatusv2 cublasCrotg(cublasHandle handle, ref ComplexF a, ref ComplexF b, ref float c, ref ComplexF s);
        CUBLASStatusv2 cublasZrotg(cublasHandle handle, ref ComplexD a, ref ComplexD b, ref double c, ref ComplexD s);
    }

    internal class CUBLASDriver32Ex : ICUBLASDriverv2Ex
    {
        [DllImport(CUBLASDriver32.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasCaxpy_v2(cublasHandle handle, int n, ref ComplexF alpha, IntPtr x, int incx, IntPtr y, int incy);
        [DllImport(CUBLASDriver32.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasZaxpy_v2(cublasHandle handle, int n, ref ComplexD alpha, IntPtr x, int incx, IntPtr y, int incy);
        [DllImport(CUBLASDriver32.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasCdotu_v2(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref ComplexF result);
        [DllImport(CUBLASDriver32.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasCdotc_v2(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref ComplexF result);
        [DllImport(CUBLASDriver32.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasZdotu_v2(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref ComplexD result);
        [DllImport(CUBLASDriver32.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasZdotc_v2(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref ComplexD result);
        [DllImport(CUBLASDriver32.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasCrot_v2(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref float c, ref ComplexF s);
        [DllImport(CUBLASDriver32.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasZrot_v2(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref double c, ref ComplexD s);
        [DllImport(CUBLASDriver32.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasCrotg_v2(cublasHandle handle, ref ComplexF a, ref ComplexF b, ref float c, ref ComplexF s);
        [DllImport(CUBLASDriver32.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasZrotg_v2(cublasHandle handle, ref ComplexD a, ref ComplexD b, ref double c, ref ComplexD s);

        public CUBLASStatusv2 cublasCaxpy(cublasHandle handle, int n, ref ComplexF alpha, IntPtr x, int incx, IntPtr y, int incy)
        {
            return cublasCaxpy_v2(handle, n, ref alpha, x, incx, y, incy);
        }

        public CUBLASStatusv2 cublasZaxpy(cublasHandle handle, int n, ref ComplexD alpha, IntPtr x, int incx, IntPtr y, int incy)
        {
            return cublasZaxpy_v2(handle, n, ref alpha, x, incx, y, incy);
        }

        public CUBLASStatusv2 cublasCdotu(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref ComplexF result)
        {
            return cublasCdotu_v2(handle, n, x, incx, y, incy, ref result);
        }

        public CUBLASStatusv2 cublasCdotc(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref ComplexF result)
        {
            return cublasCdotc_v2(handle, n, x, incx, y, incy, ref result);
        }

        public CUBLASStatusv2 cublasZdotu(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref ComplexD result)
        {
            return cublasZdotu_v2(handle, n, x, incx, y, incy, ref result);
        }

        public CUBLASStatusv2 cublasZdotc(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref ComplexD result)
        {
            return cublasZdotc_v2(handle, n, x, incx, y, incy, ref result);
        }

        public CUBLASStatusv2 cublasCrot(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref float c, ref ComplexF s)
        {
            return cublasCrot_v2(handle, n, x, incx, y, incy, ref c, ref s);
        }

        public CUBLASStatusv2 cublasZrot(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref double c, ref ComplexD s)
        {
            return cublasZrot_v2(handle, n, x, incx, y, incy, ref c, ref s);
        }

        public CUBLASStatusv2 cublasCrotg(cublasHandle handle, ref ComplexF a, ref ComplexF b, ref float c, ref ComplexF s)
        {
            return cublasCrotg_v2(handle, ref a, ref b, ref c, ref s);
        }

        public CUBLASStatusv2 cublasZrotg(cublasHandle handle, ref ComplexD a, ref ComplexD b, ref double c, ref ComplexD s)
        {
            return cublasZrotg_v2(handle, ref a, ref b, ref c, ref s);
        }
    }

    internal class CUBLASDriver64Ex : ICUBLASDriverv2Ex
    {
        [DllImport(CUBLASDriver64.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasCaxpy_v2(cublasHandle handle, int n, ref ComplexF alpha, IntPtr x, int incx, IntPtr y, int incy);
        [DllImport(CUBLASDriver64.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasZaxpy_v2(cublasHandle handle, int n, ref ComplexD alpha, IntPtr x, int incx, IntPtr y, int incy);
        [DllImport(CUBLASDriver64.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasCdotu_v2(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref ComplexF result);
        [DllImport(CUBLASDriver64.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasCdotc_v2(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref ComplexF result);
        [DllImport(CUBLASDriver64.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasZdotu_v2(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref ComplexD result);
        [DllImport(CUBLASDriver64.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasZdotc_v2(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref ComplexD result);
        [DllImport(CUBLASDriver64.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasCrot_v2(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref float c, ref ComplexF s);
        [DllImport(CUBLASDriver64.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasZrot_v2(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref double c, ref ComplexD s);
        [DllImport(CUBLASDriver64.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasCrotg_v2(cublasHandle handle, ref ComplexF a, ref ComplexF b, ref float c, ref ComplexF s);
        [DllImport(CUBLASDriver64.CUBLAS_DLL_NAME)]
        private static extern CUBLASStatusv2 cublasZrotg_v2(cublasHandle handle, ref ComplexD a, ref ComplexD b, ref double c, ref ComplexD s);

        public CUBLASStatusv2 cublasCaxpy(cublasHandle handle, int n, ref ComplexF alpha, IntPtr x, int incx, IntPtr y, int incy)
        {
            return cublasCaxpy_v2(handle, n, ref alpha, x, incx, y, incy);
        }

        public CUBLASStatusv2 cublasZaxpy(cublasHandle handle, int n, ref ComplexD alpha, IntPtr x, int incx, IntPtr y, int incy)
        {
            return cublasZaxpy_v2(handle, n, ref alpha, x, incx, y, incy);
        }

        public CUBLASStatusv2 cublasCdotu(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref ComplexF result)
        {
            return cublasCdotu_v2(handle, n, x, incx, y, incy, ref result);
        }

        public CUBLASStatusv2 cublasCdotc(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref ComplexF result)
        {
            return cublasCdotc_v2(handle, n, x, incx, y, incy, ref result);
        }

        public CUBLASStatusv2 cublasZdotu(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref ComplexD result)
        {
            return cublasZdotu_v2(handle, n, x, incx, y, incy, ref result);
        }

        public CUBLASStatusv2 cublasZdotc(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref ComplexD result)
        {
            return cublasZdotc_v2(handle, n, x, incx, y, incy, ref result);
        }

        public CUBLASStatusv2 cublasCrot(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref float c, ref ComplexF s)
        {
            return cublasCrot_v2(handle, n, x, incx, y, incy, ref c, ref s);
        }

        public CUBLASStatusv2 cublasZrot(cublasHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, ref double c, ref ComplexD s)
        {
            return cublasZrot_v2(handle, n, x, incx, y, incy, ref c, ref s);
        }

        public CUBLASStatusv2 cublasCrotg(cublasHandle handle, ref ComplexF a, ref ComplexF b, ref float c, ref ComplexF s)
        {
            return cublasCrotg_v2(handle, ref a, ref b, ref c, ref s);
        }

        public CUBLASStatusv2 cublasZrotg(cublasHandle handle, ref ComplexD a, ref ComplexD b, ref double c, ref ComplexD s)
        {
            return cublasZrotg_v2(handle, ref a, ref b, ref c, ref s);
        }
    }
    
    
    /// <summary>
    /// Wrapper around CUBLAS.
    /// </summary>
    internal class CudaBLAS : GPGPUBLAS
    {
        internal CudaBLAS(GPGPU gpu) : base()
        {
            if (IntPtr.Size == 8)
            {
                _driver = new CUBLASDriver64();
                _driverEx = new CUBLASDriver64Ex();
            }
            else
            {
                _driver = new CUBLASDriver32();
                _driverEx = new CUBLASDriver32Ex();
            }
            LastStatus = _driver.cublasCreate(ref _blas);
            _gpu = gpu;
        }

        private GPGPU _gpu;

        private cublasHandle _blas;

        private ICUBLASDriverv2 _driver;

        private ICUBLASDriverv2Ex _driverEx;

        private CUBLASStatusv2 _status;

        private CUBLASStatusv2 LastStatus
        {
            get { return _status; }
            set
            {
                _status = value; if (_status != CUBLASStatusv2.Success)
                    throw new CudafyMathException(CudafyMathException.csBLAS_ERROR_X, _status.ToString());
            }
        }

        //~CudaBLAS()
        //{
        //    Dispose();
        //}

        protected override void Shutdown()
        {
            try
            {
                LastStatus = _driver.cublasDestroy(_blas);
            }
            catch (DllNotFoundException ex)
            {
                Debug.WriteLine(ex.Message);
            }
        }

        #region Helpers

        private void CheckLastError()
        {
            CUBLASStatus status = CUBLASDriver.cublasGetError();
            if (status != CUBLASStatus.Success)
                throw new CudafyMathException(CudafyMathException.csBLAS_ERROR_X, status.ToString());
        }

        private static uint IDX2C(int i, int j, int ld)
        {
            uint v = (uint)(((j) * (ld)) + (i));
            return v;
        }

        private cuFloatComplex Convert(ComplexF cf)
        {
            return new cuFloatComplex() { real = cf.x, imag = cf.y };
        }

        private cuDoubleComplex Convert(ComplexD cf)
        {
            return new cuDoubleComplex() { real = cf.x, imag = cf.y };
        }

        private cuFloatComplex Convert(float x, float y)
        {
            return new cuFloatComplex() { real = x, imag = y };
        }

        private cuFloatComplex ConvertToFloatComplex<T>(T alpha)
        {
            ComplexF cf = (ComplexF)(object)alpha;
            return Convert(cf);
        }

        private float ConvertToFloat<T>(T alpha)
        {
            float d = (float)(object)alpha;
            return d;
        }

        private cuDoubleComplex ConvertToDoubleComplex<T>(T alpha)
        {
            ComplexD cf = (ComplexD)(object)alpha;
            return Convert(cf);
        }

        private double ConvertToDouble<T>(T alpha)
        {
            double d = (double)(object)alpha;
            return d;
        }

        private CUdeviceptr SetupVector<T>(object vector, int x, ref int n, ref int incx, out eDataType type)
        {
            CUDevicePtrEx ptrEx = _gpu.GetDeviceMemory(vector) as CUDevicePtrEx;
            if (incx == 0)
                throw new CudafyMathException(CudafyHostException.csX_NOT_SET, "incx");
            n = (n == 0 ? ptrEx.TotalSize / incx : n);
            type = GetDataType<T>();
            int size = Marshal.SizeOf(typeof(T));
            CUdeviceptr ptr = ptrEx.DevPtr + (uint)(size * x);
            return ptr;
        }

        private CUdeviceptr SetupVector<T>(object vector, int x, ref int n, ref int incx)
        {
            CUDevicePtrEx ptrEx = _gpu.GetDeviceMemory(vector) as CUDevicePtrEx;
            if (incx == 0)
                throw new CudafyMathException(CudafyHostException.csX_NOT_SET, "incx");
            n = (n == 0 ? ptrEx.TotalSize / incx : n);
            int size = Marshal.SizeOf(typeof(T));
            CUdeviceptr ptr = ptrEx.DevPtr + (uint)(size * x);
            return ptr;
        }

        private CUdeviceptr SetupVector(object vector, int x, ref int n, ref int incx, int elemSize)
        {
            CUDevicePtrEx ptrEx = _gpu.GetDeviceMemory(vector) as CUDevicePtrEx;
            if (incx == 0)
                throw new CudafyMathException(CudafyHostException.csX_NOT_SET, "incx");
            n = (n == 0 ? ptrEx.TotalSize / incx : n);
            CUdeviceptr ptr = ptrEx.DevPtr + (uint)(elemSize * x);
            return ptr;
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

        private int Setup2D<T>(object devMatrix, int n, int row, int col, bool columnWise, out CUdeviceptr ptr, ref int incx, out eDataType type)
        {
            int width, height;
            return Setup2D<T>(devMatrix, n, row, col, columnWise, out ptr, ref incx, out type, out width, out height);
        }

        private int Setup2D<T>(object devMatrix, int n, int row, int col, bool columnWise, out CUdeviceptr ptr, ref int incx, out eDataType type, out int width, out int height)
        {
            CUDevicePtrEx ptrEx = _gpu.GetDeviceMemory(devMatrix) as CUDevicePtrEx;
            width = ptrEx.XSize;
            height = ptrEx.YSize;
            int ldm = height;
            incx = (incx == 1 ? (columnWise ? height : 1) : incx);
            uint v = IDX2C(col, row, ldm);
            n = (n == 0 ? (columnWise ? height : ptrEx.TotalSize - (int)v) : n);
            
            type = GetDataType<T>();
            int elemSize = Marshal.SizeOf(typeof(T));
            ptr = ptrEx.DevPtr + (uint)(elemSize * v);

            return n;
        }

        private int Setup2DOld<T>(object devMatrix, int n, int row, int col, bool columnWise, out CUdeviceptr ptr, ref int incx, out eDataType type, out int width, out int height)
        {
            CUDevicePtrEx ptrEx = _gpu.GetDeviceMemory(devMatrix) as CUDevicePtrEx;
            width = ptrEx.XSize;
            height = ptrEx.YSize;
            int ldm = ptrEx.YSize;
            incx = (incx == 1 ? (columnWise ? 1 : ptrEx.YSize) : incx);
            uint v = IDX2C(col, row, ldm);
            n = (n == 0 ? (columnWise ? ptrEx.YSize : ptrEx.TotalSize - (int)v) : n);

            type = GetDataType<T>();
            int elemSize = Marshal.SizeOf(typeof(T));
            ptr = ptrEx.DevPtr + (uint)(elemSize * v);

            return n;
        }


        private IntPtr GetIntPtr(object hostOrDevice, out GCHandle handle)
        {
            handle = new GCHandle();
            DevicePtrEx ptrEx = _gpu.TryGetDeviceMemory(hostOrDevice);
            if (ptrEx == null)
                handle = GCHandle.Alloc(hostOrDevice, GCHandleType.Pinned);
            return ptrEx == null ? handle.AddrOfPinnedObject() : ptrEx.Pointer;
        }

        private void FreeGCHandles(params GCHandle[] handles)
        {
            foreach (var h in handles)
                if (h.IsAllocated)
                    h.Free();
        }
        

        #endregion

        #region Max

        public override int IAMAX(float[] vectorx, int n = 0, int rowx = 0, int incx = 1)
        {
            CUdeviceptr ptr = SetupVector(vectorx, rowx, ref n, ref incx, 4);
            int res = 0;
            LastStatus = _driver.cublasIsamax(_blas, n, ptr.Pointer, incx, ref res);
            return res;
        }

        public override int IAMAX(double[] vectorx, int n = 0, int rowx = 0, int incx = 1)
        {
            CUdeviceptr ptr = SetupVector(vectorx, rowx, ref n, ref incx, 8);
            int res = 0;
            LastStatus = _driver.cublasIdamax(_blas, n, ptr.Pointer, incx, ref res);
            return res;
        }

        public override int IAMAX(ComplexF[] vectorx, int n = 0, int rowx = 0, int incx = 1)
        {
            CUdeviceptr ptr = SetupVector(vectorx, rowx, ref n, ref incx, 8);
            int res = 0;
            LastStatus = _driver.cublasIcamax(_blas, n, ptr.Pointer, incx, ref res);
            return res;
        }

        public override int IAMAX(ComplexD[] vectorx, int n = 0, int rowx = 0, int incx = 1)
        {
            CUdeviceptr ptr = SetupVector(vectorx, rowx, ref n, ref incx, 16);
            int res = 0;
            LastStatus = _driver.cublasIzamax(_blas, n, ptr.Pointer, incx, ref res);
            return res;
        }

        #endregion

        #region Min

        public override int IAMIN(float[] vectorx, int n = 0, int rowx = 0, int incx = 1)
        {
            CUdeviceptr ptr = SetupVector(vectorx, rowx, ref n, ref incx, 4);
            int res = 0;
            LastStatus = _driver.cublasIsamin(_blas, n, ptr.Pointer, incx, ref res);
            return res;
        }

        public override int IAMIN(double[] vectorx, int n = 0, int rowx = 0, int incx = 1)
        {
            CUdeviceptr ptr = SetupVector(vectorx, rowx, ref n, ref incx, 8);
            int res = 0;
            LastStatus = _driver.cublasIdamin(_blas, n, ptr.Pointer, incx, ref res);
            return res;
        }

        public override int IAMIN(ComplexF[] vectorx, int n = 0, int rowx = 0, int incx = 1)
        {
            CUdeviceptr ptr = SetupVector(vectorx, rowx, ref n, ref incx, 8);
            int res = 0;
            LastStatus = _driver.cublasIcamin(_blas, n, ptr.Pointer, incx, ref res);
            return res;
        }

        public override int IAMIN(ComplexD[] vectorx, int n = 0, int rowx = 0, int incx = 1)
        {
            CUdeviceptr ptr = SetupVector(vectorx, rowx, ref n, ref incx, 16);
            int res = 0;
            LastStatus = _driver.cublasIzamin(_blas, n, ptr.Pointer, incx, ref res);
            return res;
        }

        #endregion

        #region Sum

        public override float ASUM(float[] vectorx, int n = 0, int rowx = 0, int incx = 1)
        {
            CUdeviceptr ptr = SetupVector(vectorx, rowx, ref n, ref incx, 4);
            float res = 0;
            LastStatus = _driver.cublasSasum(_blas, n, ptr.Pointer, incx, ref res);
            return res;
        }

        public override double ASUM(double[] vectorx, int n = 0, int rowx = 0, int incx = 1)
        {
            CUdeviceptr ptr = SetupVector(vectorx, rowx, ref n, ref incx, 8);
            double res = 0;
            LastStatus = _driver.cublasDasum(_blas, n, ptr.Pointer, incx, ref res);
            return res;
        }

        public override float ASUM(ComplexF[] vectorx, int n = 0, int rowx = 0, int incx = 1)
        {
            CUdeviceptr ptr = SetupVector(vectorx, rowx, ref n, ref incx, 8);
            float res = 0;
            LastStatus = _driver.cublasScasum(_blas, n, ptr.Pointer, incx, ref res);
            return res;
        }

        public override double ASUM(ComplexD[] vectorx, int n = 0, int rowx = 0, int incx = 1)
        {
            CUdeviceptr ptr = SetupVector(vectorx, rowx, ref n, ref incx, 16);
            double res = 0;
            LastStatus = _driver.cublasDzasum(_blas, n, ptr.Pointer, incx, ref res);
            return res;
        }

        #endregion

        #region AXPY



        protected override void AXPY(float[] alpha, float[] vectorx, float[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
        {
            //CUDevicePtrEx ptrAlphaEx = _gpu.TryGetDeviceMemory(alpha) as CUDevicePtrEx;
            CUdeviceptr ptrx = SetupVector(vectorx, rowx, ref n, ref incx, 4);
            CUdeviceptr ptry = SetupVector(vectory, rowy, ref n, ref incy, 4);                     
            //if (ptrAlphaEx == null)
                LastStatus = _driver.cublasSaxpy(_blas, n, ref alpha[0], ptrx.Pointer, incx, ptry.Pointer, incy);
            //else
            //    LastStatus = CUBLASDriverv2.cublasSaxpy(_blas, n, ptrAlphaEx.DevPtr.Pointer, ptrx.Pointer, incx, ptry.Pointer, incy);
        }

        protected override void AXPY(double[] alpha, double[] vectorx, double[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
        {
            //CUDevicePtrEx ptrAlphaEx = _gpu.TryGetDeviceMemory(alpha) as CUDevicePtrEx;
            CUdeviceptr ptrx = SetupVector(vectorx, rowx, ref n, ref incx, 8);
            CUdeviceptr ptry = SetupVector(vectory, rowy, ref n, ref incy, 8);
            //if (ptrAlphaEx == null)
                LastStatus = _driver.cublasDaxpy(_blas, n, ref alpha[0], ptrx.Pointer, incx, ptry.Pointer, incy);
            //else
            //    LastStatus = CUBLASDriverv2.cublasDaxpy(_blas, n, ptrAlphaEx.DevPtr.Pointer, ptrx.Pointer, incx, ptry.Pointer, incy);
        }

        protected override void AXPY(ComplexF[] alpha, ComplexD[] vectorx, ComplexD[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
        {
            //CUDevicePtrEx ptrAlphaEx = _gpu.TryGetDeviceMemory(alpha) as CUDevicePtrEx;
            CUdeviceptr ptrx = SetupVector(vectorx, rowx, ref n, ref incx, 8);
            CUdeviceptr ptry = SetupVector(vectory, rowy, ref n, ref incy, 8);
            //if (ptrAlphaEx == null)
            LastStatus = _driverEx.cublasCaxpy(_blas, n, ref alpha[0], ptrx.Pointer, incx, ptry.Pointer, incy);
            //else
            //    LastStatus = CUBLASDriverv2.cublasCaxpy(_blas, n, ptrAlphaEx.DevPtr.Pointer, ptrx.Pointer, incx, ptry.Pointer, incy);
        }

        protected override void AXPY(ComplexD[] alpha, ComplexD[] vectorx, ComplexD[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
        {
            //CUDevicePtrEx ptrAlphaEx = _gpu.TryGetDeviceMemory(alpha) as CUDevicePtrEx;
            CUdeviceptr ptrx = SetupVector(vectorx, rowx, ref n, ref incx, 8);
            CUdeviceptr ptry = SetupVector(vectory, rowy, ref n, ref incy, 8);
            //if (ptrAlphaEx == null)
            LastStatus = _driverEx.cublasZaxpy(_blas, n, ref alpha[0], ptrx.Pointer, incx, ptry.Pointer, incy);
            //else
            //    LastStatus = CUBLASDriverv2.cublasZaxpy(_blas, n, ptrAlphaEx.DevPtr.Pointer, ptrx.Pointer, incx, ptry.Pointer, incy);
        }

        #endregion

        #region COPY

        public override void COPY(float[] vectorx, float[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
        {
            CUdeviceptr ptrx = SetupVector(vectorx, rowx, ref n, ref incx, 4);
            CUdeviceptr ptry = SetupVector(vectory, rowy, ref n, ref incy, 4);
            LastStatus = _driver.cublasScopy(_blas, n, ptrx.Pointer, incx, ptry.Pointer, incy);
        }

        public override void COPY(double[] vectorx, double[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
        {
            CUdeviceptr ptrx = SetupVector(vectorx, rowx, ref n, ref incx, 8);
            CUdeviceptr ptry = SetupVector(vectory, rowy, ref n, ref incy, 8);
            LastStatus = _driver.cublasDcopy(_blas, n, ptrx.Pointer, incx, ptry.Pointer, incy);
        }

        public override void COPY(ComplexF[] vectorx, ComplexF[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
        {
            CUdeviceptr ptrx = SetupVector(vectorx, rowx, ref n, ref incx, 8);
            CUdeviceptr ptry = SetupVector(vectory, rowy, ref n, ref incy, 8);
            LastStatus = _driver.cublasCcopy(_blas, n, ptrx.Pointer, incx, ptry.Pointer, incy);
        }

        public override void COPY(ComplexD[] vectorx, ComplexD[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
        {
            CUdeviceptr ptrx = SetupVector(vectorx, rowx, ref n, ref incx, 16);
            CUdeviceptr ptry = SetupVector(vectory, rowy, ref n, ref incy, 16);
            LastStatus = _driver.cublasZcopy(_blas, n, ptrx.Pointer, incx, ptry.Pointer, incy);
        }

        #endregion

        #region DOT



        public override float DOT(float[] vectorx, float[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
        {
            CUdeviceptr ptrx = SetupVector(vectorx, rowx, ref n, ref incx, 4);
            CUdeviceptr ptry = SetupVector(vectory, rowy, ref n, ref incy, 4);
            float res = 0;
            LastStatus = _driver.cublasSdot(_blas, n, ptrx.Pointer, incx, ptry.Pointer, incy, ref res);
            return res;
        }

        public override double DOT(double[] vectorx, double[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
        {
            CUdeviceptr ptrx = SetupVector(vectorx, rowx, ref n, ref incx, 8);
            CUdeviceptr ptry = SetupVector(vectory, rowy, ref n, ref incy, 8);
            double res = 0;
            LastStatus = _driver.cublasDdot(_blas, n, ptrx.Pointer, incx, ptry.Pointer, incy, ref res);
            return res;
        }

        public override ComplexF DOTU(ComplexF[] vectorx, ComplexF[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
        {
            CUdeviceptr ptrx = SetupVector(vectorx, rowx, ref n, ref incx, 8);
            CUdeviceptr ptry = SetupVector(vectory, rowy, ref n, ref incy, 8);
            ComplexF res = new ComplexF();
            LastStatus = _driverEx.cublasCdotu(_blas, n, ptrx.Pointer, incx, ptry.Pointer, incy, ref res);
            return res;
        }

        public override ComplexF DOTC(ComplexF[] vectorx, ComplexF[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
        {
            CUdeviceptr ptrx = SetupVector(vectorx, rowx, ref n, ref incx, 8);
            CUdeviceptr ptry = SetupVector(vectory, rowy, ref n, ref incy, 8);
            ComplexF res = new ComplexF();
            LastStatus = _driverEx.cublasCdotc(_blas, n, ptrx.Pointer, incx, ptry.Pointer, incy, ref res);
            return res;
        }

        public override ComplexD DOTU(ComplexD[] vectorx, ComplexD[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
        {
            CUdeviceptr ptrx = SetupVector(vectorx, rowx, ref n, ref incx, 16);
            CUdeviceptr ptry = SetupVector(vectory, rowy, ref n, ref incy, 16);
            ComplexD res = new ComplexD();
            LastStatus = _driverEx.cublasZdotu(_blas, n, ptrx.Pointer, incx, ptry.Pointer, incy, ref res);
            return res;
        }

        public override ComplexD DOTC(ComplexD[] vectorx, ComplexD[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
        {
            CUdeviceptr ptrx = SetupVector(vectorx, rowx, ref n, ref incx, 16);
            CUdeviceptr ptry = SetupVector(vectory, rowy, ref n, ref incy, 16);
            ComplexD res = new ComplexD();
            LastStatus = _driverEx.cublasZdotc(_blas, n, ptrx.Pointer, incx, ptry.Pointer, incy, ref res);
            return res;
        }

        #endregion

        #region NRM2

        public override float NRM2(float[] vectorx, int n = 0, int rowx = 0, int incx = 1)
        {
            CUdeviceptr ptr = SetupVector(vectorx, rowx, ref n, ref incx, 4);
            float res = 0;
            LastStatus = _driver.cublasSnrm2(_blas, n, ptr.Pointer, incx, ref res);
            return res;
        }

        public override double NRM2(double[] vectorx, int n = 0, int rowx = 0, int incx = 1)
        {
            CUdeviceptr ptr = SetupVector(vectorx, rowx, ref n, ref incx, 8);
            double res = 0;
            LastStatus = _driver.cublasDnrm2(_blas, n, ptr.Pointer, incx, ref res);
            return res;
        }

        public override float NRM2(ComplexF[] vectorx, int n = 0, int rowx = 0, int incx = 1)
        {
            CUdeviceptr ptr = SetupVector(vectorx, rowx, ref n, ref incx, 8);
            float res = 0;
            LastStatus = _driver.cublasScnrm2(_blas, n, ptr.Pointer, incx, ref res);
            return res;
        }

        public override double NRM2(ComplexD[] vectorx, int n = 0, int rowx = 0, int incx = 1)
        {
            CUdeviceptr ptr = SetupVector(vectorx, rowx, ref n, ref incx, 16);
            double res = 0;
            LastStatus = _driver.cublasDznrm2(_blas, n, ptr.Pointer, incx, ref res);
            return res;
        }

        #endregion

        #region ROT



        public override void ROT(float[] vectorx, float[] vectory, float[] c, float[] s, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
        {
            CUDevicePtrEx ptrc = _gpu.TryGetDeviceMemory(c) as CUDevicePtrEx;
            CUDevicePtrEx ptrs = _gpu.TryGetDeviceMemory(s) as CUDevicePtrEx;
            CUdeviceptr ptrx = SetupVector(vectorx, rowx, ref n, ref incx, 4);
            CUdeviceptr ptry = SetupVector(vectory, rowy, ref n, ref incy, 4);
            if (ptrc == null || ptrs == null)
                LastStatus = _driver.cublasSrot(_blas, n, ptrx.Pointer, incx, ptry.Pointer, incy, ref c[0], ref s[0]);
            else
                LastStatus = _driver.cublasSrot(_blas, n, ptrx.Pointer, incx, ptry.Pointer, incy, ptrc.DevPtr.Pointer, ptrs.DevPtr.Pointer);
        }

        public override void ROT(double[] vectorx, double[] vectory, double[] c, double[] s, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
        {
            CUDevicePtrEx ptrc = _gpu.TryGetDeviceMemory(c) as CUDevicePtrEx;
            CUDevicePtrEx ptrs = _gpu.TryGetDeviceMemory(s) as CUDevicePtrEx;
            CUdeviceptr ptrx = SetupVector(vectorx, rowx, ref n, ref incx, 8);
            CUdeviceptr ptry = SetupVector(vectory, rowy, ref n, ref incy, 8);
            if (ptrc == null || ptrs == null)
                LastStatus = _driver.cublasDrot(_blas, n, ptrx.Pointer, incx, ptry.Pointer, incy, ref c[0], ref s[0]);
            else
                LastStatus = _driver.cublasDrot(_blas, n, ptrx.Pointer, incx, ptry.Pointer, incy, ptrc.DevPtr.Pointer, ptrs.DevPtr.Pointer);
        }

        public override void ROT(ComplexF[] vectorx, ComplexF[] vectory, float[] c, ComplexF[] s, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
        {
            CUDevicePtrEx ptrc = _gpu.TryGetDeviceMemory(c) as CUDevicePtrEx;
            CUDevicePtrEx ptrs = _gpu.TryGetDeviceMemory(s) as CUDevicePtrEx;
            CUdeviceptr ptrx = SetupVector(vectorx, rowx, ref n, ref incx, 8);
            CUdeviceptr ptry = SetupVector(vectory, rowy, ref n, ref incy, 8);
            if (ptrc == null || ptrs == null)
                LastStatus = _driverEx.cublasCrot(_blas, n, ptrx.Pointer, incx, ptry.Pointer, incy, ref c[0], ref s[0]);
            else
                LastStatus = _driver.cublasCrot(_blas, n, ptrx.Pointer, incx, ptry.Pointer, incy, ptrc.DevPtr.Pointer, ptrs.DevPtr.Pointer);
        }

        public override void ROT(ComplexF[] vectorx, ComplexF[] vectory, float[] c, float[] s, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
        {
            CUDevicePtrEx ptrc = _gpu.TryGetDeviceMemory(c) as CUDevicePtrEx;
            CUDevicePtrEx ptrs = _gpu.TryGetDeviceMemory(s) as CUDevicePtrEx;
            CUdeviceptr ptrx = SetupVector(vectorx, rowx, ref n, ref incx, 8);
            CUdeviceptr ptry = SetupVector(vectory, rowy, ref n, ref incy, 8);
            if (ptrc == null || ptrs == null)
                LastStatus = _driver.cublasCsrot(_blas, n, ptrx.Pointer, incx, ptry.Pointer, incy, ref c[0], ref s[0]);
            else
                LastStatus = _driver.cublasCsrot(_blas, n, ptrx.Pointer, incx, ptry.Pointer, incy, ptrc.DevPtr.Pointer, ptrs.DevPtr.Pointer);
        }

        public override void ROT(ComplexD[] vectorx, ComplexD[] vectory, double[] c, ComplexD[] s, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
        {
            CUDevicePtrEx ptrc = _gpu.TryGetDeviceMemory(c) as CUDevicePtrEx;
            CUDevicePtrEx ptrs = _gpu.TryGetDeviceMemory(s) as CUDevicePtrEx;
            CUdeviceptr ptrx = SetupVector(vectorx, rowx, ref n, ref incx, 16);
            CUdeviceptr ptry = SetupVector(vectory, rowy, ref n, ref incy, 16);
            if (ptrc == null || ptrs == null)
                LastStatus = _driverEx.cublasZrot(_blas, n, ptrx.Pointer, incx, ptry.Pointer, incy, ref c[0], ref s[0]);
            else
                LastStatus = _driver.cublasZrot(_blas, n, ptrx.Pointer, incx, ptry.Pointer, incy, ptrc.DevPtr.Pointer, ptrs.DevPtr.Pointer);
        }

        public override void ROT(ComplexD[] vectorx, ComplexD[] vectory, double[] c, double[] s, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
        {
            CUDevicePtrEx ptrc = _gpu.TryGetDeviceMemory(c) as CUDevicePtrEx;
            CUDevicePtrEx ptrs = _gpu.TryGetDeviceMemory(s) as CUDevicePtrEx;
            CUdeviceptr ptrx = SetupVector(vectorx, rowx, ref n, ref incx, 16);
            CUdeviceptr ptry = SetupVector(vectory, rowy, ref n, ref incy, 16);
            if (ptrc == null || ptrs == null)
                LastStatus = _driver.cublasZdrot(_blas, n, ptrx.Pointer, incx, ptry.Pointer, incy, ref c[0], ref s[0]);
            else
                LastStatus = _driver.cublasZdrot(_blas, n, ptrx.Pointer, incx, ptry.Pointer, incy, ptrc.Pointer, ptrs.Pointer);
        }

        #endregion

        #region ROTG


       
        public override void ROTG(float[] a, float[] b, float[] c, float[] s)
        {
            CUDevicePtrEx ptra = _gpu.TryGetDeviceMemory(a) as CUDevicePtrEx;
            CUDevicePtrEx ptrb = _gpu.TryGetDeviceMemory(b) as CUDevicePtrEx;
            CUDevicePtrEx ptrc = _gpu.TryGetDeviceMemory(c) as CUDevicePtrEx;
            CUDevicePtrEx ptrs = _gpu.TryGetDeviceMemory(s) as CUDevicePtrEx;
            if (ptrc == null || ptrs == null || ptra == null || ptrb == null)
                LastStatus = _driver.cublasSrotg(_blas, ref a[0], ref b[0], ref c[0], ref s[0]);
            else
                LastStatus = _driver.cublasSrotg(_blas, ptra.Pointer, ptrb.Pointer, ptrc.Pointer, ptrs.Pointer);
        }

        public override void ROTG(double[] a, double[] b, double[] c, double[] s)
        {
            CUDevicePtrEx ptra = _gpu.TryGetDeviceMemory(a) as CUDevicePtrEx;
            CUDevicePtrEx ptrb = _gpu.TryGetDeviceMemory(b) as CUDevicePtrEx;
            CUDevicePtrEx ptrc = _gpu.TryGetDeviceMemory(c) as CUDevicePtrEx;
            CUDevicePtrEx ptrs = _gpu.TryGetDeviceMemory(s) as CUDevicePtrEx;
            if (ptrc == null || ptrs == null || ptra == null || ptrb == null)
                LastStatus = _driver.cublasDrotg(_blas, ref a[0], ref b[0], ref c[0], ref s[0]);
            else
                LastStatus = _driver.cublasDrotg(_blas, ptra.Pointer, ptrb.Pointer, ptrc.Pointer, ptrs.Pointer);
        }

        public override void ROTG(ComplexF[] a, ComplexF[] b, float[] c, ComplexF[] s)
        {
            CUDevicePtrEx ptra = _gpu.TryGetDeviceMemory(a) as CUDevicePtrEx;
            CUDevicePtrEx ptrb = _gpu.TryGetDeviceMemory(b) as CUDevicePtrEx;
            CUDevicePtrEx ptrc = _gpu.TryGetDeviceMemory(c) as CUDevicePtrEx;
            CUDevicePtrEx ptrs = _gpu.TryGetDeviceMemory(s) as CUDevicePtrEx;
            if (ptrc == null || ptrs == null || ptra == null || ptrb == null)
                LastStatus = _driverEx.cublasCrotg(_blas, ref a[0], ref b[0], ref c[0], ref s[0]);
            else
                LastStatus = _driver.cublasCrotg(_blas, ptra.Pointer, ptrb.Pointer, ptrc.Pointer, ptrs.Pointer);
        }

        public override void ROTG(ComplexD[] a, ComplexD[] b, double[] c, ComplexD[] s)
        {
            CUDevicePtrEx ptra = _gpu.TryGetDeviceMemory(a) as CUDevicePtrEx;
            CUDevicePtrEx ptrb = _gpu.TryGetDeviceMemory(b) as CUDevicePtrEx;
            CUDevicePtrEx ptrc = _gpu.TryGetDeviceMemory(c) as CUDevicePtrEx;
            CUDevicePtrEx ptrs = _gpu.TryGetDeviceMemory(s) as CUDevicePtrEx;
            if (ptrc == null || ptrs == null || ptra == null || ptrb == null)
                LastStatus = _driverEx.cublasZrotg(_blas, ref a[0], ref b[0], ref c[0], ref s[0]);
            else
                LastStatus = _driver.cublasZrotg(_blas, ptra.Pointer, ptrb.Pointer, ptrc.Pointer, ptrs.Pointer);
        }

        #endregion

        #region ROTM

        public override void ROTM(float[] vectorx, float[] vectory, float[] param, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
        {
            CUdeviceptr ptrx = SetupVector(vectorx, rowx, ref n, ref incx, 4);
            CUdeviceptr ptry = SetupVector(vectory, rowy, ref n, ref incy, 4);
            CUDevicePtrEx ptrp = _gpu.TryGetDeviceMemory(param) as CUDevicePtrEx;
            if (ptrp == null)
            {
                GCHandle handle = GCHandle.Alloc(param, GCHandleType.Pinned);
                IntPtr hostPtr = handle.AddrOfPinnedObject();
                LastStatus = _driver.cublasSrotm(_blas, n, ptrx.Pointer, incx, ptry.Pointer, incy, hostPtr);
                handle.Free();
            }
            else
                LastStatus = _driver.cublasSrotm(_blas, n, ptrx.Pointer, incx, ptry.Pointer, incy, ptrp.Pointer);                   
        }

        public override void ROTM(double[] vectorx, double[] vectory, double[] param, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
        {
            CUdeviceptr ptrx = SetupVector(vectorx, rowx, ref n, ref incx, 8);
            CUdeviceptr ptry = SetupVector(vectory, rowy, ref n, ref incy, 8);
            CUDevicePtrEx ptrp = _gpu.TryGetDeviceMemory(param) as CUDevicePtrEx;
            if (ptrp == null)
            {
                GCHandle handle = GCHandle.Alloc(param, GCHandleType.Pinned);
                IntPtr hostPtr = handle.AddrOfPinnedObject();
                LastStatus = _driver.cublasDrotm(_blas, n, ptrx.Pointer, incx, ptry.Pointer, incy, hostPtr);
                handle.Free();
            }
            else
                LastStatus = _driver.cublasDrotm(_blas, n, ptrx.Pointer, incx, ptry.Pointer, incy, ptrp.Pointer);  
        }

        #endregion

        #region ROTMG

        public override void ROTMG(ref float d1, ref float d2, ref float x1, ref float y1, float[] param)
        {
            CUDevicePtrEx paramptr = _gpu.TryGetDeviceMemory(param) as CUDevicePtrEx;
            if (paramptr == null)
            {
                GCHandle handle = GCHandle.Alloc(param, GCHandleType.Pinned);
                IntPtr hostPtr = handle.AddrOfPinnedObject();
                LastStatus = _driver.cublasSrotmg(_blas, ref d1, ref d2, ref x1, ref y1, hostPtr);
                handle.Free();
            }
            else
                _driver.cublasSrotmg(_blas, ref d1, ref d2, ref x1, ref y1, paramptr.Pointer);
        }

        public override void ROTMG(ref double d1, ref double d2, ref double x1, ref double y1, double[] param)
        {
            CUDevicePtrEx paramptr = _gpu.TryGetDeviceMemory(param) as CUDevicePtrEx;
            if (paramptr == null)
            {
                GCHandle handle = GCHandle.Alloc(param, GCHandleType.Pinned);
                IntPtr hostPtr = handle.AddrOfPinnedObject();
                LastStatus = _driver.cublasDrotmg(_blas, ref d1, ref d2, ref x1, ref y1, hostPtr);
                handle.Free();
            }
            else
                _driver.cublasDrotmg(_blas, ref d1, ref d2, ref x1, ref y1, paramptr.Pointer);
        }

        public override void ROTMG(float[] d1, float[] d2, float[] x1, float[] y1, float[] param)
        {
            GCHandle d1gc, d2gc, x1gc, y1gc, pagc;
            IntPtr d1ptr = GetIntPtr(d1, out d1gc);
            IntPtr d2ptr = GetIntPtr(d2, out d2gc);
            IntPtr x1ptr = GetIntPtr(x1, out x1gc);
            IntPtr y1ptr = GetIntPtr(y1, out y1gc);
            IntPtr paptr = GetIntPtr(param, out pagc);
            LastStatus = _driver.cublasSrotmg(_blas, d1ptr, d2ptr, x1ptr, y1ptr, paptr);
            FreeGCHandles(d1gc, d2gc, x1gc, y1gc, pagc);
        }

        public override void ROTMG(double[] d1, double[] d2, double[] x1, double[] y1, double[] param)
        {
            GCHandle d1gc, d2gc, x1gc, y1gc, pagc;
            IntPtr d1ptr = GetIntPtr(d1, out d1gc);
            IntPtr d2ptr = GetIntPtr(d2, out d2gc);
            IntPtr x1ptr = GetIntPtr(x1, out x1gc);
            IntPtr y1ptr = GetIntPtr(y1, out y1gc);
            IntPtr paptr = GetIntPtr(param, out pagc);
            LastStatus = _driver.cublasSrotmg(_blas, d1ptr, d2ptr, x1ptr, y1ptr, paptr);
            FreeGCHandles(d1gc, d2gc, x1gc, y1gc, pagc);
        }

        #endregion

        #region SCAL

        public override void SCAL(float[] alpha, float[] vectorx, int n = 0, int rowx = 0, int incx = 1)
        {
            GCHandle agc;
            IntPtr aptr = GetIntPtr(alpha, out agc);
            IntPtr xptr = SetupVector(vectorx, rowx, ref n, ref incx, 4).Pointer;
            LastStatus = _driver.cublasSscal(_blas, n, aptr, xptr, incx);
            FreeGCHandles(agc);
        }

        public override void SCAL(double[] alpha, double[] vectorx, int n = 0, int rowx = 0, int incx = 1)
        {
            GCHandle agc;
            IntPtr aptr = GetIntPtr(alpha, out agc);
            IntPtr xptr = SetupVector(vectorx, rowx, ref n, ref incx, 8).Pointer;
            LastStatus = _driver.cublasDscal(_blas, n, aptr, xptr, incx);
            FreeGCHandles(agc);
        }

        public override void SCAL(ComplexF[] alpha, ComplexF[] vectorx, int n = 0, int rowx = 0, int incx = 1)
        {
            GCHandle agc;
            IntPtr aptr = GetIntPtr(alpha, out agc);
            IntPtr xptr = SetupVector(vectorx, rowx, ref n, ref incx, 8).Pointer;
            LastStatus = _driver.cublasCscal(_blas, n, aptr, xptr, incx);
            FreeGCHandles(agc);
        }

        public override void SCAL(float[] alpha, ComplexF[] vectorx, int n = 0, int rowx = 0, int incx = 1)
        {
            GCHandle agc;
            IntPtr aptr = GetIntPtr(alpha, out agc);
            IntPtr xptr = SetupVector(vectorx, rowx, ref n, ref incx, 8).Pointer;
            LastStatus = _driver.cublasCsscal(_blas, n, aptr, xptr, incx);
            FreeGCHandles(agc);
        }

        public override void SCAL(ComplexD[] alpha, ComplexD[] vectorx, int n = 0, int rowx = 0, int incx = 1)
        {
            GCHandle agc;
            IntPtr aptr = GetIntPtr(alpha, out agc);
            IntPtr xptr = SetupVector(vectorx, rowx, ref n, ref incx, 16).Pointer;
            LastStatus = _driver.cublasZscal(_blas, n, aptr, xptr, incx);
            FreeGCHandles(agc);
        }

        public override void SCAL(double[] alpha, ComplexD[] vectorx, int n = 0, int rowx = 0, int incx = 1)
        {
            GCHandle agc;
            IntPtr aptr = GetIntPtr(alpha, out agc);
            IntPtr xptr = SetupVector(vectorx, rowx, ref n, ref incx, 16).Pointer;
            LastStatus = _driver.cublasZdscal(_blas, n, aptr, xptr, incx);
            FreeGCHandles(agc);
        }

        #endregion

        #region SWAP

        public override void SWAP(float[] vectorx, float[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
        {
            CUdeviceptr ptrx = SetupVector(vectorx, rowx, ref n, ref incx, 4);
            CUdeviceptr ptry = SetupVector(vectory, rowy, ref n, ref incy, 4);
            LastStatus = _driver.cublasSswap(_blas, n, ptrx.Pointer, incx, ptry.Pointer, incy);
        }

        public override void SWAP(double[] vectorx, double[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
        {
            CUdeviceptr ptrx = SetupVector(vectorx, rowx, ref n, ref incx, 8);
            CUdeviceptr ptry = SetupVector(vectory, rowy, ref n, ref incy, 8);
            LastStatus = _driver.cublasDswap(_blas, n, ptrx.Pointer, incx, ptry.Pointer, incy);
        }

        public override void SWAP(ComplexF[] vectorx, ComplexF[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
        {
            CUdeviceptr ptrx = SetupVector(vectorx, rowx, ref n, ref incx, 8);
            CUdeviceptr ptry = SetupVector(vectory, rowy, ref n, ref incy, 8);
            LastStatus = _driver.cublasCswap(_blas, n, ptrx.Pointer, incx, ptry.Pointer, incy);
        }

        public override void SWAP(ComplexD[] vectorx, ComplexD[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
        {
            CUdeviceptr ptrx = SetupVector(vectorx, rowx, ref n, ref incx, 16);
            CUdeviceptr ptry = SetupVector(vectory, rowy, ref n, ref incy, 16);
            LastStatus = _driver.cublasZswap(_blas, n, ptrx.Pointer, incx, ptry.Pointer, incy);
        }

        #endregion
    }
}

//        #region Commented out

//        //        public void CopyToDevice<T>(T[] hostArray, T[] devArray)
////        {
////            CUDevicePtrEx devPtrEx = _gpu.GetDeviceMemory(devArray) as CUDevicePtrEx;
////            int n = hostArray.Length;
////            Type type = typeof(T);
////            int elemSize = Marshal.SizeOf(type);
////            unsafe
////            {
////                GCHandle handle = GCHandle.Alloc(hostArray, GCHandleType.Pinned);
////                CUBLASStatus rc = CUBLASDriver.cublasSetVector(n, elemSize, handle.AddrOfPinnedObject(), 1, devPtrEx.DevPtr, 1);
////                handle.Free();
////                if (rc != CUBLASStatus.Success)
////                    throw new CudafyHostException(rc.ToString());
////            }
////        }

////        public void CopyFromDevice<T>(T[] devArray, T[] hostArray)
////        {
////            CUDevicePtrEx devPtrEx = _gpu.GetDeviceMemory(devArray) as CUDevicePtrEx;
////            int n = hostArray.Length;
////            Type type = typeof(T);
////            int elemSize = Marshal.SizeOf(type);
////            unsafe
////            {
////                GCHandle handle = GCHandle.Alloc(hostArray, GCHandleType.Pinned);
////                CUBLASStatus rc = CUBLASDriver.cublasGetVector(n, elemSize, devPtrEx.DevPtr, 1, handle.AddrOfPinnedObject(), 1);
////                handle.Free();
////                if (rc != CUBLASStatus.Success)
////                    throw new CudafyHostException(rc.ToString());
////            }
////        }
////#warning TODO Allow arrays to differ in size
////        public void CopyToDevice<T>(T[,] hostArray, T[,] devArray)
////        {
////            CUDevicePtrEx devPtrEx = _gpu.GetDeviceMemory(devArray) as CUDevicePtrEx;
////            int rowsHost = hostArray.GetLength(0);
////            int colsHost = hostArray.GetLength(1);
////            int rowsDev = devPtrEx.XSize;
////            int colsDev = devPtrEx.YSize;
////            Type type = typeof(T);
////            unsafe
////            {
////                GCHandle handle = GCHandle.Alloc(hostArray, GCHandleType.Pinned);
////                CUBLASStatus rc = CUBLASDriver.cublasSetMatrix(rowsHost, colsHost, Marshal.SizeOf(typeof(T)), handle.AddrOfPinnedObject(), rowsHost, devPtrEx.DevPtr, rowsDev);
////                handle.Free();
////                if (rc != CUBLASStatus.Success)
////                    throw new CudafyHostException(rc.ToString());
////            }
////        }

////        public void CopyFromDevice<T>(T[,] devArray, T[,] hostArray)
////        {
////            CUDevicePtrEx devPtrEx = _gpu.GetDeviceMemory(devArray) as CUDevicePtrEx;
////            int rowsHost = hostArray.GetLength(0);
////            int colsHost = hostArray.GetLength(1);
////            int rowsDev = devPtrEx.XSize;
////            int colsDev = devPtrEx.YSize;
////            unsafe
////            {
////                GCHandle handle = GCHandle.Alloc(hostArray, GCHandleType.Pinned);
////                CUBLASStatus rc = CUBLASDriver.cublasGetMatrix(rowsDev, colsDev, Marshal.SizeOf(typeof(T)), devPtrEx.DevPtr, rowsDev, handle.AddrOfPinnedObject(), rowsHost);
////                handle.Free();
////                if (rc != CUBLASStatus.Success)
////                    throw new CudafyHostException(rc.ToString());
////            }
////        }


////        public T[] Allocate<T>(T[] x)
////        {
////            return _gpu.Allocate(x);
////        }

////        public T[] Allocate<T>(int x)
////        {
////            return _gpu.Allocate<T>(x);
////        }

////        public T[,] Allocate<T>(int x, int y)
////        {
////            return _gpu.Allocate<T>(x, y);
////        }

////        public void Free(object o)
////        {
////            _gpu.Free(o);
//        //        }

//        #endregion

//        #region IAMAX

//        protected override int IAMAXEx<T>(object vector, int n = 0, int x = 0, int incx = 1)
//        {
//            eDataType type;
//            CUdeviceptr ptr = SetupVector<T>(vector, x, ref n, ref incx, out type);
//            int res = DoIAMAXv2(n, incx, type, ptr);
//            return res;
//        }

//        //public override Tuple<int,int> IAMAX<T>(T[,] devMatrix, int n = 0, int row = 0, int col = 0, bool columnWise = false, int incx = 1)
//        //{
//        //    CUdeviceptr ptr;
//        //    eDataType type;
//        //    int width, height;
//        //    n = Setup2D<T>(devMatrix, n, row, col, columnWise, out ptr, ref incx, out type, out width, out height);
//        //    int res = DoIAMAXv2(n, incx, type, ptr);
//        //    Debug.WriteLine(res);
//        //    Tuple<int, int> tuple = Get2DLocation(res, width, height);
//        //    return tuple;
//        //}

//        //private Tuple<int, int> Get2DLocation(int pos, int width, int height)
//        //{
//        //    pos--;
//        //    int x = pos / height;
//        //    int y = (pos % height);
//        //    return new Tuple<int, int>(x, y);
//        //}

//        //private int DoIAMAX(int n, int incx, eDataType type, CUdeviceptr ptr)
//        //{
//        //    int res = 0;
//        //    if (type == eDataType.C)
//        //        res = CUBLASDriver.cublasIcamax(n, ptr, incx);
//        //    else if (type == eDataType.D)
//        //        res = CUBLASDriver.cublasIdamax(n, ptr, incx);
//        //    else if (type == eDataType.S)
//        //        res = CUBLASDriver.cublasIsamax(n, ptr, incx);
//        //    else if (type == eDataType.Z)
//        //        res = CUBLASDriver.cublasIzamax(n, ptr, incx);
//        //    CheckLastError();
//        //    return res;
//        //}

//        private int DoIAMAXv2(int n, int incx, eDataType type, CUdeviceptr ptr)
//        {
//            int result = 0;
//            if (type == eDataType.C)
//                LastStatus = CUBLASDriverv2.cublasIcamax(_blas, n, ptr.Pointer, incx, ref result);
//            else if (type == eDataType.D)
//                LastStatus = CUBLASDriverv2.cublasIdamax(_blas, n, ptr.Pointer, incx, ref result);
//            else if (type == eDataType.S)
//                LastStatus = CUBLASDriverv2.cublasIsamax(_blas, n, ptr.Pointer, incx, ref result);
//            else if (type == eDataType.Z)
//                LastStatus = CUBLASDriverv2.cublasIzamax(_blas, n, ptr.Pointer, incx, ref result);

//            return result;
//        }

//        #endregion

//        #region IAMIN

//        protected override int IAMINEx<T>(object vector, int n = 0, int row = 0, int incx = 1)
//        {
//            eDataType type;
//            CUdeviceptr ptr = SetupVector<T>(vector, row, ref n, ref incx, out type);
//            int res = DoIAMINv2(n, incx, type, ptr);
//            return res;
//        }

//        //public override int IAMIN<T>(T[,] devMatrix, int n = 0, int row = 0, int col = 0, bool columnWise = true, int incx = 1)
//        //{
//        //    CUdeviceptr ptr;
//        //    eDataType type;
//        //    n = Setup2D<T>(devMatrix, n, row, col, columnWise, out ptr, ref incx, out type);
//        //    int res = DoIAMIN(n, incx, type, ptr);
//        //    return res;
//        //}

//        private int DoIAMIN(int n, int incx, eDataType type, CUdeviceptr ptr)
//        {
//            int res = 0;
//            if (type == eDataType.C)
//                res = CUBLASDriver.cublasIcamin(n, ptr, incx);
//            else if (type == eDataType.D)
//                res = CUBLASDriver.cublasIdamin(n, ptr, incx);
//            else if (type == eDataType.S)
//                res = CUBLASDriver.cublasIsamin(n, ptr, incx);
//            else if (type == eDataType.Z)
//                res = CUBLASDriver.cublasIzamin(n, ptr, incx);
//            CheckLastError();
//            return res;
//        }

//        private int DoIAMINv2(int n, int incx, eDataType type, CUdeviceptr ptr)
//        {
//            int result = 0;
//            if (type == eDataType.C)
//                LastStatus = CUBLASDriverv2.cublasIcamin(_blas, n, ptr.Pointer, incx, ref result);
//            else if (type == eDataType.D)
//                LastStatus = CUBLASDriverv2.cublasIdamin(_blas, n, ptr.Pointer, incx, ref result);
//            else if (type == eDataType.S)
//                LastStatus = CUBLASDriverv2.cublasIsamin(_blas, n, ptr.Pointer, incx, ref result);
//            else if (type == eDataType.Z)
//                LastStatus = CUBLASDriverv2.cublasIzamin(_blas, n, ptr.Pointer, incx, ref result);

//            return result;
//        }

//        #endregion

//        #region ASUM

//        public override float ASUM<T>(float[] vector, int n = 0, int row = 0, int incx = 1)
//        {
//            eDataType type;
//            float result = 0F;
//            CUdeviceptr ptr = SetupVector<T>(vector, row, ref n, ref incx, out type);
//            LastStatus = CUBLASDriverv2.cublasSasum(_blas, n, ptr.Pointer, incx, ref result);
//            return result;
//        }

//        //public override float ASUM<T>(float[,] matrix, int n = 0, int row = 0, int col = 0, bool columnWise = true, int incx = 1)
//        //{
//        //    CUdeviceptr ptr;
//        //    eDataType type;
//        //    n = Setup2D<T>(matrix, n, row, col, columnWise, out ptr, ref incx, out type);
//        //    float res = CUBLASDriver.cublasSasum(n, ptr, incx);
//        //    CheckLastError();
//        //    return res;
//        //}

//        public override double ASUM<T>(double[] vector, int n = 0, int row = 0, int incx = 1)
//        {
//            eDataType type;
//            double result = 0;
//            CUdeviceptr ptr = SetupVector<T>(vector, row, ref n, ref incx, out type);
//            LastStatus = CUBLASDriverv2.cublasDasum(_blas, n, ptr.Pointer, incx, ref result);
//            return result;
//        }

//        //public override double ASUM<T>(double[,] matrix, int n = 0, int row = 0, int col = 0, bool columnWise = true, int incx = 1)
//        //{
//        //    CUdeviceptr ptr;
//        //    eDataType type;
//        //    n = Setup2D<T>(matrix, n, row, col, columnWise, out ptr, ref incx, out type);
//        //    double res = CUBLASDriver.cublasDasum(n, ptr, incx);
//        //    CheckLastError();
//        //    return res;
//        //}

//        public override float ASUM<T>(ComplexF[] vector, int n = 0, int row = 0, int incx = 1)
//        {
//            eDataType type;
//            float result = 0;
//            CUdeviceptr ptr = SetupVector<T>(vector, row, ref n, ref incx, out type);
//            LastStatus = CUBLASDriverv2.cublasScasum(_blas, n, ptr.Pointer, incx, ref result);
//            return result;
//        }

//        //public override float ASUM<T>(ComplexF[,] matrix, int n = 0, int row = 0, int col = 0, bool columnWise = true, int incx = 1)
//        //{
//        //    CUdeviceptr ptr;
//        //    eDataType type;
//        //    n = Setup2D<T>(matrix, n, row, col, columnWise, out ptr, ref incx, out type);
//        //    float res = CUBLASDriver.cublasScasum(n, ptr, incx);
//        //    CheckLastError();
//        //    return res;
//        //}

//        public override double ASUM<T>(ComplexD[] vector, int n = 0, int row = 0, int incx = 1)
//        {
//            eDataType type;
//            double result = 0;
//            CUdeviceptr ptr = SetupVector<T>(vector, row, ref n, ref incx, out type);
//            LastStatus = CUBLASDriverv2.cublasDzasum(_blas, n, ptr.Pointer, incx, ref result);
//            return result;
//        }

//        //public override double ASUM<T>(ComplexD[,] matrix, int n = 0, int row = 0, int col = 0, bool columnWise = true, int incx = 1)
//        //{
//        //    CUdeviceptr ptr;
//        //    eDataType type;
//        //    n = Setup2D<T>(matrix, n, row, col, columnWise, out ptr, ref incx, out type);
//        //    double res = CUBLASDriver.cublasDzasum(n, ptr, incx);
//        //    CheckLastError();
//        //    return res;
//        //}

//        #endregion

//        #region AXPY

//        public override void AXPY<T>(T alpha, T[] vectorx, T[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
//        {
//            eDataType type;
//            CUDevicePtrEx ptrAlphaEx = _gpu.TryGetDeviceMemory(alpha) as CUDevicePtrEx;
//            CUdeviceptr ptrx = SetupVector<T>(vectorx, rowx, ref n, ref incx, out type);
//            CUdeviceptr ptry = SetupVector<T>(vectory, rowy, ref n, ref incy, out type);
//            if(ptrAlphaEx == null)                
//                DoAXPY(n, alpha, incx, incy, type, ptrx, ptry);
//            else
//                DoAXPYEx<T>(n, ptrAlphaEx.DevPtr, incx, incy, type, ptrx, ptry);
//        }

//        //protected override void AXPYEx<T>(object alpha, object vectorx, object vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
//        //{
//        //    eDataType type;
//        //    CUDevicePtrEx ptrAlphaEx = _gpu.GetDeviceMemory(alpha) as CUDevicePtrEx;
//        //    CUdeviceptr ptrx = SetupVector<T>(vectorx, rowx, ref n, ref incx, out type);
//        //    CUdeviceptr ptry = SetupVector<T>(vectory, rowy, ref n, ref incy, out type);
//        //    DoAXPY(n, alpha, incx, incy, type, ptrx, ptry);
//        //}

//        //protected override void AXPY<T>(T alpha, object vectorx, object vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
//        //{
//        //    eDataType type;
//        //    CUdeviceptr ptrx = SetupVector<T>(vectorx, rowx, ref n, ref incx, out type);
//        //    CUdeviceptr ptry = SetupVector<T>(vectory, rowy, ref n, ref incy, out type);
//        //    DoAXPY(n, alpha, incx, incy, type, ptrx, ptry);
//        //}

//        //private void DoAXPYEx<T>(int n, CUdeviceptr alpha, int incx, int incy, eDataType type, CUdeviceptr ptrx, CUdeviceptr ptry)
//        //{
//        //    if (type == eDataType.C)
//        //        CUBLASDriver.cublasCaxpy(n, alpha, ptrx, incx, ptry, incy);
//        //    else if (type == eDataType.D)
//        //        CUBLASDriver.cublasDaxpy(n, alpha, ptrx, incx, ptry, incy);
//        //    else if (type == eDataType.S)
//        //        CUBLASDriver.cublasSaxpy(n, alpha, ptrx, incx, ptry, incy);
//        //    else if (type == eDataType.Z)
//        //        CUBLASDriver.cublasZaxpy(n, alpha, ptrx, incx, ptry, incy);
//        //    CheckLastError();
//        //}

//        private void DoAXPYEx<T>(int n, CUdeviceptr alpha, int incx, int incy, eDataType type, CUdeviceptr ptrx, CUdeviceptr ptry)
//        {

//        }

//        private void DoAXPY<T>(int n, T alpha, int incx, int incy, eDataType type, CUdeviceptr ptrx, CUdeviceptr ptry)
//        {
//            if (type == eDataType.C)
//            {
//                cuFloatComplex cfc = ConvertToFloatComplex(alpha);
//                CUBLASDriverv2.cublasCaxpy(_blas, n, ref cfc, ptrx.Pointer, incx, ptry.Pointer, incy);
//            }
//            else if (type == eDataType.D)
//            {
//                double d = ConvertToDouble(alpha);
//                CUBLASDriverv2.cublasDaxpy(_blas, n, ref d, ptrx.Pointer, incx, ptry.Pointer, incy);
//            }
//            else if (type == eDataType.S)
//            {
//                float f = ConvertToFloat(alpha);
//                CUBLASDriverv2.cublasSaxpy(_blas, n, ref f, ptrx.Pointer, incx, ptry.Pointer, incy);
//            }
//            else if (type == eDataType.Z)
//            {
//                cuDoubleComplex cdd = ConvertToDoubleComplex(alpha);
//                CUBLASDriverv2.cublasZaxpy(_blas, n, ref cdd, ptrx.Pointer, incx, ptry.Pointer, incy);
//            }
//        }



//        #endregion

//        #region COPY

//        protected override void COPY<T>(object vectorx, object vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
//        {
//            eDataType type;
//            CUdeviceptr ptrx = SetupVector<T>(vectorx, rowx, ref n, ref incx, out type);
//            CUdeviceptr ptry = SetupVector<T>(vectory, rowy, ref n, ref incy, out type);
//            DoCOPY<T>(n, incx, incy, type, ptrx, ptry);
//        }

//        private void DoCOPY<T>(int n, int incx, int incy, eDataType type, CUdeviceptr ptrx, CUdeviceptr ptry)
//        {
//            if (type == eDataType.C)
//                CUBLASDriver.cublasCcopy(n, ptrx, incx, ptry, incy);
//            else if (type == eDataType.D)
//                CUBLASDriver.cublasDcopy(n, ptrx, incx, ptry, incy);
//            else if (type == eDataType.S)
//                CUBLASDriver.cublasScopy(n, ptrx, incx, ptry, incy);
//            else if (type == eDataType.Z)
//                CUBLASDriver.cublasZcopy(n, ptrx, incx, ptry, incy);
//            CheckLastError();
//        }

//        #endregion

//        #region DOT

//        public override T DOT<T>(T[] vectorx, T[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
//        {
//            eDataType type;
//            CUdeviceptr ptrx = SetupVector<T>(vectorx, 0, ref n, ref incx, out type);
//            CUdeviceptr ptry = SetupVector<T>(vectory, 0, ref n, ref incy, out type);
//            float res = CUBLASDriver.cublasSdot(n, ptrx, incx, ptry, incy);
//            T result;
//            if (type == eDataType.C)
//                result = (T)(object)CUBLASDriver.cublasCdotu(n, ptrx, incx, ptry, incy);
//            else if (type == eDataType.D)
//                result = (T)(object)CUBLASDriver.cublasDdot(n, ptrx, incx, ptry, incy);
//            else if (type == eDataType.S)
//                result = (T)(object)CUBLASDriver.cublasSdot(n, ptrx, incx, ptry, incy);
//            else if (type == eDataType.Z)
//                result = (T)(object)CUBLASDriver.cublasZdotu(n, ptrx, incx, ptry, incy);
//            else
//                throw new NotSupportedException(typeof(T).Name);
//            CheckLastError();
//            return result;
//        }

//        public override T DOTC<T>(T[] vectorx, T[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
//        {
//            eDataType type;
//            CUdeviceptr ptrx = SetupVector<T>(vectorx, 0, ref n, ref incx, out type);
//            CUdeviceptr ptry = SetupVector<T>(vectory, 0, ref n, ref incy, out type);
//            T result;
//            if (type == eDataType.C)
//                result = (T)(object)CUBLASDriver.cublasCdotc(n, ptrx, incx, ptry, incy);
//            else if (type == eDataType.D)
//                result = (T)(object)CUBLASDriver.cublasDdot(n, ptrx, incx, ptry, incy);
//            else if (type == eDataType.S)
//                result = (T)(object)CUBLASDriver.cublasSdot(n, ptrx, incx, ptry, incy);
//            else if (type == eDataType.Z)
//                result = (T)(object)CUBLASDriver.cublasZdotc(n, ptrx, incx, ptry, incy);
//            else
//                throw new NotSupportedException(typeof(T).Name);
//            CheckLastError();
//            return result;
//        }

//        #endregion

//        #region NRM2

//        public override T NRM2<T>(T[] vectorx, int n = 0, int rowx = 0, int incx = 1)
//        {
//            eDataType type;
//            CUdeviceptr ptrx = SetupVector<T>(vectorx, 0, ref n, ref incx, out type);
//            T result;
//            if (type == eDataType.S)
//                result = (T)(object)CUBLASDriver.cublasSnrm2(n, ptrx, incx);            
//            if (type == eDataType.C)
//                result = (T)(object)CUBLASDriver.cublasScnrm2(n, ptrx, incx);
//            else if (type == eDataType.D)
//                result = (T)(object)CUBLASDriver.cublasDnrm2(n, ptrx, incx);
//            else if (type == eDataType.Z)
//                result = (T)(object)CUBLASDriver.cublasDznrm2(n, ptrx, incx);
//            else
//                throw new NotSupportedException(typeof(T).Name);
//            CheckLastError();
//            return result;
//        }


//        #endregion

//        #region ROT

//        public override void ROT(float[] vectorx, float[] vectory, float sc, float ss, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 0)
//        {
//            eDataType type;
//            CUdeviceptr ptrx = SetupVector<float>(vectorx, 0, ref n, ref incx, out type);
//            CUdeviceptr ptry = SetupVector<float>(vectory, 0, ref n, ref incy, out type);
//            CUBLASDriver.cublasSrot(n, ptrx, incx, ptry, incy, sc, ss);
//            CheckLastError();
//        }

//        public override void ROT(double[] vectorx, double[] vectory, double sc, double ss, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 0)
//        {
//            eDataType type;
//            CUdeviceptr ptrx = SetupVector<double>(vectorx, 0, ref n, ref incx, out type);
//            CUdeviceptr ptry = SetupVector<double>(vectory, 0, ref n, ref incy, out type);
//            CUBLASDriver.cublasDrot(n, ptrx, incx, ptry, incy, sc, ss);
//            CheckLastError();
//        }

//        public override void ROT(ComplexF[] vectorx, ComplexF[] vectory, float sc, ComplexF ss, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 0)
//        {
//            eDataType type;
//            CUdeviceptr ptrx = SetupVector<ComplexF>(vectorx, 0, ref n, ref incx, out type);
//            CUdeviceptr ptry = SetupVector<ComplexF>(vectory, 0, ref n, ref incy, out type);
//            CUBLASDriver.cublasCrot(n, ptrx, incx, ptry, incy, sc, ConvertToFloatComplex(ss));
//            CheckLastError();
//        }

//        public override void ROT(ComplexD[] vectorx, ComplexD[] vectory, float sc, ComplexD cs, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 0)
//        {
//            eDataType type;
//            CUdeviceptr ptrx = SetupVector<ComplexD>(vectorx, 0, ref n, ref incx, out type);
//            CUdeviceptr ptry = SetupVector<ComplexD>(vectory, 0, ref n, ref incy, out type);
//            CUBLASDriver.cublasZrot(n, ptrx, incx, ptry, incy, sc, ConvertToDoubleComplex(cs));
//            CheckLastError();
//        }


//        #endregion

//        #region ROTM

//        public override void ROTM(float[] vectorx, float[] vectory, float[] sparam, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 0)
//        {
//            eDataType type;
//            CUdeviceptr ptrx = SetupVector<float>(vectorx, 0, ref n, ref incx, out type);
//            CUdeviceptr ptry = SetupVector<float>(vectory, 0, ref n, ref incy, out type);
//            int len = 5;
//            int inc = 1;
//            CUdeviceptr ptr = SetupVector<float>(sparam, 0, ref len, ref inc, out type);
//            CUBLASDriver.cublasSrotm(n, ptrx, incx, ptry, incy, ptr);
//            CheckLastError();
//        }

//        public override void ROTM(double[] vectorx, double[] vectory, double[] sparam, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 0)
//        {
//            eDataType type;
//            CUdeviceptr ptrx = SetupVector<double>(vectorx, 0, ref n, ref incx, out type);
//            CUdeviceptr ptry = SetupVector<double>(vectory, 0, ref n, ref incy, out type);
//            int len = 5;
//            int inc = 1;
//            CUdeviceptr ptr = SetupVector<double>(sparam, 0, ref len, ref inc, out type);
//            CUBLASDriver.cublasDrotm(n, ptrx, incx, ptry, incy, ptr);
//            CheckLastError();
//        } 

//        #endregion


//        #region ROTG
//#warning TODO ROTG
//        //public override void ROTG(ref float host_sa, ref float host_sb, ref float host_sc, ref float host_ss)
//        //{

//        //    CUBLASDriver.cublasSrotg(

//        //}

//        //public override void ROTG(double[] host_da, double[] host_db, double[] host_dc, double[] host_ds)
//        //{
//        //    throw new NotImplementedException();
//        //}

//        //public override void ROTG(ComplexF[] host_ca, ComplexF[] host_cb, float[] host_sc, float[] host_ss)
//        //{
//        //    throw new NotImplementedException();
//        //}

//        //public override void ROTG(ComplexD[] host_ca, ComplexD[] host_cb, double[] host_dc, double[] host_ds)
//        //{
//        //    throw new NotImplementedException();
//        //}

//        #endregion

//        #region SCAL

//        public override void SCALEx<T>(T alpha, object vector, int n = 0, int row = 0, int incx = 1)
//        {
//            eDataType type;
//            Console.WriteLine(incx);
//            CUdeviceptr ptr = SetupVector<T>(vector, row, ref n, ref incx, out type);
//            DoSCAL(n, alpha, incx, type, ptr);
//        }

//        //public override void SCAL<T>(T alpha, T[,] devMatrix, int n = 0, int row = 0, int y = 0, bool columnWise = true, int incx = 1)
//        //{
//        //    CUdeviceptr ptr;
//        //    eDataType type;
//        //    n = Setup2D<T>(devMatrix, n, row, y, columnWise, out ptr, ref incx, out type);
//        //    DoSCAL(n, alpha, incx, type, ptr);
//        //}

//        private void DoSCAL<T>(int n, T alpha, int incx, eDataType type, CUdeviceptr ptr)
//        {
//            if (type == eDataType.C)
//                CUBLASDriver.cublasCscal(n, new cuFloatComplex() { real = (((ComplexF)(object)alpha)).x, imag = (((ComplexF)(object)alpha)).y }, ptr, incx);
//            else if (type == eDataType.D)
//                CUBLASDriver.cublasDscal(n, (double)(object)alpha, ptr, incx);
//            else if (type == eDataType.S)
//                CUBLASDriver.cublasSscal(n, (float)(object)alpha, ptr, incx);
//            else if (type == eDataType.Z)
//                CUBLASDriver.cublasZscal(n, new cuDoubleComplex() { real = (((ComplexD)(object)alpha)).x, imag = (((ComplexD)(object)alpha)).y }, ptr, incx);
//            CheckLastError();
//        }

//        #endregion

//        #region Swap

//        public override void SWAP<T>(T[] vectorx, T[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
//        {
//            eDataType type;
//            CUdeviceptr ptrx = SetupVector<T>(vectorx, rowx, ref n, ref incx, out type);
//            CUdeviceptr ptry = SetupVector<T>(vectory, rowy, ref n, ref incy, out type);
//            DoSWAP(n, incx, incy, type, ptrx, ptry);
//        }

//        private void DoSWAP(int n, int incx, int incy, eDataType type, CUdeviceptr ptrx, CUdeviceptr ptry)
//        {
//            if (type == eDataType.C)
//                CUBLASDriver.cublasCswap(n, ptrx, incx, ptry, incy);
//            else if (type == eDataType.D)
//                CUBLASDriver.cublasDswap(n, ptrx, incx, ptry, incy);
//            else if (type == eDataType.S)
//                CUBLASDriver.cublasSswap(n, ptrx, incx, ptry, incy); 
//            else if (type == eDataType.Z)
//                CUBLASDriver.cublasZswap(n, ptrx, incx, ptry, incy);
//            CheckLastError();
//        }

//        #endregion
