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
using Cudafy.Host;
using Cudafy.Types;
namespace Cudafy.Maths.BLAS
{
    /// <summary>
    /// Not implemented.
    /// </summary>
    internal class HostBLAS : GPGPUBLAS
    {
        internal HostBLAS(GPGPU gpu)
        {
            _gpu = gpu;
        }

        private GPGPU _gpu;
        
        protected override void Shutdown()
        {
            throw new NotImplementedException();
        }

        ////public override double ASUM<T>(Types.ComplexD[,] devMatrix, int n = 0, int row = 0, int col = 0, bool columnWise = true, int incx = 1)
        ////{
        ////    throw new NotImplementedException();
        ////}

        //public override double ASUM<T>(Types.ComplexD[] vector, int n = 0, int row = 0, int incx = 1)
        //{
        //    throw new NotImplementedException();
        //}

        ////public override float ASUM<T>(Types.ComplexF[,] devMatrix, int n = 0, int row = 0, int col = 0, bool columnWise = true, int incx = 1)
        ////{
        ////    throw new NotImplementedException();
        ////}

        //public override float ASUM<T>(Types.ComplexF[] vector, int n = 0, int row = 0, int incx = 1)
        //{
        //    throw new NotImplementedException();
        //}

        ////public override double ASUM<T>(double[,] devMatrix, int n = 0, int row = 0, int col = 0, bool columnWise = true, int incx = 1)
        ////{
        ////    throw new NotImplementedException();
        ////}

        //public override double ASUM<T>(double[] vector, int n = 0, int row = 0, int incx = 1)
        //{
        //    throw new NotImplementedException();
        //}

        ////public override float ASUM<T>(float[,] devMatrix, int n = 0, int row = 0, int col = 0, bool columnWise = true, int incx = 1)
        ////{
        ////    throw new NotImplementedException();
        ////}

        //public override float ASUM<T>(float[] vector, int n = 0, int row = 0, int incx = 1)
        //{
        //    throw new NotImplementedException();
        //}

        ////protected override void AXPYEx<T>(object alpha, object vectorx, object vectory, int n = 0, int row = 0, int incx = 1, int y = 0, int incy = 1)
        ////{
        ////    throw new NotImplementedException();
        ////}

        //public override void AXPY<T>(T alpha, T[] vectorx, T[] vectory, int n = 0, int row = 0, int incx = 1, int y = 0, int incy = 1)
        //{
        //    throw new NotImplementedException();
        //}

        //protected override void COPY<T>(object src, object dst, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
        //{
        //    throw new NotImplementedException();
        //}

        //public override T DOT<T>(T[] vectorx, T[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
        //{
        //    throw new NotImplementedException();
        //}

        //public override T NRM2<T>(T[] vectorx, int n = 0, int rowx = 0, int incx = 1)
        //{
        //    throw new NotImplementedException();
        //}

        //protected override int IAMAXEx<T>(object devArray, int n = 0, int row = 0, int incx = 1)
        //{
        //    throw new NotImplementedException();
        //}

        ////public override Tuple<int,int> IAMAX<T>(T[,] devMatrix, int n = 0, int row = 0, int col = 0, bool columnWise = true, int incx = 1)
        ////{
        ////    throw new NotImplementedException();
        ////}

        //protected override int IAMINEx<T>(object devArray, int n = 0, int row = 0, int incx = 1)
        //{
        //    throw new NotImplementedException();
        //}

        ////public override int IAMIN<T>(T[,] devMatrix, int n = 0, int row = 0, int col = 0, bool columnWise = true, int incx = 1)
        ////{
        ////    throw new NotImplementedException();
        ////}

        //public override void SCALEx<T>(T alpha, object vector, int n = 0, int row = 0, int incx = 1)
        //{
        //    throw new NotImplementedException();
        //}

        ////public override void SCAL<T>(T alpha, T[,] devMatrix, int n = 0, int row = 0, int col = 0, bool columnWise = true, int incx = 1)
        ////{
        ////    throw new NotImplementedException();
        ////}

        //public override T DOTC<T>(T[] vectorx, T[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
        //{
        //    throw new NotImplementedException();
        //}

        //public override void ROT(float[] vectorx, float[] vectory, float sc, float ss, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 0)
        //{
        //    throw new NotImplementedException();
        //}

        //public override void ROT(double[] vectorx, double[] vectory, double sc, double ss, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 0)
        //{
        //    throw new NotImplementedException();
        //}

        //public override void ROT(Types.ComplexF[] vectorx, Types.ComplexF[] vectory, float sc, Types.ComplexF ss, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 0)
        //{
        //    throw new NotImplementedException();
        //}

        //public override void ROT(Types.ComplexD[] vectorx, Types.ComplexD[] vectory, float sc, Types.ComplexD cs, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 0)
        //{
        //    throw new NotImplementedException();
        //}


        //public override void ROTM(float[] vectorx, float[] vectory, float[] sparam, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 0)
        //{
        //    throw new NotImplementedException();
        //}

        //public override void ROTM(double[] vectorx, double[] vectory, double[] sparam, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 0)
        //{
        //    throw new NotImplementedException();
        //}

        //public override void SWAP<T>(T[] vectorx, T[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 0)
        //{
        //    throw new NotImplementedException();
        //}

        private IntPtr SetupVector<T>(object vector, int x, ref int n, ref int incx, out eDataType type)
        {
            EmuDevicePtrEx ptrEx = _gpu.GetDeviceMemory(vector) as EmuDevicePtrEx;
            if (incx == 0)
                throw new CudafyHostException(CudafyHostException.csX_NOT_SET, "incx");
            n = (n == 0 ? ptrEx.TotalSize / incx : n);
            type = GetDataType<T>();
            int size = Marshal.SizeOf(typeof(T));
            IntPtr ptr = ptrEx.GetDevPtrPtr(size * x);// DevPtr + (uint)(size * x);
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
                throw new CudafyHostException(CudafyHostException.csX_NOT_SUPPORTED, typeof(T).Name);
            return type;
        }


        public override int IAMAX(float[] vector, int n = 0, int row = 0, int incx = 1)
        {
            throw new NotImplementedException();
        }

        public override int IAMAX(double[] vector, int n = 0, int row = 0, int incx = 1)
        {
            throw new NotImplementedException();
        }

        public override int IAMAX(ComplexF[] vector, int n = 0, int row = 0, int incx = 1)
        {
            throw new NotImplementedException();
        }

        public override int IAMAX(ComplexD[] vector, int n = 0, int row = 0, int incx = 1)
        {
            throw new NotImplementedException();
        }

        public override int IAMIN(float[] vector, int n = 0, int row = 0, int incx = 1)
        {
            throw new NotImplementedException();
        }

        public override int IAMIN(double[] vector, int n = 0, int row = 0, int incx = 1)
        {
            throw new NotImplementedException();
        }

        public override int IAMIN(ComplexF[] vector, int n = 0, int row = 0, int incx = 1)
        {
            throw new NotImplementedException();
        }

        public override int IAMIN(ComplexD[] vector, int n = 0, int row = 0, int incx = 1)
        {
            throw new NotImplementedException();
        }

        public override float ASUM(float[] vector, int n = 0, int row = 0, int incx = 1)
        {
            throw new NotImplementedException();
        }

        public override double ASUM(double[] vector, int n = 0, int row = 0, int incx = 1)
        {
            throw new NotImplementedException();
        }

        public override float ASUM(ComplexF[] vector, int n = 0, int row = 0, int incx = 1)
        {
            throw new NotImplementedException();
        }

        public override double ASUM(ComplexD[] vector, int n = 0, int row = 0, int incx = 1)
        {
            throw new NotImplementedException();
        }

        protected override void AXPY(float[] alpha, float[] vectorx, float[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
        {
            throw new NotImplementedException();
        }

        protected override void AXPY(double[] alpha, double[] vectorx, double[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
        {
            throw new NotImplementedException();
        }

        protected override void AXPY(ComplexF[] alpha, ComplexD[] vectorx, ComplexD[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
        {
            throw new NotImplementedException();
        }

        protected override void AXPY(ComplexD[] alpha, ComplexD[] vectorx, ComplexD[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
        {
            throw new NotImplementedException();
        }

        public override void COPY(float[] vectorx, float[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
        {
            throw new NotImplementedException();
        }

        public override void COPY(double[] vectorx, double[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
        {
            throw new NotImplementedException();
        }

        public override void COPY(ComplexF[] vectorx, ComplexF[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
        {
            throw new NotImplementedException();
        }

        public override void COPY(ComplexD[] vectorx, ComplexD[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
        {
            throw new NotImplementedException();
        }

        public override float DOT(float[] vectorx, float[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
        {
            throw new NotImplementedException();
        }

        public override double DOT(double[] vectorx, double[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
        {
            throw new NotImplementedException();
        }

        public override ComplexF DOTU(ComplexF[] vectorx, ComplexF[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
        {
            throw new NotImplementedException();
        }

        public override ComplexF DOTC(ComplexF[] vectorx, ComplexF[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
        {
            throw new NotImplementedException();
        }

        public override ComplexD DOTU(ComplexD[] vectorx, ComplexD[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
        {
            throw new NotImplementedException();
        }

        public override ComplexD DOTC(ComplexD[] vectorx, ComplexD[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
        {
            throw new NotImplementedException();
        }

        public override float NRM2(float[] vectorx, int n = 0, int rowx = 0, int incx = 1)
        {
            throw new NotImplementedException();
        }

        public override double NRM2(double[] vectorx, int n = 0, int rowx = 0, int incx = 1)
        {
            throw new NotImplementedException();
        }

        public override float NRM2(ComplexF[] vectorx, int n = 0, int rowx = 0, int incx = 1)
        {
            throw new NotImplementedException();
        }

        public override double NRM2(ComplexD[] vectorx, int n = 0, int rowx = 0, int incx = 1)
        {
            throw new NotImplementedException();
        }


        public override void ROT(float[] vectorx, float[] vectory, float[] c, float[] s, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
        {
            throw new NotImplementedException();
        }

        public override void ROT(double[] vectorx, double[] vectory, double[] c, double[] s, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
        {
            throw new NotImplementedException();
        }

        public override void ROT(ComplexF[] vectorx, ComplexF[] vectory, float[] c, ComplexF[] s, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
        {
            throw new NotImplementedException();
        }

        public override void ROT(ComplexF[] vectorx, ComplexF[] vectory, float[] c, float[] s, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
        {
            throw new NotImplementedException();
        }

        public override void ROT(ComplexD[] vectorx, ComplexD[] vectory, double[] c, ComplexD[] s, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
        {
            throw new NotImplementedException();
        }

        public override void ROT(ComplexD[] vectorx, ComplexD[] vectory, double[] c, double[] s, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
        {
            throw new NotImplementedException();
        }

        public override void ROTG(float[] a, float[] b, float[] c, float[] s)
        {
            throw new NotImplementedException();
        }

        public override void ROTG(double[] a, double[] b, double[] c, double[] s)
        {
            throw new NotImplementedException();
        }

        public override void ROTG(ComplexF[] a, ComplexF[] b, float[] c, ComplexF[] s)
        {
            throw new NotImplementedException();
        }

        public override void ROTG(ComplexD[] a, ComplexD[] b, double[] c, ComplexD[] s)
        {
            throw new NotImplementedException();
        }

        public override void ROTM(float[] vectorx, float[] vectory, float[] param, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
        {
            throw new NotImplementedException();
        }

        public override void ROTM(double[] vectorx, double[] vectory, double[] param, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
        {
            throw new NotImplementedException();
        }

        public override void ROTMG(ref float d1, ref float d2, ref float x1, ref float y1, float[] param)
        {
            throw new NotImplementedException();
        }

        public override void ROTMG(ref double d1, ref double d2, ref double x1, ref double y1, double[] param)
        {
            throw new NotImplementedException();
        }

        public override void ROTMG(float[] d1, float[] d2, float[] x1, float[] y1, float[] param)
        {
            throw new NotImplementedException();
        }

        public override void ROTMG(double[] d1, double[] d2, double[] x1, double[] y1, double[] param)
        {
            throw new NotImplementedException();
        }

        public override void SCAL(float[] alpha, float[] vectorx, int n = 0, int rowx = 0, int incx = 1)
        {
            throw new NotImplementedException();
        }

        public override void SCAL(double[] alpha, double[] vectorx, int n = 0, int rowx = 0, int incx = 1)
        {
            throw new NotImplementedException();
        }

        public override void SCAL(ComplexF[] alpha, ComplexF[] vectorx, int n = 0, int rowx = 0, int incx = 1)
        {
            throw new NotImplementedException();
        }

        public override void SCAL(float[] alpha, ComplexF[] vectorx, int n = 0, int rowx = 0, int incx = 1)
        {
            throw new NotImplementedException();
        }

        public override void SCAL(ComplexD[] alpha, ComplexD[] vectorx, int n = 0, int rowx = 0, int incx = 1)
        {
            throw new NotImplementedException();
        }

        public override void SCAL(double[] alpha, ComplexD[] vectorx, int n = 0, int rowx = 0, int incx = 1)
        {
            throw new NotImplementedException();
        }

        public override void SWAP(float[] vectorx, float[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
        {
            throw new NotImplementedException();
        }

        public override void SWAP(double[] vectorx, double[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
        {
            throw new NotImplementedException();
        }

        public override void SWAP(ComplexF[] vectorx, ComplexF[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
        {
            throw new NotImplementedException();
        }

        public override void SWAP(ComplexD[] vectorx, ComplexD[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
        {
            throw new NotImplementedException();
        }
    }
}
