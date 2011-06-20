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
using System.Diagnostics;
using Cudafy.Host;
using Cudafy.Types;

using GASS.CUDA.BLAS;


namespace Cudafy.Maths.BLAS
{
    internal enum eDataType { S, C, D, Z };

    /// <summary>
    /// Abstract base class for devices supporting BLAS.
    /// Warning: This code has received limited testing.
    /// </summary>
    public abstract class GPGPUBLAS : IDisposable
    {
        /// <summary>
        /// Creates a BLAS wrapper based on the specified gpu.
        /// </summary>
        /// <param name="gpu">The gpu.</param>
        /// <returns></returns>
        public static GPGPUBLAS Create(GPGPU gpu)
        {
            if (gpu is CudaGPU)
                return new CudaBLAS(gpu);
            else
                return new HostBLAS(gpu);
                //throw new NotImplementedException(gpu.ToString());
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="GPGPUBLAS"/> class.
        /// </summary>
        protected GPGPUBLAS()
        {
            _lock = new object();
        }

        /// <summary>
        /// Releases unmanaged resources and performs other cleanup operations before the
        /// <see cref="GPGPUBLAS"/> is reclaimed by garbage collection.
        /// </summary>
        ~GPGPUBLAS()
        {
            Dispose(false);
        }

        private object _lock;

        // Track whether Dispose has been called.
        private bool _disposed = false;

        /// <summary>
        /// Gets a value indicating whether this instance is disposed.
        /// </summary>
        /// <value>
        /// 	<c>true</c> if this instance is disposed; otherwise, <c>false</c>.
        /// </value>
        public bool IsDisposed
        {
            get { lock (_lock) { return _disposed; } }
        }

        /// <summary>
        /// Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged resources.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            // This object will be cleaned up by the Dispose method.
            // Therefore, you should call GC.SupressFinalize to
            // take this object off the finalization queue
            // and prevent finalization code for this object
            // from executing a second time.
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Shutdowns this instance.
        /// </summary>
        protected abstract void Shutdown();

        /// <summary>
        /// Releases unmanaged and - optionally - managed resources
        /// </summary>
        /// <param name="disposing"><c>true</c> to release both managed and unmanaged resources; <c>false</c> to release only unmanaged resources.</param>
        protected virtual void Dispose(bool disposing)
        {
            lock (_lock)
            {
                Debug.WriteLine(string.Format("GPGPUBLAS::Dispose({0})", disposing));
                // Check to see if Dispose has already been called.
                if (!this._disposed)
                {
                    Debug.WriteLine("Disposing");
                    // If disposing equals true, dispose all managed
                    // and unmanaged resources.
                    if (disposing)
                    {
                        // Dispose managed resources.
                    }

                    // Call the appropriate methods to clean up
                    // unmanaged resources here.
                    // If disposing is false,
                    // only the following code is executed.
                    Shutdown();

                    // Note disposing has been done.
                    _disposed = true;

                }
                else
                    Debug.WriteLine("Already disposed");
            }
        }

        #region Max

        /// <summary>
        /// IAMAXs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <returns></returns>
        public abstract int IAMAX(float[] vectorx, int n = 0, int rowx = 0, int incx = 1);

        /// <summary>
        /// IAMAXs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <returns></returns>
        public abstract int IAMAX(double[] vectorx, int n = 0, int rowx = 0, int incx = 1);

        /// <summary>
        /// IAMAXs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <returns></returns>
        public abstract int IAMAX(ComplexF[] vectorx, int n = 0, int rowx = 0, int incx = 1);

        /// <summary>
        /// IAMAXs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <returns></returns>
        public abstract int IAMAX(ComplexD[] vectorx, int n = 0, int rowx = 0, int incx = 1);

        #endregion

        #region Min

        /// <summary>
        /// IAMINs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <returns></returns>
        public abstract int IAMIN(float[] vectorx, int n = 0, int rowx = 0, int incx = 1);

        /// <summary>
        /// IAMINs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <returns></returns>
        public abstract int IAMIN(double[] vectorx, int n = 0, int rowx = 0, int incx = 1);

        /// <summary>
        /// IAMINs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <returns></returns>
        public abstract int IAMIN(ComplexF[] vectorx, int n = 0, int rowx = 0, int incx = 1);

        /// <summary>
        /// IAMINs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <returns></returns>
        public abstract int IAMIN(ComplexD[] vectorx, int n = 0, int rowx = 0, int incx = 1);

        #endregion

        #region Sum

        /// <summary>
        /// ASUMs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <returns></returns>
        public abstract float ASUM(float[] vectorx, int n = 0, int rowx = 0, int incx = 1);

        /// <summary>
        /// ASUMs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <returns></returns>
        public abstract double ASUM(double[] vectorx, int n = 0, int rowx = 0, int incx = 1);

        /// <summary>
        /// ASUMs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <returns></returns>
        public abstract float ASUM(ComplexF[] vectorx, int n = 0, int rowx = 0, int incx = 1);

        /// <summary>
        /// ASUMs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <returns></returns>
        public abstract double ASUM(ComplexD[] vectorx, int n = 0, int rowx = 0, int incx = 1);
                   
        #endregion

        #region AXPY

        /// <summary>
        /// AXPYs the specified alpha.
        /// </summary>
        /// <param name="alpha">The alpha.</param>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="vectory">The vectory.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <param name="rowy">The rowy.</param>
        /// <param name="incy">The incy.</param>
        public void AXPY(float alpha, float[] vectorx, float[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
        {
            AXPY(new [] { alpha }, vectorx, vectory, n, rowx, incx, rowy, incy);
        }

        /// <summary>
        /// AXPYs the specified alpha.
        /// </summary>
        /// <param name="alpha">The alpha.</param>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="vectory">The vectory.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <param name="rowy">The rowy.</param>
        /// <param name="incy">The incy.</param>
        public void AXPY(double alpha, double[] vectorx, double[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
        {
            AXPY(new [] { alpha }, vectorx, vectory, n, rowx, incx, rowy, incy);
        }

        /// <summary>
        /// AXPYs the specified alpha.
        /// </summary>
        /// <param name="alpha">The alpha.</param>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="vectory">The vectory.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <param name="rowy">The rowy.</param>
        /// <param name="incy">The incy.</param>
        public void AXPY(ComplexF alpha, ComplexD[] vectorx, ComplexD[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
        {
            AXPY(new [] { alpha }, vectorx, vectory, n, rowx, incx, rowy, incy);
        }

        /// <summary>
        /// AXPYs the specified alpha.
        /// </summary>
        /// <param name="alpha">The alpha.</param>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="vectory">The vectory.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <param name="rowy">The rowy.</param>
        /// <param name="incy">The incy.</param>
        public void AXPY(ComplexD alpha, ComplexD[] vectorx, ComplexD[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
        {
            AXPY(new[] { alpha }, vectorx, vectory, n, rowx, incx, rowy, incy);
        }

        /// <summary>
        /// AXPYs the specified alpha.
        /// </summary>
        /// <param name="alpha">The alpha.</param>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="vectory">The vectory.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <param name="rowy">The rowy.</param>
        /// <param name="incy">The incy.</param>
        protected abstract void AXPY(float[] alpha, float[] vectorx, float[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1);

        /// <summary>
        /// AXPYs the specified alpha.
        /// </summary>
        /// <param name="alpha">The alpha.</param>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="vectory">The vectory.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <param name="rowy">The rowy.</param>
        /// <param name="incy">The incy.</param>
        protected abstract void AXPY(double[] alpha, double[] vectorx, double[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1);

        /// <summary>
        /// AXPYs the specified alpha.
        /// </summary>
        /// <param name="alpha">The alpha.</param>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="vectory">The vectory.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <param name="rowy">The rowy.</param>
        /// <param name="incy">The incy.</param>
        protected abstract void AXPY(ComplexF[] alpha, ComplexD[] vectorx, ComplexD[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1);

        /// <summary>
        /// AXPYs the specified alpha.
        /// </summary>
        /// <param name="alpha">The alpha.</param>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="vectory">The vectory.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <param name="rowy">The rowy.</param>
        /// <param name="incy">The incy.</param>
        protected abstract void AXPY(ComplexD[] alpha, ComplexD[] vectorx, ComplexD[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1);

        #endregion

        #region Copy

        /// <summary>
        /// COPYs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="vectory">The vectory.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <param name="rowy">The rowy.</param>
        /// <param name="incy">The incy.</param>
        public abstract void COPY(float[] vectorx, float[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1);

        /// <summary>
        /// COPYs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="vectory">The vectory.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <param name="rowy">The rowy.</param>
        /// <param name="incy">The incy.</param>
        public abstract void COPY(double[] vectorx, double[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1);

        /// <summary>
        /// COPYs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="vectory">The vectory.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <param name="rowy">The rowy.</param>
        /// <param name="incy">The incy.</param>
        public abstract void COPY(ComplexF[] vectorx, ComplexF[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1);

        /// <summary>
        /// COPYs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="vectory">The vectory.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <param name="rowy">The rowy.</param>
        /// <param name="incy">The incy.</param>
        public abstract void COPY(ComplexD[] vectorx, ComplexD[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1);

        #endregion

        #region Dot

        /// <summary>
        /// DOTs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="vectory">The vectory.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <param name="rowy">The rowy.</param>
        /// <param name="incy">The incy.</param>
        /// <returns></returns>
        public abstract float DOT(float[] vectorx, float[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1);

        /// <summary>
        /// DOTs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="vectory">The vectory.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <param name="rowy">The rowy.</param>
        /// <param name="incy">The incy.</param>
        /// <returns></returns>
        public abstract double DOT(double[] vectorx, double[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1);

        /// <summary>
        /// DOTUs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="vectory">The vectory.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <param name="rowy">The rowy.</param>
        /// <param name="incy">The incy.</param>
        /// <returns></returns>
        public abstract ComplexF DOTU(ComplexF[] vectorx, ComplexF[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1);

        /// <summary>
        /// DOTCs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="vectory">The vectory.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <param name="rowy">The rowy.</param>
        /// <param name="incy">The incy.</param>
        /// <returns></returns>
        public abstract ComplexF DOTC(ComplexF[] vectorx, ComplexF[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1);

        /// <summary>
        /// DOTUs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="vectory">The vectory.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <param name="rowy">The rowy.</param>
        /// <param name="incy">The incy.</param>
        /// <returns></returns>
        public abstract ComplexD DOTU(ComplexD[] vectorx, ComplexD[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1);

        /// <summary>
        /// DOTCs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="vectory">The vectory.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <param name="rowy">The rowy.</param>
        /// <param name="incy">The incy.</param>
        /// <returns></returns>
        public abstract ComplexD DOTC(ComplexD[] vectorx, ComplexD[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1);

        #endregion

        #region NRM2

        /// <summary>
        /// NRs the m2.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <returns></returns>
        public abstract float NRM2(float[] vectorx, int n = 0, int rowx = 0, int incx = 1);

        /// <summary>
        /// NRs the m2.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <returns></returns>
        public abstract double NRM2(double[] vectorx, int n = 0, int rowx = 0, int incx = 1);

        /// <summary>
        /// NRs the m2.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <returns></returns>
        public abstract float NRM2(ComplexF[] vectorx, int n = 0, int rowx = 0, int incx = 1);

        /// <summary>
        /// NRs the m2.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <returns></returns>
        public abstract double NRM2(ComplexD[] vectorx, int n = 0, int rowx = 0, int incx = 1);

        #endregion

        #region ROT

        /// <summary>
        /// ROTs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="vectory">The vectory.</param>
        /// <param name="c">The c.</param>
        /// <param name="s">The s.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <param name="rowy">The rowy.</param>
        /// <param name="incy">The incy.</param>
        public void ROT(float[] vectorx, float[] vectory, float c, float s, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
        {
            ROT(vectorx, vectory, new[] { c }, new[] { s }, n, rowx, incx, rowy, incy);
        }

        /// <summary>
        /// ROTs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="vectory">The vectory.</param>
        /// <param name="c">The c.</param>
        /// <param name="s">The s.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <param name="rowy">The rowy.</param>
        /// <param name="incy">The incy.</param>
        public void ROT(double[] vectorx, double[] vectory, double c, double s, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
        {
            ROT(vectorx, vectory, new[] { c }, new[] { s }, n, rowx, incx, rowy, incy);
        }

        /// <summary>
        /// ROTs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="vectory">The vectory.</param>
        /// <param name="c">The c.</param>
        /// <param name="s">The s.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <param name="rowy">The rowy.</param>
        /// <param name="incy">The incy.</param>
        public void ROT(ComplexF[] vectorx, ComplexF[] vectory, float c, ComplexF s, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
        {
            ROT(vectorx, vectory, new[] { c }, new[] { s }, n, rowx, incx, rowy, incy);
        }

        /// <summary>
        /// ROTs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="vectory">The vectory.</param>
        /// <param name="c">The c.</param>
        /// <param name="s">The s.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <param name="rowy">The rowy.</param>
        /// <param name="incy">The incy.</param>
        public void ROT(ComplexF[] vectorx, ComplexF[] vectory, float c, float s, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
        {
            ROT(vectorx, vectory, new[] { c }, new[] { s }, n, rowx, incx, rowy, incy);
        }

        /// <summary>
        /// ROTs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="vectory">The vectory.</param>
        /// <param name="c">The c.</param>
        /// <param name="s">The s.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <param name="rowy">The rowy.</param>
        /// <param name="incy">The incy.</param>
        public void ROT(ComplexD[] vectorx, ComplexD[] vectory, double c, ComplexD s, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
        {
            ROT(vectorx, vectory, new[] { c }, new[] { s }, n, rowx, incx, rowy, incy);
        }

        /// <summary>
        /// ROTs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="vectory">The vectory.</param>
        /// <param name="c">The c.</param>
        /// <param name="s">The s.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <param name="rowy">The rowy.</param>
        /// <param name="incy">The incy.</param>
        public void ROT(ComplexD[] vectorx, ComplexD[] vectory, double c, double s, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
        {
            ROT(vectorx, vectory, new[] { c }, new[] { s }, n, rowx, incx, rowy, incy);
        }

        /// <summary>
        /// ROTs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="vectory">The vectory.</param>
        /// <param name="c">The c.</param>
        /// <param name="s">The s.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <param name="rowy">The rowy.</param>
        /// <param name="incy">The incy.</param>
        public abstract void ROT(float[] vectorx, float[] vectory, float[] c, float[] s, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1);

        /// <summary>
        /// ROTs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="vectory">The vectory.</param>
        /// <param name="c">The c.</param>
        /// <param name="s">The s.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <param name="rowy">The rowy.</param>
        /// <param name="incy">The incy.</param>
        public abstract void ROT(double[] vectorx, double[] vectory, double[] c, double[] s, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1);

        /// <summary>
        /// ROTs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="vectory">The vectory.</param>
        /// <param name="c">The c.</param>
        /// <param name="s">The s.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <param name="rowy">The rowy.</param>
        /// <param name="incy">The incy.</param>
        public abstract void ROT(ComplexF[] vectorx, ComplexF[] vectory, float[] c, ComplexF[] s, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1);

        /// <summary>
        /// ROTs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="vectory">The vectory.</param>
        /// <param name="c">The c.</param>
        /// <param name="s">The s.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <param name="rowy">The rowy.</param>
        /// <param name="incy">The incy.</param>
        public abstract void ROT(ComplexF[] vectorx, ComplexF[] vectory, float[] c, float[] s, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1);

        /// <summary>
        /// ROTs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="vectory">The vectory.</param>
        /// <param name="c">The c.</param>
        /// <param name="s">The s.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <param name="rowy">The rowy.</param>
        /// <param name="incy">The incy.</param>
        public abstract void ROT(ComplexD[] vectorx, ComplexD[] vectory, double[] c, ComplexD[] s, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1);

        /// <summary>
        /// ROTs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="vectory">The vectory.</param>
        /// <param name="c">The c.</param>
        /// <param name="s">The s.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <param name="rowy">The rowy.</param>
        /// <param name="incy">The incy.</param>
        public abstract void ROT(ComplexD[] vectorx, ComplexD[] vectory, double[] c, double[] s, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1);

        #endregion

        #region ROTG

        //public void ROTG(ref float a, ref float b, ref float c, ref float s)
        //{
        //    ROTG(new[] { a }, new[] { b }, new[] { c }, new[] { s });
        //}

        //public void ROTG(ref double a, ref double b, ref double c, ref double s)
        //{
        //    ROTG(new[] { a }, new[] { b }, new[] { c }, new[] { s });
        //}

        //public void ROTG(ref ComplexF a, ref ComplexF b, ref float c, ref ComplexF s)
        //{
        //    ROTG(new[] { a }, new[] { b }, new[] { c }, new[] { s });
        //}

        //public void ROTG(ref ComplexD a, ref ComplexD b, ref double c, ref ComplexD s)
        //{
        //    ROTG(new[] { a }, new[] { b }, new[] { c }, new[] { s });
        //}

        /// <summary>
        /// ROTGs the specified a.
        /// </summary>
        /// <param name="a">A.</param>
        /// <param name="b">The b.</param>
        /// <param name="c">The c.</param>
        /// <param name="s">The s.</param>
        public abstract void ROTG(float[] a, float[] b, float[] c, float[] s);

        /// <summary>
        /// ROTGs the specified a.
        /// </summary>
        /// <param name="a">A.</param>
        /// <param name="b">The b.</param>
        /// <param name="c">The c.</param>
        /// <param name="s">The s.</param>
        public abstract void ROTG(double[] a, double[] b,double[] c, double[] s);

        /// <summary>
        /// ROTGs the specified a.
        /// </summary>
        /// <param name="a">A.</param>
        /// <param name="b">The b.</param>
        /// <param name="c">The c.</param>
        /// <param name="s">The s.</param>
        public abstract void ROTG(ComplexF[] a, ComplexF[] b, float[] c, ComplexF[] s);

        /// <summary>
        /// ROTGs the specified a.
        /// </summary>
        /// <param name="a">A.</param>
        /// <param name="b">The b.</param>
        /// <param name="c">The c.</param>
        /// <param name="s">The s.</param>
        public abstract void ROTG(ComplexD[] a, ComplexD[] b, double[] c, ComplexD[] s);

        #endregion

        #region ROTM

        /// <summary>
        /// ROTMs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="vectory">The vectory.</param>
        /// <param name="param">The param.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <param name="rowy">The rowy.</param>
        /// <param name="incy">The incy.</param>
        public abstract void ROTM(float[] vectorx, float[] vectory, float[] param, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1);

        /// <summary>
        /// ROTMs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="vectory">The vectory.</param>
        /// <param name="param">The param.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <param name="rowy">The rowy.</param>
        /// <param name="incy">The incy.</param>
        public abstract void ROTM(double[] vectorx, double[] vectory, double[] param, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1);

        #endregion

        #region ROTMG

        /// <summary>
        /// ROTMGs the specified d1.
        /// </summary>
        /// <param name="d1">The d1.</param>
        /// <param name="d2">The d2.</param>
        /// <param name="x1">The x1.</param>
        /// <param name="y1">The y1.</param>
        /// <param name="param">The param.</param>
        public abstract void ROTMG(ref float d1, ref float d2, ref float x1, ref float y1, float[] param);

        /// <summary>
        /// ROTMGs the specified d1.
        /// </summary>
        /// <param name="d1">The d1.</param>
        /// <param name="d2">The d2.</param>
        /// <param name="x1">The x1.</param>
        /// <param name="y1">The y1.</param>
        /// <param name="param">The param.</param>
        public abstract void ROTMG(ref double d1, ref double d2, ref double x1, ref double y1, double[] param);

        /// <summary>
        /// ROTMGs the specified d1.
        /// </summary>
        /// <param name="d1">The d1.</param>
        /// <param name="d2">The d2.</param>
        /// <param name="x1">The x1.</param>
        /// <param name="y1">The y1.</param>
        /// <param name="param">The param.</param>
        public abstract void ROTMG(float[] d1, float[] d2, float[] x1, float[] y1, float[] param);

        /// <summary>
        /// ROTMGs the specified d1.
        /// </summary>
        /// <param name="d1">The d1.</param>
        /// <param name="d2">The d2.</param>
        /// <param name="x1">The x1.</param>
        /// <param name="y1">The y1.</param>
        /// <param name="param">The param.</param>
        public abstract void ROTMG(double[] d1, double[] d2, double[] x1, double[] y1, double[] param);

        #endregion

        #region SCAL

        /// <summary>
        /// SCALs the specified alpha.
        /// </summary>
        /// <param name="alpha">The alpha.</param>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        public void SCAL(float alpha, float[] vectorx, int n = 0, int rowx = 0, int incx = 1)
        {
            SCAL(new []{ alpha }, vectorx, n, rowx, incx);
        }

        /// <summary>
        /// SCALs the specified alpha.
        /// </summary>
        /// <param name="alpha">The alpha.</param>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        public void SCAL(double alpha, double[] vectorx, int n = 0, int rowx = 0, int incx = 1)
        {
            SCAL(new []{ alpha }, vectorx, n, rowx, incx);
        }

        /// <summary>
        /// SCALs the specified alpha.
        /// </summary>
        /// <param name="alpha">The alpha.</param>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        public void SCAL(ComplexF alpha, ComplexF[] vectorx, int n = 0, int rowx = 0, int incx = 1)
        {
            SCAL(new []{ alpha }, vectorx, n, rowx, incx);
        }

        /// <summary>
        /// SCALs the specified alpha.
        /// </summary>
        /// <param name="alpha">The alpha.</param>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        public void SCAL(float alpha, ComplexF[] vectorx, int n = 0, int rowx = 0, int incx = 1)
        {
            SCAL(new []{ alpha }, vectorx, n, rowx, incx);
        }

        /// <summary>
        /// SCALs the specified alpha.
        /// </summary>
        /// <param name="alpha">The alpha.</param>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        public void SCAL(ComplexD alpha, ComplexD[] vectorx, int n = 0, int rowx = 0, int incx = 1)
        {
            SCAL(new []{ alpha }, vectorx, n, rowx, incx);
        }

        /// <summary>
        /// SCALs the specified alpha.
        /// </summary>
        /// <param name="alpha">The alpha.</param>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        public void SCAL(double alpha, ComplexD[] vectorx, int n = 0, int rowx = 0, int incx = 1)
        {
            SCAL(new[] { alpha }, vectorx, n, rowx, incx);
        }

        /// <summary>
        /// SCALs the specified alpha.
        /// </summary>
        /// <param name="alpha">The alpha.</param>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        public abstract void SCAL(float[] alpha, float[] vectorx, int n = 0, int rowx = 0, int incx = 1);

        /// <summary>
        /// SCALs the specified alpha.
        /// </summary>
        /// <param name="alpha">The alpha.</param>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        public abstract void SCAL(double[] alpha, double[] vectorx, int n = 0, int rowx = 0, int incx = 1);

        /// <summary>
        /// SCALs the specified alpha.
        /// </summary>
        /// <param name="alpha">The alpha.</param>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        public abstract void SCAL(ComplexF[] alpha, ComplexF[] vectorx, int n = 0, int rowx = 0, int incx = 1);

        /// <summary>
        /// SCALs the specified alpha.
        /// </summary>
        /// <param name="alpha">The alpha.</param>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        public abstract void SCAL(float[] alpha, ComplexF[] vectorx, int n = 0, int rowx = 0, int incx = 1);

        /// <summary>
        /// SCALs the specified alpha.
        /// </summary>
        /// <param name="alpha">The alpha.</param>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        public abstract void SCAL(ComplexD[] alpha, ComplexD[] vectorx, int n = 0, int rowx = 0, int incx = 1);

        /// <summary>
        /// SCALs the specified alpha.
        /// </summary>
        /// <param name="alpha">The alpha.</param>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        public abstract void SCAL(double[] alpha, ComplexD[] vectorx, int n = 0, int rowx = 0, int incx = 1);

        #endregion

        #region SWAP

        /// <summary>
        /// SWAPs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="vectory">The vectory.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <param name="rowy">The rowy.</param>
        /// <param name="incy">The incy.</param>
        public abstract void SWAP(float[] vectorx, float[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1);

        /// <summary>
        /// SWAPs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="vectory">The vectory.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <param name="rowy">The rowy.</param>
        /// <param name="incy">The incy.</param>
        public abstract void SWAP(double[] vectorx, double[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1);

        /// <summary>
        /// SWAPs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="vectory">The vectory.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <param name="rowy">The rowy.</param>
        /// <param name="incy">The incy.</param>
        public abstract void SWAP(ComplexF[] vectorx, ComplexF[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1);

        /// <summary>
        /// SWAPs the specified vectorx.
        /// </summary>
        /// <param name="vectorx">The vectorx.</param>
        /// <param name="vectory">The vectory.</param>
        /// <param name="n">The n.</param>
        /// <param name="rowx">The rowx.</param>
        /// <param name="incx">The incx.</param>
        /// <param name="rowy">The rowy.</param>
        /// <param name="incy">The incy.</param>
        public abstract void SWAP(ComplexD[] vectorx, ComplexD[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1);

        #endregion

        //public void Copy<T>(T[,] src, T[,] dst, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1)
        //{
        //    Copy<T>(src, dst, n, rowx, incx, rowy, incy);
        //}

        //protected abstract void COPY<T>(object src, object dst, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1);

        //public abstract T DOT<T>(T[] vectorx, T[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1);

        //public abstract T NRM2<T>(T[] vectorx, int n = 0, int rowx = 0, int incx = 1);


        //#region Max
        ///// <summary>
        ///// Gets the index of the maximum value in the specified array.
        ///// </summary>
        ///// <typeparam name="T">One of the supported types.</typeparam>
        ///// <param name="devArray"></param>
        ///// <param name="n"></param>
        ///// <param name="row"></param>
        ///// <param name="incx"></param>
        ///// <returns></returns>
        //public int IAMAX<T>(T[] devArray, int n = 0, int row = 0, int incx = 1)
        //{
        //    return IAMAXEx<T>(devArray, n, row, incx);
        //}

        ////public abstract Tuple<int, int> IAMAX<T>(T[,] devMatrix, int n = 0, int row = 0, int col = 0, bool columnWise = false, int incx = 1);

        //protected abstract int IAMAXEx<T>(object devArray, int n = 0, int row = 0, int incx = 1);

        //#endregion

        //#region Min

        //public int IAMIN<T>(T[] devArray, int n = 0, int row = 0, int incx = 1)
        //{
        //    return IAMINEx<T>(devArray, n, row, incx);
        //}

        //public int Min<T>(T[] devArray)
        //{
        //    return IAMINEx<T>(devArray) - 1;
        //}

        //protected abstract int IAMINEx<T>(object devArray, int n = 0, int row = 0, int incx = 1);

        ////public abstract int IAMIN<T>(T[,] devMatrix, int n = 0, int row = 0, int col = 0, bool columnWise = true, int incx = 1);

        //#endregion

        //public void SCAL<T>(T alpha, T[] vector, int n = 0, int row = 0, int incx = 1)
        //{
        //    SCALEx<T>(alpha, vector, n, row, incx);
        //}

        //public abstract void SCALEx<T>(T alpha, object vector, int n = 0, int row = 0, int incx = 1);

        ////public abstract void SCAL<T>(T alpha, T[,] devMatrix, int n = 0, int row = 0, int col = 0, bool columnWise = true, int incx = 1);

        //public abstract T DOTC<T>(T[] vectorx, T[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1);


        //public abstract void ROT(float[] vectorx, float[] vectory, float sc, float ss, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1);

        //public abstract void ROT(double[] vectorx, double[] vectory, double sc, double ss, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1);

        //public abstract void ROT(ComplexF[] vectorx, ComplexF[] vectory, float sc, ComplexF ss, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1);

        //public abstract void ROT(ComplexD[] vectorx, ComplexD[] vectory, float sc, ComplexD cs, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1);


        //public abstract void ROTM(float[] vectorx, float[] vectory, float[] sparam, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1);

        //public abstract void ROTM(double[] vectorx, double[] vectory, double[] sparam, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1);


        //public abstract void SWAP<T>(T[] vectorx, T[] vectory, int n = 0, int rowx = 0, int incx = 1, int rowy = 0, int incy = 1); 

        //public abstract void ROTG(float[] host_sa, float[] host_sb, float[] host_sc, float[] host_ss)
        //{
        //    throw new NotImplementedException();
        //}

        //public abstract void ROTG(double[] host_da, double[] host_db, double[] host_dc, double[] host_ds)
        //{
        //    throw new NotImplementedException();
        //}

        //public abstract void ROTG(ComplexF[] host_ca, ComplexF[] host_cb, float[] host_sc, float[] host_ss)
        //{
        //    throw new NotImplementedException();
        //}

        //public abstract void ROTG(ComplexD[] host_ca, ComplexD[] host_cb, double[] host_dc, double[] host_ds)
        //{
        //    throw new NotImplementedException();
        //}

        //void IDisposable.Dispose()
        //{
        //    throw new NotImplementedException();
        //}
    }
}
