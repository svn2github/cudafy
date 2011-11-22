/*
 * Added by Kichang Kim
 * kkc0923@hotmail.com
 * */
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Diagnostics;

using Cudafy.Host;
using Cudafy.Types;
using Cudafy.Maths.SPARSE.Types;

namespace Cudafy.Maths.SPARSE
{
    internal enum eDataType { S, C, D, Z };

    /// <summary>
    /// Abstract base class for devices supporting SPARSE matrices.
    /// Warning: This code is alpha and incomplete.
    /// </summary>
    public abstract class GPGPUSPARSE : IDisposable
    {
        private object _lock;
        private bool _disposed = false;

        /// <summary>
        /// Gets a value indicating whether this instance is disposed.
        /// </summary>
        /// <value>
        /// 	<c>true</c> if this instance is disposed; otherwise, <c>false</c>.
        /// </value>
        public bool IsDisposed
        {
            get{lock(_lock){return _disposed;}}
        }

        /// <summary>
        /// Creates a SPARSE wrapper based on the specified gpu. Note only CudaGPU is supported.
        /// </summary>
        /// <param name="gpu">The gpu.</param>
        /// <returns></returns>
        public static GPGPUSPARSE Create(GPGPU gpu)
        {
            if (gpu is CudaGPU)
                return new CudaSPARSE(gpu);
            else
                throw new NotImplementedException(gpu.ToString());
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="GPGPUSPARSE"/> class.
        /// </summary>
        protected GPGPUSPARSE()
        {
            _lock = new object();
        }

        /// <summary>
        /// Releases unmanaged resources and performs other cleanup operations before the
        /// <see cref="GPGPUSPARSE"/> is reclaimed by garbage collection.
        /// </summary>
        ~GPGPUSPARSE()
        {
            Dispose(false);
        }

        /// <summary>
        /// Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged resources.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
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
                Debug.WriteLine(string.Format("GPGPUSPARSE::Dispose({0})", disposing));
                // Check to see if Dispose has already been called.
                if (!this._disposed)
                {
                    Debug.WriteLine("Disposing GPGPUSPARSE");
                    if (disposing)
                    {
                    }

                    Shutdown();
                    _disposed = true;

                }
                else
                    Debug.WriteLine("Already disposed");
            }
        }

#region AXPY
        /// <summary>
        /// Multiplies the vector x in sparse format by the constant alpha and adds
        /// the result to the vector y in dense format; that is, it overwrites y with alpha * x + y.
        /// </summary>
        /// <param name="alpha">constant multiplier.</param>
        /// <param name="vectorx">non‐zero values of vector x.</param>
        /// <param name="indexx">indices corresponding to non‐zero values of vector x.</param>
        /// <param name="vectory">initial vector in dense format.</param>
        /// <param name="n">number of elements of the vector x (set to 0 for all elements).</param>
        /// <param name="ibase">The index base.</param>
        public abstract void AXPY(double alpha, double[] vectorx, int[] indexx, double[] vectory, int n = 0, cusparseIndexBase ibase = cusparseIndexBase.Zero);
#endregion
    }
}
