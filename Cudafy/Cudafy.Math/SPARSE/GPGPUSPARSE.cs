/* Added by Kichang Kim (kkc0923@hotmail.com) */
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
        /// Gets the version info.
        /// </summary>
        /// <returns></returns>
        public abstract string GetVersionInfo();

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

        #region SPARSE Level 1

        #region AXPY
        /// <summary>
        /// Multiplies the vector x in sparse format by the constant alpha and adds
        /// the result to the vector y in dense format.
        /// y = alpha * x + y
        /// </summary>
        /// <param name="alpha">constant multiplier.</param>
        /// <param name="vectorx">non-zero values of vector x.</param>
        /// <param name="indexx">indices corresponding to non‐zero values of vector x.</param>
        /// <param name="vectory">initial vector in dense format.</param>
        /// <param name="n">number of elements of the vector x (set to 0 for all elements).</param>
        /// <param name="ibase">The index base.</param>
        public abstract void AXPY(float alpha, float[] vectorx, int[] indexx, float[] vectory, int n = 0, cusparseIndexBase ibase = cusparseIndexBase.Zero);
        /// <summary>
        /// Multiplies the vector x in sparse format by the constant alpha and adds
        /// the result to the vector y in dense format.
        /// y = alpha * x + y
        /// </summary>
        /// <param name="alpha">constant multiplier.</param>
        /// <param name="vectorx">non-zero values of vector x.</param>
        /// <param name="indexx">indices corresponding to non‐zero values of vector x.</param>
        /// <param name="vectory">initial vector in dense format.</param>
        /// <param name="n">number of elements of the vector x (set to 0 for all elements).</param>
        /// <param name="ibase">The index base.</param>
        public abstract void AXPY(double alpha, double[] vectorx, int[] indexx, double[] vectory, int n = 0, cusparseIndexBase ibase = cusparseIndexBase.Zero);
        #endregion

        #region DOT
        /// <summary>
        /// Returns the dot product of a vector x in sparse format and vector y in dense format.
        /// For i = 0 to n-1
        ///     result += x[i] * y[i]
        /// </summary>
        /// <param name="vectorx">non-zero values of vector x.</param>
        /// <param name="indexx">indices corresponding to non-zero values of vector x.</param>
        /// <param name="vectory">vector in dense format.</param>
        /// <param name="n">number of non-zero elements of the vector x (set to 0 for all non-zero elements).</param>
        /// <param name="ibase">The index base.</param>
        /// <returns>result.</returns>
        public abstract float DOT(float[] vectorx, int[] indexx, float[] vectory, int n = 0, cusparseIndexBase ibase = cusparseIndexBase.Zero);
        /// <summary>
        /// Returns the dot product of a vector x in sparse format and vector y in dense format.
        /// For i = 0 to n-1
        ///     result += x[i] * y[i]
        /// </summary>
        /// <param name="vectorx">non-zero values of vector x.</param>
        /// <param name="indexx">indices corresponding to non-zero values of vector x.</param>
        /// <param name="vectory">vector in dense format.</param>
        /// <param name="n">number of non-zero elements of the vector x (set to 0 for all non-zero elements).</param>
        /// <param name="ibase">The index base.</param>
        /// <returns>result.</returns>
        public abstract double DOT(double[] vectorx, int[] indexx, double[] vectory, int n = 0, cusparseIndexBase ibase = cusparseIndexBase.Zero);
        #endregion

        #region GTHR
        /// <summary>
        /// Gathers the elements of the vector y listed by the index array indexx into the array vectorx.
        /// x[i] = y[i]
        /// </summary>
        /// <param name="vectory">vector in dense format, of size greater than or equal to max(indexx)-idxBase+1</param>
        /// <param name="vectorx">pre-allocated array in device memory of size greater than or equal to n</param>
        /// <param name="indexx">indices corresponding to non-zero values of vector x.</param>
        /// <param name="n">number of non-zero elements of the vector x (set to 0 for all non-zero elements).</param>
        /// <param name="ibase">The index base.</param>
        public abstract void GTHR(float[] vectory, float[] vectorx, int[] indexx, int n = 0, cusparseIndexBase ibase = cusparseIndexBase.Zero);
        /// <summary>
        /// Gathers the elements of the vector y listed by the index array indexx into the array vectorx.
        /// x[i] = y[i]
        /// </summary>
        /// <param name="vectory">vector in dense format, of size greater than or equal to max(indexx)-idxBase+1</param>
        /// <param name="vectorx">pre-allocated array in device memory of size greater than or equal to n</param>
        /// <param name="indexx">indices corresponding to non-zero values of vector x.</param>
        /// <param name="n">number of non-zero elements of the vector x (set to 0 for all non-zero elements).</param>
        /// <param name="ibase">The index base.</param>
        public abstract void GTHR(double[] vectory, double[] vectorx, int[] indexx, int n = 0, cusparseIndexBase ibase = cusparseIndexBase.Zero);
        #endregion

        #region GTHRZ
        /// <summary>
        /// Gathers the elements of the vector y listed by the index array indexx into the array vectorx, and zeroes those elements in the vector y.
        /// x[i] = y[i]
        /// y[i] = 0
        /// </summary>
        /// <param name="vectory">vector in dense format, of size greater than or equal to max(indexx)-idxBase+1.</param>
        /// <param name="vectorx">pre-allocated array in device memory of size greater than or equal to n.</param>
        /// <param name="indexx">indices corresponding to non-zero values of vector x.</param>
        /// <param name="n">number of non-zero elements of the vector x (set to 0 for all non-zero elements).</param>
        /// <param name="ibase">The index base.</param>
        public abstract void GTHRZ(float[] vectory, float[] vectorx, int[] indexx, int n = 0, cusparseIndexBase ibase = cusparseIndexBase.Zero);
        /// <summary>
        /// Gathers the elements of the vector y listed by the index array indexx into the array vectorx, and zeroes those elements in the vector y.
        /// x[i] = y[i]
        /// y[i] = 0
        /// </summary>
        /// <param name="vectory">vector in dense format, of size greater than or equal to max(indexx)-idxBase+1.</param>
        /// <param name="vectorx">pre-allocated array in device memory of size greater than or equal to n.</param>
        /// <param name="indexx">indices corresponding to non-zero values of vector x.</param>
        /// <param name="n">number of non-zero elements of the vector x (set to 0 for all non-zero elements).</param>
        /// <param name="ibase">The index base.</param>
        public abstract void GTHRZ(double[] vectory, double[] vectorx, int[] indexx, int n = 0, cusparseIndexBase ibase = cusparseIndexBase.Zero);
        #endregion

        #region ROT
        /// <summary>
        /// Applies givens rotation, defined by values c and s, to vectors x in sparse and y in dense format.
        /// x[i] = c * x[i] + s * y[i];
        /// y[i] = c * y[i] - s * x[i];
        /// </summary>
        /// <param name="vectorx">non-zero values of the vector x.</param>
        /// <param name="indexx">indices correspoding to non-zero values of vector x.</param>
        /// <param name="vectory">vector in dense format.</param>
        /// <param name="c">scalar</param>
        /// <param name="s">scalar</param>
        /// <param name="n">number of non-zero elements of the vector x (set to 0 for all non-zero elements).</param>
        /// <param name="ibase">The index base.</param>
        public abstract void ROT(float[] vectorx, int[] indexx, float[] vectory, float c, float s, int n = 0, cusparseIndexBase ibase = cusparseIndexBase.Zero);
        /// <summary>
        /// Applies givens rotation, defined by values c and s, to vectors x in sparse and y in dense format.
        /// x[i] = c * x[i] + s * y[i];
        /// y[i] = c * y[i] - s * x[i];
        /// </summary>
        /// <param name="vectorx">non-zero values of the vector x.</param>
        /// <param name="indexx">indices correspoding to non-zero values of vector x.</param>
        /// <param name="vectory">vector in dense format.</param>
        /// <param name="c">scalar</param>
        /// <param name="s">scalar</param>
        /// <param name="n">number of non-zero elements of the vector x (set to 0 for all non-zero elements).</param>
        /// <param name="ibase">The index base.</param>
        public abstract void ROT(double[] vectorx, int[] indexx, double[] vectory, double c, double s, int n = 0, cusparseIndexBase ibase = cusparseIndexBase.Zero);
        #endregion

        #region SCTR
        /// <summary>
        /// Scatters the vector x in sparse format into the vector y in dense format.
        /// It modifies only the lements of y whose indices are listed in the array indexx.
        /// y[i] = x[i]
        /// </summary>
        /// <param name="vectorx">non-zero values of vector x.</param>
        /// <param name="indexx">indices correspoding to non-zero values of vector x.</param>
        /// <param name="vectory">pre-allocated vector in dense format, of size greater than or equal to max(indexx)-ibase+1.</param>
        /// <param name="n">number of non-zero elements of the vector x (set to 0 for all non-zero elements).</param>
        /// <param name="ibase">The index base.</param>
        public abstract void SCTR(float[] vectorx, int[] indexx, float[] vectory, int n = 0, cusparseIndexBase ibase = cusparseIndexBase.Zero);
        /// <summary>
        /// Scatters the vector x in sparse format into the vector y in dense format.
        /// It modifies only the lements of y whose indices are listed in the array indexx.
        /// y[i] = x[i]
        /// </summary>
        /// <param name="vectorx">non-zero values of vector x.</param>
        /// <param name="indexx">indices correspoding to non-zero values of vector x.</param>
        /// <param name="vectory">pre-allocated vector in dense format, of size greater than or equal to max(indexx)-ibase+1.</param>
        /// <param name="n">number of non-zero elements of the vector x (set to 0 for all non-zero elements).</param>
        /// <param name="ibase">The index base.</param>
        public abstract void SCTR(double[] vectorx, int[] indexx, double[] vectory, int n = 0, cusparseIndexBase ibase = cusparseIndexBase.Zero);
        #endregion



        #endregion
    }
}
