/*
CUDAfy.NET - LGPL 2.1 License
Please consider purchasing a commerical license - it helps development, frees you from LGPL restrictions
and provides you with support.  Thank you!
Copyright (C) 2013 Hybrid DSP Systems
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
using System.Text;
using System.Threading;

namespace Cudafy.DynamicParallelism
{
    public static class DynamicParallelismFunctions
    {
        private const string csErrorMsg = "Emulation of dynamic parallelism.";

        private static void ThrowNotSupported()
        {
            throw new NotSupportedException(csErrorMsg);
        }
        
        /// <summary>
        ///  NOTE: Compute Capability 3.5 and later only. Dynamic parallelism. Call from a single thread.
        ///  Not supported by emulator.
        /// </summary>
        /// <param name="gridSize"></param>
        /// <param name="blockSize"></param>
        /// <param name="functionName"></param>
        /// <param name="args"></param>
        public static int Launch(this GThread thread, dim3 gridSize, dim3 blockSize, string functionName, params object[] args)
        {
            ThrowNotSupported();
            return 0;
        }

        public static int SynchronizeDevice(this GThread thread)
        {
            ThrowNotSupported();
            return 0;
        }

        public static int GetLastError(this GThread thread)
        {
            ThrowNotSupported();
            return 0;
        }

        //public string GetLastErrorString(this GThread thread)
        //{
        //    ThrowNotSupported();
        //    return string.Empty;
        //}

        public static int GetDeviceCount(this GThread thread, ref int count)
        {
            ThrowNotSupported();
            return 0;
        }

        public static int GetDeviceID(this GThread thread, ref int id)
        {
            ThrowNotSupported();
            return 0;
        }

        //cudaMemcpyAsync

        //cudaMemsetAsync

        //cudaRuntimeGetVersion

        //cudaMalloc cudaError_t cudaMalloc ( void** devPtr, size_t size )

        //cudaFree
    }
}
