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
using System.Threading;
namespace Cudafy
{
    /// <summary>
    /// Represents a Cuda thread.
    /// </summary>
    public class GThread
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="GThread"/> class.
        /// </summary>
        /// <param name="xId">The x id.</param>
        /// <param name="yId">The y id.</param>
        /// <param name="parent">The parent block.</param>
        public GThread(int xId, int yId, GBlock parent)
        {
            threadIdx = new dim3(xId, yId);
            block = parent;
        }

         /// <summary>
        /// Gets the warp id this thread belongs too
        /// </summary>
        /// <value>
        /// The warp id
        /// </value>
        internal int WarpId()
        {
            //return (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) / warpSize;
            return threadIdx.x / warpSize - 1;
        }

        /// <summary>
        /// Gets the size of the warp.
        /// </summary>
        /// <value>
        /// The size of the warp.
        /// </value>
        public int warpSize
        {
            get { return 32; }
        }

        /// <summary>
        /// Gets the parent block id.
        /// </summary>
        public dim3 blockIdx
        {
            get { return block.Idx; }
        }

        /// <summary>
        /// Gets the parent block dimension.
        /// </summary>
        public dim3 blockDim
        {
            get { return block.Dim; }
        }

        /// <summary>
        /// Gets the parent grid dim.
        /// </summary>
        public dim3 gridDim
        {
            get { return block.Grid.Dim; }
        }

        /// <summary>
        /// Syncs the threads in the block.
        /// </summary>
        public void SyncThreads()
        {
            block.SyncThreads();
        }

         /// <summary>
        /// NOTE Compute Capability 2.x only. Syncs the threads in the block.
        /// </summary>
        public int SyncThreadsCount(bool condition)
        {
            return block.SyncThreadsCount(condition);
        }

        /// <summary>
        /// Syncs threads in warp, returns true if any had true predicate 
        /// </summary>
        public bool Any(bool predicate)
        {
            return block.Any(predicate, WarpId());// ? 1 : 0;
        }

        /// <summary>
        /// Syncs threads in warp, returns true if any had true predicate 
        /// </summary>
        public bool All(bool predicate)
        {
            return block.All(predicate, WarpId());
        }

        /// <summary>
        /// NOTE Compute Capability 2.x only. Syncs threads in warp, returns true if any had true predicate. 
        /// </summary>
        public int Ballot(bool predicate)
        {
            return block.Ballot(predicate, WarpId());
        }

        /// <summary>
        /// Allocates a 1D array in shared memory.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="varName">Key of the variable.</param>
        /// <param name="x">The x size.</param>
        /// <returns>Pointer to the shared memory.</returns>
        public T[] AllocateShared<T>(string varName, int x)
        {
            return block.AllocateShared<T>(varName, x);
        }

        /// <summary>
        /// Allocates a 2D array in shared memory.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="varName">Key of the variable.</param>
        /// <param name="x">The x size.</param>
        /// <param name="y">The y size.</param>
        /// <returns>Pointer to the shared memory.</returns>
        public T[,] AllocateShared<T>(string varName, int x, int y)
        {
            return block.AllocateShared<T>(varName, x, y);
        }

        /// <summary>
        /// Allocates a 3D array in shared memory.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="varName">Key of the variable.</param>
        /// <param name="x">The x size.</param>
        /// <param name="y">The y size.</param>
        /// <param name="z">The z size.</param>
        /// <returns>Pointer to the shared memory.</returns>
        public T[,,] AllocateShared<T>(string varName, int x, int y, int z)
        {
            return block.AllocateShared<T>(varName, x, y, z);
        }

        /// <summary>
        /// Gets the thread id.
        /// </summary>
        public dim3 threadIdx { get; private set; }

        /// <summary>
        /// Gets the parent block.
        /// </summary>
        internal GBlock block { get; private set; }

    }


}
