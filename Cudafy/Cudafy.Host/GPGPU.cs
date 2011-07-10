﻿/*
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
using System.Runtime.InteropServices;
using System.Dynamic;
using GASS.CUDA;
using GASS.CUDA.Types;

namespace Cudafy.Host
{    
    /// <summary>
    /// Abstract base class for General Purpose GPUs.
    /// </summary>
    public abstract class GPGPU : IDisposable
    {       
        /// <summary>
        /// Initializes a new instance of the <see cref="GPGPU"/> class.
        /// </summary>
        /// <param name="deviceId">The device id.</param>
        protected GPGPU(int deviceId = 0)
        {
            _lock = new object();

            _deviceMemory = new Dictionary<object, DevicePtrEx>();
            _streams = new Dictionary<int, object>();
            _dynamicLauncher = new DynamicLauncher(this);
            DeviceId = deviceId;
        }

        /// <summary>
        /// Releases unmanaged resources and performs other cleanup operations before the
        /// <see cref="GPGPU"/> is reclaimed by garbage collection.
        /// </summary>
        ~GPGPU()
        {
            Dispose(false);
        }

        #region Properties

        /// <summary>
        /// Gets the device id.
        /// </summary>
        public int DeviceId { get; private set; }

        #endregion

        // Track whether Dispose has been called.
        private bool _disposed = false;

        private DynamicLauncher _dynamicLauncher;

        /// <summary>
        /// Gets a value indicating whether this instance is disposed.
        /// </summary>
        /// <value>
        /// 	<c>true</c> if this instance is disposed; otherwise, <c>false</c>.
        /// </value>
        public bool IsDisposed
        {
            get { lock(_lock){ return _disposed; } }
        }

        /// <summary>
        /// Releases unmanaged and - optionally - managed resources
        /// </summary>
        /// <param name="disposing"><c>true</c> to release both managed and unmanaged resources; <c>false</c> to release only unmanaged resources.</param>
        protected virtual void Dispose(bool disposing)
        {
            lock (_lock)
            {
                Debug.WriteLine(string.Format("GPGPU::Dispose({0})", disposing));
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
                    FreeAll();
                    HostFreeAll();
                    DestroyStreams();

                    // Note disposing has been done.
                    _disposed = true;

                }
                else
                    Debug.WriteLine("Already disposed");
            }
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
        /// Internal use.
        /// </summary>
        protected object _lock;

        /// <summary>
        /// Stores pointers to data on the device.
        /// </summary>
        private Dictionary<object, DevicePtrEx> _deviceMemory;

        /// <summary>
        /// Locks this instance.
        /// </summary>
        public virtual void Lock()
        {
        }

        /// <summary>
        /// Unlocks this instance.
        /// </summary>
        public virtual void Unlock()
        {
        }

        /// <summary>
        /// Allows multiple threads to access this GPU.
        /// </summary>
        public virtual void EnableMultithreading()
        {
        }

        /// <summary>
        /// Called once multiple threads have completed work.
        /// </summary>
        public virtual void DisableMultithreading()
        {
        }

        #region Dynamic

        /// <summary>
        /// Gets the dynamic launcher with grid and block sizes equal to 1.
        /// Allows GPU functions to be called using dynamic language run-time. For example:
        /// gpgpu.Launch().myGPUFunction(x, y, res)         
        /// </summary>
        /// <returns>Dynamic launcher</returns>
        public dynamic Launch()
        {
            return Launch(1, 1, -1);
        }

        /// <summary>
        /// Gets the dynamic launcher.
        /// Allows GPU functions to be called using dynamic language run-time. For example:
        /// gpgpu.Launch(16, new dim3(8,8)).myGPUFunction(x, y, res)   
        /// </summary>
        /// <param name="gridSize">Size of the grid.</param>
        /// <param name="blockSize">Size of the block.</param>
        /// <param name="streamId">The stream id or -1 for synchronous.</param>
        /// <returns>Dynamic launcher</returns>
        public dynamic Launch(int gridSize, dim3 blockSize, int streamId = -1)
        {
            return Launch(new dim3(gridSize), blockSize, streamId);
        }

        /// <summary>
        /// Gets the dynamic launcher.
        /// Allows GPU functions to be called using dynamic language run-time. For example:
        /// gpgpu.Launch(new dim3(8,8), 16).myGPUFunction(x, y, res)   
        /// </summary>
        /// <param name="gridSize">Size of the grid.</param>
        /// <param name="blockSize">Size of the block.</param>
        /// <param name="streamId">The stream id or -1 for synchronous.</param>
        /// <returns>Dynamic launcher</returns>
        public dynamic Launch(dim3 gridSize, int blockSize, int streamId = -1)
        {
            return Launch(gridSize, new dim3(blockSize), streamId);
        }

        /// <summary>
        /// Gets the dynamic launcher.
        /// Allows GPU functions to be called using dynamic language run-time. For example:
        /// gpgpu.Launch(16, 16).myGPUFunction(x, y, res)   
        /// </summary>
        /// <param name="gridSize">Size of the grid.</param>
        /// <param name="blockSize">Size of the block.</param>
        /// <param name="streamId">The stream id or -1 for synchronous.</param>
        /// <returns>Dynamic launcher</returns>
        public dynamic Launch(int gridSize, int blockSize, int streamId = -1)
        {
            return Launch(new dim3(gridSize), new dim3(blockSize), streamId);
        }

        /// <summary>
        /// Gets the dynamic launcher.
        /// Allows GPU functions to be called using dynamic language run-time. For example:
        /// gpgpu.Launch(new dim3(8,8), new dim3(8,8)).myGPUFunction(x, y, res)   
        /// </summary>
        /// <param name="gridSize">Size of the grid.</param>
        /// <param name="blockSize">Size of the block.</param>
        /// <param name="streamId">The stream id or -1 for synchronous.</param>
        /// <returns>Dynamic launcher</returns>
        public dynamic Launch(dim3 gridSize, dim3 blockSize, int streamId = -1)
        {
            _dynamicLauncher.BlockSize = blockSize;
            _dynamicLauncher.GridSize = gridSize;
            _dynamicLauncher.StreamId = streamId;
            return _dynamicLauncher;
        }

        #endregion

        /// <summary>
        /// Adds to device memory.
        /// </summary>
        /// <param name="key">The key.</param>
        /// <param name="value">The value.</param>
        protected void AddToDeviceMemory(object key, DevicePtrEx value)
        {
            lock (_lock)
            {
                _deviceMemory.Add(key, value);
            }
        }

        /// <summary>
        /// Gets the device memory pointers.
        /// </summary>
        /// <returns>All data pointers currently on device.</returns>
        public object[] GetDeviceMemoryPointers()
        {
            lock (_lock)
            {
                return _deviceMemory.Keys.ToArray();
            }
        }

        /// <summary>
        /// Gets the device memory pointer.
        /// </summary>
        /// <param name="ptrEx">The pointer.</param>
        /// <returns></returns>
        public object GetDeviceMemoryPointer(DevicePtrEx ptrEx)
        {
            lock (_lock)
            {
                return _deviceMemory.Values.Where(v => v == ptrEx).FirstOrDefault();
            }
        }

        /// <summary>
        /// Gets the device memory for key specified.
        /// </summary>
        /// <param name="devArray">The dev array.</param>
        /// <returns>Device memory</returns>
        public DevicePtrEx GetDeviceMemory(object devArray)
        {
            DevicePtrEx ptr;
            lock (_lock)
            {
                if (devArray == null || !_deviceMemory.ContainsKey(devArray))
                    throw new CudafyHostException(CudafyHostException.csDATA_IS_NOT_ON_GPU);
                ptr = _deviceMemory[devArray] as DevicePtrEx;
            }
            return ptr;
        }

        /// <summary>
        /// Tries to get the device memory.
        /// </summary>
        /// <param name="devArray">The dev array.</param>
        /// <returns>Device memory or null if not found.</returns>
        public DevicePtrEx TryGetDeviceMemory(object devArray)
        {
            DevicePtrEx ptr;
            lock (_lock)
            {
                if (devArray == null || !_deviceMemory.ContainsKey(devArray))
                    ptr = null;
                else
                    ptr = _deviceMemory[devArray] as DevicePtrEx;
            }
            return ptr;
        }

        /// <summary>
        /// Checks if specified device memory value exists.
        /// </summary>
        /// <param name="val">The device memory instance.</param>
        /// <returns></returns>
        protected bool DeviceMemoryValueExists(object val)
        {
            return _deviceMemory.Values.Contains(val);
        }

        /// <summary>
        /// Gets the device memories.
        /// </summary>
        /// <returns></returns>
        protected IEnumerable<object> GetDeviceMemories()
        {
            lock (_lock)
            {
                return _deviceMemory.Values;
            }
        }

        /// <summary>
        /// Clears the device memory.
        /// </summary>
        protected void ClearDeviceMemory()
        {
            lock (_lock)
            {
                _deviceMemory.Clear();
            }
        }

        /// <summary>
        /// Removes from device memory.
        /// </summary>
        /// <param name="key">The key.</param>
        protected void RemoveFromDeviceMemory(object key)
        {
            lock (_lock)
            {
                _deviceMemory.Remove(key);
            }
        }

        /// <summary>
        /// Removes from device memory based on specified pointer.
        /// </summary>
        /// <param name="ptrEx">The PTR ex.</param>
        protected void RemoveFromDeviceMemoryEx(DevicePtrEx ptrEx)
        {
            lock (_lock)
            {
                var kvp = _deviceMemory.Where(k => k.Value == ptrEx).FirstOrDefault();
                foreach (var child in kvp.Value.GetAllChildren())
                {
                    var list = _deviceMemory.Where(ch => ch.Value == child).ToList();
                    foreach(var item in list)
                        _deviceMemory.Remove(item.Key);


                }
                _deviceMemory.Remove(kvp.Key);
                
            }
        }

        /// <summary>
        /// Stores streams.
        /// </summary>
        protected Dictionary<int, object> _streams;

        ///// <summary>
        ///// Currently loaded module.
        ///// </summary>
        //protected CudafyModule _module;

        /// <summary>
        /// Gets the device properties.
        /// </summary>
        /// <param name="useAdvanced">States whether to get advanced properties.</param>
        /// <returns>Device properties.</returns>
        public abstract GPGPUProperties GetDeviceProperties(bool useAdvanced = true);

        /// <summary>
        /// Gets the free memory.
        /// </summary>
        /// <value>The free memory.</value>
        public abstract ulong FreeMemory { get; }

        /// <summary>
        /// Gets the names of all global functions.
        /// </summary>
        /// <returns></returns>
        public IEnumerable<string> GetFunctionNames()
        {
            foreach (var mod in _modules)
                foreach (var f in mod.Functions.Values.Where(f => f.MethodType == eKernelMethodType.Global))
                    yield return f.Name;
        }

        /// <summary>
        /// Gets the stream object.
        /// </summary>
        /// <param name="streamId">The stream id.</param>
        /// <returns>Stream object.</returns>
        public virtual object GetStream(int streamId)
        {
            lock (_lock)
            {
                if (streamId >= 0 && !_streams.ContainsKey(streamId))
                    _streams.Add(streamId, streamId);
                return _streams[streamId];
            }
        }

        /// <summary>
        /// Copies to constant memory.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="hostArray">The host array.</param>
        /// <param name="devArray">The device array.</param>
        public void CopyToConstantMemory<T>(T[] hostArray, T[] devArray)
        {
            KernelConstantInfo ci = InitializeCopyToConstantMemory(hostArray, 0, devArray, 0, hostArray.Length);
            DoCopyToConstantMemory<T>(hostArray, 0, devArray, 0, hostArray.Length, ci);
        }

        /// <summary>
        /// Copies to constant memory.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="hostArray">The host array.</param>
        /// <param name="devArray">The device array.</param>
        public void CopyToConstantMemory<T>(T[,] hostArray, T[,] devArray)
        {
            KernelConstantInfo ci = InitializeCopyToConstantMemory(hostArray, 0, devArray, 0, hostArray.Length);
            DoCopyToConstantMemory<T>(hostArray, 0, devArray, 0, hostArray.Length, ci);
        }

        /// <summary>
        /// Copies to constant memory.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="hostArray">The host array.</param>
        /// <param name="devArray">The device array.</param>
        public void CopyToConstantMemory<T>(T[,,] hostArray, T[,,] devArray)
        {
            KernelConstantInfo ci = InitializeCopyToConstantMemory(hostArray, 0, devArray, 0, hostArray.Length);
            DoCopyToConstantMemory<T>(hostArray, 0, devArray, 0, hostArray.Length, ci);
        }

        /// <summary>
        /// Copies to constant memory.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="hostArray">The host array.</param>
        /// <param name="hostOffset">The host offset.</param>
        /// <param name="devArray">The device array.</param>
        /// <param name="devOffset">The device offset.</param>
        /// <param name="count">The number of element to copy.</param>
        public void CopyToConstantMemory<T>(T[] hostArray, int hostOffset, T[] devArray, int devOffset, int count)
        {
            KernelConstantInfo ci = InitializeCopyToConstantMemory(hostArray, hostOffset, devArray, devOffset, count);
            DoCopyToConstantMemory<T>(hostArray, hostOffset, devArray, devOffset, count, ci);
        }

        private KernelConstantInfo InitializeCopyToConstantMemory(Array hostArray, int hostOffset, Array devArray, int devOffset, int count)
        {
            object o = null;
            KernelConstantInfo ci = null;
            foreach (var module in _modules)
            {
                foreach (var kvp in module.Constants)
                {
                    ci = kvp.Value;
                    o = ci.Information.GetValue(null);
                    if (o == devArray)
                        break;
                    o = null;
                }
                if (o != null)
                    break;
            }
            if (o == null)
                throw new CudafyHostException(CudafyHostException.csCONSTANT_MEMORY_NOT_FOUND);
            if (count == 0)
                count = hostArray.Length;
            if (count > devArray.Length - devOffset)
                throw new CudafyHostException(CudafyHostException.csINDEX_OUT_OF_RANGE);
            return ci;
        }

        /// <summary>
        /// Gets the device count.
        /// </summary>
        /// <returns>Number of devices of this type.</returns>
        public static int GetDeviceCount()
        {
            return 0;
        }

        /// <summary>
        /// Synchronizes context.
        /// </summary>
        public abstract void Synchronize();

        /// <summary>
        /// Starts the timer.
        /// </summary>
        public abstract void StartTimer();

        /// <summary>
        /// Stops the timer.
        /// </summary>
        /// <returns>Elapsed time.</returns>
        public abstract float StopTimer();

        /// <summary>
        /// Loads module from file.
        /// </summary>
        /// <param name="filename">The filename.</param>
        public void LoadModule(string filename)
        {
            CudafyModule km = CudafyModule.Deserialize(filename);
            LoadModule(km);
        }

        /// <summary>
        /// Internal use.
        /// </summary>
        protected List<CudafyModule> _modules = new List<CudafyModule>();

        /// <summary>
        /// Internal use.
        /// </summary>
        protected CudafyModule _module;

        /// <summary>
        /// Internal use. Checks for duplicate members.
        /// </summary>
        /// <param name="module">The module.</param>
        protected void CheckForDuplicateMembers(CudafyModule module)
        {
            if (_modules.Contains(module))
                throw new CudafyHostException(CudafyHostException.csMODULE_ALREADY_LOADED);
            bool duplicateFunc = _modules.Any(m => m.Functions.Any(f => module.Functions.ContainsKey(f.Key)));
            if (duplicateFunc)
                throw new CudafyHostException(CudafyHostException.csDUPLICATE_X_NAME, "function");
            bool duplicateConstant = _modules.Any(m => m.Constants.Any(c => module.Constants.ContainsKey(c.Key)));
            if (duplicateConstant)
                throw new CudafyHostException(CudafyHostException.csDUPLICATE_X_NAME, "constant");
            bool duplicateType = _modules.Any(m => m.Constants.Any(t => module.Types.ContainsKey(t.Key)));
            if (duplicateType)
                throw new CudafyHostException(CudafyHostException.csDUPLICATE_X_NAME, "type");
        }

        /// <summary>
        /// Loads module from module instance optionally unloading all already loaded modules.
        /// </summary>
        /// <param name="module">The module.</param>
        /// <param name="unload">If true then unload any currently loaded modules first.</param>
        public abstract void LoadModule(CudafyModule module, bool unload = true);

        /// <summary>
        /// Unloads the specified module.
        /// </summary>
        /// <param name="module">Module to unload.</param>
        public abstract void UnloadModule(CudafyModule module);

        /// <summary>
        /// Unloads the current module.
        /// </summary>
        public virtual void UnloadModule()
        {
            if (_module != null)
                UnloadModule(_module);
        }

        /// <summary>
        /// Unloads all modules.
        /// </summary>
        public virtual void UnloadModules()
        {
            UnloadModule();
            _modules.Clear();
        }

        /// <summary>
        /// Gets the current module.
        /// </summary>
        /// <value>The current module.</value>
        public CudafyModule CurrentModule
        {
            get { return _module; }
        }

        /// <summary>
        /// Launches the specified kernel.
        /// </summary>
        /// <param name="gridSize">Size of the grid.</param>
        /// <param name="blockSize">Size of the block.</param>
        /// <param name="methodName">Name of the method.</param>
        /// <param name="arguments">The arguments.</param>
        public void Launch(int gridSize, int blockSize, string methodName, params object[] arguments)
        {
            LaunchAsync(new dim3(gridSize), new dim3(blockSize), -1, methodName, arguments);
        }


        /// <summary>
        /// Launches the specified kernel.
        /// </summary>
        /// <param name="gridSize">Size of the grid.</param>
        /// <param name="blockSize">Size of the block.</param>
        /// <param name="streamId">The stream id.</param>
        /// <param name="methodName">Name of the method.</param>
        /// <param name="arguments">The arguments.</param>
        public void LaunchAsync(int gridSize, int blockSize, int streamId, string methodName, params object[] arguments)
        {
            LaunchAsync(new dim3(gridSize), new dim3(blockSize), streamId, methodName, arguments);
        }

        /// <summary>
        /// Launches the specified kernel.
        /// </summary>
        /// <param name="gridSize">Size of the grid.</param>
        /// <param name="blockSize">Size of the block.</param>
        /// <param name="methodName">Name of the method.</param>
        /// <param name="arguments">The arguments.</param>
        public void Launch(dim3 gridSize, int blockSize, string methodName, params object[] arguments)
        {
            LaunchAsync(gridSize, new dim3(blockSize), -1, methodName, arguments);
        }

        /// <summary>
        /// Launches the specified kernel.
        /// </summary>
        /// <param name="gridSize">Size of the grid.</param>
        /// <param name="blockSize">Size of the block.</param>
        /// <param name="streamId">The stream id.</param>
        /// <param name="methodName">Name of the method.</param>
        /// <param name="arguments">The arguments.</param>
        public void LaunchAsync(dim3 gridSize, int blockSize, int streamId, string methodName, params object[] arguments)
        {
            LaunchAsync(gridSize, new dim3(blockSize), streamId, methodName, arguments);
        }

        /// <summary>
        /// Launches the specified kernel.
        /// </summary>
        /// <param name="gridSize">Size of the grid.</param>
        /// <param name="blockSize">Size of the block.</param>
        /// <param name="methodName">Name of the method.</param>
        /// <param name="arguments">The arguments.</param>
        public void Launch(int gridSize, dim3 blockSize, string methodName, params object[] arguments)
        {
            LaunchAsync(new dim3(gridSize), blockSize, -1, methodName, arguments);
        }

        /// <summary>
        /// Launches the specified kernel.
        /// </summary>
        /// <param name="gridSize">Size of the grid.</param>
        /// <param name="blockSize">Size of the block.</param>
        /// <param name="methodName">Name of the method.</param>
        /// <param name="arguments">The arguments.</param>
        public void Launch(dim3 gridSize, dim3 blockSize, string methodName, params object[] arguments)
        {
            LaunchAsync(gridSize, blockSize, -1, methodName, arguments);
        }

        /// <summary>
        /// Launches the specified kernel.
        /// </summary>
        /// <param name="gridSize">Size of the grid.</param>
        /// <param name="blockSize">Size of the block.</param>
        /// <param name="streamId">The stream id.</param>
        /// <param name="methodName">Name of the method.</param>
        /// <param name="arguments">The arguments.</param>
        public void LaunchAsync(int gridSize, dim3 blockSize, int streamId, string methodName, params object[] arguments)
        {
            LaunchAsync(new dim3(gridSize), blockSize, streamId, methodName, arguments);
        }

        /// <summary>
        /// Launches the specified kernel.
        /// </summary>
        /// <param name="gridSize">Size of the grid.</param>
        /// <param name="blockSize">Size of the block.</param>
        /// <param name="streamId">Stream id.</param>
        /// <param name="methodName">Name of the method.</param>
        /// <param name="arguments">The arguments.</param>
        public void LaunchAsync(dim3 gridSize, dim3 blockSize, int streamId, string methodName, params object[] arguments)
        {
            if (_modules.Count == 0)
                throw new CudafyHostException(CudafyHostException.csNO_MODULE_LOADED);
            CudafyModule module = _modules.Where(mod => mod.Functions.ContainsKey(methodName)).FirstOrDefault();
            if(module == null)
                throw new CudafyHostException(CudafyHostException.csCOULD_NOT_FIND_FUNCTION_X, methodName);
            _module = module;
            VerifyMembersAreOnGPU(arguments);
            KernelMethodInfo gpuMI = module.Functions[methodName];
            if (gpuMI.MethodType != eKernelMethodType.Global)
                throw new CudafyHostException(CudafyHostException.csCAN_ONLY_LAUNCH_GLOBAL_METHODS);
            DoLaunch(gridSize, blockSize, streamId, gpuMI, arguments);
        }

        /// <summary>
        /// Does the launch.
        /// </summary>
        /// <param name="gridSize">Size of the grid.</param>
        /// <param name="blockSize">Size of the block.</param>
        /// <param name="streamId">Stream id, or -1 for non-async.</param>
        /// <param name="gpuMI">The gpu MI.</param>
        /// <param name="arguments">The arguments.</param>
        protected abstract void DoLaunch(dim3 gridSize, dim3 blockSize, int streamId, KernelMethodInfo gpuMI, params object[] arguments);

        /// <summary>
        /// Does the copy to constant memory.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="hostArray">The host array.</param>
        /// <param name="devArray">The dev array.</param>
        /// <param name="ci">The ci.</param>
        protected abstract void DoCopyToConstantMemory<T>(Array hostArray, int hostOffset, Array devArray, int devOffset, int count, KernelConstantInfo ci);

        /// <summary>
        /// Does the copy to device.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="hostArray">The host array.</param>
        /// <param name="hostOffset">The host offset.</param>
        /// <param name="devArray">The dev array.</param>
        /// <param name="devOffset">The dev offset.</param>
        /// <param name="count">The count.</param>
        protected abstract void DoCopyToDevice<T>(Array hostArray, int hostOffset, Array devArray, int devOffset, int count);

        /// <summary>
        /// Does the copy from device.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="devArray">The dev array.</param>
        /// <param name="hostArray">The host array.</param>
        protected abstract void DoCopyFromDevice<T>(Array devArray, Array hostArray);

        /// <summary>
        /// Does the copy from device.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="devArray">The dev array.</param>
        /// <param name="devOffset">The dev offset.</param>
        /// <param name="hostArray">The host array.</param>
        /// <param name="hostOffset">The host offset.</param>
        /// <param name="count">The count.</param>
        protected abstract void DoCopyFromDevice<T>(Array devArray, int devOffset, Array hostArray, int hostOffset, int count);

        //protected abstract void DoCopyToDeviceAsync<T>(IntPtr hostArray, Array devArray, int streamId);

        //protected abstract void DoCopyFromDeviceAsync<T>(Array devArray, IntPtr hostArray, int streamId);

        /// <summary>
        /// Does the copy to device async.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="hostArray">The host array.</param>
        /// <param name="hostOffset">The host offset.</param>
        /// <param name="devArray">The dev array.</param>
        /// <param name="devOffset">The dev offset.</param>
        /// <param name="count">The count.</param>
        /// <param name="streamId">The stream id.</param>
        protected abstract void DoCopyToDeviceAsync<T>(IntPtr hostArray, int hostOffset, Array devArray, int devOffset, int count, int streamId);

        /// <summary>
        /// Does the copy from device async.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="devArray">The dev array.</param>
        /// <param name="devOffset">The dev offset.</param>
        /// <param name="hostArray">The host array.</param>
        /// <param name="hostOffset">The host offset.</param>
        /// <param name="count">The count.</param>
        /// <param name="streamId">The stream id.</param>
        protected abstract void DoCopyFromDeviceAsync<T>(Array devArray, int devOffset, IntPtr hostArray, int hostOffset, int count, int streamId);

        /// <summary>
        /// Copies to preallocated array on device.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="hostArray">The host array.</param>
        /// <param name="devArray">The device array.</param>
        public void CopyToDevice<T>(T[] hostArray, T[] devArray)
        {
            DoCopyToDevice<T>(hostArray, 0, devArray, 0, hostArray.Length);
        }

        /// <summary>
        /// Copies to preallocated array on device.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="hostArray">The host array.</param>
        /// <param name="hostOffset">The host offset.</param>
        /// <param name="devArray">The device array.</param>
        /// <param name="devOffset">The device offset.</param>
        /// <param name="count">The number of elements.</param>
        public void CopyToDevice<T>(T[] hostArray, int hostOffset, T[] devArray, int devOffset, int count)
        {
            DoCopyToDevice<T>(hostArray, hostOffset, devArray, devOffset, count);
        }

        //public void SmartCopyToDevice<T>(TextWriterTraceListener[])

        /// <summary>
        /// Copies to preallocated array on device.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="hostArray">The host array.</param>
        /// <param name="hostOffset">The host offset.</param>
        /// <param name="devArray">The device array.</param>
        /// <param name="devOffset">The device offset.</param>
        /// <param name="count">The number of elements.</param>
        public void CopyToDevice<T>(IntPtr hostArray, int hostOffset, T[] devArray, int devOffset, int count)
        {
            DoCopyToDeviceAsync<T>(hostArray, hostOffset, devArray, devOffset, count, -1);
        }

        /// <summary>
        /// Copies asynchronously to preallocated array on device.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="hostArray">The host array.</param>
        /// <param name="hostOffset">The host offset.</param>
        /// <param name="devArray">The device array.</param>
        /// <param name="devOffset">The device offset.</param>
        /// <param name="count">The number of elements.</param>
        /// <param name="streamId">The stream id.</param>
        public void CopyToDeviceAsync<T>(IntPtr hostArray, int hostOffset, T[] devArray, int devOffset, int count, int streamId = 0)
        {
            DoCopyToDeviceAsync<T>(hostArray, hostOffset, devArray, devOffset, count, streamId);
        }

        /// <summary>
        /// Copies asynchronously to preallocated array on device.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="hostArray">The host array.</param>
        /// <param name="hostOffset">The host offset.</param>
        /// <param name="devArray">The device array.</param>
        /// <param name="devOffset">The device offset.</param>
        /// <param name="count">The number of elements.</param>
        /// <param name="streamId">The stream id.</param>
        public void CopyToDeviceAsync<T>(IntPtr hostArray, int hostOffset, T[,] devArray, int devOffset, int count, int streamId = 0)
        {
            DoCopyToDeviceAsync<T>(hostArray, hostOffset, devArray, devOffset, count, streamId);
        }

        /// <summary>
        /// Copies asynchronously to preallocated array on device.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="hostArray">The host array.</param>
        /// <param name="hostOffset">The host offset.</param>
        /// <param name="devArray">The device array.</param>
        /// <param name="devOffset">The device offset.</param>
        /// <param name="count">The number of elements.</param>
        /// <param name="streamId">The stream id.</param>
        public void CopyToDeviceAsync<T>(IntPtr hostArray, int hostOffset, T[,,] devArray, int devOffset, int count, int streamId = 0)
        {
            DoCopyToDeviceAsync<T>(hostArray, hostOffset, devArray, devOffset, count, streamId);
        }

        /// <summary>
        /// Copies from device.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="devArray">The device array.</param>
        /// <param name="devOffset">The device offset.</param>
        /// <param name="hostArray">The host array.</param>
        /// <param name="hostOffset">The host offset.</param>
        /// <param name="count">The number of elements.</param>
        public void CopyFromDevice<T>(T[] devArray, int devOffset, T[] hostArray, int hostOffset, int count)
        {
            DoCopyFromDevice<T>(devArray, devOffset, hostArray, hostOffset, count);
        }

        /// <summary>
        /// Copies from device.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="devArray">The dev array.</param>
        /// <param name="devOffset">The device offset.</param>
        /// <param name="hostArray">The host array.</param>
        /// <param name="hostOffset">The host offset.</param>
        /// <param name="count">The number of elements.</param>
        public void CopyFromDevice<T>(T[] devArray, int devOffset, IntPtr hostArray, int hostOffset, int count)
        {
            DoCopyFromDeviceAsync<T>(devArray, devOffset, hostArray, hostOffset, count, -1);
        }

        /// <summary>
        /// Copies from device asynchronously.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="devArray">The dev array.</param>
        /// <param name="devOffset">The device offset.</param>
        /// <param name="hostArray">The host array.</param>
        /// <param name="hostOffset">The host offset.</param>
        /// <param name="count">The number of elements.</param>
        /// <param name="streamId">The stream id.</param>
        public void CopyFromDeviceAsync<T>(T[] devArray, int devOffset, IntPtr hostArray, int hostOffset, int count, int streamId = 0)
        {
            DoCopyFromDeviceAsync<T>(devArray, devOffset, hostArray, hostOffset, count, streamId);
        }

        /// <summary>
        /// Copies from device asynchronously.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="devArray">The dev array.</param>
        /// <param name="devOffset">The device offset.</param>
        /// <param name="hostArray">The host array.</param>
        /// <param name="hostOffset">The host offset.</param>
        /// <param name="count">The number of elements.</param>
        /// <param name="streamId">The stream id.</param>
        public void CopyFromDeviceAsync<T>(T[,] devArray, int devOffset, IntPtr hostArray, int hostOffset, int count, int streamId = 0)
        {
            DoCopyFromDeviceAsync<T>(devArray, devOffset, hostArray, hostOffset, count, streamId);
        }

        /// <summary>
        /// Copies from device asynchronously.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="devArray">The dev array.</param>
        /// <param name="devOffset">The device offset.</param>
        /// <param name="hostArray">The host array.</param>
        /// <param name="hostOffset">The host offset.</param>
        /// <param name="count">The number of elements.</param>
        /// <param name="streamId">The stream id.</param>
        public void CopyFromDeviceAsync<T>(T[,,] devArray, int devOffset, IntPtr hostArray, int hostOffset, int count, int streamId = 0)
        {
            DoCopyFromDeviceAsync<T>(devArray, devOffset, hostArray, hostOffset, count, streamId);
        }

        /// <summary>
        /// Synchronizes the stream.
        /// </summary>
        /// <param name="streamId">The stream id.</param>
        public abstract void SynchronizeStream(int streamId = 0);

        /// <summary>
        /// Performs a default host memory allocation.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="x">The number of elements.</param>
        /// <returns>Pointer to allocated memory.</returns>
        /// <remarks>Remember to free this memory with HostFree.</remarks>
        public abstract IntPtr HostAllocate<T>(int x);

        /// <summary>
        /// Performs a default host memory allocation.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="x">The x size.</param>
        /// <param name="y">The y size.</param>
        /// <returns>Pointer to allocated memory.</returns>
        /// <remarks>Remember to free this memory with HostFree.</remarks>
        public IntPtr HostAllocate<T>(int x, int y)
        {
            return HostAllocate<T>(x * y);
        }

        /// <summary>
        /// Performs a default host memory allocation.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="x">The x size.</param>
        /// <param name="y">The y size.</param>
        /// <param name="z">The z size.</param>
        /// <returns>Pointer to allocated memory.</returns>
        /// <remarks>Remember to free this memory with HostFree.</remarks>
        public IntPtr HostAllocate<T>(int x, int y, int z)
        {
            return HostAllocate<T>(x * y * z);
        }

        /// <summary>
        /// Frees memory allocated by HostAllocate.
        /// </summary>
        /// <param name="ptr">The pointer to free.</param>
        /// <exception cref="CudafyHostException">Pointer not found.</exception>
        public abstract void HostFree(IntPtr ptr);

        /// <summary>
        /// Frees all memory allocated by HostAllocate.
        /// </summary>
        public abstract void HostFreeAll();

        /// <summary>
        /// Copies memory on host using native CopyMemory function from kernel32.dll.
        /// </summary>
        /// <param name="Destination">The destination.</param>
        /// <param name="Source">The source.</param>
        /// <param name="Length">The length.</param>
        [DllImport("kernel32.dll", EntryPoint = "RtlMoveMemory")]
        public static extern void CopyMemory(IntPtr Destination, IntPtr Source, uint Length);

        /// <summary>
        /// Gets the value at specified index.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="devArray">The dev array.</param>
        /// <param name="x">The x.</param>
        /// <returns>Value at index.</returns>
        public T GetValue<T>(T[] devArray, int x)
        {
            T[] hostArray = new T[1];
            CopyFromDevice(devArray, x, hostArray, 0, 1);
            return hostArray[0];
        }

        /// <summary>
        /// Gets the value at specified index.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="devArray">The dev array.</param>
        /// <param name="x">The x.</param>
        /// <param name="y">The y.</param>
        /// <returns></returns>
        public T GetValue<T>(T[,] devArray, int x, int y)
        {
            T[] hostArray = new T[1];
            var ptrEx = GetDeviceMemory(devArray) as DevicePtrEx;
            DoCopyFromDevice<T>(devArray, ptrEx.GetOffset1D(x, y), hostArray, 0, 1);
            return hostArray[0];
        }

        /// <summary>
        /// Gets the value.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="devArray">The dev array.</param>
        /// <param name="x">The x.</param>
        /// <param name="y">The y.</param>
        /// <param name="z">The z.</param>
        /// <returns></returns>
        public T GetValue<T>(T[,,] devArray, int x, int y, int z)
        {
            T[] hostArray = new T[1];
            var ptrEx = GetDeviceMemory(devArray) as DevicePtrEx;
            DoCopyFromDevice<T>(devArray, ptrEx.GetOffset1D(x, y, z), hostArray, 0, 1);
            return hostArray[0];
        }

        /// <summary>
        /// Casts the specified dev array.
        /// </summary>
        /// <typeparam name="T">Type of dev array.</typeparam>
        /// <typeparam name="U">Type to cast to.</typeparam>
        /// <param name="devArray">The dev array.</param>
        /// <param name="n">The number of samples.</param>
        /// <returns></returns>
        public U[] Cast<T,U>(T[] devArray, int n)
        {
            return (U[])DoCast<T, U>(0, devArray, n);
        }

        /// <summary>
        /// Casts the specified dev array to 1D.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="devArray">The dev array.</param>
        /// <param name="n">The number of samples.</param>
        /// <returns>1D array.</returns>
        public T[] Cast<T>(T[,] devArray, int n)
        {
            return (T[])DoCast<T,T>(0, devArray, n);
        }


        /// <summary>
        /// Casts the specified dev array.
        /// </summary>
        /// <typeparam name="T">Type of dev array.</typeparam>
        /// <typeparam name="U">Type to cast to.</typeparam>
        /// <param name="devArray">The dev array.</param>
        /// <param name="n">The n.</param>
        /// <returns></returns>
        public U[] Cast<T,U>(T[,] devArray, int n)
        {
            return (U[])DoCast<T, U>(0, devArray, n);
        }

        /// <summary>
        /// Casts the specified dev array to 2D.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="devArray">The dev array.</param>
        /// <param name="x">The x.</param>
        /// <param name="y">The y.</param>
        /// <returns></returns>
        public T[,] Cast<T>(T[] devArray, int x, int y)
        {
            return (T[,])DoCast<T,T>(0, devArray, x, y);
        }

        /// <summary>
        /// Casts the specified dev array.
        /// </summary>
        /// <typeparam name="T">Type of dev array.</typeparam>
        /// <typeparam name="U">Type to cast to.</typeparam>
        /// <param name="devArray">The dev array.</param>
        /// <param name="x">The x.</param>
        /// <param name="y">The y.</param>
        /// <returns></returns>
        public U[,] Cast<T,U>(T[] devArray, int x, int y)
        {
            return (U[,])DoCast<T, U>(0, devArray, x, y);
        }

        /// <summary>
        /// Casts the specified dev array to 3D.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="devArray">The dev array.</param>
        /// <param name="x">The x.</param>
        /// <param name="y">The y.</param>
        /// <param name="z">The z.</param>
        /// <returns></returns>
        public T[,,] Cast<T>(T[] devArray, int x, int y, int z)
        {
            return (T[, ,])DoCast<T, T>(0, devArray, x, y, z);
        }

        /// <summary>
        /// Casts the specified dev array.
        /// </summary>
        /// <typeparam name="T">Type of dev array.</typeparam>
        /// <typeparam name="U">Type to cast to.</typeparam>
        /// <param name="devArray">The dev array.</param>
        /// <param name="x">The x.</param>
        /// <param name="y">The y.</param>
        /// <param name="z">The z.</param>
        /// <returns></returns>
        public U[, ,] Cast<T,U>(T[] devArray, int x, int y, int z)
        {
            return (U[, ,])DoCast<T, U>(0, devArray, x, y, z);
        }

        /// <summary>
        /// Casts the specified dev array to 1D.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="devArray">The dev array.</param>
        /// <param name="n">The n.</param>
        /// <returns></returns>
        public T[] Cast<T>(T[,,] devArray, int n)
        {
            return (T[])DoCast<T,T>(0, devArray, n);
        }

        /// <summary>
        /// Casts the specified dev array.
        /// </summary>
        /// <typeparam name="T">Type of dev array.</typeparam>
        /// <typeparam name="U">Type to cast to.</typeparam>
        /// <param name="devArray">The dev array.</param>
        /// <param name="n">The n.</param>
        /// <returns></returns>
        public U[] Cast<T,U>(T[, ,] devArray, int n)
        {
            return (U[])DoCast<T, U>(0, devArray, n);
        }

        /// <summary>
        /// Casts the specified offset.
        /// </summary>
        /// <typeparam name="T">Type of dev array.</typeparam>
        /// <param name="offset">The offset.</param>
        /// <param name="devArray">The dev array.</param>
        /// <param name="n">The n.</param>
        /// <returns></returns>
        public T[] Cast<T>(int offset, T[] devArray, int n)
        {
            return (T[])DoCast<T, T>(offset, devArray, n);
        }

        /// <summary>
        /// Casts the specified offset.
        /// </summary>
        /// <typeparam name="T">Type of dev array.</typeparam>
        /// <typeparam name="U">Type to cast to.</typeparam>
        /// <param name="offset">The offset.</param>
        /// <param name="devArray">The dev array.</param>
        /// <param name="n">The n.</param>
        /// <returns></returns>
        public U[] Cast<T,U>(int offset, T[] devArray, int n)
        {
            return (U[])DoCast<T, U>(offset, devArray, n);
        }

        /// <summary>
        /// Casts the specified dev array to 1D.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="offset">Offset into dev array.</param>
        /// <param name="devArray">The dev array.</param>
        /// <param name="n">The number of samples.</param>
        /// <returns>1D array.</returns>
        public T[] Cast<T>(int offset, T[,] devArray, int n)
        {
            return (T[])DoCast<T, T>(offset, devArray, n);
        }

        /// <summary>
        /// Casts the specified offset.
        /// </summary>
        /// <typeparam name="T">Type of dev array.</typeparam>
        /// <typeparam name="U">Type to cast to.</typeparam>
        /// <param name="offset">The offset.</param>
        /// <param name="devArray">The dev array.</param>
        /// <param name="n">The n.</param>
        /// <returns></returns>
        public U[] Cast<T, U>(int offset, T[,] devArray, int n)
        {
            return (U[])DoCast<T, U>(offset, devArray, n);
        }

        /// <summary>
        /// Casts the specified offset.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="offset">The offset.</param>
        /// <param name="devArray">The dev array.</param>
        /// <param name="x">The x.</param>
        /// <param name="y">The y.</param>
        /// <returns></returns>
        public T[,] Cast<T>(int offset, T[,] devArray, int x, int y)
        {
            return (T[,])DoCast<T, T>(offset, devArray, x, y);
        }

        /// <summary>
        /// Casts the specified offset.
        /// </summary>
        /// <typeparam name="T">Type of dev array.</typeparam>
        /// <typeparam name="U">Type to cast to.</typeparam>
        /// <param name="offset">The offset.</param>
        /// <param name="devArray">The dev array.</param>
        /// <param name="x">The x.</param>
        /// <param name="y">The y.</param>
        /// <returns></returns>
        public U[,] Cast<T,U>(int offset, T[,] devArray, int x, int y)
        {
            return (U[,])DoCast<T, U>(offset, devArray, x, y);
        }

        /// <summary>
        /// Casts the specified dev array to 2D.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="offset">Offset into dev array.</param>
        /// <param name="devArray">The dev array.</param>
        /// <param name="x">The x.</param>
        /// <param name="y">The y.</param>
        /// <returns></returns>
        public T[,] Cast<T>(int offset, T[] devArray, int x, int y)
        {
            return (T[,])DoCast<T, T>(offset, devArray, x, y);
        }

        /// <summary>
        /// Casts the specified offset.
        /// </summary>
        /// <typeparam name="T">Type of dev array.</typeparam>
        /// <typeparam name="U">Type to cast to.</typeparam>
        /// <param name="offset">The offset.</param>
        /// <param name="devArray">The dev array.</param>
        /// <param name="x">The x.</param>
        /// <param name="y">The y.</param>
        /// <returns></returns>
        public U[,] Cast<T,U>(int offset, T[] devArray, int x, int y)
        {
            return (U[,])DoCast<T, U>(offset, devArray, x, y);
        }

        /// <summary>
        /// Casts the specified offset.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="offset">The offset.</param>
        /// <param name="devArray">The dev array.</param>
        /// <param name="x">The x.</param>
        /// <param name="y">The y.</param>
        /// <param name="z">The z.</param>
        /// <returns></returns>
        public T[,,] Cast<T>(int offset, T[,,] devArray, int x, int y, int z)
        {
            return (T[, ,])DoCast<T, T>(offset, devArray, x, y, z);
        }

        /// <summary>
        /// Casts the specified offset.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <typeparam name="U"></typeparam>
        /// <param name="offset">The offset.</param>
        /// <param name="devArray">The dev array.</param>
        /// <param name="x">The x.</param>
        /// <param name="y">The y.</param>
        /// <param name="z">The z.</param>
        /// <returns></returns>
        public U[, ,] Cast<T,U>(int offset, T[, ,] devArray, int x, int y, int z)
        {
            return (U[, ,])DoCast<T, U>(offset, devArray, x, y, z);
        }

        /// <summary>
        /// Casts the specified dev array to 3D.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="offset">Offset into dev array.</param>
        /// <param name="devArray">The dev array.</param>
        /// <param name="x">The x.</param>
        /// <param name="y">The y.</param>
        /// <param name="z">The z.</param>
        /// <returns></returns>
        public T[,,] Cast<T>(int offset, T[] devArray, int x, int y, int z)
        {
            return (T[, ,])DoCast<T, T>(offset, devArray, x, y, z);
        }

        /// <summary>
        /// Casts the specified offset.
        /// </summary>
        /// <typeparam name="T">Type of dev array.</typeparam>
        /// <typeparam name="U">Type to cast to.</typeparam>
        /// <param name="offset">The offset.</param>
        /// <param name="devArray">The dev array.</param>
        /// <param name="x">The x.</param>
        /// <param name="y">The y.</param>
        /// <param name="z">The z.</param>
        /// <returns></returns>
        public U[, ,] Cast<T,U>(int offset, T[] devArray, int x, int y, int z)
        {
            return (U[, ,])DoCast<T, U>(offset, devArray, x, y, z);
        }

        /// <summary>
        /// Casts the specified dev array to 1D.
        /// </summary>
        /// <typeparam name="T">Type of dev array.</typeparam>
        /// <param name="offset">Offset into dev array.</param>
        /// <param name="devArray">The dev array.</param>
        /// <param name="n">The n.</param>
        /// <returns></returns>
        public T[] Cast<T>(int offset, T[, ,] devArray, int n)
        {
            return (T[])DoCast<T, T>(offset, devArray, n);
        }

        /// <summary>
        /// Casts the specified dev array to 1D.
        /// </summary>
        /// <typeparam name="T">Type of source array.</typeparam>
        /// <typeparam name="U">Type of destination array.</typeparam>
        /// <param name="offset">Offset into dev array.</param>
        /// <param name="devArray">The dev array.</param>
        /// <param name="n">The n.</param>
        /// <returns></returns>
        public U[] Cast<T,U>(int offset, T[, ,] devArray, int n)
        {
            return (U[])DoCast<T,U>(offset, devArray, n);
        }

        /// <summary>
        /// Does the cast.
        /// </summary>
        /// <typeparam name="T">Type of source array.</typeparam>
        /// <typeparam name="U">Type of result array.</typeparam>
        /// <param name="offset">Offset into dev array.</param>
        /// <param name="devArray">The dev array.</param>
        /// <param name="n">The n.</param>
        /// <returns></returns>
        protected abstract Array DoCast<T, U>(int offset, Array devArray, int n);

        /// <summary>
        /// Does the cast.
        /// </summary>
        /// <typeparam name="T">Type of source array.</typeparam>
        /// <typeparam name="U">Type of result array.</typeparam>
        /// <param name="offset">Offset into dev array.</param>
        /// <param name="devArray">The dev array.</param>
        /// <param name="x">The x.</param>
        /// <param name="y">The y.</param>
        /// <returns></returns>
        protected abstract Array DoCast<T, U>(int offset, Array devArray, int x, int y);

        /// <summary>
        /// Does the cast.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <typeparam name="U"></typeparam>
        /// <param name="offset">Offset into dev array.</param>
        /// <param name="devArray">The dev array.</param>
        /// <param name="x">The x.</param>
        /// <param name="y">The y.</param>
        /// <param name="z">The z.</param>
        /// <returns></returns>
        protected abstract Array DoCast<T, U>(int offset, Array devArray, int x, int y, int z);

        /// <summary>
        /// Copies data on host.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="nativeHostArraySrc">The source native host array.</param>
        /// <param name="srcOffset">The source offset.</param>
        /// <param name="hostAllocatedMemory">The destination host allocated memory.</param>
        /// <param name="dstOffset">The destination offset.</param>
        /// <param name="count">The number of elements.</param>
        public static void CopyOnHost<T>(T[] nativeHostArraySrc, int srcOffset, IntPtr hostAllocatedMemory, int dstOffset, int count)
        {
            DoCopyOnHost<T>(nativeHostArraySrc, srcOffset, hostAllocatedMemory, dstOffset, count);
        }

        /// <summary>
        /// Copies data on host.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="nativeHostArraySrc">The source native host array.</param>
        /// <param name="srcOffset">The source offset.</param>
        /// <param name="hostAllocatedMemory">The destination host allocated memory.</param>
        /// <param name="dstOffset">The destination offset.</param>
        /// <param name="count">The number of elements.</param>
        public static void CopyOnHost<T>(T[,] nativeHostArraySrc, int srcOffset, IntPtr hostAllocatedMemory, int dstOffset, int count)
        {
            DoCopyOnHost<T>(nativeHostArraySrc, srcOffset, hostAllocatedMemory, dstOffset, count);
        }

        /// <summary>
        /// Copies data on host.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="nativeHostArraySrc">The source native host array.</param>
        /// <param name="srcOffset">The source offset.</param>
        /// <param name="hostAllocatedMemory">The destination host allocated memory.</param>
        /// <param name="dstOffset">The destination offset.</param>
        /// <param name="count">The number of elements.</param>
        public static void CopyOnHost<T>(T[,,] nativeHostArraySrc, int srcOffset, IntPtr hostAllocatedMemory, int dstOffset, int count)
        {
            DoCopyOnHost<T>(nativeHostArraySrc, srcOffset, hostAllocatedMemory, dstOffset, count);
        }

        private unsafe static void DoCopyOnHost<T>(Array nativeHostArraySrc, int srcOffset, IntPtr hostAllocatedMemory, int dstOffset, int count)
        {
            //Type type = (typeof(T));

            GCHandle handle = GCHandle.Alloc(nativeHostArraySrc, GCHandleType.Pinned);
            try
            {
                int size = MSizeOf(typeof(T));
                IntPtr srcIntPtr = handle.AddrOfPinnedObject();
                IntPtr srcOffsetPtr = new IntPtr(srcIntPtr.ToInt64() + srcOffset * size);
                IntPtr dstIntPtrOffset = new IntPtr(hostAllocatedMemory.ToInt64() + dstOffset * size);
                CopyMemory(dstIntPtrOffset, srcOffsetPtr, (uint)(count * size));
            }
            finally
            {
                handle.Free();
            }

        }

        //private unsafe static IntPtr MarshalArray<T>(ref Array items, int srcOffset, IntPtr dstPtr, int dstOffset, int count = 0)
        //{
        //    int length = count <= 0 ? (items.Length - srcOffset) : count;
        //    int iSizeOfOneItemPos = Marshal.SizeOf(typeof(T));
        //    IntPtr dstIntPtrOffset = new IntPtr(dstPtr.ToInt64() + (dstOffset * length));
        //    byte* pbyUnmanagedItems = (byte*)(dstIntPtrOffset.ToPointer());
            
        //    for (int i = srcOffset; i < (srcOffset + length); i++, pbyUnmanagedItems += (iSizeOfOneItemPos))
        //    {
        //        IntPtr pOneItemPos = new IntPtr(pbyUnmanagedItems);
        //        GCHandle handle = GCHandle.Alloc(items.GetValue(i));//, GCHandleType.Pinned);
        //        CopyMemory(pOneItemPos, handle.AddrOfPinnedObject(), (uint)(iSizeOfOneItemPos));
        //        handle.Free();
        //        //Marshal.StructureToPtr(, pOneItemPos, false);
        //    }

        //    return dstPtr;
        //}

        //private unsafe static void UnMarshalArray<T>(IntPtr srcItems, int srcOffset, ref Array items, int dstOffset, int count = 0)
        //{
        //    int length = count <= 0 ? (items.Length - srcOffset) : count;
        //    int iSizeOfOneItemPos = Marshal.SizeOf(typeof(T));
        //    IntPtr srcIntPtrOffset = new IntPtr(srcItems.ToInt64() + (srcOffset * length));
        //    byte* pbyUnmanagedItems = (byte*)(srcIntPtrOffset.ToPointer());

        //    for (int i = dstOffset; i < (dstOffset + length); i++, pbyUnmanagedItems += (iSizeOfOneItemPos))
        //    {
        //        IntPtr pOneItemPos = new IntPtr(pbyUnmanagedItems);
        //        items.SetValue((T)(Marshal.PtrToStructure(pOneItemPos, typeof(T))), i);
        //    }
        //}

        /// <summary>
        /// Copies data on host.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="hostAllocatedMemory">The source host allocated memory.</param>
        /// <param name="srcOffset">The source offset.</param>
        /// <param name="nativeHostArrayDst">The destination native host array.</param>
        /// <param name="dstOffset">The destination offset.</param>
        /// <param name="count">The number of elements.</param>
        public static void CopyOnHost<T>(IntPtr hostAllocatedMemory, int srcOffset, T[] nativeHostArrayDst, int dstOffset, int count)
        {
            DoCopyOnHost<T>(hostAllocatedMemory, srcOffset, nativeHostArrayDst, dstOffset, count);
        }

        /// <summary>
        /// Copies data on host.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="hostAllocatedMemory">The source host allocated memory.</param>
        /// <param name="srcOffset">The source offset.</param>
        /// <param name="nativeHostArrayDst">The destination native host array.</param>
        /// <param name="dstOffset">The destination offset.</param>
        /// <param name="count">The number of elements.</param>
        public static void CopyOnHost<T>(IntPtr hostAllocatedMemory, int srcOffset, T[,] nativeHostArrayDst, int dstOffset, int count)
        {
            DoCopyOnHost<T>(hostAllocatedMemory, srcOffset, nativeHostArrayDst, dstOffset, count);
        }

        /// <summary>
        /// Copies data on host.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="hostAllocatedMemory">The source host allocated memory.</param>
        /// <param name="srcOffset">The source offset.</param>
        /// <param name="nativeHostArrayDst">The destination native host array.</param>
        /// <param name="dstOffset">The destination offset.</param>
        /// <param name="count">The number of elements.</param>
        public static void CopyOnHost<T>(IntPtr hostAllocatedMemory, int srcOffset, T[,,] nativeHostArrayDst, int dstOffset, int count)
        {
            DoCopyOnHost<T>(hostAllocatedMemory, srcOffset, nativeHostArrayDst, dstOffset, count);
        }

        private unsafe static void DoCopyOnHost<T>(IntPtr hostAllocatedMemory, int srcOffset, Array nativeHostArrayDst, int dstOffset, int count)
        {
            //Type type = typeof(T);
            GCHandle handle = GCHandle.Alloc(nativeHostArrayDst, GCHandleType.Pinned);
            try
            {
                int size = MSizeOf(typeof(T));
                IntPtr srcIntPtrOffset = new IntPtr(hostAllocatedMemory.ToInt64() + srcOffset * size);
                IntPtr dstIntPtr = handle.AddrOfPinnedObject();
                IntPtr dstOffsetPtr = new IntPtr(dstIntPtr.ToInt64() + srcOffset * size);
                CopyMemory(dstOffsetPtr, srcIntPtrOffset, (uint)(count * size));
            }
            finally
            {
                handle.Free();
            }     
        }

        /// <summary>
        /// Does the copy.
        /// </summary>
        /// <param name="srcArray">The source array.</param>
        /// <param name="srcOffset">The source offset.</param>
        /// <param name="dstArray">The destination array.</param>
        /// <param name="dstOffset">The destination offset.</param>
        /// <param name="count">The count.</param>
        /// <param name="size">The size.</param>
        protected static void DoCopy(Array srcArray, int srcOffset, Array dstArray, int dstOffset, int count, int size)
        {
            unsafe
            {
                GCHandle dstHandle = GCHandle.Alloc(dstArray, GCHandleType.Pinned);
                GCHandle srcHandle = GCHandle.Alloc(srcArray, GCHandleType.Pinned);
                try
                {
                    IntPtr srcIntPtr = srcHandle.AddrOfPinnedObject();
                    IntPtr srcOffsetPtr = new IntPtr(srcIntPtr.ToInt64() + srcOffset * size);
                    IntPtr dstIntPtr = dstHandle.AddrOfPinnedObject();
                    IntPtr dstOffsetPtr = new IntPtr(dstIntPtr.ToInt64() + dstOffset * size);
                    CopyMemory(dstOffsetPtr, srcOffsetPtr, (uint)(count * size));
                }
                finally
                {
                    dstHandle.Free();
                    srcHandle.Free();
                }
            }
        }

        /// <summary>
        /// Does the copy.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="srcArray">The source array.</param>
        /// <param name="srcOffset">The source offset.</param>
        /// <param name="dstArray">The destination array.</param>
        /// <param name="dstOffset">The destination offset.</param>
        /// <param name="count">The count.</param>
        protected static void DoCopy<T>(Array srcArray, int srcOffset, Array dstArray, int dstOffset, int count)
        {
            DoCopy(srcArray, srcOffset, dstArray, dstOffset, count, MSizeOf(typeof(T)));
        }

        /// <summary>
        /// Does the copy.
        /// </summary>
        /// <param name="srcArray">The source array.</param>
        /// <param name="srcOffset">The source offset.</param>
        /// <param name="dstArray">The destination array.</param>
        /// <param name="dstOffset">The destination offset.</param>
        /// <param name="count">The count.</param>
        /// <param name="type">The type.</param>
        protected static void DoCopy(Array srcArray, int srcOffset, Array dstArray, int dstOffset, int count, Type type)
        {
            DoCopy(srcArray, srcOffset, dstArray, dstOffset, count, MSizeOf(type));
        }

        /// <summary>
        /// Destroys the stream.
        /// </summary>
        /// <param name="streamId">The stream id.</param>
        public abstract void DestroyStream(int streamId);

        /// <summary>
        /// Destroys all streams.
        /// </summary>
        public abstract void DestroyStreams();

        /// <summary>
        /// Copies to preallocated array on device.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="hostArray">The host array.</param>
        /// <param name="devArray">The device array.</param>
        public void CopyToDevice<T>(T[,] hostArray, T[,] devArray)
        {
            DoCopyToDevice<T>(hostArray, 0, devArray, 0, hostArray.Length);
        }


        /// <summary>
        /// Copies to preallocated array on device.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="hostArray">The host array.</param>
        /// <param name="devArray">The device array.</param>
        public void CopyToDevice<T>(T[,,] hostArray, T[,,] devArray)
        {
            DoCopyToDevice<T>(hostArray, 0, devArray, 0, hostArray.Length);
        }

        /// <summary>
        /// Allocates Unicode character array on device, copies to device and returns pointer.
        /// </summary>
        /// <param name="text">The text.</param>
        /// <returns>The device array.</returns>
        public char[] CopyToDevice(string text)
        {
            return CopyToDevice(text.ToCharArray());
        }

        /// <summary>
        /// Allocates array on device, copies to device and returns pointer.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="hostArray">The host array.</param>
        /// <returns>The device array.</returns>
        public abstract T[] CopyToDevice<T>(T[] hostArray);

        /// <summary>
        /// Allocates array on device, copies to device and returns pointer.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="hostArray">The host array.</param>
        /// <returns>The device array.</returns>
        public abstract T[,] CopyToDevice<T>(T[,] hostArray);

        /// <summary>
        /// Allocates array on device, copies to device and returns pointer.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="hostArray">The host array.</param>
        /// <returns>The device array.</returns>
        public abstract T[,,] CopyToDevice<T>(T[,,] hostArray);

        ///// <summary>
        ///// Copies from device.
        ///// </summary>
        ///// <typeparam name="T">Blittable type.</typeparam>
        ///// <param name="dev">The device array.</param>
        ///// <param name="host">The host array.</param>
        //public abstract void CopyFromDevice<T>(T dev, out T host);

        /// <summary>
        /// Copies from device.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="devArray">The device array.</param>
        /// <param name="hostData">The host data.</param>
        public void CopyFromDevice<T>(T[] devArray, out T hostData)
        {
            T[] hostArray = new T[1];
            DoCopyFromDevice<T>(devArray, 0, hostArray, 0, 1);
            hostData = hostArray[0];
        }

        /// <summary>
        /// Copies the complete device array to the host array.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="devArray">The device array.</param>
        /// <param name="hostArray">The host array.</param>
        public void CopyFromDevice<T>(T[] devArray, T[] hostArray)
        {
            DoCopyFromDevice<T>(devArray, hostArray);
        }

        /// <summary>
        /// Copies the complete device array to the host array.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="devArray">The device array.</param>
        /// <param name="hostArray">The host array.</param>
        public void CopyFromDevice<T>(T[,] devArray, T[,] hostArray)
        {
            DoCopyFromDevice<T>(devArray, hostArray);
        }

        /// <summary>
        /// Copies from device.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="devArray">The dev array.</param>
        /// <param name="devOffset">The device offset.</param>
        /// <param name="hostArray">The host array.</param>
        /// <param name="hostOffset">The host offset.</param>
        /// <param name="count">The number of elements.</param>
        public void CopyFromDevice<T>(T[,] devArray, int devOffset, T[] hostArray, int hostOffset, int count)
        {
            DoCopyFromDevice<T>(devArray, devOffset, hostArray, hostOffset, count);
        }

        /// <summary>
        /// Copies the complete device array to the host array.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="devArray">The device array.</param>
        /// <param name="hostArray">The host array.</param>
        public void CopyFromDevice<T>(T[,,] devArray, T[,,] hostArray)
        {
            DoCopyFromDevice<T>(devArray, hostArray);
        }

        /// <summary>
        /// Copies between preallocated arrays on device.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="srcDevArray">The source device array.</param>
        /// <param name="dstDevArray">The destination device array.</param>
        public abstract void CopyOnDevice<T>(T[] srcDevArray, T[] dstDevArray);

        /// <summary>
        /// Copies between preallocated arrays on device.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="srcDevArray">The source device array.</param>
        /// <param name="srcOffset">The source offset.</param>
        /// <param name="dstDevArray">The destination device array.</param>
        /// <param name="dstOffset">The destination offet.</param>
        /// <param name="count">The number of element.</param>
        public void CopyOnDevice<T>(T[] srcDevArray, int srcOffset, T[] dstDevArray, int dstOffset, int count)
        {
            DoCopyOnDevice<T>(srcDevArray, srcOffset, dstDevArray, dstOffset, count);
        }

        /// <summary>
        /// Copies between preallocated arrays on device.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="srcDevArray">The source device array.</param>
        /// <param name="srcOffset">The source offset.</param>
        /// <param name="dstDevArray">The destination device array.</param>
        /// <param name="dstOffset">The destination offet.</param>
        /// <param name="count">The number of element.</param>
        public void CopyOnDevice<T>(T[,] srcDevArray, int srcOffset, T[,] dstDevArray, int dstOffset, int count)
        {
            DoCopyOnDevice<T>(srcDevArray, srcOffset, dstDevArray, dstOffset, count);
        }

        /// <summary>
        /// Copies between preallocated arrays on device.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="srcDevArray">The source device array.</param>
        /// <param name="srcOffset">The source offset.</param>
        /// <param name="dstDevArray">The destination device array.</param>
        /// <param name="dstOffset">The destination offet.</param>
        /// <param name="count">The number of element.</param>
        public void CopyOnDevice<T>(T[,,] srcDevArray, int srcOffset, T[,,] dstDevArray, int dstOffset, int count)
        {
            DoCopyOnDevice<T>(srcDevArray, srcOffset, dstDevArray, dstOffset, count);
        }

        /// <summary>
        /// Copies between preallocated arrays on device.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="srcDevArray">The source device array.</param>
        /// <param name="srcOffset">The source offset.</param>
        /// <param name="dstDevArray">The destination device array.</param>
        /// <param name="dstOffet">The destination offet.</param>
        /// <param name="count">The number of element.</param>
        protected abstract void DoCopyOnDevice<T>(Array srcDevArray, int srcOffset, Array dstDevArray, int dstOffet, int count);
        //public abstract void CopyOnDevice<T>(T[] srcDevArray, int srcOffset, T[] dstDevArray, int dstOffet, int count);

        

        /// <summary>
        /// Allocates on device.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <returns>Device array of length 1.</returns>
        public virtual T[] Allocate<T>()
        {
            return Allocate<T>(1);
        }

        /// <summary>
        /// Allocates array on device.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="x">Length of 1D array.</param>
        /// <returns>Device array of length x.</returns>
        public abstract T[] Allocate<T>(int x);

        /// <summary>
        /// Allocates array on device.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="x">The x dimension.</param>
        /// <param name="y">The y dimension.</param>
        /// <returns>2D device array.</returns>
        public abstract T[,] Allocate<T>(int x, int y);

        /// <summary>
        /// Allocates array on device.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="x">The x dimension.</param>
        /// <param name="y">The y dimension.</param>
        /// <param name="z">The z dimension.</param>
        /// <returns>3D device array.</returns>
        public abstract T[,,] Allocate<T>(int x, int y, int z);

        /// <summary>
        /// Allocates array on device of same size as supplied host array.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="hostArray">The host array.</param>
        /// <returns>1D device array.</returns>
        public abstract T[] Allocate<T>(T[] hostArray);

        /// <summary>
        /// Allocates array on device of same size as supplied host array.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="hostArray">The host array.</param>
        /// <returns>1D device array.</returns>
        public abstract T[,] Allocate<T>(T[,] hostArray);

        /// <summary>
        /// Allocates array on device of same size as supplied host array.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="hostArray">The host array.</param>
        /// <returns>1D device array.</returns>
        public abstract T[,,] Allocate<T>(T[,,] hostArray);


        /// <summary>
        /// Sets the specified device array to zero.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="devArray">The device array.</param>
        public void Set<T>(T[] devArray)
        {
            DoSet<T>(devArray);
        }

        /// <summary>
        /// Sets the specified device array to zero.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="devArray">The device array.</param>
        public void Set<T>(T[,] devArray)
        {
            DoSet<T>(devArray);
        }

        /// <summary>
        /// Sets the specified device array to zero.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="devArray">The device array.</param>
        public void Set<T>(T[,,] devArray)
        {
            DoSet<T>(devArray);
        }

        /// <summary>
        /// Sets the specified device array to zero.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="devArray">The device array.</param>
        /// <param name="offset">The offset.</param>
        /// <param name="count">The number of elements.</param>
        public void Set<T>(T[] devArray, int offset, int count)
        {
            DoSet<T>(devArray, offset, count);
        }

        /// <summary>
        /// Sets the specified device array to zero.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="devArray">The device array.</param>
        /// <param name="offset">The offset.</param>
        /// <param name="count">The number of elements.</param>
        public void Set<T>(T[,] devArray, int offset, int count)
        {
            DoSet<T>(devArray, offset, count);
        }

        /// <summary>
        /// Sets the specified device array to zero.
        /// </summary>
        /// <typeparam name="T">Blittable type.</typeparam>
        /// <param name="devArray">The device array.</param>
        /// <param name="offset">The offset.</param>
        /// <param name="count">The number of elements.</param>
        public void Set<T>(T[, ,] devArray, int offset, int count)
        {
            DoSet<T>(devArray, offset, count);
        }


        /// <summary>
        /// Does the set.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="devArray">The dev array.</param>
        /// <param name="offset">The offset.</param>
        /// <param name="count">The count.</param>
        protected abstract void DoSet<T>(Array devArray, int offset = 0, int count = 0);

        /// <summary>
        /// Frees the specified data array on device.
        /// </summary>
        /// <param name="devArray">The device array to free.</param>
        public abstract void Free(object devArray);

        /// <summary>
        /// Frees all data arrays on device.
        /// </summary>
        public abstract void FreeAll();

        /// <summary>
        /// Verifies launch arguments are on GPU and are supported.
        /// </summary>
        /// <param name="args">The arguments.</param>
        /// <exception cref="ArgumentException">Argument is either not on GPU or not supported.</exception>
        protected void VerifyMembersAreOnGPU(params object[] args)
        {
            int i = 1;
            lock (_lock)
            {
                foreach (object o in args)
                {
                    Type type = o.GetType();
                    //if (type == typeof(uint) || type == typeof(float) || type == typeof(int) || type == typeof(GThread))
                    if(type.IsValueType || type == typeof(GThread))
                        continue;

                    if (!_deviceMemory.ContainsKey(o))
                        throw new ArgumentException(string.Format("Argument {0} of type {1} is not on the GPU or not supported.", i, type));

                    i++;
                }
            }
        }

        /// <summary>
        /// Verifies the specified data is on GPU.
        /// </summary>
        /// <param name="data">The data.</param>
        /// <exception cref="CudafyHostException">Data is not on GPU.</exception>
        public void VerifyOnGPU(object data)
        {
            if (!IsOnGPU(data))
                throw new CudafyHostException(CudafyHostException.csDATA_IS_NOT_ON_GPU);
        }


        /// <summary>
        /// Determines whether the specified data is on GPU.
        /// </summary>
        /// <param name="data">The data.</param>
        /// <returns>
        /// 	<c>true</c> if the specified data is on GPU; otherwise, <c>false</c>.
        /// </returns>
        public bool IsOnGPU(object data)
        {
            lock (_lock)
            {
                return data != null && _deviceMemory.ContainsKey(data);
            }
        }

        /// <summary>
        /// Gets the pointer to the native GPU data.
        /// </summary>
        /// <param name="data">The data.</param>
        /// <returns>Pointer to the actual data.</returns>
        public virtual object GetGPUData(object data)
        {
            VerifyOnGPU(data);
            return _deviceMemory[data];
        }

        /// <summary>
        /// Gets the size of the type specified. Note that this differs from Marshal.SizeOf for System.Char (it returns 2 instead of 1).
        /// </summary>
        /// <param name="t">The type to get the size of.</param>
        /// <returns>Size of type in bytes.</returns>
        public static int MSizeOf(Type t)
        {
            if (t == typeof(char))
                return 2;
            else
                return Marshal.SizeOf(t);
        }

        /// <summary>
        /// Gets the version.
        /// </summary>
        /// <returns></returns>
        public virtual int GetDriverVersion()
        {
            return 1010;
        }

        //public virtual void HostRegister<T>(T[] hostArray)
        //{           
        //}

        //public virtual void Unregister<T>(T[] hostArray)
        //{
        //}

        ///// <summary>
        ///// Convert 2D to 1D array.
        ///// </summary>
        ///// <typeparam name="T">Blittable type.</typeparam>
        ///// <param name="array2d">The 2D array.</param>
        ///// <returns>1D array.</returns>
        //protected T[] Convert2Dto1DArray<T>(T[,] array2d)
        //{
        //    int x = array2d.GetUpperBound(0) - 1;
        //    int y = array2d.GetUpperBound(1) - 1;
        //    T[] array1d = new T[x * y];
        //    int dstIndex = 0;
        //    for (int i = 0; i < x; i++)
        //    {
        //        for (int j = 0; j < y; j++)
        //        {
        //            array1d[dstIndex] = array2d[x, y];
        //            dstIndex++;
        //        }
        //    }
        //    return array1d;
        //}

        ///// <summary>
        ///// Convert 3D to 1D array.
        ///// </summary>
        ///// <typeparam name="T">Blittable type.</typeparam>
        ///// <param name="array3d">The 3D array.</param>
        ///// <returns>1D array.</returns>
        //protected T[] Convert3Dto1DArray<T>(T[,,] array3d)
        //{
        //    int x = array3d.GetUpperBound(0) - 1;
        //    int y = array3d.GetUpperBound(1) - 1;
        //    int z = array3d.GetUpperBound(2) - 1;
        //    T[] array1d = new T[x * y * z];
        //    int dstIndex = 0;
        //    for (int i = 0; i < x; i++)
        //    {
        //        for (int j = 0; j < y; j++)
        //        {
        //            for (int k = 0; k < y; k++)
        //            {
        //                array1d[dstIndex] = array3d[x, y, k];
        //                dstIndex++;
        //            }
        //        }
        //    }
        //    return array1d;
        //}

        ///// <summary>
        /////Convert 2D to 1D array.
        ///// </summary>
        ///// <typeparam name="T">Blittable type.</typeparam>
        ///// <param name="array2d">The 2D array.</param>
        ///// <returns>1D array.</returns>
        //protected T[] Convert2Dto1DArrayPrimitive<T>(T[,] array2d)
        //{
        //    int len = array2d.Length;
        //    T[] array1d = new T[array2d.Length];
        //    System.Buffer.BlockCopy(array2d, 0, array1d, 0, len * Marshal.SizeOf(typeof(T))); 

        //    return array1d;
        //}

        ///// <summary>
        ///// Convert 3D to 1D array.
        ///// </summary>
        ///// <typeparam name="T">Blittable type.</typeparam>
        ///// <param name="array3d">The 3D array.</param>
        ///// <returns>1D array.</returns>
        //protected T[] Convert3Dto1DArrayPrimitive<T>(T[,,] array3d)
        //{
        //    int len = array3d.Length;
        //    T[] array1d = new T[array3d.Length];
        //    System.Buffer.BlockCopy(array3d, 0, array1d, 0, len * Marshal.SizeOf(typeof(T)));

        //    return array1d;
        //}

    }

    /// <summary>
    /// Base class for Device data pointers
    /// </summary>
    public abstract class DevicePtrEx
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="DevicePtrEx"/> class.
        /// </summary>
        public DevicePtrEx()
        {
            Disposed = false;
            _children = new List<DevicePtrEx>();
        }
        
        /// <summary>
        /// Gets the size of the X.
        /// </summary>
        /// <value>
        /// The size of the X.
        /// </value>
        public int XSize { get; protected set; }
        /// <summary>
        /// Gets the size of the Y.
        /// </summary>
        /// <value>
        /// The size of the Y.
        /// </value>
        public int YSize { get; protected set; }
        /// <summary>
        /// Gets the size of the Z.
        /// </summary>
        /// <value>
        /// The size of the Z.
        /// </value>
        public int ZSize { get; protected set; }
 
        /// <summary>
        /// Gets the number of dimensions (rank).
        /// </summary>
        public int Dimensions { get; protected set; }

        /// <summary>
        /// Gets the total size.
        /// </summary>
        public int TotalSize
        {
            get { return XSize * YSize * ZSize; }
        }

        /// <summary>
        /// Gets the offset1 D.
        /// </summary>
        /// <param name="x">The x.</param>
        /// <returns></returns>
        public int GetOffset1D(int x)
        {
            return x;
        }

        /// <summary>
        /// Gets the offset1 D.
        /// </summary>
        /// <param name="x">The x.</param>
        /// <param name="y">The y.</param>
        /// <returns></returns>
        public int GetOffset1D(int x, int y)
        {
            int v = (x * YSize) + y;
            return v;
        }

        /// <summary>
        /// Gets the offset1 D.
        /// </summary>
        /// <param name="x">The x.</param>
        /// <param name="y">The y.</param>
        /// <param name="z">The z.</param>
        /// <returns></returns>
        public int GetOffset1D(int x, int y, int z)
        {
            return (x * YSize * ZSize) + (y * ZSize) + z;//i*length*width + j*width + k
        }

        //public DevicePtrEx Original { get; set; }

        /// <summary>
        /// Gets the pointer when overridden.
        /// </summary>
        public virtual IntPtr Pointer
        {
            get { return IntPtr.Zero; }
        }

        /// <summary>
        /// Gets or sets the offset.
        /// </summary>
        /// <value>
        /// The offset.
        /// </value>
        public virtual int Offset { get; protected set; }

        /// <summary>
        /// Gets or sets a value indicating whether this <see cref="DevicePtrEx"/> is disposed.
        /// </summary>
        /// <value>
        ///   <c>true</c> if disposed; otherwise, <c>false</c>.
        /// </value>
        public bool Disposed { get; set; }

        /// <summary>
        /// Gets the dimensions.
        /// </summary>
        /// <returns></returns>
        public int[] GetDimensions()
        {
            int[] dims = new int[Dimensions];
            if (Dimensions > 0)
                dims[0] = XSize;
            if (Dimensions > 1)
                dims[1] = YSize;
            if (Dimensions > 2)
                dims[2] = ZSize;
            return dims;
        }

        /// <summary>
        /// Adds the child.
        /// </summary>
        /// <param name="ptrEx">The PTR ex.</param>
        public void AddChild(DevicePtrEx ptrEx)
        {
            _children.Add(ptrEx);
        }

        /// <summary>
        /// Removes the children.
        /// </summary>
        public void RemoveChildren()
        {
            foreach (var ptr in _children)
                ptr.RemoveChildren();
            _children.Clear();
        }

        /// <summary>
        /// Gets the level 1 children.
        /// </summary>
        public IEnumerable<DevicePtrEx> Children
        {
            get { return _children; }
        }

        /// <summary>
        /// Gets all children.
        /// </summary>
        /// <returns></returns>
        public IEnumerable<DevicePtrEx> GetAllChildren()
        {
            foreach (var ptr in _children)
            {
                yield return ptr;
                foreach (var child in ptr.GetAllChildren())
                    yield return child;
            }
            
        }

        private List<DevicePtrEx> _children;
    }
}