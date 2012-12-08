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
using System.IO;
using System.Collections.ObjectModel;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Diagnostics;
using System.Threading;
using Cloo;
namespace Cudafy.Host
{
    public class OpenCLDevice : GPGPU
    {
                /// <summary>
        /// Initializes a new instance of the <see cref="CudaGPU"/> class.
        /// </summary>
        /// <param name="deviceId">The device id.</param>
        public OpenCLDevice(int deviceId = 0)
            : base(deviceId)
        {
            try
            {
                _computeDevice = GetComputeDevice(deviceId);
            }
            catch (IndexOutOfRangeException)
            {
                throw new CudafyHostException(CudafyHostException.csDEVICE_ID_OUT_OF_RANGE);
            }

        }

        static OpenCLDevice()
        {
            var tempComputeDevices = new List<ComputeDevice>();
            foreach (var platform in ComputePlatform.Platforms)
                foreach (var device in platform.Devices)
                    tempComputeDevices.Add(device);
            ComputeDevices = new ReadOnlyCollection<ComputeDevice>(tempComputeDevices);
        }

        internal static ReadOnlyCollection<ComputeDevice> ComputeDevices;

        private ComputeDevice GetComputeDevice(int id)
        {
            if (id < 0 || id > ComputeDevices.Count - 1)
                throw new ArgumentOutOfRangeException("id");
            return ComputeDevices[id];
        }

        private ComputeDevice _computeDevice;


        public override GPGPUProperties GetDeviceProperties(bool useAdvanced = true)
        {
            return GetDeviceProperties(_computeDevice, DeviceId);
        }

        internal static GPGPUProperties GetDeviceProperties(ComputeDevice computeDevice, int deviceId)
        {
            GPGPUProperties props = new GPGPUProperties();
            props.Capability = computeDevice.Version;
            props.ClockRate = (int)computeDevice.MaxClockFrequency;
            props.DeviceId = deviceId;
            props.DeviceOverlap = false;//TODO
            props.ECCEnabled = computeDevice.ErrorCorrectionSupport;
            props.HighPerformanceDriver = false;
            props.Integrated = computeDevice.Type != ComputeDeviceTypes.Cpu;
            props.IsSimulated = false;
            props.KernelExecTimeoutEnabled = false;// TODO
            props.MaxGridSize = new dim3(1);//TODO
            props.MaxThreadsPerBlock = (int)computeDevice.MaxWorkGroupSize;//CHECK
            props.MaxThreadsPerMultiProcessor = 1; //TODO
            props.MaxThreadsSize = new dim3(1);//TODO
            props.MemoryPitch = 1;//TODO
            props.MultiProcessorCount = (int)computeDevice.MaxComputeUnits;
            props.Name = computeDevice.Name;
            props.PciBusID = 1;//TODO
            props.PciDeviceID = 1;//TODO
            props.RegistersPerBlock = 1;//TODO
            props.SharedMemoryPerBlock = (int)computeDevice.LocalMemorySize;//CHECK
            props.TextureAlignment = 1;//TODO
            props.TotalConstantMemory = (int)computeDevice.MaxConstantBufferSize;
            props.TotalGlobalMem = computeDevice.GlobalMemorySize;
            props.TotalMemory = (ulong)props.TotalConstantMemory + (ulong)props.TotalGlobalMem;
            props.UseAdvanced = true;
            props.WarpSize = 32;// TODO
            return props;
        }

        
        public override bool CanAccessPeer(GPGPU peer)
        {
            throw new NotImplementedException();
        }

        protected override void DoCopyDeviceToDevice<T>(Array srcDevArray, int srcOffset, GPGPU peer, Array dstDevArray, int dstOffet, int count)
        {
            throw new NotImplementedException();
        }

        protected override void DoCopyDeviceToDeviceAsync<T>(Array srcDevArray, int srcOffset, GPGPU peer, Array dstDevArray, int dstOffet, int count, int stream)
        {
            throw new NotImplementedException();
        }

        public override void CreateStream(int streamId)
        {
            throw new NotImplementedException();
        }

        public override ulong FreeMemory
        {
            get { throw new NotImplementedException(); }
        }

        public override ulong TotalMemory
        {
            get { throw new NotImplementedException(); }
        }

        public override void Synchronize()
        {
            throw new NotImplementedException();
        }

        public override void StartTimer()
        {
            throw new NotImplementedException();
        }

        public override float StopTimer()
        {
            throw new NotImplementedException();
        }

        public override void LoadModule(CudafyModule module, bool unload = true)
        {
            throw new NotImplementedException();
        }

        public override void UnloadModule(CudafyModule module)
        {
            throw new NotImplementedException();
        }

        protected override void DoLaunch(dim3 gridSize, dim3 blockSize, int streamId, KernelMethodInfo gpuMI, params object[] arguments)
        {
            throw new NotImplementedException();
        }

        protected override void DoCopyToConstantMemory<T>(Array hostArray, int hostOffset, Array devArray, int devOffset, int count, KernelConstantInfo ci)
        {
            throw new NotImplementedException();
        }

        protected override void DoCopyToConstantMemoryAsync<T>(IntPtr hostArray, int hostOffset, Array devArray, int devOffset, int count, KernelConstantInfo ci, int streamId)
        {
            throw new NotImplementedException();
        }

        protected override void DoCopyToDevice<T>(Array hostArray, int hostOffset, Array devArray, int devOffset, int count)
        {
            throw new NotImplementedException();
        }

        protected override void DoCopyFromDevice<T>(Array devArray, Array hostArray)
        {
            throw new NotImplementedException();
        }

        protected override void DoCopyFromDevice<T>(Array devArray, int devOffset, Array hostArray, int hostOffset, int count)
        {
            throw new NotImplementedException();
        }

        protected override void DoCopyFromDeviceAsync<T>(Array devArray, int devOffset, Array hostArray, int hostOffset, int count, int streamId)
        {
            throw new NotImplementedException();
        }

        protected override void DoCopyToDeviceAsync<T>(IntPtr hostArray, int hostOffset, Array devArray, int devOffset, int count, int streamId)
        {
            throw new NotImplementedException();
        }

        protected override void DoCopyToDeviceAsync<T>(Array hostArray, int hostOffset, Array devArray, int devOffset, int count, int streamId)
        {
            throw new NotImplementedException();
        }

        protected override void DoCopyToDeviceAsync<T>(IntPtr hostArray, int hostOffset, DevicePtrEx devArray, int devOffset, int count, int streamId)
        {
            throw new NotImplementedException();
        }

        protected override void DoCopyToDeviceAsync<T>(Array hostArray, int hostOffset, Array devArray, int devOffset, int count, int streamId, IntPtr stagingPost, bool isConstantMemory = false)
        {
            throw new NotImplementedException();
        }

        protected override void DoCopyFromDeviceAsync<T>(Array devArray, int devOffset, IntPtr hostArray, int hostOffset, int count, int streamId)
        {
            throw new NotImplementedException();
        }

        protected override void DoCopyFromDeviceAsync<T>(DevicePtrEx devArray, int devOffset, IntPtr hostArray, int hostOffset, int count, int streamId)
        {
            throw new NotImplementedException();
        }

        protected override void DoCopyFromDeviceAsync<T>(Array devArray, int devOffset, Array hostArray, int hostOffset, int count, int streamId, IntPtr stagingPost)
        {
            throw new NotImplementedException();
        }

        public override void SynchronizeStream(int streamId = 0)
        {
            throw new NotImplementedException();
        }

        public override IntPtr HostAllocate<T>(int x)
        {
            throw new NotImplementedException();
        }

        public override void HostFree(IntPtr ptr)
        {
            throw new NotImplementedException();
        }

        public override void HostFreeAll()
        {
            throw new NotImplementedException();
        }

        protected override Array DoCast<T, U>(int offset, Array devArray, int n)
        {
            throw new NotImplementedException();
        }

        protected override Array DoCast<T, U>(int offset, Array devArray, int x, int y)
        {
            throw new NotImplementedException();
        }

        protected override Array DoCast<T, U>(int offset, Array devArray, int x, int y, int z)
        {
            throw new NotImplementedException();
        }

        public override void DestroyStream(int streamId)
        {
            throw new NotImplementedException();
        }

        public override void DestroyStreams()
        {
            throw new NotImplementedException();
        }

        public override T[] CopyToDevice<T>(T[] hostArray)
        {
            throw new NotImplementedException();
        }

        public override T[,] CopyToDevice<T>(T[,] hostArray)
        {
            throw new NotImplementedException();
        }

        public override T[, ,] CopyToDevice<T>(T[, ,] hostArray)
        {
            throw new NotImplementedException();
        }

        public override void CopyOnDevice<T>(T[] srcDevArray, T[] dstDevArray)
        {
            throw new NotImplementedException();
        }

        protected override void DoCopyOnDevice<T>(Array srcDevArray, int srcOffset, Array dstDevArray, int dstOffet, int count)
        {
            throw new NotImplementedException();
        }

        protected override void DoCopyOnDevice<T>(DevicePtrEx srcDevArray, int srcOffset, DevicePtrEx dstDevArray, int dstOffet, int count)
        {
            throw new NotImplementedException();
        }

        protected override void DoCopyOnDeviceAsync<T>(Array srcDevArray, int srcOffset, Array dstDevArray, int dstOffet, int count, int streamId)
        {
            throw new NotImplementedException();
        }

        protected override void DoCopyOnDeviceAsync<T>(DevicePtrEx srcDevArray, int srcOffset, DevicePtrEx dstDevArray, int dstOffet, int count, int streamId)
        {
            throw new NotImplementedException();
        }

        public override T[] Allocate<T>(int x)
        {
            throw new NotImplementedException();
        }

        public override T[,] Allocate<T>(int x, int y)
        {
            throw new NotImplementedException();
        }

        public override T[, ,] Allocate<T>(int x, int y, int z)
        {
            throw new NotImplementedException();
        }

        public override T[] Allocate<T>(T[] hostArray)
        {
            throw new NotImplementedException();
        }

        public override T[,] Allocate<T>(T[,] hostArray)
        {
            throw new NotImplementedException();
        }

        public override T[, ,] Allocate<T>(T[, ,] hostArray)
        {
            throw new NotImplementedException();
        }

        protected override void DoSet<T>(Array devArray, int offset = 0, int count = 0)
        {
            throw new NotImplementedException();
        }

        public override void Free(object devArray)
        {
            throw new NotImplementedException();
        }

        public override void FreeAll()
        {
            throw new NotImplementedException();
        }
    }
}
