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
using Cloo.Bindings;
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
                _kernels = new List<ComputeKernel>();
                ComputeContextPropertyList properties = new ComputeContextPropertyList(_computeDevice.Platform);
                _context = new ComputeContext(new[] { _computeDevice }, properties, null, IntPtr.Zero);
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

        private ComputeContext _context;

        private List<ComputeKernel> _kernels;

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
            return false;
        }

        protected override void DoCopyDeviceToDevice<T>(Array srcDevArray, int srcOffset, GPGPU peer, Array dstDevArray, int dstOffet, int count)
        {
            throw new NotSupportedException();
        }

        protected override void DoCopyDeviceToDeviceAsync<T>(Array srcDevArray, int srcOffset, GPGPU peer, Array dstDevArray, int dstOffet, int count, int stream)
        {
            throw new NotSupportedException();
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


        private string clTestProgramSource = @"
kernel void VectorAdd(
    global  read_only int* a,
    global  read_only int* b,
    global write_only int* c )
{
    int index = get_global_id(0);
    c[index] = a[index] + b[index];
}
";

        public override void LoadModule(CudafyModule module, bool unload = true)
        {
            // Create and build the opencl program.
            Debug.WriteLine(module.CudaSourceCode);
            ComputeProgram program = new ComputeProgram(_context, module.CudaSourceCode);
            try
            {
                program.Build(null, null, null, IntPtr.Zero);
            }
            catch (Exception)
            {
                throw;
            }
            finally
            {
                module.CompilerOutput = program.GetBuildLog(_computeDevice);
                Debug.WriteLine(module.CompilerOutput);
            }

            if (unload)
                UnloadModules();
            else
                CheckForDuplicateMembers(module);
                
            // Create the kernel function and set its arguments.
            foreach (ComputeKernel kernel in program.CreateAllKernels())
                _kernels.Add(kernel);
           
            // Load constants
            foreach (var kvp in module.Constants)
            {
                if (!kvp.Value.IsDummy)
                {
                    int elemSize = MSizeOf(kvp.Value.Information.FieldType.GetElementType());
                    int totalLength = kvp.Value.GetTotalLength();
                    ComputeBuffer<byte> a = new ComputeBuffer<byte>(_context, ComputeMemoryFlags.ReadOnly, totalLength * elemSize);
                    module.Constants[kvp.Key].CudaPointer = a.Handle;
                }
            }

            _modules.Add(module);
        }

        public override void UnloadModule(CudafyModule module)
        {
            //throw new NotImplementedException();
            
        }

        public override void UnloadModules()
        {
            _kernels.Clear();
            base.UnloadModules();
        }

        //private CLDevicePtrEx<T> GetDeviceMemoryCL<T>(object devArray)
        //{
        //    return GetDeviceMemory(devArray) as CLDevicePtrEx<Type>
        //}

        protected override void DoLaunch(dim3 gridSize, dim3 blockSize, int streamId, KernelMethodInfo gpuMI, params object[] arguments)
        {
            ComputeKernel kernel = _kernels.Where(k => k.FunctionName == gpuMI.Name).FirstOrDefault();//
            int totalArgs = arguments.Length;
            int actualArgCtr = 0;
            int i = 0;
            for (i = 0; i < totalArgs; i++, actualArgCtr++)
            {
                object arg = arguments[actualArgCtr];
                if (arg is Array)
                {
                    var ptrEx = GetDeviceMemory(arg) as CLDevicePtrExInter;
                    kernel.SetMemoryArgument(i, ptrEx.Handle);
                    int[] dims = ptrEx.GetDimensions();
                    for (int d = 0; d < ptrEx.Dimensions; d++, totalArgs++)
                        kernel.SetValueArgument(++i, dims[d]);

                }
                else
                {
                    kernel.SetValueArgument(i, MSizeOf(arg.GetType()), arg);
                }
            }
            
            // Add constants
            foreach (KeyValuePair<string, KernelConstantInfo> kvp in gpuMI.ParentModule.Constants)
            {
                kernel.SetMemoryArgument(i++, (CLMemoryHandle)kvp.Value.CudaPointer);
            }
           
            //foreach(KernelConstantInfo kci in _module)

            // Create the event wait list. An event list is not really needed for this example but it is important to see how it works.
            // Note that events (like everything else) consume OpenCL resources and creating a lot of them may slow down execution.
            // For this reason their use should be avoided if possible.
            ComputeEventList eventList = new ComputeEventList();

            // Create the command queue. This is used to control kernel execution and manage read/write/copy operations.
            ComputeCommandQueue commands = new ComputeCommandQueue(_context, _computeDevice, ComputeCommandQueueFlags.None);

            // Execute the kernel "count" times. After this call returns, "eventList" will contain an event associated with this command.
            // If eventList == null or typeof(eventList) == ReadOnlyCollection<ComputeEventBase>, a new event will not be created.

            // Convert from CUDA grid and block size to OpenCL grid size
            int gridDims = gridSize.ToArray().Length; 
            int blockDims = blockSize.ToArray().Length;
            int maxDims = Math.Max(gridDims, blockDims);

            long[] blockSizeArray = blockSize.ToFixedSizeArray(maxDims);
            long[] gridSizeArray = gridSize.ToFixedSizeArray(maxDims);
            for(i = 0; i < maxDims; i++)
                gridSizeArray[i] *= blockSizeArray[i];
            commands.Execute(kernel, null, gridSizeArray, blockSizeArray, eventList);

            commands.Finish();
        }

        protected override void DoCopyToConstantMemory<T>(Array hostArray, int hostOffset, Array devArray, int devOffset, int count, KernelConstantInfo ci)
        {
            ComputeEventList eventList = new ComputeEventList(); 
            ComputeCommandQueue commands = new ComputeCommandQueue(_context, _computeDevice, ComputeCommandQueueFlags.None);
            commands.WriteToBufferEx<T>(hostArray, (CLMemoryHandle)ci.CudaPointer, true, hostOffset, devOffset, count, eventList);
            commands.Finish();
        }

        protected override void DoCopyToConstantMemoryAsync<T>(IntPtr hostArray, int hostOffset, Array devArray, int devOffset, int count, KernelConstantInfo ci, int streamId)
        {
            throw new NotImplementedException();
        }

        protected override void DoCopyToDevice<T>(Array hostArray, int hostOffset, Array devArray, int devOffset, int count)
        {
            ComputeEventList eventList = new ComputeEventList();
            ComputeCommandQueue commands = new ComputeCommandQueue(_context, _computeDevice, ComputeCommandQueueFlags.None);
            CLDevicePtrEx<T> ptr = GetDeviceMemory(devArray) as CLDevicePtrEx<T>;
            commands.WriteToBufferEx(hostArray, ptr.DevPtr, true, hostOffset, devOffset, count, eventList);
            commands.Finish();
        }

        protected override void DoCopyFromDevice<T>(Array devArray, Array hostArray)
        {
            DoCopyFromDevice<T>(devArray, 0, hostArray, 0, -1);
        }

        protected override void DoCopyFromDevice<T>(Array devArray, int devOffset, Array hostArray, int hostOffset, int count)
        {
            ComputeEventList eventList = new ComputeEventList();
            ComputeCommandQueue commands = new ComputeCommandQueue(_context, _computeDevice, ComputeCommandQueueFlags.None);
            CLDevicePtrEx<T> ptr = GetDeviceMemory(devArray) as CLDevicePtrEx<T>;
            commands.ReadFromBufferEx(ptr.DevPtr, ref hostArray, true, devOffset, hostOffset, count < 0 ? ptr.TotalSize : count, eventList);
            commands.Finish();
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
            //throw new NotImplementedException();
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
            //throw new NotImplementedException();
        }

        public override T[] CopyToDevice<T>(T[] hostArray)
        {
            T[] devMem = new T[0];      
            ComputeBuffer<T> a = new ComputeBuffer<T>(_context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.CopyHostPointer, hostArray);
            AddToDeviceMemory(devMem, new CLDevicePtrEx<T>(a, hostArray.Length, _context));
            return devMem;
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
            T[] devMem = new T[0];
            ComputeBuffer<T> a = new ComputeBuffer<T>(_context, ComputeMemoryFlags.ReadWrite, x);
            AddToDeviceMemory(devMem, new CLDevicePtrEx<T>(a, x, _context));
            return devMem;
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
            return Allocate<T>(hostArray.Length);
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
            //throw new NotImplementedException();
        }

        public override void FreeAll()
        {
            //throw new NotImplementedException();
        }
    }

    public abstract class CLDevicePtrExInter : DevicePtrEx
    {
        public abstract CLMemoryHandle Handle { get; }
    }

    /// <summary>
    /// Internal use.
    /// </summary>
    public class CLDevicePtrEx<T> : CLDevicePtrExInter where T  : struct
    {
        //protected CUDevicePtrEx(CUcontext? context)
        //{
        //    Context = context;
        //}

        /// <summary>
        /// Initializes a new instance of the <see cref="CUDevicePtrEx"/> class.
        /// </summary>
        /// <param name="devPtr">The dev PTR.</param>
        /// <param name="context">The context.</param>       
        public CLDevicePtrEx(ComputeBuffer<T> devPtr, ComputeContext context)
            : this(devPtr, 1, 1, 1, context)
        {
            Dimensions = 0;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="CUDevicePtrEx"/> class.
        /// </summary>
        /// <param name="devPtr">The dev PTR.</param>
        /// <param name="xSize">Size of the x.</param>
        /// <param name="context">The context.</param>     
        public CLDevicePtrEx(ComputeBuffer<T> devPtr, int xSize, ComputeContext context)
            : this(devPtr, xSize, 1, 1, context)
        {
            Dimensions = 1;
        }

        ///// <summary>
        ///// Casts the specified pointer.
        ///// </summary>
        ///// <typeparam name="T"></typeparam>
        ///// <param name="ptrEx">The pointer.</param>
        ///// <param name="offset">The offset.</param>
        ///// <param name="xSize">Size of the x.</param>
        ///// <returns></returns>
        //public CLDevicePtrEx Cast<T>(CUDevicePtrEx ptrEx, int offset, int xSize)
        //{
        //    int size = GPGPU.MSizeOf(typeof(T));
        //    CUdeviceptr ptrOffset = ptrEx.DevPtr + (long)(offset * size);
        //    CUDevicePtrEx newPtrEx = new CUDevicePtrEx(ptrOffset, xSize, ptrEx.Context);
        //    newPtrEx.CreatedFromCast = true;
        //    ptrEx.AddChild(newPtrEx);
        //    return newPtrEx;
        //}

        /// <summary>
        /// Initializes a new instance of the <see cref="CUDevicePtrEx"/> class.
        /// </summary>
        /// <param name="devPtr">The dev PTR.</param>
        /// <param name="xSize">Size of the x.</param>
        /// <param name="ySize">Size of the y.</param>
        /// <param name="context">The context.</param>       
        public CLDevicePtrEx(ComputeBuffer<T> devPtr, int xSize, int ySize, ComputeContext context)
            : this(devPtr, xSize, ySize, 1, context)
        {
            Dimensions = 2;
        }

        ///// <summary>
        ///// Casts the specified pointer.
        ///// </summary>
        ///// <typeparam name="T"></typeparam>
        ///// <param name="ptrEx">The pointer.</param>
        ///// <param name="offset">The offset.</param>
        ///// <param name="xSize">Size of the x.</param>
        ///// <param name="ySize">Size of the y.</param>
        ///// <returns></returns>
        //public CLDevicePtrEx Cast<T>(CUDevicePtrEx ptrEx, int offset, int xSize, int ySize)
        //{
        //    int size = GPGPU.MSizeOf(typeof(T));
        //    CUdeviceptr ptrOffset = ptrEx.DevPtr + (long)(offset * size);
        //    CUDevicePtrEx newPtrEx = new CUDevicePtrEx(ptrOffset, xSize, ySize, ptrEx.Context);
        //    newPtrEx.CreatedFromCast = true;
        //    ptrEx.AddChild(newPtrEx);
        //    return newPtrEx;
        //}

        /// <summary>
        /// Initializes a new instance of the <see cref="CUDevicePtrEx"/> class.
        /// </summary>
        /// <param name="devPtr">The dev PTR.</param>
        /// <param name="xSize">Size of the x.</param>
        /// <param name="ySize">Size of the y.</param>
        /// <param name="zSize">Size of the z.</param>
        /// <param name="context">The context.</param>       
        public CLDevicePtrEx(ComputeBuffer<T> devPtr, int xSize, int ySize, int zSize, ComputeContext context)
        {
            CreatedFromCast = false;
            DevPtr = devPtr;
            XSize = xSize;
            YSize = ySize;
            ZSize = zSize;
            Dimensions = 3;
            Context = context;
            //OriginalSize = originalSize < 0 ? TotalSize : originalSize;
        }

        ///// <summary>
        ///// Casts the specified pointer.
        ///// </summary>
        ///// <typeparam name="T"></typeparam>
        ///// <param name="ptrEx">The pointer.</param>
        ///// <param name="offset">The offset.</param>
        ///// <param name="xSize">Size of the x.</param>
        ///// <param name="ySize">Size of the y.</param>
        ///// <param name="zSize">Size of the z.</param>
        ///// <returns></returns>
        //public CLDevicePtrEx Cast<T>(CUDevicePtrEx ptrEx, int offset, int xSize, int ySize, int zSize)
        //{
        //    int size = GPGPU.MSizeOf(typeof(T));
        //    CUdeviceptr ptrOffset = ptrEx.DevPtr + (long)(offset * size);
        //    CUDevicePtrEx newPtrEx = new CUDevicePtrEx(ptrOffset, xSize, ySize, zSize, ptrEx.Context);
        //    newPtrEx.CreatedFromCast = true;
        //    ptrEx.AddChild(newPtrEx);
        //    return newPtrEx;
        //}


        /// <summary>
        /// Gets the dev PTR.
        /// </summary>
        public ComputeBuffer<T> DevPtr { get; private set; }

        public override CLMemoryHandle Handle
        {
            get
            {
                return DevPtr.Handle;
            }
        }

        /// <summary>
        /// Gets the IntPtr in DevPtr.
        /// </summary>
        public override IntPtr Pointer
        {
            get { return DevPtr.Handle.Value; }
        }


        /// <summary>
        /// Gets the context.
        /// </summary>
        public ComputeContext Context { get; private set; }


    }
}
