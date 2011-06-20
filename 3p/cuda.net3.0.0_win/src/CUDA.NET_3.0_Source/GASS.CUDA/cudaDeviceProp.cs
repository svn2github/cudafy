namespace GASS.CUDA
{
    using GASS.Types;
    using System;
    using System.Runtime.InteropServices;
    using System.Text;

    // 20-06-2011
    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
    public struct cudaDeviceProp
    {
        public string name
        {
            get { return (new string(nameChar)).Trim('\0'); }
        }

        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 256)]
        public char[] nameChar;
        public SizeT totalGlobalMem;
        public SizeT sharedMemPerBlock;
        public int regsPerBlock;
        public int warpSize;
        public SizeT memPitch;
        public int maxThreadsPerBlock;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
        public int[] maxThreadsDim;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
        public int[] maxGridSize;

        public int clockRate;

        public SizeT totalConstMem;
        public int major;
        public int minor;
        
        public SizeT textureAlignment;
        public int deviceOverlap;
        public int multiProcessorCount;
        public int kernelExecTimeoutEnabled;
        public int integrated;
        public int canMapHostMemory;
        public int computeMode;

        public int maxTexture1D;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 2)]
        public int[] maxTexture2D;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
        public int[] maxTexture3D;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 2)]
        public int[] maxTexture1DLayered;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
        public int[] maxTexture2DLayered;
        public SizeT surfaceAlignment;
        public int concurrentKernels;
        public int ECCEnabled;
        public int pciBusID;
        public int pciDeviceID;
        public int pciDomainID;
        public int tccDriver;
        public int asyncEngineCount;
        public int unifiedAddressing;
        public int memoryClockRate;
        public int memoryBusWidth;
        public int l2CacheSize;
        public int maxThreadsPerMultiProcessor;


 
        //[MarshalAs(UnmanagedType.ByValArray, SizeConst = 31)]
        //public int[] __cudaReserved;
    }



    //[StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
    //public struct cudaDeviceProp
    //{
    //    public string name
    //    {
    //        get { return (new string(nameChar)).Trim('\0'); }
    //    }

    //    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 256)]
    //    public char[] nameChar; 
    //    public SizeT totalGlobalMem;
    //    public SizeT sharedMemPerBlock;
    //    public int regsPerBlock;
    //    public int warpSize;
    //    public SizeT memPitch;
    //    public int maxThreadsPerBlock;
    //    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
    //    public int[] maxThreadsDim;
    //    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
    //    public int[] maxGridSize;

    //    public SizeT totalConstMem;
    //    public int major;
    //    public int minor;
    //    public int clockRate;
    //    public SizeT textureAlignment;
    //    public int deviceOverlap;
    //    public int multiProcessorCount;
    //    public int kernelExecTimeoutEnabled;
    //    public int integrated;
    //    public int canMapHostMemory;
    //    public int computeMode;

    //    public int concurrentKernels;
    //    public int ECCEnabled;
    //    public int pciBusID;
    //    public int pciDeviceID;
    //    public int tccDriver;
    //    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 31)]
    //    public int[] __cudaReserved;
    //}

    //[StructLayout(LayoutKind.Sequential)]
    //public struct cudaDeviceProp
    //{
    //    [MarshalAs(UnmanagedType.LPStr)]
    //    public string name;
    //    public SizeT totalGlobalMem;
    //    public SizeT sharedMemPerBlock;
    //    public int regsPerBlock;
    //    public int warpSize;
    //    public SizeT memPitch;
    //    public int maxThreadsPerBlock;
    //    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
    //    public int[] maxThreadsDim;
    //    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
    //    public int[] maxGridSize;
    //    public int clockRate;
    //    public SizeT totalConstMem;
    //    public int major;
    //    public int minor;
    //    public SizeT textureAlignment;
    //    public int deviceOverlap;
    //    public int multiProcessorCount;
    //    public int kernelExecTimeoutEnabled;
    //    public int integrated;
    //    public int canMapHostMemory;
    //    public int computeMode;
    //    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 0x24)]
    //    public int[] __cudaReserved;
    //}
}

