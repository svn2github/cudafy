namespace GASS.CUDA
{
    using GASS.Types;
    using System;
    using System.Runtime.InteropServices;
    using System.Text;


//    struct __device_builtin__ cudaDeviceProp
//{
//    char   name[256];                  /**< ASCII string identifying device */
//    size_t totalGlobalMem;             /**< Global memory available on device in bytes */
//    size_t sharedMemPerBlock;          /**< Shared memory available per block in bytes */
//    int    regsPerBlock;               /**< 32-bit registers available per block */
//    int    warpSize;                   /**< Warp size in threads */
//    size_t memPitch;                   /**< Maximum pitch in bytes allowed by memory copies */
//    int    maxThreadsPerBlock;         /**< Maximum number of threads per block */
//    int    maxThreadsDim[3];           /**< Maximum size of each dimension of a block */
//    int    maxGridSize[3];             /**< Maximum size of each dimension of a grid */
//    int    clockRate;                  /**< Clock frequency in kilohertz */
//    size_t totalConstMem;              /**< Constant memory available on device in bytes */
//    int    major;                      /**< Major compute capability */
//    int    minor;                      /**< Minor compute capability */
//    size_t textureAlignment;           /**< Alignment requirement for textures */
//    size_t texturePitchAlignment;      /**< Pitch alignment requirement for texture references bound to pitched memory */
//    int    deviceOverlap;              /**< Device can concurrently copy memory and execute a kernel. Deprecated. Use instead asyncEngineCount. */
//    int    multiProcessorCount;        /**< Number of multiprocessors on device */
//    int    kernelExecTimeoutEnabled;   /**< Specified whether there is a run time limit on kernels */
//    int    integrated;                 /**< Device is integrated as opposed to discrete */
//    int    canMapHostMemory;           /**< Device can map host memory with cudaHostAlloc/cudaHostGetDevicePointer */
//    int    computeMode;                /**< Compute mode (See ::cudaComputeMode) */
//    int    maxTexture1D;               /**< Maximum 1D texture size */
//    int    maxTexture1DLinear;         /**< Maximum size for 1D textures bound to linear memory */
//    int    maxTexture2D[2];            /**< Maximum 2D texture dimensions */
//    int    maxTexture2DLinear[3];      /**< Maximum dimensions (width, height, pitch) for 2D textures bound to pitched memory */
//    int    maxTexture2DGather[2];      /**< Maximum 2D texture dimensions if texture gather operations have to be performed */
//    int    maxTexture3D[3];            /**< Maximum 3D texture dimensions */
//    int    maxTextureCubemap;          /**< Maximum Cubemap texture dimensions */
//    int    maxTexture1DLayered[2];     /**< Maximum 1D layered texture dimensions */
//    int    maxTexture2DLayered[3];     /**< Maximum 2D layered texture dimensions */
//    int    maxTextureCubemapLayered[2];/**< Maximum Cubemap layered texture dimensions */
//    int    maxSurface1D;               /**< Maximum 1D surface size */
//    int    maxSurface2D[2];            /**< Maximum 2D surface dimensions */
//    int    maxSurface3D[3];            /**< Maximum 3D surface dimensions */
//    int    maxSurface1DLayered[2];     /**< Maximum 1D layered surface dimensions */
//    int    maxSurface2DLayered[3];     /**< Maximum 2D layered surface dimensions */
//    int    maxSurfaceCubemap;          /**< Maximum Cubemap surface dimensions */
//    int    maxSurfaceCubemapLayered[2];/**< Maximum Cubemap layered surface dimensions */
//    size_t surfaceAlignment;           /**< Alignment requirements for surfaces */
//    int    concurrentKernels;          /**< Device can possibly execute multiple kernels concurrently */
//    int    ECCEnabled;                 /**< Device has ECC support enabled */
//    int    pciBusID;                   /**< PCI bus ID of the device */
//    int    pciDeviceID;                /**< PCI device ID of the device */
//    int    pciDomainID;                /**< PCI domain ID of the device */
//    int    tccDriver;                  /**< 1 if device is a Tesla device using TCC driver, 0 otherwise */
//    int    asyncEngineCount;           /**< Number of asynchronous engines */
//    int    unifiedAddressing;          /**< Device shares a unified address space with the host */
//    int    memoryClockRate;            /**< Peak memory clock frequency in kilohertz */
//    int    memoryBusWidth;             /**< Global memory bus width in bits */
//    int    l2CacheSize;                /**< Size of L2 cache in bytes */
//    int    maxThreadsPerMultiProcessor;/**< Maximum resident threads per multiprocessor */
//};

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
        public SizeT texturePitchAlignment;/*4.2*/
        public int deviceOverlap;
        public int multiProcessorCount;
        public int kernelExecTimeoutEnabled;
        public int integrated;
        public int canMapHostMemory;
        public int computeMode;

        public int maxTexture1D;
        public int maxTexture1DLinear;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 2)]
        public int[] maxTexture2D;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
        public int[] maxTexture2DLinear;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 2)]
        public int[] maxTexture2DGather;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
        public int[] maxTexture3D;
        public int maxTextureCubemap;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 2)]
        public int[] maxTexture1DLayered;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
        public int[] maxTexture2DLayered;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 2)]
        public int[] maxTextureCubemapLayered;

        public int    maxSurface1D;               /**< Maximum 1D surface size */
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 2)]
        public int[]    maxSurface2D;            /**< Maximum 2D surface dimensions */
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
        public int[]    maxSurface3D;            /**< Maximum 3D surface dimensions */
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 2)]
        public int[]    maxSurface1DLayered;     /**< Maximum 1D layered surface dimensions */
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
        public int[]    maxSurface2DLayered;     /**< Maximum 2D layered surface dimensions */
        public int    maxSurfaceCubemap;          /**< Maximum Cubemap surface dimensions */
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 2)]
        public int[]    maxSurfaceCubemapLayered;/**< Maximum Cubemap layered surface dimensions */


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

