namespace GASS.CUDA
{
    using System;

    public enum CUDeviceAttribute
    {
        CanMapHostMemory = 0x13,
        ClockRate = 13,
        ComputeMode = 20,
        ConcurrentKernels = 0x1f,
        ECCEnabled = 0x20,
        GPUOverlap = 15,
        Integrated = 0x12,
        KernelExecTimeout = 0x11,
        MaxBlockDimX = 2,
        MaxBlockDimY = 3,
        MaxBlockDimZ = 4,
        MaxGridDimX = 5,
        MaxGridDimY = 6,
        MaxGridDimZ = 7,
        MaximumTexture1DWidth = 0x15,
        MaximumTexture2DArrayHeight = 0x1c,
        MaximumTexture2DArrayNumSlices = 0x1d,
        MaximumTexture2DArrayWidth = 0x1b,
        MaximumTexture2DHeight = 0x17,
        MaximumTexture2DWidth = 0x16,
        MaximumTexture3DDepth = 0x1a,
        MaximumTexture3DHeight = 0x19,
        MaximumTexture3DWidth = 0x18,
        MaxPitch = 11,
        MaxRegistersPerBlock = 12,
        MaxSharedMemoryPerBlock = 8,
        MaxThreadsPerBlock = 1,
        MultiProcessorCount = 0x10,
        [Obsolete("Use MaxRegistersPerBlock")]
        RegistersPerBlock = 12,
        [Obsolete("Use MaxSharedMemoryPerBlock")]
        SharedMemoryPerBlock = 8,
        SurfaceAlignment = 30,
        TextureAlignment = 14,
        TotalConstantMemory = 9,
        WarpSize = 10
    }
}

