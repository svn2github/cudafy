namespace GASS.CUDA.Direct3D
{
    using GASS.CUDA;
    using GASS.CUDA.Types;
    using System;
    using System.Runtime.InteropServices;

    public class CUD3D11Driver
    {
        [DllImport("nvcuda")]
        public static extern CUResult cuD3D11CtxCreate(ref CUcontext pCtx, ref CUdevice pCudaDevice, uint Flags, IntPtr pD3DDevice);
        [DllImport("nvcuda")]
        public static extern CUResult cuD3D11GetDevice(ref CUdevice pCudaDevice, IntPtr pAdapter);
        [DllImport("nvcuda")]
        public static extern CUResult cuGraphicsD3D11RegisterResource(ref CUgraphicsResource pCudaResource, IntPtr pD3DResource, uint Flags);
    }
}

