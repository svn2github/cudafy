namespace GASS.CUDA.OpenGL
{
    using GASS.CUDA;
    using GASS.CUDA.Types;
    using System;
    using System.Runtime.InteropServices;

    public class CUOpenGLDriver
    {
        [Obsolete("Use new graphics API."), DllImport("nvcuda")]
        public static extern CUResult cuGLCtxCreate(ref CUcontext pCtx, CUCtxFlags Flags, CUdevice device);
        [DllImport("nvcuda")]
        public static extern CUResult cuGLCtxCreate(ref CUcontext pCtx, uint Flags, CUdevice device);
        [Obsolete("Use new graphics API."), DllImport("nvcuda")]
        public static extern CUResult cuGLInit();
        [Obsolete("Use new graphics API."), DllImport("nvcuda")]
        public static extern CUResult cuGLMapBufferObject(ref CUdeviceptr dptr, ref uint size, uint bufferobj);
        [Obsolete("Use new graphics API."), DllImport("nvcuda")]
        public static extern CUResult cuGLMapBufferObjectAsync(ref CUdeviceptr dptr, ref uint size, uint bufferobj, CUstream hStream);
        [Obsolete("Use new graphics API."), DllImport("nvcuda")]
        public static extern CUResult cuGLRegisterBufferObject(uint bufferobj);
        [Obsolete("Use new graphics API."), DllImport("nvcuda")]
        public static extern CUResult cuGLSetBufferObjectMapFlags(uint bufferobj, uint Flags);
        [Obsolete("Use new graphics API."), DllImport("nvcuda")]
        public static extern CUResult cuGLUnmapBufferObject(uint bufferobj);
        [Obsolete("Use new graphics API."), DllImport("nvcuda")]
        public static extern CUResult cuGLUnmapBufferObjectAsync(uint bufferobj, CUstream hStream);
        [Obsolete("Use new graphics API."), DllImport("nvcuda")]
        public static extern CUResult cuGLUnregisterBufferObject(uint bufferobj);
        [DllImport("nvcuda")]
        public static extern CUResult cuGraphicsGLRegisterBuffer(ref CUgraphicsResource pCudaResource, uint buffer, uint Flags);
        [DllImport("nvcuda")]
        public static extern CUResult cuGraphicsGLRegisterImage(ref CUgraphicsResource pCudaResource, uint image, uint target, uint Flags);
        [DllImport("nvcuda")]
        public static extern CUResult cuWGLGetDevice(ref CUdevice pDevice, IntPtr hGpu);
    }
}

