/*
 * Added by Kichang Kim
 * kkc0923@hotmail.com
 * */
namespace Cudafy.Maths.SPARSE.Types
{
    using System;
    using System.Runtime.InteropServices;

    [StructLayout(LayoutKind.Sequential)]
    public struct cusparseHandle
    {
        public uint handle;
    }
}
