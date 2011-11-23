/* Added by Kichang Kim (kkc0923@hotmail.com) */
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Cudafy.Maths.SPARSE.Types
{
    using System;
    using System.Runtime.InteropServices;

    [StructLayout(LayoutKind.Sequential)]
    public struct cusparseSolveAnalysisInfo
    {
        public uint ptr;
    }
}
