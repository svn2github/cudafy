using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Reflection;
namespace Cudafy
{
    public static class AssemblyExtensions
    {
        public static bool HasCudafyModule(this Assembly assembly)
        {
            return CudafyModule.HasCudafyModule(assembly);
        }

        public static CudafyModule GetCudafyModule(this Assembly assembly)
        {
            return CudafyModule.GetFromAssembly(assembly);
        }
    }
}
