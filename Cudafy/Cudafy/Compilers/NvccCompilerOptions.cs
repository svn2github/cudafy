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
using System.Diagnostics;
namespace Cudafy.Compilers
{
    /// <summary>
    /// Compiler options.
    /// </summary>
    public class NvccCompilerOptions : CompilerOptions
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="NvccCompilerOptions"/> class.
        /// </summary>
        /// <param name="name">The name.</param>
        public NvccCompilerOptions(string name)
            : base(name, "nvcc.exe", string.Empty)
        {

        }

        /// <summary>
        /// Initializes a new instance of the <see cref="NvccCompilerOptions"/> class.
        /// </summary>
        /// <param name="name">The name.</param>
        /// <param name="compiler">The compiler.</param>
        /// <param name="includeDirectory">The include directory.</param>
        public NvccCompilerOptions(string name, string compiler, string includeDirectory)
            : base(name, compiler, includeDirectory)
        {

        }

        /// <summary>
        /// Gets the arguments.
        /// </summary>
        /// <returns></returns>
        public override string GetArguments()
        {
            string command = string.Empty;

            string includeDir = string.IsNullOrEmpty(Include) ? string.Empty : @" -I""" + Include + @"""";
            command += includeDir;
            foreach (string opt in Options)
                command += string.Format(" {0} ", opt);

            if (GenerateDebugInfo)
                command += " -G0 ";

            if (Sources.Count() == 0)
                throw new CudafyCompileException(CudafyCompileException.csNO_SOURCES);
            foreach (string src in Sources)
                command += string.Format(@" ""{0}"" ", src);

            if(Outputs.Count() == 1)
                command += string.Format(@" -o ""{0}"" ", Outputs.Take(1).FirstOrDefault());

            command += " --ptx";
            return command;
        }


        private const string csGPUTOOLKIT = @"NVIDIA GPU Computing Toolkit\CUDA\";

        /// <summary>
        /// Creates a default x86 instance. Architecture is 1.2.
        /// </summary>
        /// <returns></returns>
        public static NvccCompilerOptions Create()
        {
            NvccCompilerOptions opt = Createx86(null, eArchitecture.sm_12);
            opt.CanEdit = true;
            return opt;
        }

        /// <summary>
        /// Creates a default x86 instance. Architecture is 1.2.
        /// </summary>
        /// <returns></returns>
        public static NvccCompilerOptions Createx86()
        {
            return Createx86(null, eArchitecture.sm_12);
        }

        private static void AddArchOptions(CompilerOptions co, eArchitecture arch)
        {
            if (arch == eArchitecture.sm_11)
                co.AddOption("-arch=sm_11");
            else if (arch == eArchitecture.sm_12)
                co.AddOption("-arch=sm_12");
            else if (arch == eArchitecture.sm_13)
                co.AddOption("-arch=sm_13");
            else if (arch == eArchitecture.sm_20)
                co.AddOption("-arch=sm_20");
            else
                throw new NotImplementedException(arch.ToString());
        }

        /// <summary>
        /// Creates a default x86 instance for specified architecture.
        /// </summary>
        /// <param name="arch">The architecture.</param>
        /// <returns></returns>
        public static NvccCompilerOptions Createx86(eArchitecture arch)
        {
            return Createx86(null, arch);
        }

        /// <summary>
        /// Creates am x86 instance based on the specified cuda version.
        /// </summary>
        /// <param name="cudaVersion">The cuda version.</param>
        /// <param name="arch">Architecture.</param>
        /// <returns></returns>
        public static NvccCompilerOptions Createx86(Version cudaVersion, eArchitecture arch)
        {
            string progFiles = Utility.ProgramFilesx86();
            string toolkitbasedir = progFiles + Path.DirectorySeparatorChar + csGPUTOOLKIT;
            string cvStr = GetCudaVersion(cudaVersion, toolkitbasedir);
            Debug.WriteLineIf(!string.IsNullOrEmpty(cvStr), "Compiler version: " + cvStr);
            string gpuToolKit = progFiles + Path.DirectorySeparatorChar + csGPUTOOLKIT + cudaVersion;
            string compiler = gpuToolKit + Path.DirectorySeparatorChar + cvStr + Path.DirectorySeparatorChar + @"bin\nvcc.exe";
            string includeDir = gpuToolKit + Path.DirectorySeparatorChar + cvStr + Path.DirectorySeparatorChar + @"include";
            NvccCompilerOptions opt = new NvccCompilerOptions("NVidia CC (x86)", compiler, includeDir);
            if (!opt.TryTest())
            {
                opt = new NvccCompilerOptions("NVidia CC (x86)", "nvcc.exe", string.Empty);
#if DEBUG
                throw new CudafyCompileException("Test failed for NvccCompilerOptions for x86");
#endif
            }
            opt.Platform = ePlatform.x86;
            AddArchOptions(opt, arch);
            return opt;
        }

        /// <summary>
        /// Creates a default x64 instance. Architecture is 1.2.
        /// </summary>
        /// <returns></returns>
        public static NvccCompilerOptions Createx64()
        {
            return Createx64(null, eArchitecture.sm_12);
        }

        /// <summary>
        /// Creates a default x64 instance for specified architecture.
        /// </summary>
        /// <param name="arch">The architecture.</param>
        /// <returns></returns>
        public static NvccCompilerOptions Createx64(eArchitecture arch)
        {
            return Createx64(null, arch);
        }

        /// <summary>
        /// Creates an x64 instance based on the specified cuda version.
        /// </summary>
        /// <param name="cudaVersion">The cuda version or null for auto.</param>
        /// <param name="arch">Architecture.</param>
        /// <returns></returns>
        /// <exception cref="NotSupportedException">ProgramFilesx64 not found.</exception>
        public static NvccCompilerOptions Createx64(Version cudaVersion, eArchitecture arch)
        {
            string progFiles = Utility.ProgramFilesx64();
            string toolkitbasedir = progFiles + Path.DirectorySeparatorChar + csGPUTOOLKIT;
            string cvStr = GetCudaVersion(cudaVersion, toolkitbasedir);
            Debug.WriteLineIf(!string.IsNullOrEmpty(cvStr), "Compiler version: " + cvStr);
            string gpuToolKit = progFiles + Path.DirectorySeparatorChar + csGPUTOOLKIT + cudaVersion;
            string compiler = gpuToolKit + Path.DirectorySeparatorChar + cvStr + Path.DirectorySeparatorChar + @"bin\nvcc.exe";
            string includeDir = gpuToolKit + Path.DirectorySeparatorChar + cvStr + Path.DirectorySeparatorChar + @"include";
            NvccCompilerOptions opt = new NvccCompilerOptions("NVidia CC (x64)", compiler, includeDir);
            if (!opt.TryTest())
            {
                opt = new NvccCompilerOptions("NVidia CC (x64)", "nvcc.exe", string.Empty);
#if DEBUG
                throw new CudafyCompileException("Test failed for NvccCompilerOptions for x86");
#endif
            }
            opt.AddOption("-m64");
            //opt.AddOption("-DCUDA_FORCE_API_VERSION=3010"); //For mixed bitness mode
            //if(Directory.Exists(@"C:\Program Files (x86)\Microsoft Visual Studio 10.0\VC\include"))
            //    opt.AddOption(@"-I""C:\Program Files (x86)\Microsoft Visual Studio 10.0\VC\include""");
            //else
            //    opt.AddOption(@"-I""C:\Program Files (x86)\Microsoft Visual Studio 9.0\VC\include""");
            opt.Platform = ePlatform.x64;
            AddArchOptions(opt, arch);
            return opt;
        }


        private static string GetCudaVersion(Version cudaVersion, string gpuToolKitDir)
        {
            string s = "v{0}.{1}";
            if (cudaVersion != null)
                return string.Format(s, cudaVersion.Major, cudaVersion.Minor);

            // Version 4.x Support            
            for (int i = 9; i >= 0; i--)
            {
                string version = string.Format(s, 4, i);
                string dir = gpuToolKitDir + version;
                if (System.IO.Directory.Exists(dir))
                    return version;
            }
            //// Version 3.2 Support            
            //for (int i = 9; i >= 2; i--)
            //{
            //    string version = string.Format(s, 3, i);
            //    string dir = gpuToolKitDir + version;
            //    if (System.IO.Directory.Exists(dir))
            //        return version;
            //}

            return string.Empty;
        }
    }
}
