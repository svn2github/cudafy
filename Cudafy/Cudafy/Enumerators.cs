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

namespace Cudafy
{

    /// <summary>
    /// Flag for code generator.
    /// </summary>
    [Flags]
    public enum eGPUCodeGenerator
    {
        /// <summary>
        /// None selected.
        /// </summary>
        None = 0,
        /// <summary>
        /// Cuda C code.
        /// </summary>
        CudaC = 1,
        /// <summary>
        /// Generate code for all.
        /// </summary>
        All = 255
    };

    /// <summary>
    /// GPU target type.
    /// </summary>
    public enum eGPUType
    {
        /// <summary>
        /// Target GPU kernel emulator.
        /// </summary>
        Emulator,
        /// <summary>
        /// Target a Cuda GPU.
        /// </summary>
        Cuda
    }

    /// <summary>
    /// High level enumerator encapsulation eGPUType and eGPUCodeGenerator.
    /// </summary>
    public enum eCudafyQuickMode
    {
        /// <summary>
        /// Cuda emulator.
        /// </summary>
        CudaEmulate,
        /// <summary>
        /// Cuda.
        /// </summary>
        Cuda
    }

    /// <summary>
    /// Convenience class for storing three main types of enumerators.
    /// </summary>
    public class CudafyModes
    {
        /// <summary>
        /// Target GPU.
        /// </summary>
        public static eGPUType Target;
        /// <summary>
        /// Target compiler.
        /// </summary>
        public static eGPUCompiler Compiler;
        /// <summary>
        /// Target code generator.
        /// </summary>
        public static eGPUCodeGenerator CodeGen;

        /// <summary>
        /// Quick mode.
        /// </summary>
        public static eCudafyQuickMode Mode;

        /// <summary>
        /// Warning message if CRC check fails.
        /// </summary>
        public static string csCRCWARNING = "The Cudafy module was created from a different version of the .NET assembly.";

        /// <summary>
        /// Static constructor for the <see cref="CudafyModes"/> class.
        /// Sets CodeGen to CudaC, Compiler to CudaNvcc, Target to Cuda and Mode to Cuda.
        /// </summary>
        static CudafyModes()
        {
            CodeGen = eGPUCodeGenerator.CudaC;
            Compiler = eGPUCompiler.CudaNvcc;
            Target = eGPUType.Cuda;
            Mode = eCudafyQuickMode.Cuda;
        }
    }

    /// <summary>
    /// Enumerator for the type of CudafyAttribute.
    /// </summary>
    public enum eCudafyType
    {
        /// <summary>
        /// Auto. The code generator will determine it.
        /// </summary>
        Auto,
        /// <summary>
        /// Used to indicate a method that should be made into a Cuda C device function.
        /// </summary>
        Device,
        /// <summary>
        /// Used to indicate a method that should be made into a Cuda C global function.
        /// </summary>
        Global,
        /// <summary>
        /// Used to indicate a structure that should be converted to Cuda C.
        /// </summary>
        Struct,
        /// <summary>
        /// Used to indicate a static field that should be converted to Cuda C.
        /// </summary>
        Constant
    };


    /// <summary>
    /// Target platform.
    /// </summary>
    public enum ePlatform
    {
        /// <summary>
        /// x86
        /// </summary>
        x86,
        /// <summary>
        /// x64
        /// </summary>
        x64,

        /// <summary>
        /// None selected.
        /// </summary>
        Auto,

        /// <summary>
        /// Both x86 and x64
        /// </summary>
        All
    }


    /// <summary>
    /// CUDA Architecture
    /// </summary>
    public enum eArchitecture
    {
        /// <summary>
        /// sm_11
        /// </summary>
        sm_11,
        /// <summary>
        /// sm_12
        /// </summary>
        sm_12,
        /// <summary>
        /// sm_13
        /// </summary>
        sm_13,
        /// <summary>
        /// sm_20
        /// </summary>
        sm_20,
        /// <summary>
        /// sm_21
        /// </summary>
        sm_21,
        /// <summary>
        /// sm_30
        /// </summary>
        sm_30,
        /// <summary>
        /// sm_35
        /// </summary>
        sm_35
    }
}
