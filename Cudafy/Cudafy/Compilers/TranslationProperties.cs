/*
CUDAfy.NET - LGPL 2.1 License
Please consider purchasing a commerical license - it helps development, frees you from LGPL restrictions
and provides you with support.  Thank you!
Copyright (C) 2013 Hybrid DSP Systems
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
using System.IO;
using System.Xml;
using System.Xml.Linq;
using System.Text;
using System.Reflection;
namespace Cudafy
{

    
    public class CompileProperties
    {
        //-I"C:\Program Files (x86)\NVIDIA GPU Computing Toolkit\CUDA\v4.0\include" -m64  -arch=sm_20  -cubin  cudadevrt.lib  cublas_device.lib  -dlink  "C:\Sandbox\HybridDSPSystems\Codeplex\Cudafy\Cudafy.Host.UnitTests\bin\Debug\CUDAFYSOURCETEMP.cu"  -o "C:\Sandbox\HybridDSPSystems\Codeplex\Cudafy\Cudafy.Host.UnitTests\bin\Debug\CUDAFYSOURCETEMP.ptx"  -ptx
        // -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.0\include" -m64  -arch=sm_13  "C:\Sandbox\HybridDSPSystems\Codeplex\Cudafy\Cudafy.Host.UnitTests\bin\Debug\CUDAFYSOURCETEMP.cu"  -o "C:\Sandbox\HybridDSPSystems\Codeplex\Cudafy\Cudafy.Host.UnitTests\bin\Debug\CUDAFYSOURCETEMP.ptx"  -ptx

        public CompileProperties()
        {
            CompilerPath = "";
            IncludeDirectoryPath = "";
            WorkingDirectory = "";
            OutputFile = "CUDAFYSOURCETEMP.ptx";
            AdditionalInputArgs = "";
            AdditionalOutputArgs = "";
            TimeOut = 20000;
            Platform = ePlatform.Auto;
            Architecture = eArchitecture.sm_13;
            CompileMode = eCudafyCompileMode.Default;
            InputFile = "CUDAFYSOURCETEMP.cu";

        }

        private const string csM64 = "-m64";

        private const string csM32 = "-m32";



        public string CompilerPath { get; set; }
        
        public string IncludeDirectoryPath { get; set; }

        public string WorkingDirectory { get; set; }

        public string InputFile { get; set; }

        private string _outputFile = "";

        public string OutputFile { 
            get{ return _outputFile; }
            set{ ChangeOutputFilename(value);}
        }
                  
        public ePlatform Platform { get; set; }

        public eArchitecture Architecture { get; set; }

        public eLanguage Language
        {
            get
            {
                return Architecture.HasFlag((eArchitecture)32768) ? eLanguage.OpenCL : eLanguage.Cuda;
            }
        }

        public string PlatformArg 
        { 
            get 
            { 
                if(Platform == ePlatform.x64)
                    return csM64;
                else if(Platform == ePlatform.x86)
                    return csM32;
                else
                    return IntPtr.Size == 4 ? csM32 : csM64;;
            }
        }
        
        public eCudafyCompileMode CompileMode { get; set; }

        public string AdditionalInputArgs { get; set; }

        public string AdditionalOutputArgs { get; set; }

        public bool GenerateDebugInfo { get; set; }

        /// <summary>
        /// Gets or sets the time out for compilation.
        /// </summary>
        /// <value>
        /// The time out in milliseconds.
        /// </value>
        public int TimeOut { get; set; }

        //-m64  -arch=sm_20  -cubin  cudadevrt.lib  cublas_device.lib  -dlink 
        public string GetCommandString()
        {
            bool binary = (CompileMode & eCudafyCompileMode.Binary) == eCudafyCompileMode.Binary;
            string format = string.Format(@"{0} -I""{1}"" {2} -arch={3} {4} {5} {6}", 
                "", IncludeDirectoryPath, PlatformArg, //0,1,2
                Architecture, GenerateDebugInfo ? "-G" : "", (CompileMode & eCudafyCompileMode.Binary) == eCudafyCompileMode.Binary ? "-cubin " : "-ptx",//3,4,5
                AdditionalInputArgs//6
                );

            format += string.Format(@" ""{0}"" ", WorkingDirectory + DSChar + InputFile);

            if (!binary)
                format += string.Format(@" -o ""{0}"" ", WorkingDirectory + DSChar + OutputFile);
            return format;
        }

        private string ChangeOutputFilename(string newname)
        {          
            if (Path.HasExtension(OutputFile))
            {
                string ext = Path.GetExtension(OutputFile);
                newname = Path.GetFileNameWithoutExtension(newname) +  ext;
            }
            _outputFile = newname;
            return newname;
        }

        private char DSChar
        {
            get { return Path.DirectorySeparatorChar; }
        }
    }



    public class CompilerHelper
    {
        private static readonly string csGPUTOOLKIT = @"NVIDIA GPU Computing Toolkit"+Path.DirectorySeparatorChar+"CUDA"+Path.DirectorySeparatorChar;

        private static readonly string csNVCC = "nvcc";

        public static eLanguage GetLanguage(eArchitecture arch)
        {
            //return (((uint)arch & (uint)eArchitecture.OpenCL) == (uint)32768) ? eLanguage.OpenCL : eLanguage.Cuda;
            if (arch == eArchitecture.Unknown)
                return CudafyModes.Language;
            return arch.HasFlag((eArchitecture)32768) ? eLanguage.OpenCL : eLanguage.Cuda;
        }

        public static eGPUType GetGPUType(eArchitecture arch)
        {
            return arch.HasFlag((eArchitecture)32768) ? eGPUType.OpenCL : eGPUType.Cuda;
        }
        
        public static CompileProperties Create(ePlatform platform = ePlatform.Auto, eArchitecture arch = eArchitecture.sm_13, eCudafyCompileMode mode = eCudafyCompileMode.Default, string workingDir = null, bool debugInfo = false)
        {
            CompileProperties tp = new CompileProperties();
            eLanguage language = GetLanguage(arch);
            if (language == eLanguage.Cuda)
            {
                // Get ProgramFiles directory and CUDA directories
                // Get architecture
                string progFiles = null;
                switch (platform)
                {
                    case ePlatform.x64:
                        progFiles = Utility.ProgramFilesx64();
                        break;
                    case ePlatform.x86:
                        progFiles = Utility.ProgramFilesx86();
                        break;
                    default:
                        progFiles = Utility.ProgramFiles();
                        if (platform == ePlatform.Auto)
                            platform = IntPtr.Size == 4 ? ePlatform.x86 : ePlatform.x64;
                        break;
                }
                string toolkitbasedir = progFiles + Path.DirectorySeparatorChar + csGPUTOOLKIT;
                Version selVer;
                string cvStr = GetCudaVersion(toolkitbasedir, out selVer);
                if (string.IsNullOrEmpty(cvStr))
                    throw new CudafyCompileException(CudafyCompileException.csCUDA_DIR_NOT_FOUND);
                string gpuToolKit = progFiles + Path.DirectorySeparatorChar + csGPUTOOLKIT + cvStr;
                tp.CompilerPath = gpuToolKit + Path.DirectorySeparatorChar + @"bin" + Path.DirectorySeparatorChar + csNVCC;
                tp.IncludeDirectoryPath = gpuToolKit + Path.DirectorySeparatorChar + @"include";
                tp.Architecture = (arch == eArchitecture.Unknown) ? eArchitecture.sm_13 : arch;
                bool binary = ((mode & eCudafyCompileMode.Binary) == eCudafyCompileMode.Binary);
                string tempFileName = "CUDAFYSOURCETEMP.tmp";
                string cuFileName = tempFileName.Replace(".tmp", ".cu");
                string outputFileName = tempFileName.Replace(".tmp", binary ? ".cubin" : ".ptx");
                tp.InputFile = cuFileName;
                tp.OutputFile = outputFileName;
                if ((mode & eCudafyCompileMode.DynamicParallelism) == eCudafyCompileMode.DynamicParallelism)
                    tp.AdditionalInputArgs = "cudadevrt.lib  cublas_device.lib  -dlink";
            }
            else
            {
                mode = eCudafyCompileMode.TranslateOnly;
                tp.Architecture = (arch == eArchitecture.Unknown) ? eArchitecture.OpenCL : arch;
            }
            tp.WorkingDirectory = Directory.Exists(workingDir) ? workingDir : Environment.CurrentDirectory;

            tp.Platform = platform;
            tp.CompileMode = mode;         
            tp.GenerateDebugInfo = debugInfo;

            
            return tp;
        }

        private static string GetCudaVersion(string gpuToolKitDir, out Version selectedVersion)
        {
            string s = "v{0}.{1}";
            Version cudaVersion = null;
            selectedVersion = cudaVersion;
            for (int j = 5; j >= 4; j--)
            {
                for (int i = 9; i >= 0; i--)
                {
                    string version = string.Format(s, j, i);
                    string dir = gpuToolKitDir + version;
                    if (System.IO.Directory.Exists(dir))
                    {
                        selectedVersion = new Version(string.Format("{0}.{1}", j, i));
                        return version;
                    }
                }
            }
            return string.Empty;
        }
    }










    #region Proposed
    public abstract class ProgramModule
    {
        //-I"C:\Program Files (x86)\NVIDIA GPU Computing Toolkit\CUDA\v4.0\include" -m64  -arch=sm_20  -cubin  cudadevrt.lib  cublas_device.lib  -dlink  "C:\Sandbox\HybridDSPSystems\Codeplex\Cudafy\Cudafy.Host.UnitTests\bin\Debug\CUDAFYSOURCETEMP.cu"  -o "C:\Sandbox\HybridDSPSystems\Codeplex\Cudafy\Cudafy.Host.UnitTests\bin\Debug\CUDAFYSOURCETEMP.ptx"  -ptx
        // -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.0\include" -m64  -arch=sm_13  "C:\Sandbox\HybridDSPSystems\Codeplex\Cudafy\Cudafy.Host.UnitTests\bin\Debug\CUDAFYSOURCETEMP.cu"  -o "C:\Sandbox\HybridDSPSystems\Codeplex\Cudafy\Cudafy.Host.UnitTests\bin\Debug\CUDAFYSOURCETEMP.ptx"  -ptx

        protected readonly string csPROGRAMMODULE = "ProgramModule";

        public ProgramModule()
        {
            CompilerPath = "";
            IncludeDirectoryPath = "";
            WorkingDirectory = "";
            AdditionalInputArgs = "";

            TimeOut = 20000;
            Platform = ePlatform.Auto;
            Architecture = eArchitecture.sm_13;

            InputFile = "";
        }

        public string SourceID { get; set; }

        public virtual XElement PopulateXElement(XElement xe)
        {
            xe.Add(new XElement("InputFile", InputFile));
            xe.SetAttributeValue("Type", GetType());
            xe.SetAttributeValue("TimeOut", TimeOut);
            xe.SetAttributeValue("Platform", Platform);
            xe.SetAttributeValue("Architecture", Architecture);
            return xe;
        }

        public void SetFromXElement(XElement xe)
        {
            InputFile = xe.Element("InputFile").Value;
            TimeOut = xe.TryGetAttributeInt32Value("TimeOut").Value;
            Platform = xe.TryGetAttributeEnum<ePlatform>("Platform");
            Architecture = xe.TryGetAttributeEnum<eArchitecture>("Architecture");
        }


        public string CompilerPath { get; set; }

        public string IncludeDirectoryPath { get; set; }

        public string WorkingDirectory { get; set; }

        public string InputFile { get; set; }

        public ePlatform Platform { get; set; }

        public eArchitecture Architecture { get; set; }

        public eLanguage Language
        {
            get
            {
                return ((Architecture & eArchitecture.OpenCL) == (eArchitecture)32768) ? eLanguage.OpenCL : eLanguage.Cuda;
            }
        }

        public bool SupportDouble { get; set; }


        public string AdditionalInputArgs { get; set; }

        public bool GenerateDebugInfo { get; set; }

        /// <summary>
        /// Gets or sets the time out for compilation.
        /// </summary>
        /// <value>
        /// The time out in milliseconds.
        /// </value>
        public int TimeOut { get; set; }

        public abstract string GetCommandString();

        protected char DSChar
        {
            get { return Path.DirectorySeparatorChar; }
        }
    }

    public class OpenCLModule : ProgramModule
    {
        public override string GetCommandString()
        {
            return null;
        }

        //public override XElement PopulateXElement(XElement parent)
        //{
        //    XElement xe = parent.Element(csPROGRAMMODULE);
        //}
    }

    public abstract class CompilableModule : ProgramModule
    {
        public CompilableModule()
        {
            AdditionalOutputArgs = "";
            CompileMode = eCudafyCompileMode.Default;
        }

        protected byte[] _binary = new byte[0];

        public string AdditionalOutputArgs { get; set; }


        public eCudafyCompileMode CompileMode { get; set; }

        public void SetFromBase64(string base64)
        {
            _binary = Convert.FromBase64String(base64);
        }

        public void Set(string text)
        {
            _binary = Encoding.Default.GetBytes(text);
        }

        public void Set(byte[] binary)
        {
            _binary = binary;
        }

        public string GetBinaryAsString()
        {
            return Encoding.Default.GetString(_binary);
        }


        protected string ChangeOutputFilename(string newname)
        {
            if (Path.HasExtension(OutputFile))
            {
                string ext = Path.GetExtension(OutputFile);
                newname = newname + "." + ext;
            }
            _outputFile = newname;
            return newname;
        }

        private string _outputFile = "";

        public string OutputFile
        {
            get { return _outputFile; }
            set { ChangeOutputFilename(value); }
        }
    }

    public class CUDAModule : CompilableModule
    {

        private const string csM64 = "-m64";

        private const string csM32 = "-m32";


        public override string GetCommandString()
        {
            bool binary = (CompileMode & eCudafyCompileMode.Binary) == eCudafyCompileMode.Binary;
            string format = string.Format(@"{0} -I""{1}"" {2} -arch={3} {4} {5}",
                "", IncludeDirectoryPath, PlatformArg, //0,1,2
                Architecture, GenerateDebugInfo ? "-G" : "", (CompileMode & eCudafyCompileMode.Binary) == eCudafyCompileMode.Binary ? "-cubin " : "-ptx",//3,4
                AdditionalInputArgs//5
                );

            format += string.Format(@" ""{0}"" ", WorkingDirectory + DSChar + InputFile);

            if (!binary)
                format += string.Format(@" -o ""{0}"" ", WorkingDirectory + DSChar + OutputFile);
            return format;
        }

        private string PlatformArg
        {
            get
            {
                if (Platform == ePlatform.x64)
                    return csM64;
                else if (Platform == ePlatform.x86)
                    return csM32;
                else
                    return IntPtr.Size == 4 ? csM32 : csM64; ;
            }
        }



    }
    #endregion

}
