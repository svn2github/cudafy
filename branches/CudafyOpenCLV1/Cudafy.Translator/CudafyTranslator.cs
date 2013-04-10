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
using System.IO;
using System.Text;
using System.Reflection;
using System.Diagnostics;
using ICSharpCode.Decompiler;
using ICSharpCode.ILSpy;
using Cudafy;
using Cudafy.Compilers;
using Mono.Cecil;
namespace Cudafy.Translator
{


    /// <summary>
    /// Implements translation of .NET code to CUDA C.
    /// </summary>
    public class CudafyTranslator
    {
        static CudafyTranslator()
        {
            TimeOut = 60000;
        }
        
        private static CUDALanguage _cl = new CUDALanguage(eLanguage.Cuda);

        internal static CUDAfyLanguageSpecifics LanguageSpecifics = new CUDAfyLanguageSpecifics();

        private static IEnumerable<Type> GetNestedTypes(Type type)
        {
            foreach (var nestedType in type.GetNestedTypes(System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.NonPublic))
            {
                if (nestedType.GetNestedTypes().Count() > 0)
                {
                    foreach (var nestedNestedType in GetNestedTypes(nestedType))
                        yield return nestedNestedType;
                }
                else
                {
                    //if(nestedType.IsClass)
                        yield return nestedType;
                }
            }
        }

        private static IEnumerable<Type> GetWithNestedTypes(Type[] types)
        {
            List<Type> typesList = new List<Type>();
            foreach (Type type in types.Distinct())
            {
                if (type == null)
                    continue;
                foreach (Type nestedType in GetNestedTypes(type))
                    typesList.Add(nestedType);
                typesList.Add(type);
            }
            return typesList.Distinct();
        }

        /// <summary>
        /// Gets or sets the language to generate.
        /// </summary>
        /// <value>
        /// The language.
        /// </value>
        public static eLanguage Language
        {
            get { return LanguageSpecifics.Language; }
            set 
            { 
                if (value != LanguageSpecifics.Language) 
                    _cl = new CUDALanguage(value); 
                LanguageSpecifics.Language = value; 
            }
        }

        /// <summary>
        /// Gets or sets the time out for compilation.
        /// </summary>
        /// <value>
        /// The time out in milliseconds.
        /// </value>
        public static int TimeOut { get; set; }

        /// <summary>
        /// Gets or sets a value indicating whether to compile for debug.
        /// </summary>
        /// <value>
        ///   <c>true</c> if compile for debug; otherwise, <c>false</c>.
        /// </value>
        public static bool GenerateDebug { get; set; }

        /// <summary>
        /// Tries to use a previous serialized CudafyModule else cudafies and compiles the type in which the calling method is located. 
        /// CUDA architecture is 1.3; platform is set to the current application's (x86 or x64); and the CUDA version is the 
        /// latest official release found on the current machine. 
        /// </summary>
        /// <returns>A CudafyModule.</returns>
        public static CudafyModule Cudafy()
        {
            StackTrace stackTrace = new StackTrace();
            Type type = stackTrace.GetFrame(1).GetMethod().ReflectedType;
            CudafyModule km = CudafyModule.TryDeserialize(type.Name);
            if (km == null || !km.TryVerifyChecksums())
            {
                km = Cudafy(ePlatform.Auto, eArchitecture.sm_13, type);
                km.Name = type.Name;
                km.TrySerialize();
            }
            return km;
        }

        /// <summary>
        /// Tries to use a previous serialized CudafyModule else cudafies and compiles the type in which the calling method is located. 
        /// CUDA architecture is as specified; platform is set to the current application's (x86 or x64); and the CUDA version is the 
        /// latest official release found on the current machine. 
        /// </summary>
        /// <param name="arch">The architecture.</param>
        /// <returns>A CudafyModule.</returns>
        public static CudafyModule Cudafy(eArchitecture arch)
        {
            StackTrace stackTrace = new StackTrace();
            Type type = stackTrace.GetFrame(1).GetMethod().ReflectedType;
            CudafyModule km = CudafyModule.TryDeserialize(type.Name);
            if (km == null || !km.TryVerifyChecksums())
            {
                km = Cudafy(ePlatform.Auto, arch, type);
                km.Name = type.Name;
                km.TrySerialize();
            }
            return km;
        }

        /// <summary>
        /// Tries to use a previous serialized CudafyModule else cudafies and compiles the type in which the calling method is located. 
        /// CUDA architecture is 1.3; platform is as specified; and the CUDA version is the 
        /// latest official release found on the current machine. 
        /// </summary>
        /// <param name="platform">The platform.</param>
        /// <returns></returns>
        public static CudafyModule Cudafy(ePlatform platform)
        {
            StackTrace stackTrace = new StackTrace();
            Type type = stackTrace.GetFrame(1).GetMethod().ReflectedType;
            CudafyModule km = CudafyModule.TryDeserialize(type.Name);
            if (km == null || !km.TryVerifyChecksums() || !km.HasPTXForPlatform(platform))
            {
                km = Cudafy(platform, eArchitecture.sm_13, type);
                km.Name = type.Name;
                km.TrySerialize();
            }
            return km;
        }

        /// <summary>
        /// Cudafies for the specified platform.
        /// </summary>
        /// <param name="platform">The platform.</param>
        /// <param name="arch">The architecture.</param>
        /// <returns>A CudafyModule.</returns>
        public static CudafyModule Cudafy(ePlatform platform, eArchitecture arch)
        {
            StackTrace stackTrace = new StackTrace();
            Type type = stackTrace.GetFrame(1).GetMethod().ReflectedType;
            CudafyModule km = CudafyModule.TryDeserialize(type.Name);
            if (km == null || !km.TryVerifyChecksums())
            {
                km = Cudafy(platform, arch, type);
                km.Name = type.Name;
                km.TrySerialize();
            }
            return km;
        }

        /// <summary>
        /// Cudafies and compiles the type of the specified object with default settings. 
        /// CUDA architecture is 1.3; platform is set to the current application's (x86 or x64); and the CUDA version is the 
        /// latest official release found on the current machine. 
        /// </summary>
        /// <param name="o">An instance of the type to cudafy. Typically pass 'this'.</param>
        /// <returns>A CudafyModule.</returns>
        public static CudafyModule Cudafy(object o)
        {
            Type currentType = o.GetType();
            return Cudafy(currentType);
        }

        /// <summary>
        /// Cudafies and compiles the specified types with default settings. 
        /// CUDA architecture is 1.3; platform is set to the current application's (x86 or x64); and the CUDA version is the 
        /// latest official release found on the current machine. 
        /// </summary>
        /// <param name="types">The types.</param>
        /// <returns>A CudafyModule.</returns>
        public static CudafyModule Cudafy(params Type[] types)
        {
            return Cudafy(ePlatform.Auto, eArchitecture.sm_13, null, true, types);
        }

        /// <summary>
        /// Cudafies the specified types for the specified platform.
        /// </summary>
        /// <param name="platform">The platform.</param>
        /// <param name="arch">The architecture.</param>
        /// <param name="types">The types.</param>
        /// <returns>A CudafyModule.</returns>
        public static CudafyModule Cudafy(ePlatform platform, eArchitecture arch, params Type[] types)
        {
            return Cudafy(platform, arch, null, true, types);
        }

        /// <summary>
        /// Cudafies the specified types for the specified architecture on automatic platform.
        /// </summary>
        /// <param name="arch">The architecture.</param>
        /// <param name="types">The types.</param>
        /// <returns>A CudafyModule.</returns>
        public static CudafyModule Cudafy(eArchitecture arch, params Type[] types)
        {
            return Cudafy(ePlatform.Auto, arch, null, true, types);
        }


        /// <summary>
        /// Cudafies the specified types.
        /// </summary>
        /// <param name="platform">The platform.</param>
        /// <param name="arch">The architecture.</param>
        /// <param name="cudaVersion">The cuda version.</param>
        /// <param name="compile">if set to <c>true</c> compile to PTX.</param>
        /// <param name="types">The types.</param>
        /// <returns>A CudafyModule.</returns>
        public static CudafyModule Cudafy(ePlatform platform, eArchitecture arch, Version cudaVersion, bool compile, params Type[] types)
        {
            CudafyModule km = null;
            CUDALanguage.ComputeCapability = GetComputeCapability(arch);
            km = DoCudafy(types);
            if (km == null)
                throw new CudafyFatalException(CudafyFatalException.csUNEXPECTED_STATE_X, "CudafyModule km = null");
            if (compile && LanguageSpecifics.Language == eLanguage.Cuda)
            {
                if (platform == ePlatform.Auto)
                    platform = IntPtr.Size == 8 ? ePlatform.x64 : ePlatform.x86;
                if (platform != ePlatform.x86)
                    km.CompilerOptionsList.Add(NvccCompilerOptions.Createx64(cudaVersion, arch));
                if (platform != ePlatform.x64)
                    km.CompilerOptionsList.Add(NvccCompilerOptions.Createx86(cudaVersion, arch));
                km.GenerateDebug = GenerateDebug;
                km.TimeOut = TimeOut;
                km.Compile(eGPUCompiler.CudaNvcc, false);
            }
            Type lastType = types.Last(t => t != null);
            if(lastType != null)
                km.Name = lastType.Name;
            return km;
        }

        private static Version GetComputeCapability(eArchitecture arch)
        {
            if (arch == eArchitecture.sm_11)
                return new Version(1, 1);
            else if (arch == eArchitecture.sm_12)
                return new Version(1, 2);
            else if (arch == eArchitecture.sm_13)
                return new Version(1, 3);
            else if (arch == eArchitecture.sm_20)
                return new Version(2, 0);
            else if (arch == eArchitecture.sm_21)
                return new Version(2, 1);
            else if (arch == eArchitecture.sm_30)
                return new Version(3, 0);
            else if (arch == eArchitecture.sm_35)
                return new Version(3, 5);
            throw new ArgumentException("Unknown architecture.");
        }
        
        private static CudafyModule DoCudafy(params Type[] types)
        {
            MemoryStream output = new MemoryStream();
            var outputSw = new StreamWriter(output);
            
            MemoryStream structs = new MemoryStream();
            var structsSw = new StreamWriter(structs);
            var structsPto = new PlainTextOutput(structsSw);
            
            MemoryStream declarations = new MemoryStream();
            var declarationsSw = new StreamWriter(declarations);
            var declarationsPto = new PlainTextOutput(declarationsSw);

            MemoryStream code = new MemoryStream();
            var codeSw = new StreamWriter(code);
            var codePto = new PlainTextOutput(codeSw);

            bool isDummy = false;
            eCudafyDummyBehaviour behaviour = eCudafyDummyBehaviour.Default;

            Dictionary<string, ModuleDefinition> modules = new Dictionary<string,ModuleDefinition>();

            var compOpts = new DecompilationOptions { FullDecompilation = true };

            CUDALanguage.Reset();

            CudafyModule cm = new CudafyModule();

            // Test structs
            //foreach (var strct in types.Where(t => !t.IsClass))
            //    if (strct.GetCustomAttributes(typeof(CudafyAttribute), false).Length == 0)
            //        throw new CudafyLanguageException(CudafyLanguageException.csCUDAFY_ATTRIBUTE_IS_MISSING_ON_X, strct.Name);

            IEnumerable<Type> typeList = GetWithNestedTypes(types);
            foreach (var type in typeList)
            {
                if(!modules.ContainsKey(type.Assembly.Location))
                    modules.Add(type.Assembly.Location, ModuleDefinition.ReadModule(type.Assembly.Location));                
            }
            
            foreach (var kvp in modules)
            {
                foreach (var td in kvp.Value.Types)
                {
                    List<TypeDefinition> tdList = new List<TypeDefinition>();
                    tdList.Add(td);
                    tdList.AddRange(td.NestedTypes);
                    
                    Type type = null;
                    foreach (var t in tdList)
                    {                        
                        type = typeList.Where(tt => tt.FullName.Replace("+", "") == t.FullName.Replace("/", "")).FirstOrDefault();

                        if (type == null)
                            continue;
                        Debug.WriteLine(t.FullName);
                        // Types                      
                        var attr = t.GetCudafyType(out isDummy, out behaviour);
                        if (attr != null)
                        {
                            _cl.DecompileType(t, structsPto, compOpts);
                            cm.Types.Add(type.FullName.Replace("+", ""), new KernelTypeInfo(type, isDummy, behaviour));
                        }
                        else if (t.Name == td.Name)
                        {
                            // Fields
                            foreach (var fi in td.Fields)
                            {
                                attr = fi.GetCudafyType(out isDummy, out behaviour);
                                if (attr != null)
                                {
                                    VerifyMemberName(fi.Name);
                                    System.Reflection.FieldInfo fieldInfo = type.GetField(fi.Name, BindingFlags.Static|BindingFlags.Public | BindingFlags.NonPublic);
                                    if(fieldInfo == null)
                                        throw new CudafyLanguageException(CudafyLanguageException.csX_ARE_NOT_SUPPORTED, "Non-static fields");
                                    int[] dims = _cl.GetFieldInfoDimensions(fieldInfo);
                                    _cl.DecompileCUDAConstantField(fi, dims, codePto, compOpts);
                                    var kci = new KernelConstantInfo(fi.Name, fieldInfo, isDummy);
                                    cm.Constants.Add(fi.Name, kci);
                                    CUDALanguage.AddConstant(kci);
                                }
                            }
#warning TODO Only Global Methods can be called from host
#warning TODO For OpenCL may need to do Methods once all Constants have been handled
                            // Methods
                            foreach (var med in td.Methods)
                            {
                                attr = med.GetCudafyType(out isDummy, out behaviour);
                                if (attr != null)
                                {
                                    if (!med.IsStatic)
                                        throw new CudafyLanguageException(CudafyLanguageException.csX_ARE_NOT_SUPPORTED, "Non-static methods");
                                    _cl.DecompileMethodDeclaration(med, declarationsPto, new DecompilationOptions { FullDecompilation = false });
                                    _cl.DecompileMethod(med, codePto, compOpts);
                                    MethodInfo mi = type.GetMethod(med.Name, BindingFlags.Static|BindingFlags.Public|BindingFlags.NonPublic);
                                    if (mi == null)
                                        continue;
                                    VerifyMemberName(med.Name);
                                    eKernelMethodType kmt = eKernelMethodType.Device;
                                    kmt = GetKernelMethodType(attr, mi);
                                    cm.Functions.Add(med.Name, new KernelMethodInfo(type, mi, kmt, isDummy, behaviour, cm));
                                }
                            }
                        }
                    }
                }
            }

            codeSw.Flush();

            if (CudafyTranslator.Language == eLanguage.OpenCL)
            {
                outputSw.WriteLine("#if defined(cl_khr_fp64)");
                outputSw.WriteLine("#pragma OPENCL EXTENSION cl_khr_fp64: enable");
                outputSw.WriteLine("#elif defined(cl_amd_fp64)");
                outputSw.WriteLine("#pragma OPENCL EXTENSION cl_amd_fp64: enable");
                outputSw.WriteLine("#endif");
            }

            foreach (var oh in CUDALanguage.OptionalHeaders)
                if (oh.Used)
                    outputSw.WriteLine(oh.IncludeLine);
            foreach (var oh in CUDALanguage.OptionalFunctions)
                if (oh.Used)
                    outputSw.WriteLine(oh.Code);
            //outputSw.WriteLine(@"#include <curand_kernel.h>");


            declarationsSw.WriteLine();
            declarationsSw.Flush();

            structsSw.WriteLine();
            structsSw.Flush();

            foreach (var def in cm.GetDummyDefines())
                outputSw.WriteLine(def);
            foreach (var inc in cm.GetDummyStructIncludes())
                outputSw.WriteLine(inc);
            foreach (var inc in cm.GetDummyIncludes())
                outputSw.WriteLine(inc);
            outputSw.Flush();

            output.Write(structs.GetBuffer(), 0, (int)structs.Length);
            output.Write(declarations.GetBuffer(), 0, (int)declarations.Length);
            output.Write(code.GetBuffer(), 0, (int)code.Length);
            outputSw.Flush();
#if DEBUG
            using (FileStream fs = new FileStream("output.cu", FileMode.Create))
            {
                fs.Write(output.GetBuffer(), 0, (int)output.Length);
            }
#endif
            String s = Encoding.UTF8.GetString(output.GetBuffer(), 0, (int)output.Length);
            cm.CudaSourceCode = s;

            return cm;
        }

        private static string[] OpenCLReservedNames = new string[] { "kernel", "global" };

        private static void VerifyMemberName(string name)
        {
            if (LanguageSpecifics.Language == eLanguage.OpenCL && OpenCLReservedNames.Any(rn => rn == name))
                throw new CudafyLanguageException(CudafyLanguageException.csX_IS_A_RESERVED_KEYWORD, name);
        }

        private static eKernelMethodType GetKernelMethodType(eCudafyType? attr, MethodInfo mi)
        {
            eKernelMethodType kmt;
            if (attr == eCudafyType.Auto)
                kmt = mi.ReturnType.Name == "Void" ? eKernelMethodType.Global : eKernelMethodType.Device;
            else if (attr == eCudafyType.Device)
                kmt = eKernelMethodType.Device;
            else if (attr == eCudafyType.Global && mi.ReturnType.Name != "Void")
                throw new CudafyException(CudafyException.csX_NOT_SUPPORTED, "Return values on global methods");
            else if (attr == eCudafyType.Global)
                kmt = eKernelMethodType.Global;
            else if (attr == eCudafyType.Struct)
                throw new CudafyException(CudafyException.csX_NOT_SUPPORTED, "Cudafy struct attribute on methods");
            else
                throw new CudafyFatalException(attr.ToString());
            return kmt;
        }
    }

    public class CUDAfyLanguageSpecifics
    {
        public eLanguage Language { get; set; }

        public string KernelFunctionModifiers
        {
            get
            {
                if (Language == eLanguage.Cuda)
                    return @"extern ""C"" __global__ ";
                else
                    return "__kernel ";
            }
        }

        public string DeviceFunctionModifiers
        {
            get
            {
                if (Language == eLanguage.Cuda)
                    return "__device__ ";
                else
                    return " ";
            }
        }

        public string MemorySpaceSpecifier
        {
            get
            {
                if (Language == eLanguage.Cuda)
                    return "";
                else
                    return "global";
            }
        }

        public string SharedModifier
        {
            get
            {
                if (Language == eLanguage.Cuda)
                    return "__shared__";
                else
                    return "__local";
            }
        }

        public string ConstantModifier
        {
            get
            {
                if (Language == eLanguage.Cuda)
                    return "__constant__";
                else
                    return "__constant";
            }
        }

        public string GetAddressSpaceQualifier(eCudafyAddressSpace qualifier)
        {
            string addressSpaceQualifier = string.Empty;
            if (Language == eLanguage.OpenCL)
            {
                if ((qualifier & eCudafyAddressSpace.Global) == eCudafyAddressSpace.Global)
                {
                    return "global";
                }
                else if ((qualifier & eCudafyAddressSpace.Constant) == eCudafyAddressSpace.Constant)
                {
                    return "constant";
                }
                else if ((qualifier & eCudafyAddressSpace.Shared) == eCudafyAddressSpace.Shared)
                {
                    return "local";
                }
                else if ((qualifier & eCudafyAddressSpace.Private) == eCudafyAddressSpace.Private)
                {
                    return "private";
                }
            }
            return addressSpaceQualifier;
        }

        public string Int64Translation
        {
            get
            {
                if (Language == eLanguage.Cuda)
                    return "long long";
                else
                    return "long";
            }
        }

        public string UInt64Translation
        {
            get
            {
                if (Language == eLanguage.Cuda)
                    return "unsigned long long";
                else
                    return "ulong";
            }
        }

        public string PositiveInfinitySingle
        {
            get
            {
                if (Language == eLanguage.Cuda)
                    return "0x7ff00000";
                else
                    return "INFINITY";
            }
        }

        public string NegativeInfinitySingle
        {
            get
            {
                if (Language == eLanguage.Cuda)
                    return "0xfff00000";
                else
                    return "INFINITY";
            }
        }

        public string NaNSingle
        {
            get
            {
                if (Language == eLanguage.Cuda)
                    return "logf(-1.0F)";
                else
                    return "NAN";
            }
        }

        public string PositiveInfinityDouble
        {
            get
            {
                if (Language == eLanguage.Cuda)
                    return "0x7ff0000000000000";
                else
                    return "INFINITY";
            }
        }

        public string NegativeInfinityDouble
        {
            get
            {
                if (Language == eLanguage.Cuda)
                    return "0xfff0000000000000";
                else
                    return "INFINITY";
            }
        }

        public string NaNDouble
        {
            get
            {
                if (Language == eLanguage.Cuda)
                    return "log(-1.0)";
                else
                    return "NAN";
            }
        }
    }
}
