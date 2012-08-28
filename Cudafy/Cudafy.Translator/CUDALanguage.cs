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
using System.Collections;
using System.Collections.Generic;
using System.ComponentModel.Composition;
using System.Linq;
using System.Text;
using System.Globalization;
using System.Diagnostics;
using ICSharpCode.ILSpy;
using Mono.Cecil;
using ICSharpCode.NRefactory.CSharp;
using ICSharpCode.Decompiler;
using ICSharpCode.Decompiler.Ast.Transforms;
namespace Cudafy.Translator
{
#pragma warning disable 1591
    public class SpecialMember
    {
        public SpecialMember(string declaringType, string original, Func<MemberReferenceExpression, object, string> func, bool callFunc = true)
        {
            OriginalName = original;
            //_translation = translation;
            DeclaringType = declaringType;
            Function = func;
            CallFunction = callFunc;
        }

        public bool CallFunction { get; private set; }

        public string DeclaringType { get; private set; }
        
        public string OriginalName { get; private set; }

        //private string _translation;

        public Func<MemberReferenceExpression, object, string> Function { get; private set; }

        public virtual string GetTranslation(MemberReferenceExpression mre, object data = null)
        {
            return Function(mre, data);
        }
    }

    public class OptionalHeader
    {
        public OptionalHeader(string name, string includeLine)
        {
            Name = name;
            IncludeLine = includeLine;
        }
        
        public string Name { get; private set; }
        public string IncludeLine { get; private set; }
        public bool Used { get; set; }
    }
    
    //[Export(typeof(Language))]
    public class CUDALanguage : Language
    {
        public CUDALanguage()
        {
            
        }
        
        private Predicate<IAstTransform> transformAbortCondition = null;

        public static bool DisableSmartArray { get; set; }
        
        public override string FileExtension
        {
            get { return ".cu"; }
        }

        public override string Name
        {
            get { return "CUDA"; }
        }

        public static Version ComputeCapability { get; set; }

        public int[] GetFieldInfoDimensions(System.Reflection.FieldInfo fieldInfo)
        {            
            Array array = fieldInfo.GetValue(null) as Array;
            if (array == null)
                return new int[0];

            List<int> dims = new List<int>();
            for (int i = 0; i < array.Rank; i++)
            {
                int len = array.GetLength(i);// GetUpperBound(i) + 1;
                dims.Add(len);
            }
            return dims.ToArray();
        }

        public override void DecompileField(FieldDefinition field, ITextOutput output, DecompilationOptions options)
        {
            WriteCommentLine(output, TypeToString(field.DeclaringType, includeNamespace: true));
            CUDAAstBuilder codeDomBuilder = CreateCUDAAstBuilder(options, currentType: field.DeclaringType, isSingleMember: true);
            codeDomBuilder.AddField(field);
            RunTransformsAndGenerateCode(codeDomBuilder, output, options);
        }

        public void DecompileCUDAConstantField(FieldDefinition field, int[] dims, ITextOutput output, DecompilationOptions options)
        {
            WriteCommentLine(output, TypeToString(field.DeclaringType, includeNamespace: true));
            CUDAAstBuilder codeDomBuilder = CreateCUDAAstBuilder(options, currentType: field.DeclaringType, isSingleMember: true);
            codeDomBuilder.AddField(field);
            RunTransformsAndGenerateCode(codeDomBuilder, output, options, dims);
        }

        public override void DecompileMethod(MethodDefinition method, ICSharpCode.Decompiler.ITextOutput output, DecompilationOptions options)
        {
            WriteCommentLine(output, TypeToString(method.DeclaringType, includeNamespace: true));
            CUDAAstBuilder codeDomBuilder = CreateCUDAAstBuilder(options, currentType: method.DeclaringType, isSingleMember: true);
            codeDomBuilder.AddMethod(method);
            RunTransformsAndGenerateCode(codeDomBuilder, output, options);
        }

        public void DecompileMethodDeclaration(MethodDefinition method, ICSharpCode.Decompiler.ITextOutput output, DecompilationOptions options)
        {
            WriteCommentLine(output, TypeToString(method.DeclaringType, includeNamespace: true));
            CUDAAstBuilder codeDomBuilder = CreateCUDAAstBuilder(options, currentType: method.DeclaringType, isSingleMember: true);
            codeDomBuilder.DecompileMethodBodies = false;
            codeDomBuilder.AddMethod(method);
            RunTransformsAndGenerateCode(codeDomBuilder, output, options);
        }

        public override void DecompileType(TypeDefinition type, ITextOutput output, DecompilationOptions options)
        {
            CUDAAstBuilder codeDomBuilder = CreateCUDAAstBuilder(options, currentType: type);
            codeDomBuilder.AddType(type);
            RunTransformsAndGenerateCode(codeDomBuilder, output, options);
        }

        void RunTransformsAndGenerateCode(CUDAAstBuilder astBuilder, ITextOutput output, DecompilationOptions options, int[] lastDims = null)
        {
            astBuilder.RunTransformations(transformAbortCondition);
            //if (options.DecompilerSettings.ShowXmlDocumentation)
            //    AddXmlDocTransform.Run(astBuilder.CompilationUnit);
            astBuilder.ConstantDims = lastDims;
            astBuilder.GenerateCode(output);
        }

        CUDAAstBuilder CreateCUDAAstBuilder(DecompilationOptions options, ModuleDefinition currentModule = null, TypeDefinition currentType = null, bool isSingleMember = false)
        {
            if (currentModule == null)
                currentModule = currentType.Module;
            DecompilerSettings settings = options.DecompilerSettings;
            if (isSingleMember)
            {
                settings = settings.Clone();
                settings.UsingDeclarations = false;
            }
            return new CUDAAstBuilder(
                new DecompilerContext(currentModule)
                {
                    CancellationToken = options.CancellationToken,
                    CurrentType = currentType,
                    Settings = settings
                });
        }

        /// <summary>
        /// Initializes the <see cref="CUDALanguage"/> class.
        /// </summary>
        static CUDALanguage()
        {
            ComputeCapability = new Version(1, 2);
            
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "SyncThreads", new Func<MemberReferenceExpression, object, string>(TranslateSyncThreads)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "SyncThreadsCount", new Func<MemberReferenceExpression, object, string>(TranslateSyncThreadsCount)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "All", new Func<MemberReferenceExpression, object, string>(TranslateAll)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "Any", new Func<MemberReferenceExpression, object, string>(TranslateAny)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "Ballot", new Func<MemberReferenceExpression, object, string>(TranslateBallot)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "atomicAdd", new Func<MemberReferenceExpression, object, string>(GetMemberName)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "atomicSub", new Func<MemberReferenceExpression, object, string>(GetMemberName)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "atomicExch", new Func<MemberReferenceExpression, object, string>(GetMemberName)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "atomicAdd", new Func<MemberReferenceExpression, object, string>(GetMemberName)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "atomicMin", new Func<MemberReferenceExpression, object, string>(GetMemberName)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "atomicMax", new Func<MemberReferenceExpression, object, string>(GetMemberName)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "atomicInc", new Func<MemberReferenceExpression, object, string>(GetMemberName)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "atomicDec", new Func<MemberReferenceExpression, object, string>(GetMemberName)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "atomicCAS", new Func<MemberReferenceExpression, object, string>(GetMemberName)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "atomicAnd", new Func<MemberReferenceExpression, object, string>(GetMemberName)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "atomicOr", new Func<MemberReferenceExpression, object, string>(GetMemberName)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "atomicXor", new Func<MemberReferenceExpression, object, string>(GetMemberName)));

            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "curand_init", new Func<MemberReferenceExpression, object, string>(GetCURANDMemberName)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "curand", new Func<MemberReferenceExpression, object, string>(GetCURANDMemberName)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "curand_log_normal", new Func<MemberReferenceExpression, object, string>(GetCURANDMemberName)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "curand_log_normal_double", new Func<MemberReferenceExpression, object, string>(GetCURANDMemberName)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "curand_normal", new Func<MemberReferenceExpression, object, string>(GetCURANDMemberName)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "curand_normal_double", new Func<MemberReferenceExpression, object, string>(GetCURANDMemberName)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "curand_uniform", new Func<MemberReferenceExpression, object, string>(GetCURANDMemberName)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "curand_uniform_double", new Func<MemberReferenceExpression, object, string>(GetCURANDMemberName)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "skipahead", new Func<MemberReferenceExpression, object, string>(GetCURANDMemberName)));
            SpecialMethods.Add(new SpecialMember("Cudafy.GThread", "skipahead_sequence", new Func<MemberReferenceExpression, object, string>(GetCURANDMemberName)));

            SpecialMethods.Add(new SpecialMember("GMath", null, new Func<MemberReferenceExpression, object, string>(TranslateGMath)));
            SpecialMethods.Add(new SpecialMember("Math", null, new Func<MemberReferenceExpression, object, string>(TranslateMath)));

            SpecialMethods.Add(new SpecialMember("ComplexD", null, new Func<MemberReferenceExpression, object, string>(TranslateComplexD)));
            SpecialMethods.Add(new SpecialMember("ComplexF", null, new Func<MemberReferenceExpression, object, string>(TranslateComplexF)));

            SpecialMethods.Add(new SpecialMember("ArrayType", "GetLength", new Func<MemberReferenceExpression, object, string>(TranslateArrayGetLength), false));

            SpecialMethods.Add(new SpecialMember("ComplexD", "ctor", new Func<MemberReferenceExpression, object, string>(TranslateComplexDCtor)));
            SpecialMethods.Add(new SpecialMember("ComplexF", "ctor", new Func<MemberReferenceExpression, object, string>(TranslateComplexFCtor)));
            
            //SpecialMethods.Add(new SpecialMember("Debug", null, new Func<MemberReferenceExpression, object, string>(CommentMeOut), false));
            //SpecialMethods.Add(new SpecialMember("Console", null, new Func<MemberReferenceExpression, object, string>(CommentMeOut), false));

            SpecialMethods.Add(new SpecialMember("Debug", "Write", new Func<MemberReferenceExpression, object, string>(TranslateToPrintF), false));
            SpecialMethods.Add(new SpecialMember("Debug", "WriteIf", new Func<MemberReferenceExpression, object, string>(TranslateToPrintF), false));
            SpecialMethods.Add(new SpecialMember("Debug", "WriteLine", new Func<MemberReferenceExpression, object, string>(TranslateToPrintF), false));
            SpecialMethods.Add(new SpecialMember("Debug", "WriteLineIf", new Func<MemberReferenceExpression, object, string>(TranslateToPrintF), false));
            SpecialMethods.Add(new SpecialMember("Debug", null, new Func<MemberReferenceExpression, object, string>(CommentMeOut), false));
            SpecialMethods.Add(new SpecialMember("Console", "Write", new Func<MemberReferenceExpression, object, string>(TranslateToPrintF), false));
            SpecialMethods.Add(new SpecialMember("Console", "WriteLine", new Func<MemberReferenceExpression, object, string>(TranslateToPrintF), false));
            SpecialMethods.Add(new SpecialMember("Console", null, new Func<MemberReferenceExpression, object, string>(CommentMeOut), false));
            SpecialMethods.Add(new SpecialMember("Debug", "Assert", new Func<MemberReferenceExpression, object, string>(TranslateAssert), false));
            SpecialMethods.Add(new SpecialMember("Trace", null, new Func<MemberReferenceExpression, object, string>(CommentMeOut), false));

            SpecialProperties.Add(new SpecialMember("ArrayType", "Length", new Func<MemberReferenceExpression, object, string>(TranslateArrayLength)));
            SpecialProperties.Add(new SpecialMember("ArrayType", "LongLength", new Func<MemberReferenceExpression, object, string>(TranslateArrayLength)));
            SpecialProperties.Add(new SpecialMember("ArrayType", "IsFixedSize", new Func<MemberReferenceExpression, object, string>(TranslateToTrue)));
            SpecialProperties.Add(new SpecialMember("ArrayType", "IsReadOnly", new Func<MemberReferenceExpression, object, string>(TranslateToFalse)));
            SpecialProperties.Add(new SpecialMember("ArrayType", "IsSynchronized", new Func<MemberReferenceExpression, object, string>(TranslateToFalse)));
            SpecialProperties.Add(new SpecialMember("ArrayType", "Rank", new Func<MemberReferenceExpression, object, string>(TranslateArrayRank)));
            SpecialProperties.Add(new SpecialMember("Cudafy.GThread", "warpSize", new Func<MemberReferenceExpression, object, string>(GetMemberName)));
            //
            SpecialProperties.Add(new SpecialMember("System.String", "Length", new Func<MemberReferenceExpression, object, string>(TranslateStringLength)));
            
            SpecialProperties.Add(new SpecialMember("Math", "E", new Func<MemberReferenceExpression, object, string>(TranslateMathE)));
            SpecialProperties.Add(new SpecialMember("Math", "PI", new Func<MemberReferenceExpression, object, string>(TranslateMathPI)));
            SpecialProperties.Add(new SpecialMember("GMath", "E", new Func<MemberReferenceExpression, object, string>(TranslateGMathE)));
            SpecialProperties.Add(new SpecialMember("GMath", "PI", new Func<MemberReferenceExpression, object, string>(TranslateGMathPI)));

            SpecialTypes.Add("ComplexD", new SpecialTypeProps() { Name = "cuDoubleComplex", OptionalHeader = "cuComplex" });
            SpecialTypes.Add("ComplexF", new SpecialTypeProps() { Name = "cuFloatComplex", OptionalHeader = "cuComplex" });

            SpecialTypes.Add("RandStateXORWOW", new SpecialTypeProps() { Name = "curandStateXORWOW", OptionalHeader = csCURAND_KERNEL });
            SpecialTypes.Add("RandStateSobol32", new SpecialTypeProps() { Name = "curandStateSobol32", OptionalHeader = csCURAND_KERNEL });
            SpecialTypes.Add("RandStateScrambledSobol32", new SpecialTypeProps() { Name = "curandStateScrambledSobol32", OptionalHeader = csCURAND_KERNEL });
            SpecialTypes.Add("RandStateSobol64", new SpecialTypeProps() { Name = "curandStateSobol64", OptionalHeader = csCURAND_KERNEL });
            SpecialTypes.Add("RandStateScrambledSobol64", new SpecialTypeProps() { Name = "curandStateScrambledSobol64", OptionalHeader = csCURAND_KERNEL });

            OptionalHeaders = new List<OptionalHeader>();
            OptionalHeaders.Add(new OptionalHeader("cuComplex", @"#include <cuComplex.h>"));
            OptionalHeaders.Add(new OptionalHeader(csCURAND_KERNEL, @"#include <curand_kernel.h>"));
            OptionalHeaders.Add(new OptionalHeader(csSTDIO, @"#include <stdio.h>"));
            OptionalHeaders.Add(new OptionalHeader(csASSERT, @"#include <assert.h>"));
            DisableSmartArray = false;
        }

        private const string csCURAND_KERNEL = "curand_kernel";

        private const string csSTDIO = "stdio";

        private const string csASSERT = "assert";

        public struct SpecialTypeProps
        {
            public string Name;
            public string OptionalHeader;
        }

        static string NormalizeDeclaringType(string declaringType)
        {
            if (declaringType.Contains("["))
                return "ArrayType";
            return declaringType;
        }

        public static bool IsSpecialProperty(string memberName, string declaringType)
        {
            return GetSpecialProperty(memberName, declaringType) != null;
        }

        public static SpecialMember GetSpecialProperty(string memberName, string declaringType)
        {
            declaringType = NormalizeDeclaringType(declaringType);           
            foreach (var item in SpecialProperties)
                if (item.DeclaringType == declaringType && memberName == item.OriginalName)
                    return item;
            return null;
        }

        public static bool IsSpecialMethod(string memberName, string declaringType)
        {
            return GetSpecialMethod(memberName, declaringType) != null;
        }

        public static SpecialMember GetSpecialMethod(string memberName, string declaringType)
        {
            declaringType = NormalizeDeclaringType(declaringType);
            foreach (var item in SpecialMethods)
                if (item.DeclaringType == declaringType && memberName == item.OriginalName)
                    return item;
            // We don't want to take a default method when there is a special property
            var prop = GetSpecialProperty(memberName, declaringType);
            if (prop == null)
            {
                foreach (var item in SpecialMethods)
                    if (item.DeclaringType == declaringType && item.OriginalName == null)
                        return item;
            }
            return prop;
        }

        public static string TranslateSpecialType(string declaringType)
        {
            declaringType = NormalizeDeclaringType(declaringType);
            if (SpecialTypes.ContainsKey(declaringType))
            {
                var stp = SpecialTypes[declaringType];
                if (!string.IsNullOrEmpty(stp.OptionalHeader))
                    UseOptionalHeader(stp.OptionalHeader);
                return stp.Name;
            }
            else
                return declaringType;
        }

        public static void Reset()
        {
            foreach (var oh in OptionalHeaders)
                oh.Used = false;
            DisableSmartArray = false;
        }

        private static void UseOptionalHeader(string name)
        {
            var oh = OptionalHeaders.Where(o => o.Name == name).FirstOrDefault();
            Debug.Assert(oh != null);
            oh.Used = true;
        }


        public readonly static string csSyncThreads = "SyncThreads";
        public readonly static string csSyncThreadsCount = "SyncThreadsCount";
        
        public readonly static string csAll = "All";
        public readonly static string csAny = "Any";
        public readonly static string csBallot = "Ballot";
        public readonly static string csAllocateShared = "AllocateShared";

        public readonly static List<SpecialMember> SpecialMethods = new List<SpecialMember>();
        public readonly static List<SpecialMember> SpecialProperties = new List<SpecialMember>();
        public readonly static Dictionary<string, SpecialTypeProps> SpecialTypes = new Dictionary<string, SpecialTypeProps>();
        public readonly static List<OptionalHeader> OptionalHeaders;

        static string TranslateStringLength(MemberReferenceExpression mre, object data)
        {
            string length = mre.Target.ToString() + "Len";
            return length;
        }

        static string TranslateArrayLength(MemberReferenceExpression mre, object data)
        {
            string rank, length;
            bool rc = mre.TranslateArrayLengthAndRank(out length, out rank);
            Debug.Assert(rc);
            return length;
        }

        static string TranslateArrayRank(MemberReferenceExpression mre, object data)
        {
            string rank, length;
            bool rc = mre.TranslateArrayLengthAndRank(out length, out rank);
            Debug.Assert(rc);
            return rank;
        }

        static string TranslateToTrue(MemberReferenceExpression mre, object data)
        {
            return "true";
        }

        static string TranslateToFalse(MemberReferenceExpression mre, object data)
        {
            return "false";
        }

        static string TranslateSyncThreads(MemberReferenceExpression mre, object data)
        {
            return "__syncthreads";
        }

        static string TranslateSyncThreadsCount(MemberReferenceExpression mre, object data)
        {
            if (ComputeCapability < new Version(2, 0))
                throw new CudafyLanguageException(CudafyLanguageException.csX_IS_NOT_SUPPORTED_FOR_COMPUTE_X, mre.MemberName, ComputeCapability);
            return "__syncthreads_count";
        }

        static string TranslateAll(MemberReferenceExpression mre, object data)
        {
            return "__all";
        }

        static string TranslateAny(MemberReferenceExpression mre, object data)
        {
            return "__any";
        }

        static string TranslateBallot(MemberReferenceExpression mre, object data)
        {
            if (ComputeCapability < new Version(2, 0))
                throw new CudafyLanguageException(CudafyLanguageException.csX_IS_NOT_SUPPORTED_FOR_COMPUTE_X, mre.MemberName, ComputeCapability);
            return "__ballot";
        }

        static string TranslateAtomicAddFloat(MemberReferenceExpression mre, object data)
        {
            if (ComputeCapability < new Version(2, 0))
                throw new CudafyLanguageException(CudafyLanguageException.csX_IS_NOT_SUPPORTED_FOR_COMPUTE_X, mre.MemberName, ComputeCapability);
            return "atomicAdd";
        }

        static string TranslateToPrintF(MemberReferenceExpression mre, object data)
        {
            if (ComputeCapability < new Version(2, 0))
                return CommentMeOut(mre, data);
            UseOptionalHeader(csSTDIO);
            string dbugwrite = string.Empty;
            dbugwrite = mre.TranslateToPrintF(data);
            return dbugwrite;
        }

        static string TranslateAssert(MemberReferenceExpression mre, object data)
        {
            if (ComputeCapability < new Version(2, 0))
                return CommentMeOut(mre, data);
            UseOptionalHeader(csASSERT);
            string assert = string.Empty;
            assert = mre.TranslateAssert(data);
            return assert;
        }

        static string GetMemberName(MemberReferenceExpression mre, object data)
        {
            return mre.MemberName;
        }

        static string GetCURANDMemberName(MemberReferenceExpression mre, object data)
        {
            UseOptionalHeader("curand_kernel");
            DisableSmartArray = true;
            return mre.MemberName;
        }

        static string TranslateComplexDCtor(MemberReferenceExpression mre, object data)
        {
            UseOptionalHeader("cuComplex");
            return "make_cuDoubleComplex";
        }

        static string TranslateComplexFCtor(MemberReferenceExpression mre, object data)
        {
            UseOptionalHeader("cuComplex");
            return "make_cuFloatComplex";
        }

        static string TranslateArrayGetLength(MemberReferenceExpression mre, object data)
        {
            string length = string.Empty;
            length = mre.TranslateArrayGetLength(data);
            return length;
        }

        static string CommentMeOut(MemberReferenceExpression mre, object data)
        {
            return string.Format("// {0}", mre.ToString());
        }

        static string TranslateGMath(MemberReferenceExpression mre, object data)
        {
            switch (mre.MemberName)
            {
                case "Abs":
                    return "fabsf";
                case "Max":
                    return "fmaxf";
                case "Min":
                    return "fminf";
                default:
                    break;
            }
            return TranslateMath(mre, data) + "f";
        }

        static string TranslateMath(MemberReferenceExpression mre, object data)
        {            
            switch (mre.MemberName)
            {
                case "Round":
                    return "rint";
                case "Truncate":
                    return "trunc";
                case "Ceiling":
                    return "ceil";
                    //Math.Sign
                case "DivRem":
                    throw new NotSupportedException(mre.MemberName);
                case "IEEERemainder":
                    throw new NotSupportedException(mre.MemberName);
                case "Sign"://http://stackoverflow.com/questions/1903954/is-there-a-standard-sign-function-signum-sgn-in-c-c
                    throw new NotSupportedException(mre.MemberName);
                case "BigMul":
                    throw new NotSupportedException(mre.MemberName);
                default:
                    break;
            }
            return mre.MemberName.ToLower();
        }

        static string TranslateMathE(MemberReferenceExpression mre, object data)
        {
            return Math.E.ToString(CultureInfo.InvariantCulture);
        }

        static string TranslateMathPI(MemberReferenceExpression mre, object data)
        {
            return Math.PI.ToString(CultureInfo.InvariantCulture);
        }

        static string TranslateGMathE(MemberReferenceExpression mre, object data)
        {
            return GMath.E.ToString(CultureInfo.InvariantCulture);
        }

        static string TranslateGMathPI(MemberReferenceExpression mre, object data)
        {
            return GMath.PI.ToString(CultureInfo.InvariantCulture);
        }

        static string TranslateComplexD(MemberReferenceExpression mre, object data)
        {
            switch (mre.MemberName)
            {
                case "Conj":
                    return "cuConj";
                case "Add":
                    return "cuCadd";
                case "Subtract":
                    return "cuCsub";
                case "Multiply":
                    return "cuCmul";
                case "Divide":
                    return "cuCdiv";
                case "Abs":
                    return "cuCabs";
                default:
                    throw new NotSupportedException(mre.MemberName);
            }
        }

        static string TranslateComplexF(MemberReferenceExpression mre, object data)
        {
            return TranslateComplexD(mre, data) + "f";
        }
    }
#pragma warning restore 1591
}
