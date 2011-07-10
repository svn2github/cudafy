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
using Mono.Cecil;
using ICSharpCode.NRefactory.CSharp;

using CL = Cudafy.Translator.CUDALanguage;
namespace Cudafy.Translator
{
    /// <summary>
    /// Internal use.
    /// </summary>
    public static class ExtensionMethods
    {
        private static string csCUDAFYATTRIBUTE = typeof(CudafyAttribute).Name;

        private static string csCUDAFYDUMMYATTRIBUTE = typeof(CudafyDummyAttribute).Name;

        private static string csCUDAFYIGNOREATTRIBUTE = typeof(CudafyIgnoreAttribute).Name;

#pragma warning disable 1591
        public static eCudafyType? GetCudafyType(this ICustomAttributeProvider med, out bool isDummy)
        {
            bool ignore;
            return GetCudafyType(med, out isDummy, out ignore);
        }

        public static eCudafyType? GetCudafyType(this ICustomAttributeProvider med, out bool isDummy, out bool ignore)
        {
            isDummy = false;
            ignore = false;
            if (med is TypeDefinition)
                med = med as TypeDefinition;
            var customAttr = med.CustomAttributes.Where(ca => ca.AttributeType.Name == csCUDAFYATTRIBUTE).FirstOrDefault();
            if (customAttr == null)
            {
                customAttr = med.CustomAttributes.Where(ca => ca.AttributeType.Name == csCUDAFYDUMMYATTRIBUTE).FirstOrDefault();
                isDummy = customAttr != null;
            }
            if (customAttr == null)
            {
                customAttr = med.CustomAttributes.Where(ca => ca.AttributeType.Name == csCUDAFYIGNOREATTRIBUTE).FirstOrDefault();
                ignore = true;
            }
            else
            {
                eCudafyType et = eCudafyType.Auto;
                if (customAttr.ConstructorArguments.Count() > 0)
                    et = (eCudafyType)customAttr.ConstructorArguments.First().Value;
                return et;
            }
            return null;
        }

        //public static CudafyDummyAttribute GetCudafyDummyAttribute(this ICustomAttributeProvider med)
        //{
        //    customAttr = med.CustomAttributes.Where(ca => ca.AttributeType.Name == csCUDAFYDUMMYATTRIBUTE).FirstOrDefault();
        //}

        public static eCudafyType? GetCudafyType(this ICustomAttributeProvider med)
        {
            var customAttr = med.CustomAttributes.Where(ca => ca.AttributeType.Name == csCUDAFYATTRIBUTE).FirstOrDefault();
            if (customAttr != null)
            {
                eCudafyType et = eCudafyType.Auto;
                if (customAttr.ConstructorArguments.Count() > 0)
                    et = (eCudafyType)customAttr.ConstructorArguments.First().Value;
                return et;
            }
            else
                return null;
        }

        public static eCudafyType? GetCudafyDummyType(this ICustomAttributeProvider med)
        {
            var customAttr = med.CustomAttributes.Where(ca => ca.AttributeType.Name == csCUDAFYDUMMYATTRIBUTE).FirstOrDefault();
            if (customAttr != null)
            {
                eCudafyType et = eCudafyType.Auto;
                if (customAttr.ConstructorArguments.Count() > 0)
                    et = (eCudafyType)customAttr.ConstructorArguments.First().Value;
                return et;
            }
            else
                return null;
        }

        public static bool HasCudafyAttribute(this ICustomAttributeProvider med)
        {
            return med.HasCustomAttributes && med.CustomAttributes.Count(ca => ca.AttributeType.Name == csCUDAFYATTRIBUTE) > 0;
        }

        public static bool HasCudafyDummyAttribute(this ICustomAttributeProvider med)
        {
            return med.HasCustomAttributes && med.CustomAttributes.Count(ca => ca.AttributeType.Name == csCUDAFYDUMMYATTRIBUTE) > 0;
        }

        public static bool HasCudafyIgnoreAttribute(this ICustomAttributeProvider med)
        {
            return med.HasCustomAttributes && med.CustomAttributes.Count(ca => ca.AttributeType.Name == csCUDAFYIGNOREATTRIBUTE) > 0;
        }

        public static bool IsThreadIdVar(this MemberReferenceExpression mre)
        {
            if (mre.Target.Annotations.Count() > 0)
            {
                foreach (var ann in mre.Target.Annotations)
                {
                    PropertyDefinition pd = ann as PropertyDefinition;
                    if (pd != null && pd.DeclaringType.ToString().Contains("Cudafy.GThread"))
                        return true;

                }
            }
            return false;
        }

        public static bool IsSpecialProperty(this MemberReferenceExpression mre)
        {
            IEnumerable<object> annotations = mre.Annotations.Any() ? mre.Annotations : mre.Target.Annotations;
            if (annotations.Count() > 0)
            {
                foreach (var ann in mre.Target.Annotations)
                {
                    var pd = ann as ICSharpCode.Decompiler.ILAst.ILVariable;
                    if (pd != null)
                    {
                        if (CUDALanguage.IsSpecialProperty(mre.MemberName, pd.Type.FullName))
                            return true;
                    }
                    else
                    {
                        var fd = ann as FieldDefinition;
                        if(fd != null)
                            if (CUDALanguage.IsSpecialProperty(mre.MemberName, fd.FieldType.GetType().Name))
                                return true;
                    }
                }
            }
            return false;
        }

        public static bool IsSpecialMethod(this MemberReferenceExpression mre)
        {
            if (mre.Target.Annotations.Count() > 0)
            {
                foreach (var ann in mre.Target.Annotations)
                {
                    var pd = ann as ICSharpCode.Decompiler.ILAst.ILVariable;
                    if (pd != null)
                        if (CUDALanguage.IsSpecialMethod(mre.MemberName, pd.Type.FullName))//.GetType().Name))
                            return true;
                }
            }
            else
                return CUDALanguage.IsSpecialMethod(mre.MemberName, mre.Target.ToString());
            return false;
        }

        public static SpecialMember GetSpecialMethod(this MemberReferenceExpression mre)
        {
            SpecialMember sm = null;
            if (mre.Target.Annotations.Count() > 0)
            {
                foreach (var ann in mre.Target.Annotations)
                {
                    var pd = ann as ICSharpCode.Decompiler.ILAst.ILVariable;
                    if (pd != null)
                    {
                        sm = CUDALanguage.GetSpecialMethod(mre.MemberName, pd.Type.FullName);//.GetType().Name))
                        if(sm != null)
                            return sm;
                    }
                }
            }
            else
                return CUDALanguage.GetSpecialMethod(mre.MemberName, mre.Target.ToString());
            return sm;
        }

        public static string TranslateSpecialProperty(this MemberReferenceExpression mre)
        {
            IEnumerable<object> annotations = mre.Target.Annotations;//mre.Annotations.Any() ? mre.Annotations : 
            if (annotations.Count() > 0)
            {
                foreach (var ann in mre.Target.Annotations)
                {
                    var pd = ann as ICSharpCode.Decompiler.ILAst.ILVariable;
                    if (pd != null)
                    {
                        SpecialMember sm = CUDALanguage.GetSpecialProperty(mre.MemberName, pd.Type.FullName);
                        return sm.GetTranslation(mre);
                    }
                    else
                    {
                        var fd = ann as FieldDefinition;
                        if (fd != null)
                        {
                            SpecialMember sm = CUDALanguage.GetSpecialProperty(mre.MemberName, fd.FieldType.GetType().Name);
                            return sm.GetTranslation(mre);
                        }
                    }
                }
            }
            throw new InvalidOperationException("SpecialProperty not found.");
        }

        public static string TranslateSpecialMethod(this MemberReferenceExpression mre, object data)//, out bool callFunc)
        {
            //callFunc = true;
            if (mre.Target.Annotations.Count() > 0)
            {
                foreach (var ann in mre.Target.Annotations)
                {
                    var pd = ann as ICSharpCode.Decompiler.ILAst.ILVariable;
                    if (pd != null)
                    {
                        SpecialMember sm = CUDALanguage.GetSpecialMethod(mre.MemberName, pd.Type.FullName);// .GetType().Name);
                        //callFunc = sm.CallFunction;
                        return sm.GetTranslation(mre, data);
                    }
                }
            }
            else
            {
                SpecialMember sm = CUDALanguage.GetSpecialMethod(mre.MemberName, mre.Target.ToString());// .GetType().Name);
                return sm.GetTranslation(mre, data);
            }
            throw new InvalidOperationException("SpecialMethod not found.");
        }

        public static bool TranslateArrayLengthAndRank(this MemberReferenceExpression mre, out string length, out string rank)
        {
            length = string.Empty;
            rank = string.Empty;
            if (mre.Target.Annotations.Count() > 0)
            {
                foreach (var ann in mre.Target.Annotations)
                {
                    var pd = ann as ICSharpCode.Decompiler.ILAst.ILVariable;
                    if (pd != null)
                    {
                        var at = pd.Type as Mono.Cecil.ArrayType;
                        if (at != null)
                        {
                            rank = at.Rank.ToString();
                            string s = string.Empty;
                            for (int i = 0; i < at.Rank; i++)
                            {
                                s += string.Format("{0}Len{1}", mre.Target, i);
                                if (i < at.Rank - 1)
                                    s += " * ";
                            }
                            length = s;
                            return true;
                        }
                    }
                    else
                    {
                        var fd = ann as FieldDefinition;
                        var at = fd.FieldType as Mono.Cecil.ArrayType;
                        if (at != null)
                        {
                            rank = at.Rank.ToString();
                            string s = string.Empty;
                            for (int i = 0; i < at.Rank; i++)
                            {
                                s += string.Format("{0}Len{1}", (mre.Target as MemberReferenceExpression).MemberName, i);
                                if (i < at.Rank - 1)
                                    s += " * ";
                            }
                            length = s;
                            return true;
                        }
                    }
                }
            }
            return false;
        }

        public static string TranslateArrayGetLength(this MemberReferenceExpression mre, object data)
        {
            var ex = data as InvocationExpression;
            if (ex == null)
                throw new ArgumentNullException("data as InvocationExpression");
            PrimitiveExpression pe = ex.Arguments.First() as PrimitiveExpression;
            if (pe == null)
                throw new ArgumentNullException("PrimitiveExpression pe");
            foreach (var ann in mre.Target.Annotations)
            {
                var pd = ann as ICSharpCode.Decompiler.ILAst.ILVariable;
                if (pd != null)
                {
                    var at = pd.Type as Mono.Cecil.ArrayType;
                    if (at != null)
                    {
                        return string.Format("{0}Len{1}", mre.Target, pe.Value);
                    }
                }
            }
            return string.Empty;
        }

        public static bool IsSyncThreads(this MemberReferenceExpression mre)
        {
            return mre.MemberName == CL.csSyncThreads;
        }

        public static bool IsAllocateShared(this MemberReferenceExpression mre)
        {
            return mre.MemberName == CL.csAllocateShared;
        }
#pragma warning restore 1591
    }
}
