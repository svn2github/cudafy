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
using Cudafy.Host;
using Cudafy.UnitTests;
using NUnit.Framework;
using Cudafy.Translator;

namespace Cudafy.Host.UnitTests
{
    [TestFixture]
    public abstract class CudafiedUnitTests : CudafyUnitTest, ICudafyUnitTest
    {
        private CudafyModule _cm;

        protected GPGPU _gpu;

        [TestFixtureSetUp]
        public virtual void SetUp()
        {
            _cm = CudafyTranslator.Cudafy(this);
            _gpu = CudafyHost.GetDevice(CudafyModes.Target);
            _gpu.LoadModule(_cm);
            //Console.WriteLine(_cm.CompilerOutput);
        }

        [TestFixtureTearDown]
        public virtual void TearDown()
        {
            _gpu.FreeAll();
        }

        [SetUp]
        public virtual void TestSetUp()
        {

        }

        [TearDown]
        public virtual void TestTearDown()
        {

        }
    }  
    
    [TestFixture]
    public class GMathUnitTests : CudafiedUnitTests, ICudafyUnitTest
    {

        private const int N = 64;

        [Test]
        public void Test_Math()
        {
            double[] data = new double[N];
            double[] dev_data = _gpu.Allocate<double>(data);
#if !NET35
            _gpu.Launch().mathtest(dev_data);
#else
            _gpu.Launch(1, 1, "mathtest", dev_data);
#endif
            _gpu.CopyFromDevice(dev_data, data);
            double[] control = new double[N];
            mathtest(control);
            for (int i = 0; i < N; i++)
                Assert.AreEqual(control[i], data[i], 0.00001, "Index={0}", i);
        }

        [Test]
        public void Test_GMath()
        {
            float[] data = new float[N];
            float[] dev_data = _gpu.Allocate<float>(data);
#if !NET35
            _gpu.Launch().gmathtest(dev_data);
#else
            _gpu.Launch(1, 1, "gmathtest", dev_data);
#endif
            _gpu.CopyFromDevice(dev_data, data);
            float[] control = new float[N];
            gmathtest(control);
            for (int i = 0; i < N; i++)
                Assert.AreEqual(control[i], data[i], 0.00001, "Index={0}", i);
        }

        [Cudafy]
        public static void mathtest(double[] c)
        {
            int i = 0;
            c[i++] = Math.Abs(-42.3);
            c[i++] = Math.Acos(42.3);
            c[i++] = Math.Asin(42.3);
            c[i++] = Math.Atan(42.3);
            c[i++] = Math.Atan2(42.3, 3.8);
            c[i++] = Math.Cos(42.3);
            c[i++] = Math.Cosh(2.3);
            c[i++] = Math.E;
            c[i++] = Math.Exp(1.3);
            c[i++] = Math.Floor(3.9);
            c[i++] = Math.Log(5.8);
            c[i++] = Math.Log10(3.5);
            c[i++] = Math.Max(4.8, 4.9);
            c[i++] = Math.Min(4.8, 4.9);
            c[i++] = Math.PI;
            c[i++] = Math.Pow(4.4, 2.3);
            c[i++] = Math.Round(5.5);
            c[i++] = Math.Sin(4.2);
            c[i++] = Math.Sinh(3.1);
            c[i++] = Math.Sqrt(8.1);
            c[i++] = Math.Tan(4.3);
            c[i++] = Math.Tanh(8.1);
            c[i++] = Math.Truncate(10.14334325);
        }

        [Cudafy]
        public static void gmathtest(float[] c)
        {
            int i = 0;
            c[i++] = GMath.Abs(-42.3F);
            c[i++] = GMath.Acos(42.3F);
            c[i++] = GMath.Asin(42.3F);
            c[i++] = GMath.Atan(42.3F);
            c[i++] = GMath.Atan2(42.3F, 3.8F);
            c[i++] = GMath.Cos(42.3F);
            c[i++] = GMath.Cosh(2.3F);
            c[i++] = GMath.E;
            c[i++] = GMath.Exp(1.3F);
            c[i++] = GMath.Floor(3.9F);
            c[i++] = GMath.Log(5.8F);
            c[i++] = GMath.Log10(3.5F);
            c[i++] = GMath.Max(4.8F, 4.9F);
            c[i++] = GMath.Min(4.8F, 4.9F);
            c[i++] = GMath.PI;
            c[i++] = GMath.Pow(4.4F, 2.3F);
            c[i++] = GMath.Round(5.5F);
            c[i++] = GMath.Sin(4.2F);
            c[i++] = GMath.Sinh(3.1F);
            c[i++] = GMath.Sqrt(8.1F);
            c[i++] = GMath.Tan(4.3F);
            c[i++] = GMath.Tanh(8.1F);
            c[i++] = GMath.Truncate(10.14334325F);
        }
    }
}
