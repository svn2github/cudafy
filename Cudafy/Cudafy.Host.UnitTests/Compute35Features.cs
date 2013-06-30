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
using System.Runtime.InteropServices;
using System.Diagnostics;
using Cudafy.Host;
using Cudafy.UnitTests;
using NUnit.Framework;
using Cudafy.Translator;
using Cudafy.Compilers;

namespace Cudafy.Host.UnitTests
{
    [TestFixture]
    public class Compute35Features : CudafyUnitTest, ICudafyUnitTest
    {
        private CudafyModule _cm;

        private GPGPU _gpu;

        private const int N = 1024;

        [TestFixtureSetUp]        
        public void SetUp()
        {
            _cm = CudafyTranslator.Translate(eArchitecture.sm_35, typeof(Compute35Features));
            var options = NvccCompilerOptions.Createx64(eArchitecture.sm_35);
            options.AddOption("-cubin");
            //options.AddOption("-rdc=true");
            options.AddOption("cudadevrt.lib");
            options.AddOption("cublas_device.lib");
            options.AddOption("-dlink");
            //options.AddOption("-gencode");
            //options.AddOption(@"-L""C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.5\lib\x64""");
            //options.AddOption("-lcublas_device");
            _cm.CompilerOptionsList.Add(options);
            _cm.Compile(eGPUCompiler.CudaNvcc, false, true);
            //Console.WriteLine(_cm.CompilerOutput);

            _gpu = CudafyHost.GetDevice(CudafyModes.Target, CudafyModes.DeviceId);
            _gpu.LoadModule(_cm);
        }

        [TestFixtureTearDown]
        public void TearDown()
        {
            _gpu.FreeAll();
        }



        [Cudafy]
        public static void childKernel(GThread thread, int[] a, int[] c, short coeff)
        {
            int tid = thread.blockIdx.x + N - N;
            if (tid < a.Length)
            //for(int tid = 0; tid < a.Length; tid++)
                c[tid] = a[tid] * coeff;
        }

        [Cudafy]
        public static void parentKernel(GThread thread, int[] a, int[] c, short coeff)
        {
            GThread.InsertCode("childKernel<<<1024,1>>>(a, 1024, c, 1024, coeff);");
        }

        [SetUp]
        public void TestSetUp()
        {

        }

        [TearDown]
        public void TestTearDown()
        {

        }

        [Test]
        public void TestDynamicParallelism()
        {
            int[] a = new int[N];
            int[] c = new int[N];
            short coeff = 8;
            for (int i = 0; i < N; i++)
                a[i] = i;
            int[] dev_a = _gpu.CopyToDevice(a);
            int[] dev_c = _gpu.Allocate<int>(c);
            _gpu.Launch(N, 1, "parentKernel", dev_a, dev_c, coeff);
            _gpu.CopyFromDevice(dev_c, c);
            for (int i = 0; i < N; i++)
                Assert.AreEqual(i * coeff, c[i]);
            _gpu.Free(dev_a);      
        }
    }
}
