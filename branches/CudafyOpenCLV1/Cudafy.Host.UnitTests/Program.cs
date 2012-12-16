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
using System.Reflection;
using Cudafy.UnitTests;
using GASS.CUDA.FFT;
using NUnit.Framework;
using Cudafy.Host;
using Cudafy.Translator;
namespace Cudafy.Host.UnitTests
{
    class Program
    {
        static void Main(string[] args)
        {
            try
            {
                CudafyModes.Target = eGPUType.Cuda;

                TempOpenCLVectorAddTest();

                //CURANDTests.Basics();

                //StringTests st = new StringTests();
                //CudafyUnitTest.PerformAllTests(st);

                //BasicFunctionTests bft = new BasicFunctionTests();
                //CudafyUnitTest.PerformAllTests(bft);

                //GMathUnitTests gmu = new GMathUnitTests();
                //CudafyUnitTest.PerformAllTests(gmu);

                //MultithreadedTests mtt = new MultithreadedTests();
                //CudafyUnitTest.PerformAllTests(mtt);

                //CopyTests1D ct1d = new CopyTests1D();
                //CudafyUnitTest.PerformAllTests(ct1d);

                //GPGPUTests gput = new GPGPUTests();
                //CudafyUnitTest.PerformAllTests(gput);

                //if (CudafyHost.GetDeviceCount(CudafyModes.Target) > 1)
                //{
                //    MultiGPUTests mgt = new MultiGPUTests();
                //    CudafyUnitTest.PerformAllTests(mgt);
                //}

                Console.WriteLine("Done");
                Console.ReadLine();
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
                Console.ReadLine();
            }
            
            
        }

        const int N = 1024;

        static void TempOpenCLTest()
        {
            int[] inputData = new int[N];
            int[] outputData = new int[N];
            Random rand = new Random();
            for (int i = 0; i < N; i++)
                inputData[i] = rand.Next();

            GPGPU gpu = CudafyHost.GetDevice(eGPUType.OpenCL);
            int[] dev_data = gpu.CopyToDevice(inputData);
            gpu.CopyFromDevice(dev_data, 0, outputData, 0, N);

            for (int i = 0; i < N; i++)
                Assert.AreEqual(inputData[i], outputData[i], string.Format("Error at {0}", i));

        }

        static void TempOpenCLVectorAddTest()
        {
            int[] inputData1 = new int[N];
            int[] inputData2 = new int[N];
            int[] inputData3 = new int[N];
            int[] outputData = new int[N];
            Random rand = new Random();
            for (int i = 0; i < N; i++)
            {
                inputData1[i] = rand.Next(128);
                inputData2[i] = rand.Next(128);
                inputData3[i] = 2;
            }

            GPGPU gpu = CudafyHost.GetDevice(eGPUType.OpenCL, 0);
            Console.WriteLine(gpu.GetDeviceProperties().Name);
            CudafyTranslator.Language = eLanguage.OpenCL;
            var mod = CudafyTranslator.Cudafy(typeof(OpenCLTestClass));
            //mod.CudaSourceCode
            Console.WriteLine(mod.CudaSourceCode);
            gpu.LoadModule(mod);
            int[] dev_data1 = gpu.CopyToDevice(inputData1);
            int[] dev_data2 = gpu.CopyToDevice(inputData2);
            gpu.CopyToConstantMemory(inputData3, OpenCLTestClass.ConstantMemory);
            int[] dev_res = gpu.Allocate<int>(N);
#warning Work group and local size mess! http://stackoverflow.com/questions/7996537/cl-invalid-work-group-size-error-should-be-solved-though
            gpu.Launch(2, 512).VectorAdd(dev_data1, dev_data2, dev_res);
            gpu.CopyFromDevice(dev_res, 0, outputData, 0, N);

            for (int i = 0; i < N; i++)
                Assert.AreEqual((inputData1[i] + inputData2[i]) * inputData3[i], outputData[i], string.Format("Error at {0}", i));

            
        }
    }


    public class OpenCLTestClass
    {
        [Cudafy]
        public static int[] ConstantMemory = new int[1024];
        
        [Cudafy]
        public static void VectorAdd(GThread thread,
                                int[] a,
                                int[] b,
                                int[] c )
        {
            int[] shared = thread.AllocateShared<int>("shared", 1024);
            int index = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            //int index = thread.get_local_id(0);
            c[index] = (a[index] + b[index]) * ConstantMemory[index];
            thread.SyncThreads();
        }

    }
}
