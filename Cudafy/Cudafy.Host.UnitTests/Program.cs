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
using Cudafy.Compilers;
namespace Cudafy.Host.UnitTests
{
    class Program
    {
        static void Main(string[] args)
        {
            try
            {
               
                CudafyModes.DeviceId = 0;
                CudafyModes.Architecture = eArchitecture.sm_13; // *** Change this to the architecture of your target board ***
                CudafyModes.Target = CompilerHelper.GetGPUType(CudafyModes.Architecture);

                if (CudafyModes.Target != eGPUType.OpenCL)
                {
                    CURANDTests.Basics();
                }

                StringTests st = new StringTests();
                CudafyUnitTest.PerformAllTests(st);

                BasicFunctionTests bft = new BasicFunctionTests();
                CudafyUnitTest.PerformAllTests(bft);

                GMathUnitTests gmu = new GMathUnitTests();
                CudafyUnitTest.PerformAllTests(gmu);

                MultithreadedTests mtt = new MultithreadedTests();
                CudafyUnitTest.PerformAllTests(mtt);

                CopyTests1D ct1d = new CopyTests1D();
                CudafyUnitTest.PerformAllTests(ct1d);

                GPGPUTests gput = new GPGPUTests();
                CudafyUnitTest.PerformAllTests(gput);

                if (CudafyHost.GetDeviceCount(CudafyModes.Target) > 1)
                {
                    MultiGPUTests mgt = new MultiGPUTests();
                    CudafyUnitTest.PerformAllTests(mgt);
                }

                //if (CudafyModes.Architecture == eArchitecture.sm_35)
                //{
                //    Compute35Features c35f = new Compute35Features();
                //    CudafyUnitTest.PerformAllTests(c35f);
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

        public const int N = 1024;

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

            GPGPU gpu = CudafyHost.GetDevice(eGPUType.Cuda, 0);
            Console.WriteLine(gpu.GetDeviceProperties().Name);
            CudafyTranslator.Language = eLanguage.Cuda;
            var mod = CudafyTranslator.Cudafy(typeof(OpenCLTestClass));
            //mod.CudaSourceCode
            Console.WriteLine(mod.SourceCode);
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

        static void popcTest()
        {
            var km = CudafyModule.TryDeserialize(typeof(OpenCLTestClass).Name);
            if (km == null || !km.TryVerifyChecksums())
            {
                km = CudafyTranslator.Cudafy(typeof(OpenCLTestClass));
                km.TrySerialize();
            }
            Console.WriteLine(km.SourceCode);

            GPGPU gpu = CudafyHost.GetDevice(CudafyModes.Target);
            gpu.LoadModule(km);

            uint[] v = new uint[N];
            int[] c = new int[N];

            // allocate the memory on the GPU
            int[] dev_c = gpu.Allocate<int>(c);

            // fill the array 'v'
            for (int i = 0; i < N; i++)
            {
                v[i] = (uint)i;
            }

            // copy the array 'v' to the GPU
            uint[] dev_v = gpu.CopyToDevice(v);
            gpu.Launch(1, N).popVect(dev_v, dev_c);

            // copy the array 'c' back from the GPU to the CPU
            gpu.CopyFromDevice(dev_c, c);

            // display the results
            for (int i = 0; i < N; i++)
            {
                //Console.WriteLine("__popc{0} = {1}", v[i], c[i]);
            }

            // free the memory allocated on the GPU
            gpu.FreeAll();
        }

    }


    public class OpenCLTestClass
    {
        [Cudafy]
        public static int[] ConstantMemory = new int[Program.N];
        
        [Cudafy]
        public static void VectorAdd(GThread thread,
                                [CudafyAddressSpace(eCudafyAddressSpace.Global)] int[] a,
                                int[] b,
                                int[] c )
        {
            int[] shared = thread.AllocateShared<int>("shared", Program.N);
            int index = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            //int index = thread.get_local_id(0);
            c[index] = (a[index] + b[index]) * ConstantMemory[index];
            thread.SyncThreads();
        }

        [Cudafy]
        public static void popVect(GThread thread, uint[] v, int[] c)
        {
            int tid = thread.threadIdx.x;
            if (tid < Program.N)
                c[tid] = __popc(v[tid]);
        }

        //[CudafyDummy(eCudafyType.Auto, eCudafyDummyBehaviour.SuppressInclude)]
        public static int __popc(uint x)
        {
            int tot = 0;
            for (int f = 0; f < 32; f++)
                tot += (int)((x >> f) & 1);
            return tot;
        }

        //private static string __popc_formatter(eGPUType gpuType, params string[] args)
        //{
        //    if (args.Length != 1)
        //        throw new ApplicationException("Invalid number of arguments");
        //    switch (gpuType)
        //    {
        //        case eGPUType.Cuda:
        //            return string.Format("__popc({0})", args[0]);
        //        //case eGPUType.OpenCL:
        //        //    return string.Format("popcount({0})", args[0]);
        //        default:
        //            throw new NotImplementedException();
        //    }
        //}


    }
}
