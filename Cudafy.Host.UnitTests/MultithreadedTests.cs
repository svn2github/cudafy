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
using System.Threading;
using Cudafy.Types;
using Cudafy.Host;
using Cudafy.UnitTests;
using NUnit.Framework;

using GASS.CUDA;
using GASS.CUDA.Tools;
namespace Cudafy.Host.UnitTests
{
    [TestFixture]
    public class MultithreadedTests : CudafyUnitTest, ICudafyUnitTest
    {
        private const int N = 1024 * 1024;

        [TestFixtureSetUp]
        public void SetUp()
        {
            _gpu = CudafyHost.CreateDevice(CudafyModes.Target);
            _uintBufferIn1 = new uint[N];
            _uintBufferOut1 = new uint[N];
            _uintBufferIn2 = new uint[N];
            _uintBufferOut2 = new uint[N];
        }

        [TestFixtureTearDown]
        public void TearDown()
        {
            _gpu.FreeAll();
        }

        private GPGPU _gpu;

        private uint[] _gpuuintBufferIn1;

        private uint[] _uintBufferIn1;

        private uint[] _uintBufferOut1;

        private uint[] _gpuuintBufferIn2;

        private uint[] _uintBufferIn2;

        private uint[] _uintBufferOut2;

        private void SetInputs()
        {
            Random rand = new Random(DateTime.Now.Millisecond);
            for (int i = 0; i < N; i++)
            {
                double r = rand.NextDouble();
                _uintBufferIn1[i] = (uint)(r * uint.MaxValue);
            }
        }

        private void ClearOutputsAndGPU()
        {
            for (int i = 0; i < N; i++)
            {
                _uintBufferOut1[i] = 0;
            }
            _gpu.FreeAll();
        }

        //[Test]
        public void Test_SingleThreadCopy()
        {
            _gpuuintBufferIn1 = _gpu.CopyToDevice(_uintBufferIn1);
            _gpu.CopyFromDevice(_gpuuintBufferIn1, _uintBufferOut1);
            Assert.IsTrue(Compare(_uintBufferIn1, _uintBufferOut1));
            ClearOutputsAndGPU();
        }

        [Test]
        public void Test_TwoThreadCopy()
        {
            //CUDA cuda = ((_gpu as CudaGPU).CudaDotNet as CUDA);
            //_ccs = new CUDAContextSynchronizer(cuda.CurrentContext);
            //_ccs.MakeFloating();

            _gpu = CudafyHost.GetDevice(eGPUType.Cuda);
            _gpu.EnableMultithreading();
            bool j1 = false;
            bool j2 = false;
            for (int i = 0; i < 10; i++)
            {
                Console.WriteLine(i);
                Thread t1 = new Thread(Test_TwoThreadCopy_Thread1);
                Thread t2 = new Thread(Test_TwoThreadCopy_Thread2);
                t1.Start();
                t2.Start();
                j1 = t1.Join(10000);
                j2 = t2.Join(10000);
                if (!j1 || !j2)
                    break;
            }

            _gpu.DisableMultithreading();           
            _gpu.FreeAll();
            Assert.IsTrue(j1);
            Assert.IsTrue(j2);
        }

        private CUDAContextSynchronizer _ccs;

        private void Test_TwoThreadCopy_Thread1()
        {
            _gpu.Lock();
            _gpuuintBufferIn1 = _gpu.CopyToDevice(_uintBufferIn1);
            _gpu.CopyFromDevice(_gpuuintBufferIn1, _uintBufferOut1);
            Assert.IsTrue(Compare(_uintBufferIn1, _uintBufferOut1));
            _gpu.Free(_gpuuintBufferIn1);
            _gpu.Unlock();
        }
        
        private void Test_TwoThreadCopy_Thread2()
        {
            _gpu.Lock();
            _gpuuintBufferIn2 = _gpu.CopyToDevice(_uintBufferIn2);
            _gpu.CopyFromDevice(_gpuuintBufferIn2, _uintBufferOut2);
            Assert.IsTrue(Compare(_uintBufferIn2, _uintBufferOut2));
            _gpu.Free(_gpuuintBufferIn2);
            _gpu.Unlock();
        }


        public void TestSetUp()
        {
         
        }

        public void TestTearDown()
        {
            
        }
    }
}
