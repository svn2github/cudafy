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
using System.Diagnostics;
using Cudafy.Types;
using Cudafy.Host;
using Cudafy.UnitTests;
using NUnit.Framework;

namespace Cudafy.Host.UnitTests
{
    [TestFixture]
    public class CopyTests1D : CudafyUnitTest, ICudafyUnitTest
    {
        private CudafyModule _cm;

        private GPGPU _gpu;

        private const int N = 1024 * 1024;

        private byte[] _byteBufferIn;

        private byte[] _byteBufferOut;

        private byte[] _gpubyteBufferIn;

        private byte[] _gpubyteBufferOut;

        private sbyte[] _sbyteBufferIn;

        private sbyte[] _sbyteBufferOut;

        private sbyte[] _gpusbyteBufferIn;

        private sbyte[] _gpusbyteBufferOut;

        private ushort[] _ushortBufferIn;

        private ushort[] _ushortBufferOut;

        private ushort[] _gpuushortBufferIn;

        private ushort[] _gpuushortBufferOut;

        private uint[] _uintBufferIn;

        private uint[] _uintBufferOut;

        private uint[] _gpuuintBufferIn;

        private uint[] _gpuuintBufferOut;

        private ulong[] _ulongBufferIn;

        private ulong[] _ulongBufferOut;

        private ulong[] _gpuulongBufferIn;

        private ulong[] _gpuulongBufferOut;

        private ComplexD[] _cplxDBufferIn;

        private ComplexD[] _cplxDBufferOut;

        private ComplexD[] _gpucplxDBufferIn;

        private ComplexD[] _gpucplxDBufferOut;

        private ComplexF[] _cplxFBufferIn;

        private ComplexF[] _cplxFBufferOut;

        private ComplexF[] _gpucplxFBufferIn;

        private ComplexF[] _gpucplxFBufferOut;

        [TestFixtureSetUp]
        public void SetUp()
        {
            _gpu = CudafyHost.GetDevice(CudafyModes.Target);

            _byteBufferIn = new byte[N];
            _byteBufferOut = new byte[N];            

            _sbyteBufferIn = new sbyte[N];
            _sbyteBufferOut = new sbyte[N];

            _ushortBufferIn = new ushort[N];
            _ushortBufferOut = new ushort[N];

            _uintBufferIn = new uint[N];
            _uintBufferOut = new uint[N];

            _ulongBufferIn = new ulong[N];
            _ulongBufferOut = new ulong[N];

            _cplxDBufferIn = new ComplexD[N];
            _cplxDBufferOut = new ComplexD[N];

            _cplxFBufferIn = new ComplexF[N];
            _cplxFBufferOut = new ComplexF[N];

            SetInputs();
            ClearOutputsAndGPU();
        }

        [TestFixtureTearDown]
        public void TearDown()
        {
            _gpu.FreeAll();
        }

        private void SetInputs()
        {
            Random rand = new Random(DateTime.Now.Millisecond);
            for (int i = 0; i < N; i++)
            {
                double r = rand.NextDouble();
                double j = rand.NextDouble();
                _byteBufferIn[i] = (byte)(r * Byte.MaxValue);
                _sbyteBufferIn[i] = (sbyte)((r * Byte.MaxValue) - SByte.MaxValue);
                _ushortBufferIn[i] = (ushort)(r * ushort.MaxValue);
                _uintBufferIn[i] = (uint)(r * uint.MaxValue);
                _ulongBufferIn[i] = (ulong)(r * ulong.MaxValue);
                _cplxDBufferIn[i].x = r * short.MaxValue;
                _cplxDBufferIn[i].y = j * short.MaxValue - 1.0;
                _cplxFBufferIn[i].x = (float)r * short.MaxValue;
                _cplxFBufferIn[i].y = (float)j * short.MaxValue - 1.0F;
            }  
        }

        private void ClearOutputsAndGPU()
        {
            for (int i = 0; i < N; i++)
            {
                _byteBufferOut[i] = 0;
                _sbyteBufferOut[i] = 0;
                _ushortBufferOut[i] = 0;
                _uintBufferOut[i] = 0;
                _ulongBufferOut[i] = 0;
                _cplxDBufferOut[i].x = 0;
                _cplxDBufferOut[i].y = 0;
                _cplxFBufferOut[i].x = 0;
                _cplxFBufferOut[i].y = 0;
            }
            _gpu.FreeAll();
            GC.Collect();
        }

        [Test]
        public void Test_getValue_int2D()
        {
            int[,] data2D = new int[16,12];
            for (int i = 0, ctr = 0; i < 16; i++)
                for (int j = 0; j < 12; j++)
                    data2D[i, j] = ctr++;
            int[,] dev2D = _gpu.CopyToDevice(data2D);
            int v = _gpu.GetValue(dev2D, 14, 9);
            Assert.AreEqual(data2D[14,9], v);
            ClearOutputsAndGPU();
        }

        [Test]
        public void Test_getValue_int3D()
        {
            int[,,] data3D = new int[16, 12, 8];
            for (int i = 0, ctr = 0; i < 16; i++)
                for (int j = 0; j < 12; j++)
                    for (int k = 0; k < 8; k++)
                    data3D[i, j, k] = ctr++;
            int[,,] dev3D = _gpu.CopyToDevice(data3D);
            int v = _gpu.GetValue(dev3D, 14, 9, 6);
            Assert.AreEqual(data3D[14, 9, 6], v);
            ClearOutputsAndGPU();
        }

        [Test]
        public void Test_getValue_complexD()
        {
            _gpucplxDBufferIn = _gpu.Allocate(_cplxDBufferIn);
            _gpu.CopyToDevice(_cplxDBufferIn, _gpucplxDBufferIn);
            ComplexD cd = _gpu.GetValue(_gpucplxDBufferIn, N/32);
            Assert.AreEqual(_cplxDBufferIn[N / 32], cd);
            ClearOutputsAndGPU();
        }

        [Test]
        public void Test_autoCopyToFromGPU_byte()
        {
            _gpubyteBufferIn = _gpu.CopyToDevice(_byteBufferIn);
            _gpu.CopyFromDevice(_gpubyteBufferIn, _byteBufferOut);
            Assert.IsTrue(Compare(_byteBufferIn, _byteBufferOut));
            ClearOutputsAndGPU();
        }

        [Test]
        public void Test_autoCopyToFromGPU_sbyte()
        {
            _gpusbyteBufferIn = _gpu.CopyToDevice(_sbyteBufferIn);
            _gpu.CopyFromDevice(_gpusbyteBufferIn, _sbyteBufferOut);
            Assert.IsTrue(Compare(_sbyteBufferIn, _sbyteBufferOut));
            ClearOutputsAndGPU();
        }

        [Test]
        public void Test_autoCopyToFromGPU_ushort()
        {
            _gpuushortBufferIn = _gpu.CopyToDevice(_ushortBufferIn);
            _gpu.CopyFromDevice(_gpuushortBufferIn, _ushortBufferOut);
            Assert.IsTrue(Compare(_ushortBufferIn, _ushortBufferOut));
            ClearOutputsAndGPU();
        }

        [Test]
        public void Test_autoCopyToFromGPU_uint()
        {
            _gpuuintBufferIn = _gpu.CopyToDevice(_uintBufferIn);
            _gpu.CopyFromDevice(_gpuuintBufferIn, _uintBufferOut);
            Assert.IsTrue(Compare(_uintBufferIn, _uintBufferOut));
            ClearOutputsAndGPU();
        }

        [Test]
        public void Test_autoCopyToFromGPU_ulong()
        {
            _gpuulongBufferIn = _gpu.CopyToDevice(_ulongBufferIn);
            _gpu.CopyFromDevice(_gpuulongBufferIn, _ulongBufferOut);
            Assert.IsTrue(Compare(_ulongBufferIn, _ulongBufferOut));
            ClearOutputsAndGPU();
        }

        [Test]
        public void Test_autoCopyToFromGPU_cplxF()
        {
            _gpucplxFBufferIn = _gpu.CopyToDevice(_cplxFBufferIn);
            _gpu.CopyFromDevice(_gpucplxFBufferIn, _cplxFBufferOut);
            Assert.IsTrue(Compare(_cplxFBufferIn, _cplxFBufferOut));
            ClearOutputsAndGPU();
        }

        [Test]
        public void Test_autoCopyToFromGPU_cplxD()
        {
            _gpucplxDBufferIn = _gpu.CopyToDevice(_cplxDBufferIn);
            _gpu.CopyFromDevice(_gpucplxDBufferIn, _cplxDBufferOut);
            Assert.IsTrue(Compare(_cplxDBufferIn, _cplxDBufferOut));
            ClearOutputsAndGPU();
        }



        [Test]
        public void Test_copyToFromOffsetGPU_byte()
        {
            _gpubyteBufferIn = _gpu.Allocate(_byteBufferIn);
            _gpu.CopyToDevice(_byteBufferIn, _gpubyteBufferIn);
            _gpu.CopyFromDevice(_gpubyteBufferIn, N / 16, _byteBufferOut, N / 8, N / 2);
            Assert.IsTrue(Compare(_byteBufferIn, _byteBufferOut, N / 16, N / 8, N / 2));
            ClearOutputsAndGPU();
        }

        [Test]
        public void Test_cast_byte()
        {
            if (_gpu is EmulatedGPU)
            {
                Console.WriteLine("Emulated not supporting cast with offset, so skip.");
                return;
            }
            _gpubyteBufferIn = _gpu.Allocate(_byteBufferIn);
            _gpu.CopyToDevice(_byteBufferIn, _gpubyteBufferIn);
            byte[] offsetArray = _gpu.Cast(N / 16, _gpubyteBufferIn, N / 2);
            _gpu.CopyFromDevice(offsetArray, 0, _byteBufferOut, 0, N / 2);
            Assert.IsTrue(Compare(_byteBufferIn, _byteBufferOut, N / 16, 0, N / 2));
            ClearOutputsAndGPU();
        }

        [Test]
        public void Test_cast_cast_byte()
        {
            if (_gpu is EmulatedGPU)
            {
                Console.WriteLine("Emulated not supporting cast with offset, so skip.");
                return;
            }
            _gpubyteBufferIn = _gpu.Allocate(_byteBufferIn);
            _gpu.CopyToDevice(_byteBufferIn, _gpubyteBufferIn);
            byte[] offsetArray = _gpu.Cast(N / 16, _gpubyteBufferIn, N / 2);
            byte[] array2 = _gpu.Cast(0, offsetArray, N / 2);
            _gpu.CopyFromDevice(array2, 0, _byteBufferOut, 0, N / 2);
            Assert.IsTrue(Compare(_byteBufferIn, _byteBufferOut, N / 16, 0, N / 2));
            ClearOutputsAndGPU();
        }

        [Test]
        public void Test_cast_byte_to_sbyte()
        {
            _gpubyteBufferIn = _gpu.Allocate(_byteBufferIn);
            _gpu.CopyToDevice(_byteBufferIn, _gpubyteBufferIn);
            sbyte[] sbyteBufferOut = new sbyte[N];
            sbyte[] offsetArray = _gpu.Cast<byte,sbyte>(_gpubyteBufferIn, N);
            _gpu.CopyFromDevice(offsetArray, sbyteBufferOut);
            for (int i = 0; i < N; i++)
                Assert.AreEqual((sbyte)_byteBufferIn[i], sbyteBufferOut[i]);
            ClearOutputsAndGPU();
        }

        [Test]
        public void Test_cast_cplxD_to_double()
        {
            _gpucplxDBufferIn = _gpu.Allocate(_cplxDBufferIn);
            _gpu.CopyToDevice(_cplxDBufferIn, _gpucplxDBufferIn);
            double[] doubleBufferOut = new double[N*2];
            double[] offsetArray = _gpu.Cast<ComplexD, double>(_gpucplxDBufferIn, N*2);
            _gpu.CopyFromDevice(offsetArray, doubleBufferOut);
            for (int i = 0; i < N; i++)
            {
                Assert.AreEqual(_cplxDBufferIn[i].x, doubleBufferOut[i * 2]);
                Assert.AreEqual(_cplxDBufferIn[i].y, doubleBufferOut[i * 2 + 1]);
            }
            ClearOutputsAndGPU();
        }

        [Test]
        public void Test_copyToFromOffsetGPU_cplxD()
        {
            _gpucplxDBufferIn = _gpu.Allocate(_cplxDBufferIn);
            _gpu.CopyToDevice(_cplxDBufferIn, _gpucplxDBufferIn);
            _gpu.CopyFromDevice(_gpucplxDBufferIn, N / 16, _cplxDBufferOut, N / 8, N / 2);
            Assert.IsTrue(Compare(_cplxDBufferIn, _cplxDBufferOut, N / 16, N / 8, N / 2));
            ClearOutputsAndGPU();
        }

        [Test]
        public void Test_cast_uint_to_2d()
        {
            if (N > 32768)
            {
                Debug.WriteLine("Skipping Test_cast_uint_to_2d due to N being too large");
                return;
            }
            _gpuuintBufferIn = _gpu.CopyToDevice(_uintBufferIn);
            uint[,] devarray2D = _gpu.Cast(N / 2, _gpuuintBufferIn, N / 32, N / 64);
            uint[,] hostArray2D = new uint[N / 32, N / 64];
            _gpu.CopyFromDevice(devarray2D, hostArray2D);
            Assert.IsTrue(CompareEx<uint>(_uintBufferIn, hostArray2D, N / 2, 0, N / 2));
            ClearOutputsAndGPU();

        }


        [Test]
        public void Test_set_uint()
        {
            _gpuuintBufferIn = _gpu.CopyToDevice(_uintBufferIn);
            _gpu.Set(_gpuuintBufferIn);
            _gpu.CopyFromDevice(_gpuuintBufferIn, _uintBufferOut);
            Assert.IsTrue(Compare((uint)0, _uintBufferOut));
            ClearOutputsAndGPU();
        }

        [Test]
        public void Test_set_cplxF()
        {
            _gpucplxFBufferIn = _gpu.CopyToDevice(_cplxFBufferIn);
            _gpu.Set(_gpucplxFBufferIn);
            _gpu.CopyFromDevice(_gpucplxFBufferIn, _cplxFBufferOut);
            Assert.IsTrue(Compare(new ComplexF(), _cplxFBufferOut));
            ClearOutputsAndGPU();
        }

        [Test]
        public void Test_set_selection_uint()
        {
            _gpuuintBufferIn = _gpu.CopyToDevice(_uintBufferIn);
            _gpu.Set(_gpuuintBufferIn, N / 4, N / 2);
            _gpu.CopyFromDevice(_gpuuintBufferIn, _uintBufferOut);
            Assert.IsTrue(Compare(_uintBufferIn, _uintBufferOut, 0, 0, N / 4));
            Assert.IsTrue(Compare((uint)0, _uintBufferOut, N / 4, N / 2));
            Assert.IsTrue(Compare(_uintBufferIn, _uintBufferOut, N - N / 4, N - N / 4, N / 4));
            ClearOutputsAndGPU();
        }

        [Test]
        public void Test_set_selection_cplxF()
        {
            _gpucplxFBufferIn = _gpu.CopyToDevice(_cplxFBufferIn);
            _gpu.Set(_gpucplxFBufferIn, N / 4, N / 2);
            _gpu.CopyFromDevice(_gpucplxFBufferIn, _cplxFBufferOut);
            Assert.IsTrue(Compare(_cplxFBufferIn, _cplxFBufferOut, 0, 0, N / 4));
            Assert.IsTrue(Compare(new ComplexF(), _cplxFBufferOut, N / 4, N / 2));
            Assert.IsTrue(Compare(_cplxFBufferIn, _cplxFBufferOut, N - N / 4, N - N / 4, N / 4));
            ClearOutputsAndGPU();
        }

        [Test]
        public void Test_copyToOffsetFromGPU_ulong()
        {
            _gpuulongBufferIn = _gpu.Allocate(_ulongBufferIn);
            _gpu.Set(_gpuulongBufferIn);
            _gpu.CopyToDevice(_ulongBufferIn, N / 4, _gpuulongBufferIn, N / 16, N / 2);
            _gpu.CopyFromDevice(_gpuulongBufferIn, _ulongBufferOut);
            Assert.IsTrue(Compare(_ulongBufferIn, _ulongBufferOut, N / 4, N / 16, N / 2));
        }

        [Test]
        public void Test_copyToOffsetFromGPU_byte()
        {
            _gpubyteBufferIn = _gpu.Allocate(_byteBufferIn);
            _gpu.Set(_gpubyteBufferIn);
            _gpu.CopyToDevice(_byteBufferIn, N / 4, _gpubyteBufferIn, N / 16, N / 2);
            _gpu.CopyFromDevice(_gpubyteBufferIn, _byteBufferOut);
            Assert.IsTrue(Compare(_byteBufferIn, _byteBufferOut, N / 4, N / 16, N / 2));
        }

        [Test]
        public void Test_copyOnGPU_cplxD()
        {
            _gpucplxDBufferIn = _gpu.CopyToDevice(_cplxDBufferIn);
            _gpucplxDBufferOut = _gpu.Allocate<ComplexD>(N);
            _gpu.CopyOnDevice(_gpucplxDBufferIn, _gpucplxDBufferOut);
            _gpu.Set(_gpucplxDBufferIn);
            _gpu.CopyFromDevice(_gpucplxDBufferOut, _cplxDBufferOut);
            Assert.IsTrue(Compare(_cplxDBufferIn, _cplxDBufferOut));
            ClearOutputsAndGPU();
        }

        [Test]
        public void Test_copyOnOffsetGPU_ushort()
        {
            _gpuushortBufferIn = _gpu.CopyToDevice(_ushortBufferIn);
            _gpuushortBufferOut = _gpu.Allocate<ushort>(N / 2);
            _gpu.CopyOnDevice(_gpuushortBufferIn, N / 4, _gpuushortBufferOut, N / 8, N / 3);
            _gpu.Set(_gpuushortBufferIn);
            _gpu.CopyFromDevice(_gpuushortBufferOut, 0, _ushortBufferOut, 0, N / 2);
            Assert.IsTrue(Compare(_ushortBufferIn, _ushortBufferOut, N / 4, N / 8, N / 3));
            ClearOutputsAndGPU();
        }

        [Test]
        public void Test_copyFromPinned()
        {
            IntPtr srcPtr = _gpu.HostAllocate<ushort>(N);
            IntPtr dstPtr = _gpu.HostAllocate<ushort>(N);
            srcPtr.Write(_ushortBufferIn);
            _gpuushortBufferIn = _gpu.Allocate<ushort>(N);
            _gpuushortBufferOut = _gpu.Allocate<ushort>(N);
            _gpu.CopyToDeviceAsync(srcPtr, 0, _gpuushortBufferIn, 0, N, 0);
            _gpu.CopyOnDevice(_gpuushortBufferIn, 0, _gpuushortBufferOut, 0, N);
            _gpu.CopyFromDeviceAsync(_gpuushortBufferOut, 0, dstPtr, 0, N, 0);
            _gpu.SynchronizeStream(0);
            dstPtr.Read(_ushortBufferOut);
            _gpu.HostFreeAll();
            Assert.IsTrue(Compare(_ushortBufferIn, _ushortBufferOut, 0, 0, N));
            ClearOutputsAndGPU();
        }


        public void TestSetUp()
        {

        }

        public void TestTearDown()
        {

        }
    }
}
