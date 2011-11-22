/*
 * Added by Kichang Kim
 * kkc0923@hotmail.com
 * */
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Diagnostics;
using System.Reflection;
using Cudafy.Types;
using Cudafy.Host;
using Cudafy.Maths.SPARSE;
using Cudafy.UnitTests;
using NUnit.Framework;

namespace Cudafy.Maths.UnitTests
{
    [TestFixture]
    public class SPARSE1 : ICudafyUnitTest
    {
        private double[] _hostInput1;

        private int[] _hostInputIndex1;

        private double[] _hostInput2;

        private double[] _hostOutput1;

        private double[] _hostOutput2;

        private double[] _devPtr1;

        private int[] _devPtr2;

        private double[] _devPtr3;

        private const int ciN = 1024 * 4;
        private const int nzR = 10; // Non-zero element count : ciN / niN

        private GPGPU _gpu;

        private GPGPUSPARSE _sparse;

        [TestFixtureSetUp]
        public void SetUp()
        {
            _gpu = CudafyHost.CreateDevice(CudafyModes.Target);
            _sparse = GPGPUSPARSE.Create(_gpu);
            _hostInput1 = new double[ciN / nzR];
            _hostInputIndex1 = new int[ciN / nzR];
            _hostInput2 = new double[ciN];
            _hostOutput1 = new double[ciN];
            _hostOutput2 = new double[ciN];
            _devPtr1 = _gpu.Allocate<double>(_hostInput1);
            _devPtr2 = _gpu.Allocate<int>(_hostInputIndex1);
            _devPtr3 = _gpu.Allocate<double>(_hostInput2);
        }

        [TestFixtureTearDown]
        public void TearDown()
        {
            _sparse.Dispose();

            _gpu.Free(_devPtr1);
            _gpu.Free(_devPtr2);
            _gpu.Free(_devPtr3);
        }

        public void TestSetUp()
        {

        }

        public void TestTearDown()
        {

        }

        private void CreateRandomData(double[] buffer)
        {
            Random rand = new Random(Environment.TickCount);
            for (int i = 0; i < buffer.Length; i++)
            {
                buffer[i] = (double)rand.Next(512);
            }
        }

        private void CreateSparseNonzeroIndex(int[] buffer, int max)
        {
            for (int i = 0; i < buffer.Length; i++)
            {
                buffer[i] = i * max / buffer.Length;
            }
        }

        [Test]
        public void TestAXPY()
        {
            CreateRandomData(_hostInput1);
            CreateRandomData(_hostInput2);
            CreateSparseNonzeroIndex(_hostInputIndex1, ciN);
            
            _gpu.CopyToDevice(_hostInput1, _devPtr1);
            _gpu.CopyToDevice(_hostInputIndex1, _devPtr2);
            _gpu.CopyToDevice(_hostInput2, _devPtr3);
            _sparse.AXPY(10.0, _devPtr1, _devPtr2, _devPtr3);

            _gpu.CopyFromDevice(_devPtr3, _hostOutput1);

            int indicesIndex = 0;

            for (int i = 0; i < ciN; i++)
            {
                if (indicesIndex < _hostInputIndex1.Length && i == _hostInputIndex1[indicesIndex])
                {
                    Assert.AreEqual(10.0 * _hostInput1[indicesIndex] + _hostInput2[i], _hostOutput1[i]);
                    indicesIndex++;
                }
                else
                {
                    Assert.AreEqual(_hostInput2[i], _hostOutput1[i]);
                }
            }

        }
    }
}
