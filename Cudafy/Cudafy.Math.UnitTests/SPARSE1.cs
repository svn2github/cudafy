/* Added by Kichang Kim (kkc0923@hotmail.com) */
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
        private double[] _hostInput1Double;
        private int[] _hostInputIndex1Double;
        private double[] _hostInput2Double;

        private double[] _hostOutput1Double;
        private double[] _hostOutput2Double;

        private double[] _devVectorxDouble;
        private int[] _devIndexxDouble;
        private double[] _devVectoryDouble;

        private float[] _hostInput1Single;
        private int[] _hostInputIndex1Single;
        private float[] _hostInput2Single;

        private float[] _hostOutput1Single;
        private float[] _hostOutput2Single;

        private float[] _devVectorxSingle;
        private int[] _devIndexxSingle;
        private float[] _devVectorySingle;

        private const int ciN = 1024 * 4;
        private const int nzR = 10; // Non-zero element count : ciN / niN

        private GPGPU _gpu;

        private GPGPUSPARSE _sparse;

        [TestFixtureSetUp]
        public void SetUp()
        {
            _gpu = CudafyHost.CreateDevice(CudafyModes.Target);
            _sparse = GPGPUSPARSE.Create(_gpu);
            _hostInput1Double = new double[ciN / nzR];
            _hostInputIndex1Double = new int[ciN / nzR];
            _hostInput2Double = new double[ciN];
            _hostOutput1Double = new double[ciN];
            _hostOutput2Double = new double[ciN];
            _devVectorxDouble = _gpu.Allocate<double>(_hostInput1Double);
            _devIndexxDouble = _gpu.Allocate<int>(_hostInputIndex1Double);
            _devVectoryDouble = _gpu.Allocate<double>(_hostInput2Double);

            _hostInput1Single = new float[ciN / nzR];
            _hostInputIndex1Single = new int[ciN / nzR];
            _hostInput2Single = new float[ciN];
            _hostOutput1Single = new float[ciN];
            _hostOutput2Single = new float[ciN];
            _devVectorxSingle = _gpu.Allocate<float>(_hostInput1Single);
            _devIndexxSingle = _gpu.Allocate<int>(_hostInputIndex1Single);
            _devVectorySingle = _gpu.Allocate<float>(_hostInput2Single);
        }

        [TestFixtureTearDown]
        public void TearDown()
        {
            _sparse.Dispose();

            _gpu.Free(_devVectorxDouble);
            _gpu.Free(_devIndexxDouble);
            _gpu.Free(_devVectoryDouble);
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

        private void CreateRandomData(float[] buffer)
        {
            Random rand = new Random(Environment.TickCount);
            for (int i = 0; i < buffer.Length; i++)
            {
                buffer[i] = (float)rand.Next(512);
            }
        }

        [Test]
        public void TestSparseVersion()
        {
            Console.WriteLine(_sparse.GetVersionInfo());
        }

        [Test]
        public void TestSparseAXPY()
        {
            CreateRandomData(_hostInput1Double);
            CreateRandomData(_hostInput2Double);
            CreateSparseNonzeroIndex(_hostInputIndex1Double, ciN);

            CreateRandomData(_hostInput1Single);
            CreateRandomData(_hostInput2Single);
            CreateSparseNonzeroIndex(_hostInputIndex1Single, ciN);
            
            _gpu.CopyToDevice(_hostInput1Double, _devVectorxDouble);
            _gpu.CopyToDevice(_hostInputIndex1Double, _devIndexxDouble);
            _gpu.CopyToDevice(_hostInput2Double, _devVectoryDouble);
            _sparse.AXPY(10.0, _devVectorxDouble, _devIndexxDouble, _devVectoryDouble);

            _gpu.CopyFromDevice(_devVectoryDouble, _hostOutput1Double);

            _gpu.CopyToDevice(_hostInput1Single, _devVectorxSingle);
            _gpu.CopyToDevice(_hostInputIndex1Single, _devIndexxSingle);
            _gpu.CopyToDevice(_hostInput2Single, _devVectorySingle);
            _sparse.AXPY(10.0f, _devVectorxSingle, _devIndexxSingle, _devVectorySingle);

            _gpu.CopyFromDevice(_devVectorySingle, _hostOutput1Single);

            int indicesIndex = 0;

            for (int i = 0; i < ciN; i++)
            {
                if (indicesIndex < _hostInputIndex1Double.Length && i == _hostInputIndex1Double[indicesIndex])
                {
                    Assert.AreEqual(10.0 * _hostInput1Double[indicesIndex] + _hostInput2Double[i], _hostOutput1Double[i]);
                    Assert.AreEqual(10.0 * _hostInput1Single[indicesIndex] + _hostInput2Single[i], _hostOutput1Single[i]);
                    indicesIndex++;
                }
                else
                {
                    Assert.AreEqual(_hostInput2Double[i], _hostOutput1Double[i]);
                    Assert.AreEqual(_hostInput2Single[i], _hostOutput1Single[i]);
                }
            }

        }

        [Test]
        public void TestSparseDOT()
        {
            CreateRandomData(_hostInput1Double);
            CreateSparseNonzeroIndex(_hostInputIndex1Double, ciN);
            CreateRandomData(_hostInput2Double);

            CreateRandomData(_hostInput1Single);
            CreateRandomData(_hostInput2Single);
            CreateSparseNonzeroIndex(_hostInputIndex1Single, ciN);

            _gpu.CopyToDevice(_hostInput1Double, _devVectorxDouble);
            _gpu.CopyToDevice(_hostInputIndex1Double, _devIndexxDouble);
            _gpu.CopyToDevice(_hostInput2Double, _devVectoryDouble);

            _gpu.CopyToDevice(_hostInput1Single, _devVectorxSingle);
            _gpu.CopyToDevice(_hostInputIndex1Single, _devIndexxSingle);
            _gpu.CopyToDevice(_hostInput2Single, _devVectorySingle);

            double gpuResultDouble = _sparse.DOT(_devVectorxDouble, _devIndexxDouble, _devVectoryDouble);
            double cpuResultDouble = 0.0;

            float gpuResultSingle = _sparse.DOT(_devVectorxSingle, _devIndexxSingle, _devVectorySingle);
            float cpuResultSingle = 0.0f;

            for (int i = 0; i < _hostInputIndex1Double.Length; i++)
            {
                cpuResultSingle += _hostInput1Single[i] * _hostInput2Single[_hostInputIndex1Single[i]];
                cpuResultDouble += _hostInput1Double[i] * _hostInput2Double[_hostInputIndex1Double[i]];
            }

            Assert.AreEqual(cpuResultDouble, gpuResultDouble);
        }

        [Test]
        public void TestSparseGTHR()
        {
            CreateRandomData(_hostInput1Double);
            CreateSparseNonzeroIndex(_hostInputIndex1Double, ciN);
            CreateRandomData(_hostInput2Double);

            _gpu.CopyToDevice(_hostInput1Double, _devVectorxDouble);
            _gpu.CopyToDevice(_hostInputIndex1Double, _devIndexxDouble);
            _gpu.CopyToDevice(_hostInput2Double, _devVectoryDouble);

            _sparse.GTHR(_devVectoryDouble, _devVectorxDouble, _devIndexxDouble);
            _gpu.CopyFromDevice(_devVectorxDouble, _hostOutput1Double);

            CreateRandomData(_hostInput1Single);
            CreateSparseNonzeroIndex(_hostInputIndex1Single, ciN);
            CreateRandomData(_hostInput2Single);

            _gpu.CopyToDevice(_hostInput1Single, _devVectorxSingle);
            _gpu.CopyToDevice(_hostInputIndex1Single, _devIndexxSingle);
            _gpu.CopyToDevice(_hostInput2Single, _devVectorySingle);

            _sparse.GTHR(_devVectorySingle, _devVectorxSingle, _devIndexxSingle);
            _gpu.CopyFromDevice(_devVectorxSingle, _hostOutput1Single);

            for (int i = 0; i < _hostInputIndex1Double.Length; i++)
            {
                Assert.AreEqual(_hostOutput1Double[i], _hostInput2Double[_hostInputIndex1Double[i]]);
                Assert.AreEqual(_hostOutput1Single[i], _hostInput2Single[_hostInputIndex1Single[i]]);
            }
        }

        [Test]
        public void TestSparseGTHRZ()
        {
            CreateRandomData(_hostInput1Double);
            CreateSparseNonzeroIndex(_hostInputIndex1Double, ciN);
            CreateRandomData(_hostInput2Double);

            _gpu.CopyToDevice(_hostInput1Double, _devVectorxDouble);
            _gpu.CopyToDevice(_hostInputIndex1Double, _devIndexxDouble);
            _gpu.CopyToDevice(_hostInput2Double, _devVectoryDouble);

            _sparse.GTHRZ(_devVectoryDouble, _devVectorxDouble, _devIndexxDouble);
            _gpu.CopyFromDevice(_devVectorxDouble, _hostOutput1Double);
            _gpu.CopyFromDevice(_devVectoryDouble, _hostOutput2Double);

            CreateRandomData(_hostInput1Single);
            CreateSparseNonzeroIndex(_hostInputIndex1Double, ciN);
            CreateRandomData(_hostInput2Double);

            _gpu.CopyToDevice(_hostInput1Single, _devVectorxSingle);
            _gpu.CopyToDevice(_hostInputIndex1Single, _devIndexxSingle);
            _gpu.CopyToDevice(_hostInput2Single, _devVectorySingle);

            _sparse.GTHRZ(_devVectorySingle, _devVectorxSingle, _devIndexxSingle);
            _gpu.CopyFromDevice(_devVectorxSingle, _hostOutput1Single);
            _gpu.CopyFromDevice(_devVectorySingle, _hostOutput2Single);

            for (int i = 0; i < _hostInputIndex1Double.Length; i++)
            {
                Assert.AreEqual(_hostOutput1Double[i], _hostInput2Double[_hostInputIndex1Double[i]]);
                Assert.AreEqual(0.0, _hostOutput2Double[_hostInputIndex1Double[i]]);

                Assert.AreEqual(_hostOutput1Single[i], _hostInput2Single[_hostInputIndex1Single[i]]);
                Assert.AreEqual(0.0f, _hostOutput2Single[_hostInputIndex1Single[i]]);
            }
        }

        [Test]
        public void TestSparseROT()
        {
            double cDouble = 5.0;
            double sDouble = 12.0;
            float cSingle = 5.0f;
            float sSingle = 12.0f;

            CreateRandomData(_hostInput1Double);
            CreateSparseNonzeroIndex(_hostInputIndex1Double, ciN);
            CreateRandomData(_hostInput2Double);

            _gpu.CopyToDevice(_hostInput1Double, _devVectorxDouble);
            _gpu.CopyToDevice(_hostInputIndex1Double, _devIndexxDouble);
            _gpu.CopyToDevice(_hostInput2Double, _devVectoryDouble);

            _sparse.ROT(_devVectorxDouble, _devIndexxDouble, _devVectoryDouble, cDouble, sDouble);

            _gpu.CopyFromDevice(_devVectorxDouble, _hostOutput1Double);
            _gpu.CopyFromDevice(_devVectoryDouble, _hostOutput2Double);

            CreateRandomData(_hostInput1Single);
            CreateSparseNonzeroIndex(_hostInputIndex1Single, ciN);
            CreateRandomData(_hostInput2Single);

            _gpu.CopyToDevice(_hostInput1Single, _devVectorxSingle);
            _gpu.CopyToDevice(_hostInputIndex1Single, _devIndexxSingle);
            _gpu.CopyToDevice(_hostInput2Single, _devVectorySingle);

            _sparse.ROT(_devVectorxSingle, _devIndexxSingle, _devVectorySingle, cSingle, sSingle);

            _gpu.CopyFromDevice(_devVectorxSingle, _hostOutput1Single);
            _gpu.CopyFromDevice(_devVectorySingle, _hostOutput2Single);

            for (int i = 0; i < _hostInputIndex1Double.Length; i++)
            {
                Assert.AreEqual(_hostOutput2Double[_hostInputIndex1Double[i]], cDouble * _hostInput2Double[_hostInputIndex1Double[i]] - sDouble * _hostInput1Double[i]);
                Assert.AreEqual(_hostOutput1Double[i], cDouble * _hostInput1Double[i] + sDouble * _hostInput2Double[_hostInputIndex1Double[i]]);

                Assert.AreEqual(_hostOutput2Single[_hostInputIndex1Single[i]], cSingle * _hostInput2Single[_hostInputIndex1Single[i]] - sSingle * _hostInput1Single[i]);
                Assert.AreEqual(_hostOutput1Single[i], cSingle * _hostInput1Single[i] + sSingle * _hostInput2Single[_hostInputIndex1Single[i]]);
            }
        }

        [Test]
        public void TestSparseSCTR()
        {
            CreateRandomData(_hostInput1Double);
            CreateSparseNonzeroIndex(_hostInputIndex1Double, ciN);
            CreateRandomData(_hostInput2Double);

            _gpu.CopyToDevice(_hostInput1Double, _devVectorxDouble);
            _gpu.CopyToDevice(_hostInputIndex1Double, _devIndexxDouble);
            _gpu.CopyToDevice(_hostInput2Double, _devVectoryDouble);

            _sparse.SCTR(_devVectorxDouble, _devIndexxDouble, _devVectoryDouble);
            _gpu.CopyFromDevice(_devVectoryDouble, _hostOutput2Double);

            CreateRandomData(_hostInput1Single);
            CreateSparseNonzeroIndex(_hostInputIndex1Single, ciN);
            CreateRandomData(_hostInput2Single);

            _gpu.CopyToDevice(_hostInput1Single, _devVectorxSingle);
            _gpu.CopyToDevice(_hostInputIndex1Single, _devIndexxSingle);
            _gpu.CopyToDevice(_hostInput2Single, _devVectorySingle);

            _sparse.SCTR(_devVectorxSingle, _devIndexxSingle, _devVectorySingle);
            _gpu.CopyFromDevice(_devVectorySingle, _hostOutput2Single);

            for (int i = 0; i < _hostInputIndex1Double.Length; i++)
            {
                Assert.AreEqual(_hostOutput2Double[_hostInputIndex1Double[i]], _hostInput1Double[i]);
                Assert.AreEqual(_hostOutput2Single[_hostInputIndex1Single[i]], _hostInput1Single[i]);
            }
        }
    }
}
