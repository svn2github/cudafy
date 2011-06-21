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
using System.Diagnostics;
using System.Text;
using Cudafy.Host;
using Cudafy.UnitTests;
using NUnit.Framework;
using Cudafy.Translator;
using Cudafy.Compilers;
namespace Cudafy.Host.UnitTests
{
    [TestFixture]
    public class StringTests : CudafyUnitTest, ICudafyUnitTest
    {
        private CudafyModule _cm;

        private GPGPU _gpu;

        private const int N = 1024;

        [TestFixtureSetUp]
        public void SetUp()
        {
            CudafyTranslator.GenerateDebug = true;
            _cm = CudafyTranslator.Cudafy();
            _gpu = CudafyHost.GetDevice(CudafyModes.Target, 1);
            _gpu.LoadModule(_cm);
        }

        [TestFixtureTearDown]
        public void TearDown()
        {
            _gpu.FreeAll();
        }

        [Test]
        public void TestTransferUnicodeChar()
        {
            char a = '€';
            char c;
            char[] dev_c = _gpu.Allocate<char>();

            _gpu.Launch(1, 1, "TransferUnicodeChar", a, dev_c);
            _gpu.CopyFromDevice(dev_c, out c);
            _gpu.FreeAll();
            Assert.AreEqual(a, c);
            Debug.WriteLine(c);

        }

        [Cudafy]
        public static void TransferUnicodeChar(char a, char[] c)
        {
            c[0] = a;            
        }

        [Test]
        public void TestTransferUnicodeCharArray()
        {
            string a = "I believe it costs €155,95 in Düsseldorf";
            char[] dev_a = _gpu.CopyToDevice(a);
            char[] dev_c = _gpu.Allocate(a.ToCharArray());
            char[] host_c = new char[a.Length];
            _gpu.Launch(1, 1, "TransferUnicodeCharArray", dev_a, dev_c);
            _gpu.CopyFromDevice(dev_c, host_c);
            string c = new string(host_c);
            _gpu.FreeAll();
            Assert.AreEqual(a, c);
            Debug.WriteLine(c);

        }

        [Cudafy]
        public static void TransferUnicodeCharArray(char[] a, char[] c)
        {
            for(int i = 0; i < a.Length; i++)
                c[i] = a[i];
        }

        [Test]
        public void TestTransferASCIIArray()
        {
            string a = "I believe it costs 155,95 in Duesseldorf";
            byte[] bytes = Encoding.ASCII.GetBytes(a);
            byte[] dev_a = _gpu.CopyToDevice(bytes);
            byte[] dev_c = _gpu.Allocate(bytes);
            byte[] host_c = new byte[a.Length];
            _gpu.Launch(1, 1, "TransferASCIIArray", dev_a, dev_c);
            _gpu.CopyFromDevice(dev_c, host_c);
            string c = Encoding.ASCII.GetString(host_c);
            _gpu.FreeAll();
            Assert.AreEqual(a, c);
            Debug.WriteLine(c);
        }

        [Cudafy]
        public static void TransferASCIIArray(byte[] a, byte[] c)
        {
            for (int i = 0; i < a.Length; i++)
                c[i] = a[i];         
        }


        [Test]
        public void TestWriteHelloOnGPU()
        {
            string a = "€ello\r\nyou";
            char[] dev_c = _gpu.Allocate<char>(a.Length);
            char[] host_c = new char[a.Length];

            _gpu.Launch(1, 1, "WriteHelloOnGPU", dev_c);
            _gpu.CopyFromDevice(dev_c, host_c);
            string c = new string(host_c);
            _gpu.FreeAll();
            Assert.AreEqual(a, c);
            Debug.WriteLine(c);               
        }

        [Cudafy]
        public static void WriteHelloOnGPU(char[] c)
        {
            c[0] = '€';
            c[1] = 'e';
            c[2] = 'l';
            c[3] = 'l';
            c[4] = 'o';
            c[5] = '\r';
            c[6] = '\n';
            c[7] = 'y';
            c[8] = 'o';
            c[9] = 'u';
        }


        [Test]
        public void TestStringSearch()
        {
            string string2Search = "I believe it costs €155,95 in Düsseldorf";
            char[] string2Search_dev = _gpu.CopyToDevice(string2Search);
            
            char char2Find = '€';

            int pos = -1;
            int[] pos_dev = _gpu.Allocate<int>();

            _gpu.Launch(1, 1, "StringSearch", string2Search_dev, char2Find, pos_dev);
            _gpu.CopyFromDevice(pos_dev, out pos);
            _gpu.FreeAll();
            Assert.Greater(pos, 0);
            Assert.AreEqual(string2Search.IndexOf(char2Find), pos);
            Debug.WriteLine(pos);

        }

        [Cudafy]
        public static void StringSearch(char[] text, char match, int[] pos)
        {
            pos[0] = -1;
            for (int i = 0; i < text.Length; i++)
            {
                if (text[i] == match)
                {
                    pos[0] = i;
                    break;
                }
            }
        }


        public void TestSetUp()
        {
        
        }

        public void TestTearDown()
        {
          
        }
    }
}
