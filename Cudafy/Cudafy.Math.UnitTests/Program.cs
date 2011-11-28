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
using NUnit.Framework;
using Cudafy.UnitTests;
using Cudafy.Maths.BLAS;
using Cudafy.Maths.UnitTests;
namespace Cudafy.Host.UnitTests
{
    class Program
    {
        static void Main(string[] args)
        {
            CudafyModes.Target = eGPUType.Cuda;
            try
            {

                BLAS2 b2 = new BLAS2();
                CudafyUnitTest.PerformAllTests(b2);

                BLAS3 b3 = new BLAS3();
                CudafyUnitTest.PerformAllTests(b3);
                
                SPARSE1 sparse = new SPARSE1();
                CudafyUnitTest.PerformAllTests(sparse);

                CURANDHostTests rt = new CURANDHostTests();
                CudafyUnitTest.PerformAllTests(rt);

                BLAS1_1D bt = new BLAS1_1D();
                CudafyUnitTest.PerformAllTests(bt);

                BLAS1_2D bt2 = new BLAS1_2D();
                CudafyUnitTest.PerformAllTests(bt2);

                FFTSingleTests st = new FFTSingleTests();
                CudafyUnitTest.PerformAllTests(st);

                FFTDoubleTests dt = new FFTDoubleTests();
                CudafyUnitTest.PerformAllTests(dt);



            } 
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
            }
            Console.WriteLine("Done");
            Console.ReadLine();
        }
    }
}
