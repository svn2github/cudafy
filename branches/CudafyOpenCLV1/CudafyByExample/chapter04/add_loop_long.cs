/* 
 * This software is based upon the book CUDA By Example by Sanders and Kandrot
 * and source code provided by NVIDIA Corporation.
 * It is a good idea to read the book while studying the examples!
*/
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;

namespace CudafyByExample
{
    public class add_loop_long
    {
        public const int N = 32 * 1024;

        public static void Execute()
        {
            CudafyModule km = CudafyTranslator.Cudafy();

            GPGPU gpu = CudafyHost.GetDevice(CudafyModes.Target);
            gpu.LoadModule(km);

            int[] a = new int[N];
            int[] b = new int[N];
            int[] c = new int[N];

            // allocate the memory on the GPU of same size as specified arrays
            int[] dev_a = gpu.Allocate<int>(a);
            int[] dev_b = gpu.Allocate<int>(b);
            int[] dev_c = gpu.Allocate<int>(c);

            // fill the arrays 'a' and 'b' on the CPU
            for (int i = 0; i < N; i++)
            {
                a[i] = i;
                b[i] = 2 * i;
            }

            // copy the arrays 'a' and 'b' to the GPU
            gpu.CopyToDevice(a, dev_a);
            gpu.CopyToDevice(b, dev_b);

            gpu.Launch(128, 1).add(dev_a, dev_b, dev_c);

            // copy the array 'c' back from the GPU to the CPU
            gpu.CopyFromDevice(dev_c, c);

            // verify that the GPU did the work we requested
            bool success = true;
            for (int i = 0; i < N; i++)
            {
                if ((a[i] + b[i]) != c[i])
                {
                    Console.WriteLine("{0} + {1} != {2}", a[i], b[i], c[i]);
                    success = false;
                    break;
                }
            }
            if (success)
                Console.WriteLine("We did it!");

            // free the memory allocated on the GPU
            gpu.Free(dev_a);
            gpu.Free(dev_b);
            gpu.Free(dev_c);

            // free the memory we allocated on the CPU
            // Not necessary, this is .NET
        }

        [Cudafy]
        public static void add(GThread thread, int[] a, int[] b, int[] c)
        {
            int tid = thread.blockIdx.x;
            while (tid < N)
            {
                c[tid] = a[tid] + b[tid];
                tid += thread.gridDim.x;
            }
        }
    }
}
