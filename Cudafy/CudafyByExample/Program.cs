/* 
 * This software is based upon the book CUDA By Example by Sanders and Kandrot
 * and source code provided by NVIDIA Corporation.
 * It is a good idea to read the book while studying the examples!
*/
using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using System.Text;
using System.Diagnostics;
using System.Runtime.InteropServices;
using Cudafy;

namespace CudafyByExample
{
    class Program
    {
        [STAThread]
        static void Main(string[] args)
        {
            Console.WriteLine(IntPtr.Size);
            CudafyModes.CodeGen = eGPUCodeGenerator.CudaC;
            CudafyModes.Target = eGPUType.Cuda;
            try
            {

               
                
                // Chapter 3
                Console.WriteLine("\r\nChapter 3");
                //Console.WriteLine("\r\nhello_world");
                //hello_world.Execute();
                //Console.WriteLine("\r\nsimple_kernel");
                //simple_kernel.Execute();
                Console.WriteLine("\r\nsimple_kernel_params");
                simple_kernel_params.Execute();
                Console.WriteLine("\r\nenum_gpu");
                enum_gpu.Execute();

                // Chapter 4
                Console.WriteLine("\r\nChapter 4");
                Console.WriteLine("\r\nadd_loop_cpu");
                add_loop_cpu.Execute();
                Console.WriteLine("\r\nadd_loop_gpu");
                add_loop_gpu.Execute();
                Console.WriteLine("\r\nadd_loop_gpu_alt");
                add_loop_gpu_alt.Execute();
                Console.WriteLine("\r\nadd_loop_long");
                add_loop_long.Execute();
                Console.WriteLine("\r\njulia (cpu)");
                new julia_gui(false).ShowDialog();
                Console.WriteLine("\r\njulia (gpu)");
                new julia_gui(true).ShowDialog();

                // Chapter 5
                Console.WriteLine("\r\nChapter 5");
                Console.WriteLine("\r\nadd_loop_blocks");
                add_loop_blocks.Execute();
                Console.WriteLine("\r\nadd_loop_long_blocks");
                add_loop_long_blocks.Execute();
                Console.WriteLine("\r\nripple");
                ripple r = new ripple();
                r.Execute();
                Console.WriteLine("\r\ndot");
                dot.Execute();

                // Chapter 6
                Console.WriteLine("\r\nChapter 6");
                Console.WriteLine("\r\nray (no constant memory)");
                new ray_gui(false).ShowDialog(); // no const
                Console.WriteLine("\r\nray (constant memory)");
                new ray_gui(true).ShowDialog();  // const

                // Chapter 9
                Console.WriteLine("\r\nChapter 9");
                Console.WriteLine("\r\nhist_gpu_shmem_atomics");
                hist_gpu_shmem_atomics.Execute();

                // Chapter 10
                Console.WriteLine("\r\nChapter 10");
                Console.WriteLine("\r\nbasic_double_stream_correct");
                basic_double_stream_correct.Execute();
                Console.WriteLine("\r\ncopy_timed");
                new copy_timed().Execute();

                Console.WriteLine("Done!");
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex);
            }
            Console.ReadKey();
        }

    }
}
