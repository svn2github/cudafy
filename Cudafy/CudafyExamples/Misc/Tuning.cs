using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;
namespace CudafyExamples.Misc
{


    public class TypeTest
    {
        protected const int _threadsPerBlock = 256;
        protected const int _blocksPerGrid = 256;

        [Cudafy]
        public struct AnswerStruct { public float distance; public long pathNo; }

        public static AnswerStruct GetAnswer()
        {
            using (var gpu = CudafyHost.GetDevice())
            {
                gpu.LoadModule(CudafyTranslator.Cudafy());

                var answer = new AnswerStruct[_blocksPerGrid]; ;
                var gpuAnswer = gpu.Allocate(answer);

                gpu.Launch(_blocksPerGrid, _threadsPerBlock,
                   GpuFindPathDistance, gpuAnswer);

                gpu.Synchronize();
                gpu.CopyFromDevice(gpuAnswer, answer);
                gpu.FreeAll();

                var bestDistance = float.MaxValue;
                var bestPermutation = 0L;
                for (var i = 0; i < _blocksPerGrid; i++)
                {
                    if (answer[i].distance < bestDistance)
                    {
                        bestDistance = answer[i].distance;
                        bestPermutation = answer[i].pathNo;
                    }
                }

                return new AnswerStruct
                {
                    distance = bestDistance,
                    pathNo = bestPermutation,
                };
            }
        }

        [Cudafy]
        public static void GpuFindPathDistance(GThread thread, AnswerStruct[] answer)
        {
            var answerLocal = thread.AllocateShared<AnswerStruct>("ansL", _threadsPerBlock);

            var bestDistance = float.MaxValue;
            var bestPermutation = 0L;

            answerLocal[thread.threadIdx.x].distance = bestDistance;
            answerLocal[thread.threadIdx.x].pathNo = bestPermutation;
            thread.SyncThreads();

            if (thread.threadIdx.x == 0)
            {
                answer[thread.blockIdx.x] = answerLocal[0];
            }
        }
    }
}
