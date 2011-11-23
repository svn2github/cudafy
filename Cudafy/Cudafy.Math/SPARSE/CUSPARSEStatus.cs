/* Added by Kichang Kim (kkc0923@hotmail.com) */
namespace Cudafy.Maths.SPARSE
{
    using System;

    public enum CUSPARSEStatus
    {
        Success = 0,
        NotInitialized = 1,
        AllocFailed = 2,
        InvalidValue = 3,
        ArchMismatch = 4,
        MappingError = 5,
        ExecutionFailed = 6,
        InternalError = 7,
        MatrixTypeNotSupported = 8
    }
}

