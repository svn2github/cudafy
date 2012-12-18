using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Cudafy
{
    /// <summary>
    /// Static methods, static fields and structures to be converted to CUDA C should be decorated with this attribute.
    /// </summary>
    [AttributeUsage(AttributeTargets.Method | AttributeTargets.Struct | AttributeTargets.Field | AttributeTargets.Enum)]
    public class CudafyAttribute : Attribute
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="CudafyAttribute"/> class with type set to eCudafyType.Auto.
        /// </summary>
        public CudafyAttribute()
        {
            CudafyType = eCudafyType.Auto;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="CudafyAttribute"/> class.
        /// </summary>
        /// <param name="type">The type.</param>
        public CudafyAttribute(eCudafyType type)
        {
            CudafyType = type;
        }

        /// <summary>
        /// Gets the type of the cudafy attribute.
        /// </summary>
        /// <value>
        /// The type of the cudafy.
        /// </value>
        public eCudafyType CudafyType { get; private set; }
     
    }

    /// <summary>
    /// Methods, structures and fields that already have an equivalent in Cuda C should be decorated with this attribute.
    /// The item should have the same name and be in a Cuda C (.cu) file of the same name.
    /// </summary>
    [AttributeUsage(AttributeTargets.Method | AttributeTargets.Struct | AttributeTargets.Field)]
    public class CudafyDummyAttribute : Attribute
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="CudafyDummyAttribute"/> class.
        /// </summary>
        /// <param name="type">The type.</param>
        public CudafyDummyAttribute(eCudafyType type)
        {
            CudafyType = type;
            SupportsEmulation = true;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="CudafyDummyAttribute"/> class.
        /// </summary>
        public CudafyDummyAttribute()
        {
            CudafyType = eCudafyType.Auto;
            SupportsEmulation = true;
        }
        
        ///// <summary>
        ///// Initializes a new instance of the <see cref="CudafyDummyAttribute"/> class.
        ///// </summary>
        //public CudafyDummyAttribute(bool supportsEmulation = true)
        //    : this(eCudafyType.Auto, supportsEmulation)
        //{
        //}

        ///// <summary>
        ///// Initializes a new instance of the <see cref="CudafyDummyAttribute"/> class.
        ///// </summary>
        ///// <param name="type">The type.</param>
        //public CudafyDummyAttribute(eCudafyType type, bool supportsEmulation = true)
        //{
        //    CudafyType = type;
        //    SupportsEmulation = supportsEmulation;
        //    //SourceFile = string.Empty;
        //}

        //public CudafyDummyAttribute(eCudafyType type, string sourceFile)
        //{
        //    CudafyType = type;
        //    SourceFile = sourceFile;
        //}

        //public string SourceFile { get; private set; }

        /// <summary>
        /// Gets the type of the cudafy attribute.
        /// </summary>
        /// <value>
        /// The type of the cudafy.
        /// </value>
        public eCudafyType CudafyType { get; private set; }

        /// <summary>
        /// Gets a value indicating whether supports emulation.
        /// </summary>
        /// <value>
        ///   <c>true</c> if supports emulation; otherwise, <c>false</c>.
        /// </value>
        public bool SupportsEmulation { get; private set; }
    }

    /// <summary>
    /// Informs the CudafyTranslator to ignore the member of a struct.
    /// </summary>
    [AttributeUsage(AttributeTargets.Method | AttributeTargets.Property | AttributeTargets.Constructor)]
    public class CudafyIgnoreAttribute : Attribute
    {
    }


    /// <summary>
    /// Placed on parameters to indicate the OpenCL address space. Note that if not specified then arrays will
    /// automatically be marked global. Ignored when translating to CUDA.
    /// </summary>
    [AttributeUsage(AttributeTargets.Parameter)]
    public class CudafyAddressSpaceAttribute : Attribute
    {
        public CudafyAddressSpaceAttribute(eCudafyAddressSpace qualifier)
        {
            Qualifier = qualifier;
        }
        
        public eCudafyAddressSpace Qualifier { get; private set; }
    }
}
