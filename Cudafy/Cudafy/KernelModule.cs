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
using System.Reflection;
using System.Diagnostics;
using System.IO;
using System.Threading;
using System.Xml;
using System.Xml.Linq;
using System.Linq;
using System.Text;
using Cudafy.Compilers;
//using GASS.CUDA.Types;
namespace Cudafy
{
    /// <summary>
    /// Flags for compilers.
    /// </summary>
    [Flags]
    public enum eGPUCompiler
    {
        /// <summary>
        /// None.
        /// </summary>
        None = 0,
        /// <summary>
        /// Nvcc Cuda compiler.
        /// </summary>
        CudaNvcc = 1,
        /// <summary>
        /// Compile for all targets.
        /// </summary>
        All = 255
    };

    /// <summary>
    /// Internal use.
    /// </summary>
    public class PTXModule
    {
        /// <summary>
        /// Gets the platform.
        /// </summary>
        public ePlatform Platform { get; internal set; }
#if DEBUG
        public string PTX { get; set; }
#else
        /// <summary>
        /// Gets the PTX.
        /// </summary>
        public string PTX { get; internal set; }
#endif
        /// <summary>
        /// Returns a <see cref="System.String"/> that represents this instance.
        /// </summary>
        /// <returns>
        /// A <see cref="System.String"/> that represents this instance.
        /// </returns>
        public override string ToString()
        {
            return Platform.ToString();
        }
    }


    /// <summary>
    /// Cudafy module.
    /// </summary>
    public class CudafyModule
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="CudafyModule"/> class.
        /// </summary>
        public CudafyModule()
        {
            _is64bit = IntPtr.Size == 8;
            CudaSourceCode = string.Empty;
            Name = "cudafymodule";
            CompilerOutput = string.Empty;
            CompilerArguments = string.Empty;
            _options = new List<CompilerOptions>();
            _PTXModules = new List<PTXModule>();
            Reset();
        }

        private bool _is64bit;

        /// <summary>
        /// Gets or sets the name.
        /// </summary>
        /// <value>
        /// The name.
        /// </value>
        public string Name { get; set; }

        /// <summary>
        /// Gets or sets optional extra data (CUmodule).
        /// </summary>
        /// <value>
        /// The data.
        /// </value>
        public object Tag { get; set; }

        /// <summary>
        /// Gets the functions.
        /// </summary>
        public Dictionary<string, KernelMethodInfo> Functions { get; internal set; }

        /// <summary>
        /// Gets the constants.
        /// </summary>
        public Dictionary<string, KernelConstantInfo> Constants { get; internal set; }

        /// <summary>
        /// Gets the types.
        /// </summary>
        public Dictionary<string, KernelTypeInfo> Types { get; internal set; }

        /// <summary>
        /// Gets the member names.
        /// </summary>
        public IEnumerable<string> GetMemberNames()
        {
            foreach (var v in Functions)
                yield return v.Key;
            foreach (var v in Constants)
                yield return v.Key;
            foreach (var v in Types)
                yield return v.Key;
        }

        /// <summary>
        /// NOT IMPLEMENTED YET. Gets or sets a value indicating whether this instance can print to console.
        /// </summary>
        /// <value>
        ///   <c>true</c> if this instance can print; otherwise, <c>false</c>.
        /// </value>
        public bool CanPrint { get; set; }

        //public string Architecture { get; set; }

        private List<PTXModule> _PTXModules;

        /// <summary>
        /// Gets the PTX modules.
        /// </summary>
        public PTXModule[] PTXModules
        {
            get { return _PTXModules.ToArray(); }
        }

        /// <summary>
        /// Removes the PTX modules.
        /// </summary>
        public void RemovePTXModules()
        {
            _PTXModules.Clear();
        }

        /// <summary>
        /// Gets the current platform.
        /// </summary>
        public ePlatform CurrentPlatform
        {
            get { return _is64bit ? ePlatform.x64 : ePlatform.x86; }
        }

        /// <summary>
        /// Gets the first PTX suitable for the current platform.
        /// </summary>
        public PTXModule PTX
        {
            get { return _PTXModules.Where(ptx => ptx.Platform == CurrentPlatform).FirstOrDefault(); }
        }

        /// <summary>
        /// Gets a value indicating whether this instance has suitable PTX.
        /// </summary>
        /// <value>
        ///   <c>true</c> if this instance has PTX; otherwise, <c>false</c>.
        /// </value>
        public bool HasSuitablePTX
        {
            get { return _PTXModules.Count(ptx => ptx.Platform == CurrentPlatform) > 0; }
        }

        /// <summary>
        /// Gets a value indicating whether this instance has one or more PTX.
        /// </summary>
        /// <value>
        ///   <c>true</c> if this instance has PTX; otherwise, <c>false</c>.
        /// </value>
        public bool HasPTX
        {
            get { return _PTXModules.Count > 0; }
        }

        /// <summary>
        /// Determines whether module has PTX for the specified platform.
        /// </summary>
        /// <param name="platform">The platform.</param>
        /// <returns>
        ///   <c>true</c> if module has PTX for the specified platform; otherwise, <c>false</c>.
        /// </returns>
        public bool HasPTXForPlatform(ePlatform platform)
        {
            return _PTXModules.Count(ptx => ptx.Platform == platform) > 0;
        }

        internal void StorePTXFile(ePlatform platform, string path)
        {
            using (StreamReader sr = File.OpenText(path))
            {
                string ptx = sr.ReadToEnd();
                _PTXModules.Add(new PTXModule() { Platform = platform, PTX = ptx });
            }
        }

        /// <summary>
        /// Gets or sets the cuda source code.
        /// </summary>
        /// <value>
        /// The cuda source code.
        /// </value>
        public string CudaSourceCode { get; set; }

        /// <summary>
        /// Gets a value indicating whether this instance has cuda source code.
        /// </summary>
        /// <value>
        /// 	<c>true</c> if this instance has cuda source code; otherwise, <c>false</c>.
        /// </value>
        public bool HasCudaSourceCode
        {
            get { return !string.IsNullOrEmpty(CudaSourceCode); }
        }

        /// <summary>
        /// Resets this instance.
        /// </summary>
        public void Reset()
        {
            Functions = new Dictionary<string, KernelMethodInfo>();
            Constants = new Dictionary<string, KernelConstantInfo>();
            Types = new Dictionary<string, KernelTypeInfo>();
            _PTXModules.Clear();
            CanPrint = false;
            CompilerOptionsList.Clear();
        }

        private const string csCUDAFYMODULE = "CudafyModule";
        private const string csVERSION = "Version";
        private const string csCUDASOURCECODE = "CudaSourceCode";
        private const string csHASCUDASOURCECODE = "HasCudaSourceCode";
        private const string csPTX = "PTX";
        private const string csPTXMODULE = "PTXMODULE";
        private const string csHASPTX = "HasPTX";
        private const string csFUNCTIONS = "Functions";
        private const string csCONSTANTS = "Constants";
        private const string csCONSTANT = "Constant";
        private const string csTYPES = "Types";
        private const string csFILEEXT = ".cdfy";
        private const string csPLATFORM = "Platform";
        private const string csARCH = "Arch";
        private const string csNAME = "Name";
        private const string csDEBUGINFO = "DebugInfo";

        /// <summary>
        /// Trues to serialize this instance to file based on Name.
        /// </summary>
        /// <returns>True if successful, else false.</returns>
        public bool TrySerialize()
        {
            try
            {
                Serialize();
                return true;
            }
            catch (Exception ex)
            {
                Debug.WriteLine(ex.Message);
#if DEBUG
                throw;
#endif
            }
            return false;
        }

        /// <summary>
        /// Serializes this instance to file based on Name.
        /// </summary>
        public void Serialize()
        {
            Serialize(Name);
        }

        /// <summary>
        /// Serializes the module to the specified filename.
        /// </summary>
        /// <param name="filename">The filename.</param>
        public void Serialize(string filename)
        {
            XDocument doc = new XDocument(new XDeclaration("1.0", "utf-8", null));

            byte[] cudaSrcBa = UnicodeEncoding.ASCII.GetBytes(CudaSourceCode);
            string cudaSrcB64 = Convert.ToBase64String(cudaSrcBa);

            XElement root = new XElement(csCUDAFYMODULE);
            root.SetAttributeValue(csVERSION, this.GetType().Assembly.GetName().Version.ToString());
            root.SetAttributeValue(csNAME, Name == null ? "cudafymodule" : Name);
            root.SetAttributeValue(csDEBUGINFO, GenerateDebug);
            
            XElement cudaSrc = new XElement(csCUDASOURCECODE, cudaSrcB64);
            root.SetAttributeValue(csHASCUDASOURCECODE, XmlConvert.ToString(HasCudaSourceCode));
            root.Add(cudaSrc);
            
            root.SetAttributeValue(csHASPTX, XmlConvert.ToString(_PTXModules.Count > 0));
            foreach (var ptxMod in _PTXModules)
            {
                byte[] ba = UnicodeEncoding.ASCII.GetBytes(ptxMod.PTX);
                string b64 = Convert.ToBase64String(ba);
                XElement ptxXe = new XElement(csPTXMODULE, 
                    new XElement(csPTX, b64));
                ptxXe.SetAttributeValue(csPLATFORM, ptxMod.Platform);
                root.Add(ptxXe);
            }
          
            XElement funcs = new XElement(csFUNCTIONS);
            root.Add(funcs);
            foreach (var kvp in Functions)
            {
                XElement xe = kvp.Value.GetXElement();
                funcs.Add(xe);
            }

            XElement constants = new XElement(csCONSTANTS);
            root.Add(constants);
            foreach (var kvp in Constants)
            {
                XElement xe = kvp.Value.GetXElement();
                constants.Add(xe);
            }

            XElement types = new XElement(csTYPES);
            root.Add(types);
            foreach (var kvp in Types)
            {
                XElement xe = kvp.Value.GetXElement();
                types.Add(xe);
            }

            doc.Add(root);
            if (!Path.HasExtension(filename))
                filename += csFILEEXT;
            doc.Save(filename);
        }

        /// <summary>
        /// Deletes the specified filename (with or without default .cdfy extension).
        /// </summary>
        /// <param name="filename">The filename.</param>
        /// <returns>True if file was deleted else false.</returns>
        public static bool Clean(string filename)
        {
            filename = GetFilename(filename);
            bool exists = File.Exists(filename);
            if (exists)
                File.Delete(filename);
            return exists;
        }

        private static string GetFilename(string filename)
        {
            if (!File.Exists(filename))
                filename += csFILEEXT;
            return filename;
        }

        /// <summary>
        /// Gets the dummy struct includes.
        /// </summary>
        /// <returns>Strings representing the Cuda include files.</returns>
        public IEnumerable<string> GetDummyStructIncludes()
        {
            foreach (var kvp in Types.Where(k => k.Value.IsDummy))
                yield return kvp.Value.GetDummyInclude();
        }

        /// <summary>
        /// Gets the dummy includes.
        /// </summary>
        /// <returns>Strings representing the Cuda include files.</returns>
        public IEnumerable<string> GetDummyIncludes()
        {
            foreach (var kvp in Functions.Where(k => k.Value.IsDummy))
                yield return kvp.Value.GetDummyInclude();
        }

        /// <summary>
        /// Gets the dummy defines.
        /// </summary>
        /// <returns>Strings representing the Cuda defines files.</returns>
        public IEnumerable<string> GetDummyDefines()
        {
            foreach (var kvp in Constants.Where(k => k.Value.IsDummy))
                yield return kvp.Value.GetDummyDefine();
        }

        /// <summary>
        /// Tries to deserialize from a file with the same name as the calling type.
        /// </summary>
        /// <returns>Cudafy module or null if failed.</returns>
        public static CudafyModule TryDeserialize()
        {
            StackTrace stackTrace = new StackTrace();
            Type type = stackTrace.GetFrame(1).GetMethod().ReflectedType;
            return TryDeserialize(type.Name);
        }

        /// <summary>
        /// Tries to deserialize from the specified file.
        /// </summary>
        /// <param name="filename">The filename.</param>
        /// <returns>Cudafy module or null if failed.</returns>
        public static CudafyModule TryDeserialize(string filename)
        {
            string ts;
            return TryDeserialize(filename, out ts);
        }

        /// <summary>
        /// Tries to deserialize from the specified file.
        /// </summary>
        /// <param name="filename">The filename.</param>
        /// <param name="errorMsg">The error message if fails, else empty string.</param>
        /// <returns>Cudafy module or null if failed.</returns>
        public static CudafyModule TryDeserialize(string filename, out string errorMsg)
        {
            errorMsg = string.Empty;
            filename = GetFilename(filename);
            if (!File.Exists(filename))
                return null;
            CudafyModule km = null;
            try
            {
                km = Deserialize(filename);
            }
            catch (Exception ex)
            {
                errorMsg = ex.Message;
            }
            return km;
        }

        /// <summary>
        /// Deserializes from a file with the same name as the calling type.
        /// </summary>
        /// <returns>Cudafy module.</returns>
        public static CudafyModule Deserialize()
        {
            StackTrace stackTrace = new StackTrace();
            Type type = stackTrace.GetFrame(1).GetMethod().ReflectedType;
            return Deserialize(type.Name);
        }
#warning TODO http://www.codeproject.com/KB/dotnet/AppDomain_quick_start.aspx
        /// <summary>
        /// Deserializes the specified file.
        /// </summary>
        /// <param name="filename">The filename.</param>
        /// <returns>Cudafy module.</returns>
        public static CudafyModule Deserialize(string filename)
        {
            CudafyModule km = new CudafyModule();
            if (!File.Exists(filename))
                filename += csFILEEXT;

            XDocument doc = XDocument.Load(filename);

            string path = Path.GetDirectoryName(filename);

            return Deserialize(km, doc, path);
        }

        public static CudafyModule Deserialize(Stream stream)
        {
            CudafyModule km = new CudafyModule();
            XDocument doc = XDocument.Load(stream);
            return Deserialize(km, doc, null);
        }

        private static CudafyModule Deserialize(CudafyModule km, XDocument doc, string path)
        {
            XElement root = doc.Element(csCUDAFYMODULE);
            if (root == null)
                throw new XmlException(string.Format(GES.csELEMENT_X_NOT_FOUND, csCUDAFYMODULE));

            string vStr = root.GetAttributeValue(csVERSION);
            Version version = new Version(vStr);
            Version curVers = typeof(CudafyModule).Assembly.GetName().Version;
            if (version.Major != curVers.Major || version.Minor > curVers.Minor)
                throw new CudafyException(CudafyException.csVERSION_MISMATCH_EXPECTED_X_GOT_X, curVers, version);

            string name = root.TryGetAttributeValue(csNAME);
            km.Name = name != null ? name : km.Name;

            // Cuda Source
            bool? hasCudaSrc = root.TryGetAttributeBoolValue(csHASCUDASOURCECODE);
            if (hasCudaSrc == true)
            {
                string src = root.Element(csCUDASOURCECODE).Value;
                byte[] ba = Convert.FromBase64String(src);
                km.CudaSourceCode = UnicodeEncoding.ASCII.GetString(ba);
            }
            else
            {
                km.CudaSourceCode = string.Empty;
            }

            bool? hasDebug = root.TryGetAttributeBoolValue(csDEBUGINFO);
            km.GenerateDebug = hasDebug.HasValue && hasDebug.Value;

            // PTX
            bool? hasPtx = root.TryGetAttributeBoolValue(csHASPTX);
            if (hasPtx == true && root.Element(csPTX) != null) // legacy support V0.3 or less
            {
                string ptx = root.Element(csPTX).Value;
                byte[] ba = Convert.FromBase64String(ptx);
                km._PTXModules.Add(new PTXModule() { PTX = UnicodeEncoding.ASCII.GetString(ba), Platform = km.CurrentPlatform });
            }
            else if (hasPtx == true)
            {
                foreach (XElement xe in root.Elements(csPTXMODULE))
                {
                    string ptx = xe.Element(csPTX).Value;
                    string platformStr = xe.GetAttributeValue(csPLATFORM);
                    ePlatform platform = (ePlatform)Enum.Parse(typeof(ePlatform), platformStr);
                    byte[] ba = Convert.FromBase64String(ptx);
                    km._PTXModules.Add(new PTXModule() { PTX = UnicodeEncoding.ASCII.GetString(ba), Platform = platform });
                }
            }

            // Functions
            XElement funcs = root.Element(csFUNCTIONS);
            if (funcs != null)
            {
                foreach (var xe in funcs.Elements(KernelMethodInfo.csCUDAFYKERNELMETHOD))
                {
                    KernelMethodInfo kmi = KernelMethodInfo.Deserialize(xe, path);
                    km.Functions.Add(kmi.Method.Name, kmi);
                }
            }

            // Constants
            XElement constants = root.Element(csCONSTANTS);
            if (constants != null)
            {
                foreach (var xe in constants.Elements(KernelConstantInfo.csCUDAFYCONSTANTINFO))
                {
                    KernelConstantInfo kci = KernelConstantInfo.Deserialize(xe, path);
                    km.Constants.Add(kci.Name, kci);
                }
            }

            // Types
            XElement types = root.Element(csTYPES);
            if (constants != null)
            {
                foreach (var xe in types.Elements(KernelTypeInfo.csCUDAFYTYPE))
                {
                    KernelTypeInfo kti = KernelTypeInfo.Deserialize(xe, path);
                    km.Types.Add(kti.Name, kti);
                }
            }

            return km;
        }

        /// <summary>
        /// Verifies the checksums of all functions, constants and types.
        /// </summary>
        /// <exception cref="CudafyException">Check sums don't match or total number of members is less than one, .</exception>
        public void VerifyChecksums()
        {
            if (GetTotalMembers() == 0)
                throw new CudafyException(CudafyException.csNO_MEMBERS_FOUND);
            if (!HasSuitablePTX)
                throw new CudafyException(CudafyException.csNO_SUITABLE_X_PRESENT_IN_CUDAFY_MODULE, "PTX");
            foreach (var kvp in Functions)
                kvp.Value.VerifyChecksums();
            foreach (var kvp in Constants)
                kvp.Value.VerifyChecksums();
            foreach (var kvp in Types)
                kvp.Value.VerifyChecksums();       
        }


        /// <summary>
        /// Verifies the checksums of all functions, constants and types.
        /// </summary>
        /// <returns>True if checksums match and total number of members is greater than one, else false.</returns>
        public bool TryVerifyChecksums()
        {
            if (GetTotalMembers() == 0 || !HasSuitablePTX)
                return false;
            foreach (var kvp in Functions)
                if (kvp.Value.TryVerifyChecksums() == false)
                    return false;
            foreach (var kvp in Constants)
                if (kvp.Value.TryVerifyChecksums() == false)
                    return false;
            foreach (var kvp in Types)
                if (kvp.Value.TryVerifyChecksums() == false)
                    return false;
            return true;
        }

        private int GetTotalMembers()
        {
            return Functions.Count + Constants.Count + Types.Count;
        }

        /// <summary>
        /// Gets the compiler options.
        /// </summary>
        /// <value>
        /// The compiler options.
        /// </value>
        public List<CompilerOptions> CompilerOptionsList 
        { 
            get 
            { 
                return _options; 
            } 
        }

        //public void AddCompilerOptions(CompilerOptions co)
        //{
        //    var existingOpt = _options.Where(opt => opt.
        //}


        private List<CompilerOptions> _options;

        /// <summary>
        /// Gets the compiler output.
        /// </summary>
        public string CompilerOutput { get; private set; }

        /// <summary>
        /// Gets the last arguments passed to compiler.
        /// </summary>
        public string CompilerArguments { get; private set; }

        private string _workingDirectory;

        /// <summary>
        /// Gets or sets the working directory for the compiler.
        /// </summary>
        public string WorkingDirectory
        {
            get { return (string.IsNullOrEmpty(_workingDirectory) || !Directory.Exists(_workingDirectory)) ? Environment.CurrentDirectory : _workingDirectory; }
            set { _workingDirectory = value; }
        }

        /// <summary>
        /// Gets or sets a value indicating whether to compile for debug.
        /// </summary>
        /// <value>
        ///   <c>true</c> if compile for debug; otherwise, <c>false</c>.
        /// </value>
        public bool GenerateDebug { get; set; }

        ///// <summary>
        ///// Gets or sets a value indicating whether to start the compilation in a new window.
        ///// </summary>
        ///// <value>
        /////   <c>true</c> if suppress a new window; otherwise, <c>false</c>.
        ///// </value>
        //public bool SuppressWindow { get; set; }

        /// <summary>
        /// Compiles the module based on current Cuda source code and options.
        /// </summary>
        /// <param name="mode">The mode.</param>
        /// <param name="deleteGeneratedCode">if set to <c>true</c> delete generated code on success.</param>
        /// <returns>The compile arguments.</returns>
        /// <exception cref="CudafyCompileException">No source code or compilation error.</exception>
        public string Compile(eGPUCompiler mode, bool deleteGeneratedCode = false)
        {
            string ts = string.Empty;
            if ((mode & eGPUCompiler.CudaNvcc) == eGPUCompiler.CudaNvcc)
            {
                CompilerOutput = string.Empty;
                _PTXModules.Clear();
                if (!HasCudaSourceCode)
                    throw new CudafyCompileException(CudafyCompileException.csNO_X_SOURCE_CODE_PRESENT_IN_CUDAFY_MODULE, "Cuda");

                if (CompilerOptionsList.Count == 0)
                {
                    CompilerOptionsList.Add(IntPtr.Size == 4 ? NvccCompilerOptions.Createx86() : NvccCompilerOptions.Createx64());
                }
                
                // Write to temp file
                string tempFileName = "CUDAFYSOURCETEMP.tmp";
                string cuFileName = WorkingDirectory + Path.DirectorySeparatorChar + tempFileName.Replace(".tmp", ".cu");
                string ptxFileName = WorkingDirectory + Path.DirectorySeparatorChar + tempFileName.Replace(".tmp", ".ptx");
                File.WriteAllText(cuFileName, CudaSourceCode, Encoding.Default);

                foreach (CompilerOptions co in CompilerOptionsList)
                {
                    co.GenerateDebugInfo = GenerateDebug;
                    
                    co.ClearSources();
                    co.AddSource(cuFileName);

                    co.ClearOutputs();
                    co.AddOutput(ptxFileName);

                    CompilerOutput += "\r\n" + co.GetSummary();
                    CompilerOutput += "\r\n" + co.GetArguments();

                    // Convert to ptx
                    Process process = new Process();
                    process.StartInfo.UseShellExecute = false;
                    process.StartInfo.RedirectStandardOutput = true;
                    process.StartInfo.RedirectStandardError = true;
                    //process.StartInfo.CreateNoWindow = SuppressWindow;//WindowStyle = System.Diagnostics.ProcessWindowStyle.Hidden;
                    process.StartInfo.FileName = co.GetFileName();
                    process.StartInfo.Arguments = co.GetArguments();
                    CompilerArguments = process.StartInfo.Arguments;
                    Debug.WriteLine(process.StartInfo.FileName);
                    Debug.WriteLine(CompilerArguments);
                    process.Start();

                    while (!process.HasExited)
                        Thread.Sleep(10);

                    if (process.ExitCode != 0)
                    {
                        string s = process.StandardError.ReadToEnd();
                        
                        CompilerOutput += "\r\n" + s;
                        if (s.Contains("Cannot find compiler 'cl.exe' in PATH"))
                            CompilerOutput += "\r\nPlease add the Visual Studio VC bin directory to PATH in Environment Variables.";
                        Debug.WriteLine(s);
                        throw new CudafyCompileException(CudafyCompileException.csCOMPILATION_ERROR_X, s);
                    }
                    else
                    {
                        string s = process.StandardError.ReadToEnd() + "\r\n" + process.StandardOutput.ReadToEnd();
                        CompilerOutput += "\r\n" + s;
                        Debug.WriteLine(s);
                    }

                    // Load ptx file
                    this.StorePTXFile(co.Platform, ptxFileName);
#if DEBUG

#else
                    if (deleteGeneratedCode)
                        Delete(cuFileName, ptxFileName);
#endif
                }
            }
            return CompilerArguments;
        }

        private static void Delete(string cuFileName, string ptxFileName)
        {
            // Delete files
            try
            {
                if(File.Exists(cuFileName))
                    File.Delete(cuFileName);
                if(File.Exists(ptxFileName))
                    File.Delete(ptxFileName);
            }
            catch (Exception ex)
            {
                Debug.WriteLine(ex.ToString());
            }
        }
    }
}
