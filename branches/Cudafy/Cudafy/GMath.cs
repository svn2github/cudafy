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

namespace Cudafy
{
#warning TODO complete float versions where necessary    
    
    /// <summary>
    /// Many of the .NET math methods are double only.  When single point (float) is used 
    /// this results in an unwanted cast to double. 
    /// </summary>
    public static class GMath
    {

        /// <summary>
        /// Returns the absolute value of a single precision floating point number.
        /// </summary>
        /// <param name="value">The value to find absolute value of.</param>
        /// <returns>Absolute of specified value.</returns>
        public static float Abs(float value)
        {
            return Math.Abs(value);
        }

        /// <summary>
        /// Returns the square root of a specified number.
        /// </summary>
        /// <param name="value"></param>
        /// <returns></returns>
        public static float Sqrt(float value)
        {
            return (float)Math.Sqrt(value);
        }

        /// <summary>
        /// Returns the cosine of the specified angle. 
        /// </summary>
        /// <param name="value">An angle, measured in radians.</param>
        /// <returns>The cosine of value. If value is equal to NaN, NegativeInfinity, or PositiveInfinity, this method returns NaN.</returns>
        public static float Cos(float value)
        {
            return (float)Math.Cos(value);
        }

        /// <summary>
        /// Acoses the specified value.
        /// </summary>
        /// <param name="value">The value.</param>
        /// <returns></returns>
        public static float Acos(float value)
        {
            return (float)Math.Acos(value);
        }

        /// <summary>
        /// Returns the hyperbolic cosine of the specified angle.
        /// </summary>
        /// <param name="value">An angle, measured in radians.</param>
        /// <returns>The hyperbolic cosine of value. If value is equal to NegativeInfinity or PositiveInfinity, PositiveInfinity is returned. If value is equal to NaN, NaN is returned.</returns>
        public static float Cosh(float value)
        {
            return (float)Math.Cosh(value);
        }

        /// <summary>
        /// Returns the sine of the specified angle. 
        /// </summary>
        /// <param name="value">An angle, measured in radians.</param>
        /// <returns>The sine of value. If value is equal to NaN, NegativeInfinity, or PositiveInfinity, this method returns NaN.</returns>
        public static float Sin(float value)
        {
            return (float)Math.Sin(value);
        }

        /// <summary>
        /// Returns the sine of the specified angle. 
        /// </summary>
        /// <param name="value">An angle, measured in radians.</param>
        /// <returns>The sine of value. If value is equal to NaN, NegativeInfinity, or PositiveInfinity, this method returns NaN.</returns>
        public static float Asin(float value)
        {
            return (float)Math.Asin(value);
        }

        /// <summary>
        /// Returns the hyperbolic sine of the specified angle. 
        /// </summary>
        /// <param name="value">An angle, measured in radians.</param>
        /// <returns>The hyperbolic sine of value. If value is equal to NaN, NegativeInfinity, or PositiveInfinity, this method returns NaN.</returns> 
        public static float Sinh(float value)
        {
            return (float)Math.Sinh(value);
        }

        /// <summary>
        /// Returns the tan of the specified angle. 
        /// </summary>
        /// <param name="value">An angle, measured in radians.</param>
        /// <returns>The tan of value.</returns>
        public static float Tan(float value)
        {
            return (float)Math.Tan(value);
        }

        /// <summary>
        /// Returns the tan of the specified angle. 
        /// </summary>
        /// <param name="value">An angle, measured in radians.</param>
        /// <returns>The tan of value.</returns>
        public static float Atan(float value)
        {
            return (float)Math.Atan(value);
        }

        /// <summary>
        /// Returns the angle whose tangent is the quotient of two specified numbers.
        /// </summary>
        /// <param name="y">The y coordinate of a point.</param>
        /// <param name="x">The x coordinate of a point.</param>
        /// <returns>Type: System.Double</returns>
        public static float Atan2(float y, float x)
        {
            return (float)Math.Atan2(y, x);
        }

        /// <summary>
        /// Returns hyperbolic the tangent of the specified angle. 
        /// </summary>
        /// <param name="value">An angle, measured in radians.</param>
        /// <returns>The hyperbolic the tangent of value.</returns>
        public static float Tanh(float value)
        {
            return (float)Math.Tanh(value);
        }

        /// <summary>
        /// Rounds the specified value.
        /// </summary>
        /// <param name="value">The value.</param>
        /// <returns>Rounded value.</returns>
        public static float Round(float value)
        {
            return (float)Math.Round(value);
        }

        /// <summary>
        /// Ceilings the specified value.
        /// </summary>
        /// <param name="value">The value.</param>
        /// <returns></returns>
        public static float Ceiling(float value)
        {
            return (float)Math.Ceiling(value);
        }

        /// <summary>
        /// Floors the specified value.
        /// </summary>
        /// <param name="value">The value.</param>
        /// <returns></returns>
        public static float Floor(float value)
        {
            return (float)Math.Floor(value);
        }

        /// <summary>
        /// Represents the ratio of the circumference of a circle to its diameter, specified by the constant, π.
        /// </summary>
        public static float PI
        {
            get { return (float)Math.PI; }
        }

        /// <summary>
        /// Represents the natural logarithmic base, specified by the constant, e.
        /// </summary>
        public static float E
        {
            get { return (float)Math.E; }
        }

        /// <summary>
        /// Returns the specified number raised to the specified power.
        /// </summary>
        /// <param name="x">Number to be raised to a power.</param>
        /// <param name="y">Number that specifies the power.</param>
        /// <returns>X to the power of y.</returns>
        public static float Pow(float x, float y)
        {
            return (float)Math.Pow(x, y);
        }

        /// <summary>
        /// Returns the base 10 log of the specified number.
        /// </summary>
        /// <param name="value">A number whose logarithm is to be found.</param>
        /// <returns>Result.</returns>
        public static float Log10(float value)
        {
            return (float)Math.Log10(value);
        }
#warning TODO Ensure that two args are not passed to Math.Log
        /// <summary>
        /// Returns the natural (base e) logarithm of a specified number.
        /// </summary>
        /// <param name="value">A number whose logarithm is to be found.</param>
        /// <returns>Result.</returns>
        public static float Log(float value)
        {
            return (float)Math.Log(value);
        }

        /// <summary>
        /// Returns e raised to the specified power.
        /// </summary>
        /// <param name="value">A number specifying a power. </param>
        /// <returns>The number e raised to the power d. If d equals NaN or PositiveInfinity, that value is returned. If d equals NegativeInfinity, 0 is returned.</returns>
        public static float Exp(float value)
        {
            return (float)Math.Exp(value);
        }


        /// <summary>
        /// Truncates the specified value.
        /// </summary>
        /// <param name="value">The value.</param>
        /// <returns></returns>
        public static float Truncate(float value)
        {
            return (float)Math.Truncate(value);
        }

        /// <summary>
        /// Returns the larger of two single float precision numbers.
        /// </summary>
        /// <param name="x">The first number to compare.</param>
        /// <param name="y">The second number to compare.</param>
        /// <returns>The larger of the two numbers.</returns>
        public static float Max(float x, float y)
        {
            return (float)Math.Max(x, y);
        }

        /// <summary>
        /// Returns the smaller of two single float precision numbers.
        /// </summary>
        /// <param name="x">The first number to compare.</param>
        /// <param name="y">The second number to compare.</param>
        /// <returns>The smaller of the two numbers.</returns>
        public static float Min(float x, float y)
        {
            return (float)Math.Min(x, y);
        }

        ///// <summary>
        ///// Exp10s the specified value.
        ///// </summary>
        ///// <param name="value">The value.</param>
        ///// <returns></returns>
        //public static float Exp10(float value)
        //{
        //    return (float)Math.Pow(Math.E, value);
        //}

        ///// <summary>
        ///// Exp10s the specified d.
        ///// </summary>
        ///// <param name="d">The d.</param>
        ///// <returns></returns>
        //public static double Exp10(double d)
        //{
        //    return Math.Pow(Math.E, d);
        //}
    }
}
