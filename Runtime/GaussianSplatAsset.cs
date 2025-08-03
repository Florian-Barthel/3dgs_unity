// SPDX-License-Identifier: MIT

using System;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Experimental.Rendering;

namespace GaussianSplatting.Runtime
{
    public class GaussianSplatAsset : ScriptableObject
    {
        public const int kCurrentVersion = 2023_10_20;
        public const int kChunkSize = 256;
        public const int kTextureWidth = 2048; // allows up to 32M splats on desktop GPU (2k width x 16k height)
        public const int kMaxSplats = 8_600_000; // mostly due to 2GB GPU buffer size limit when exporting a splat (2GB / 248B is just over 8.6M)

        [SerializeField] int m_FormatVersion;
        [SerializeField] int m_SplatCount;
        [SerializeField] Vector3 m_BoundsMin;
        [SerializeField] Vector3 m_BoundsMax;
        [SerializeField] Hash128 m_DataHash;

        public int formatVersion => m_FormatVersion;
        public int splatCount => m_SplatCount;
        public Vector3 boundsMin => m_BoundsMin;
        public Vector3 boundsMax => m_BoundsMax;
        public Hash128 dataHash => m_DataHash;

        // Match VECTOR_FMT_* in HLSL
        public enum VectorFormat
        {
            Float32, // 12 bytes: 32F.32F.32F
            Norm16, // 6 bytes: 16.16.16
            Norm11, // 4 bytes: 11.10.11
            Norm6   // 2 bytes: 6.5.5
        }

        public static int GetVectorSize(VectorFormat fmt)
        {
            return fmt switch
            {
                VectorFormat.Float32 => 12,
                VectorFormat.Norm16 => 6,
                VectorFormat.Norm11 => 4,
                VectorFormat.Norm6 => 2,
                _ => throw new ArgumentOutOfRangeException(nameof(fmt), fmt, null)
            };
        }

        public enum ColorFormat
        {
            Float32x4,
            Float16x4,
            Norm8x4,
            BC7,
        }
        public static int GetColorSize(ColorFormat fmt)
        {
            return fmt switch
            {
                ColorFormat.Float32x4 => 16,
                ColorFormat.Float16x4 => 8,
                ColorFormat.Norm8x4 => 4,
                ColorFormat.BC7 => 1,
                _ => throw new ArgumentOutOfRangeException(nameof(fmt), fmt, null)
            };
        }

        public void Initialize(int splats, VectorFormat formatPos, VectorFormat formatScale, ColorFormat formatColor, Vector3 bMin, Vector3 bMax)
        {
            m_SplatCount = splats;
            m_FormatVersion = kCurrentVersion;
            m_PosFormat = formatPos;
            m_ScaleFormat = formatScale;
            m_ColorFormat = formatColor;
            m_BoundsMin = bMin;
            m_BoundsMax = bMax;
        }

        public void SetDataHash(Hash128 hash)
        {
            m_DataHash = hash;
        }

        public void SetAssetFiles(TextAsset dataPos, TextAsset dataOther, TextAsset dataColor)
        {
            m_PosData = dataPos;
            m_OtherData = dataOther;
            m_ColorData = dataColor;
        }

        public static int GetOtherSizeNoSHIndex(VectorFormat scaleFormat)
        {
            return 4 + GetVectorSize(scaleFormat);
        }

        public static (int,int) CalcTextureSize(int splatCount)
        {
            int width = kTextureWidth;
            int height = math.max(1, (splatCount + width - 1) / width);
            // our swizzle tiles are 16x16, so make texture multiple of that height
            int blockHeight = 16;
            height = (height + blockHeight - 1) / blockHeight * blockHeight;
            return (width, height);
        }

        public static GraphicsFormat ColorFormatToGraphics(ColorFormat format)
        {
            return format switch
            {
                ColorFormat.Float32x4 => GraphicsFormat.R32G32B32A32_SFloat,
                ColorFormat.Float16x4 => GraphicsFormat.R16G16B16A16_SFloat,
                ColorFormat.Norm8x4 => GraphicsFormat.R8G8B8A8_UNorm,
                ColorFormat.BC7 => GraphicsFormat.RGBA_BC7_UNorm,
                _ => throw new ArgumentOutOfRangeException(nameof(format), format, null)
            };
        }

        public static long CalcPosDataSize(int splatCount, VectorFormat formatPos)
        {
            return splatCount * GetVectorSize(formatPos);
        }
        public static long CalcOtherDataSize(int splatCount, VectorFormat formatScale)
        {
            return splatCount * GetOtherSizeNoSHIndex(formatScale);
        }
        public static long CalcColorDataSize(int splatCount, ColorFormat formatColor)
        {
            var (width, height) = CalcTextureSize(splatCount);
            return width * height * GetColorSize(formatColor);
        }
        public static long CalcChunkDataSize(int splatCount)
        {
            int chunkCount = (splatCount + kChunkSize - 1) / kChunkSize;
            return chunkCount * UnsafeUtility.SizeOf<ChunkInfo>();
        }

        [SerializeField] VectorFormat m_PosFormat = VectorFormat.Norm11;
        [SerializeField] VectorFormat m_ScaleFormat = VectorFormat.Norm11;
        [SerializeField] ColorFormat m_ColorFormat;

        [SerializeField] TextAsset m_PosData;
        [SerializeField] TextAsset m_ColorData;
        [SerializeField] TextAsset m_OtherData;
        // Chunk data is optional (if data formats are fully lossless then there's no chunking)
        [SerializeField] TextAsset m_ChunkData;
        
        public VectorFormat posFormat => m_PosFormat;
        public VectorFormat scaleFormat => m_ScaleFormat;
        public ColorFormat colorFormat => m_ColorFormat;

        public TextAsset posData => m_PosData;
        public TextAsset colorData => m_ColorData;
        public TextAsset otherData => m_OtherData;
        public TextAsset chunkData => m_ChunkData;

        public struct ChunkInfo
        {
            public uint colR, colG, colB, colA;
            public float2 posX, posY, posZ;
            public uint sclX, sclY, sclZ;
            public uint shR, shG, shB;
        }

        [Serializable]
        public struct CameraInfo
        {
            public Vector3 pos;
            public Vector3 axisX, axisY, axisZ;
            public float fov;
        }
    }
}
