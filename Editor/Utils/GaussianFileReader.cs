using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using GaussianSplatting.Runtime;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Assertions;

namespace GaussianSplatting.Editor.Utils
{
    // input file splat data is read into this format
    public struct InputSplatData
    {
        public Vector3 pos;
        public Vector3 nor;
        public Vector3 dc0;
        public float opacity;
        public Vector3 scale;
        public Quaternion rot;
    }

    [BurstCompile]
    public class GaussianFileReader
    {
        // Returns splat count
        public static int ReadFileHeader(string filePath)
        {
            int vertexCount = 0;
            if (File.Exists(filePath))
            {                if (isPLY(filePath))

                    PLYFileReader.ReadFileHeader(filePath, out vertexCount, out _, out _);
            }
            return vertexCount;
        }

        public static unsafe void ReadFile(string filePath, out NativeArray<InputSplatData> splats)
        {
            if (isPLY(filePath))
            {
                NativeArray<byte> plyRawData;
                List<(string, PLYFileReader.ElementType)> attributes;
                PLYFileReader.ReadFile(filePath, out var splatCount, out var vertexStride, out attributes, out plyRawData);
                string attrError = CheckPLYAttributes(attributes);
                if (!string.IsNullOrEmpty(attrError))
                    throw new IOException($"PLY file is probably not a Gaussian Splat file? Missing properties: {attrError}");
                splats = PLYDataToSplats(plyRawData, splatCount, vertexStride, attributes);
                LinearizeData(splats);
                return;
            }
            throw new IOException($"File {filePath} is not a supported format");
        }

        static bool isPLY(string filePath) => filePath.EndsWith(".ply", true, CultureInfo.InvariantCulture);

        static string CheckPLYAttributes(List<(string, PLYFileReader.ElementType)> attributes)
        {
            string[] required =
            {
                "x", "y", "z", 
                "f_dc_0", "f_dc_1", "f_dc_2",
                "opacity", "scale_0", 
                "scale_1", "scale_2",
                "rot_0", "rot_1", "rot_2", "rot_3"
            };
            List<string> missing = required.Where(req => !attributes.Contains((req, PLYFileReader.ElementType.Float))).ToList();
            if (missing.Count == 0)
                return null;
            return string.Join(",", missing);
        }

        static unsafe NativeArray<InputSplatData> PLYDataToSplats(NativeArray<byte> input, int count, int stride, List<(string, PLYFileReader.ElementType)> attributes)
        {
            NativeArray<int> fileAttrOffsets = new NativeArray<int>(attributes.Count, Allocator.Temp);
            int offset = 0;
            for (var ai = 0; ai < attributes.Count; ai++)
            {
                var attr = attributes[ai];
                fileAttrOffsets[ai] = offset;
                offset += PLYFileReader.TypeToSize(attr.Item2);
            }

            string[] splatAttributes =
            {
                "x",
                "y",
                "z",
                "nx",
                "ny",
                "nz",
                "f_dc_0",
                "f_dc_1",
                "f_dc_2",
                "opacity",
                "scale_0",
                "scale_1",
                "scale_2",
                "rot_0",
                "rot_1",
                "rot_2",
                "rot_3",                
            };
            Assert.AreEqual(UnsafeUtility.SizeOf<InputSplatData>() / 4, splatAttributes.Length);
            NativeArray<int> srcOffsets = new NativeArray<int>(splatAttributes.Length, Allocator.Temp);
            for (int ai = 0; ai < splatAttributes.Length; ai++)
            {
                int attrIndex = attributes.IndexOf((splatAttributes[ai], PLYFileReader.ElementType.Float));
                int attrOffset = attrIndex >= 0 ? fileAttrOffsets[attrIndex] : -1;
                srcOffsets[ai] = attrOffset;
            }
            
            NativeArray<InputSplatData> dst = new NativeArray<InputSplatData>(count, Allocator.Persistent);
            ReorderPLYData(count, (byte*)input.GetUnsafeReadOnlyPtr(), stride, (byte*)dst.GetUnsafePtr(), UnsafeUtility.SizeOf<InputSplatData>(), (int*)srcOffsets.GetUnsafeReadOnlyPtr());
            return dst;
        }

        [BurstCompile]
        static unsafe void ReorderPLYData(int splatCount, byte* src, int srcStride, byte* dst, int dstStride, int* srcOffsets)
        {
            for (int i = 0; i < splatCount; i++)
            {
                for (int attr = 0; attr < dstStride / 4; attr++)
                {
                    if (srcOffsets[attr] >= 0)
                        *(int*)(dst + attr * 4) = *(int*)(src + srcOffsets[attr]);
                }
                src += srcStride;
                dst += dstStride;
            }
        }

        [BurstCompile]
        struct LinearizeDataJob : IJobParallelFor
        {
            public NativeArray<InputSplatData> splatData;
            public void Execute(int index)
            {
                var splat = splatData[index];

                // rot
                var q = splat.rot;
                var qq = GaussianUtils.NormalizeSwizzleRotation(new float4(q.x, q.y, q.z, q.w));
                qq = GaussianUtils.PackSmallest3Rotation(qq);
                splat.rot = new Quaternion(qq.x, qq.y, qq.z, qq.w);

                // scale
                splat.scale = GaussianUtils.LinearScale(splat.scale);

                // color
                splat.dc0 = GaussianUtils.SH0ToColor(splat.dc0);
                splat.opacity = GaussianUtils.Sigmoid(splat.opacity);

                splatData[index] = splat;
            }
        }

        static void LinearizeData(NativeArray<InputSplatData> splatData)
        {
            LinearizeDataJob job = new LinearizeDataJob();
            job.splatData = splatData;
            job.Schedule(splatData.Length, 4096).Complete();
        }
    }
}