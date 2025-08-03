// SPDX-License-Identifier: MIT

using System;
using System.Collections.Generic;
using System.IO;
using GaussianSplatting.Editor.Utils;
using GaussianSplatting.Runtime;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEditor;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.WSA;

namespace GaussianSplatting.Editor
{
    [BurstCompile]
    public class GaussianSplatAssetCreatorFolder : EditorWindow
    {
        const string kProgressTitle = "Creating Gaussian Splat Asset";
        const string kPrefOutputFolder = "nesnausk.GaussianSplatting.CreatorOutputFolder";
        
        readonly FilePickerControl m_FilePicker = new();
        List<string> m_allFiles = new ();


        [SerializeField] string m_InputFolder;
        [SerializeField] string m_OutputFolder = "Assets/GaussianAssets";
        [SerializeField] GaussianSplatAsset.VectorFormat m_FormatPos;
        [SerializeField] GaussianSplatAsset.VectorFormat m_FormatScale;
        [SerializeField] GaussianSplatAsset.ColorFormat m_FormatColor;

        string m_ErrorMessage;
        string m_PrevFolder;
        int m_PrevVertexCount;
        long m_PrevFileSize;
        int m_plyFileCount;

        [MenuItem("3DGS/Create from Folder")]
        public static void Init()
        {
            var window = GetWindowWithRect<GaussianSplatAssetCreatorFolder>(new Rect(50, 50, 360, 340), false, "3DIL Splat Creator", true);
            window.minSize = new Vector2(320, 320);
            window.maxSize = new Vector2(1500, 1500);
            window.Show();
        }

        void Awake()
        {
            m_OutputFolder = EditorPrefs.GetString(kPrefOutputFolder, "Assets/GaussianAssets");
        }

        void OnGUI()
        {
            EditorGUILayout.Space();
            GUILayout.Label("Input data", EditorStyles.boldLabel);
            var rect = EditorGUILayout.GetControlRect(true);
            m_InputFolder = m_FilePicker.PathFieldGUI(rect, new GUIContent("Input Folder"), m_InputFolder, null, "GaussianAssetInputFolder");

            if (m_InputFolder != m_PrevFolder && !string.IsNullOrWhiteSpace(m_InputFolder))
            {
                m_allFiles = new List<string>();
                foreach (string file in Directory.GetFiles(m_InputFolder))
                {
                    if (file.EndsWith(".ply"))
                        m_allFiles.Add(file);
                }
                m_PrevFolder = m_InputFolder;
            }

            EditorGUILayout.LabelField("Ply files found", $"{m_allFiles.Count}");

            EditorGUILayout.Space();
            GUILayout.Label("Output", EditorStyles.boldLabel);
            rect = EditorGUILayout.GetControlRect(true);
            string newOutputFolder = m_FilePicker.PathFieldGUI(rect, new GUIContent("Output Folder"), m_OutputFolder, null, "GaussianAssetOutputFolder");
            if (newOutputFolder != m_OutputFolder)
            {
                m_OutputFolder = newOutputFolder;
                EditorPrefs.SetString(kPrefOutputFolder, m_OutputFolder);
            }
            
            EditorGUILayout.Space();
            GUILayout.BeginHorizontal();
            GUILayout.Space(30);
            if (GUILayout.Button($"Create {m_allFiles.Count} Assets"))
            {
                foreach (var plyFile in m_allFiles)
                {
                    CreateAsset(plyFile);
                }
            }
            GUILayout.Space(30);
            GUILayout.EndHorizontal();

            if (!string.IsNullOrWhiteSpace(m_ErrorMessage))
            {
                EditorGUILayout.HelpBox(m_ErrorMessage, MessageType.Error);
            }
        }


        static T CreateOrReplaceAsset<T>(T asset, string path) where T : UnityEngine.Object
        {
            T result = AssetDatabase.LoadAssetAtPath<T>(path);
            if (result == null)
            {
                AssetDatabase.CreateAsset(asset, path);
                result = asset;
            }
            else
            {
                if (typeof(Mesh).IsAssignableFrom(typeof(T))) { (result as Mesh)?.Clear(); }
                EditorUtility.CopySerialized(asset, result);
            }
            return result;
        }

        unsafe void CreateAsset(String ply_file)
        {
            m_ErrorMessage = null;
            if (string.IsNullOrWhiteSpace(ply_file))
            {
                m_ErrorMessage = $"Select input PLY file";
                return;
            }

            if (string.IsNullOrWhiteSpace(m_OutputFolder) || !m_OutputFolder.StartsWith("Assets/"))
            {
                m_ErrorMessage = $"Output folder must be within project, was '{m_OutputFolder}'";
                return;
            }
            Directory.CreateDirectory(m_OutputFolder);

            EditorUtility.DisplayProgressBar(kProgressTitle, "Reading data files", 0.0f);
            using NativeArray<InputSplatData> inputSplats = LoadInputSplatFile(ply_file);
            if (inputSplats.Length == 0)
            {
                EditorUtility.ClearProgressBar();
                return;
            }

            float3 boundsMin, boundsMax;
            var boundsJob = new CalcBoundsJob
            {
                m_BoundsMin = &boundsMin,
                m_BoundsMax = &boundsMax,
                m_SplatData = inputSplats
            };
            boundsJob.Schedule().Complete();

            EditorUtility.DisplayProgressBar(kProgressTitle, "Morton reordering", 0.05f);
            ReorderMorton(inputSplats, boundsMin, boundsMax);

            string baseName = Path.GetFileNameWithoutExtension(FilePickerControl.PathToDisplayString(ply_file));

            EditorUtility.DisplayProgressBar(kProgressTitle, "Creating data objects", 0.7f);
            GaussianSplatAsset asset = ScriptableObject.CreateInstance<GaussianSplatAsset>();
            asset.Initialize(inputSplats.Length, m_FormatPos, m_FormatScale, m_FormatColor, boundsMin, boundsMax);
            asset.name = baseName;

            var dataHash = new Hash128((uint)asset.splatCount, (uint)asset.formatVersion, 0, 0);
            string pathPos = $"{m_OutputFolder}/{baseName}_pos.bytes";
            string pathOther = $"{m_OutputFolder}/{baseName}_oth.bytes";
            string pathCol = $"{m_OutputFolder}/{baseName}_col.bytes";
            
            CreatePositionsData(inputSplats, pathPos, ref dataHash);
            CreateOtherData(inputSplats, pathOther, ref dataHash);//, splatSHIndices);
            CreateColorData(inputSplats, pathCol, ref dataHash);
            asset.SetDataHash(dataHash);

            // files are created, import them so we can get to the imported objects, ugh
            EditorUtility.DisplayProgressBar(kProgressTitle, "Initial texture import", 0.85f);
            AssetDatabase.Refresh(ImportAssetOptions.ForceUncompressedImport);

            EditorUtility.DisplayProgressBar(kProgressTitle, "Setup data onto asset", 0.95f);
            asset.SetAssetFiles(
                AssetDatabase.LoadAssetAtPath<TextAsset>(pathPos),
                AssetDatabase.LoadAssetAtPath<TextAsset>(pathOther),
                AssetDatabase.LoadAssetAtPath<TextAsset>(pathCol)
                );

            var assetPath = $"{m_OutputFolder}/{baseName}.asset";
            var savedAsset = CreateOrReplaceAsset(asset, assetPath);

            EditorUtility.DisplayProgressBar(kProgressTitle, "Saving assets", 0.99f);
            AssetDatabase.SaveAssets();
            EditorUtility.ClearProgressBar();

            Selection.activeObject = savedAsset;
        }

        NativeArray<InputSplatData> LoadInputSplatFile(string filePath)
        {
            NativeArray<InputSplatData> data = default;
            if (!File.Exists(filePath))
            {
                m_ErrorMessage = $"Did not find {filePath} file";
                return data;
            }
            try
            {
                GaussianFileReader.ReadFile(filePath, out data);
            }
            catch (Exception ex)
            {
                m_ErrorMessage = ex.Message;
            }
            return data;
        }

        [BurstCompile]
        struct CalcBoundsJob : IJob
        {
            [NativeDisableUnsafePtrRestriction] public unsafe float3* m_BoundsMin;
            [NativeDisableUnsafePtrRestriction] public unsafe float3* m_BoundsMax;
            [ReadOnly] public NativeArray<InputSplatData> m_SplatData;

            public unsafe void Execute()
            {
                float3 boundsMin = float.PositiveInfinity;
                float3 boundsMax = float.NegativeInfinity;

                for (int i = 0; i < m_SplatData.Length; ++i)
                {
                    float3 pos = m_SplatData[i].pos;
                    boundsMin = math.min(boundsMin, pos);
                    boundsMax = math.max(boundsMax, pos);
                }
                *m_BoundsMin = boundsMin;
                *m_BoundsMax = boundsMax;
            }
        }

        [BurstCompile]
        struct ReorderMortonJob : IJobParallelFor
        {
            const float kScaler = (float) ((1 << 21) - 1);
            public float3 m_BoundsMin;
            public float3 m_InvBoundsSize;
            [ReadOnly] public NativeArray<InputSplatData> m_SplatData;
            public NativeArray<(ulong,int)> m_Order;

            public void Execute(int index)
            {
                float3 pos = ((float3)m_SplatData[index].pos - m_BoundsMin) * m_InvBoundsSize * kScaler;
                uint3 ipos = (uint3) pos;
                ulong code = GaussianUtils.MortonEncode3(ipos);
                m_Order[index] = (code, index);
            }
        }

        struct OrderComparer : IComparer<(ulong, int)> {
            public int Compare((ulong, int) a, (ulong, int) b)
            {
                if (a.Item1 < b.Item1) return -1;
                if (a.Item1 > b.Item1) return +1;
                return a.Item2 - b.Item2;
            }
        }

        static void ReorderMorton(NativeArray<InputSplatData> splatData, float3 boundsMin, float3 boundsMax)
        {
            ReorderMortonJob order = new ReorderMortonJob
            {
                m_SplatData = splatData,
                m_BoundsMin = boundsMin,
                m_InvBoundsSize = 1.0f / (boundsMax - boundsMin),
                m_Order = new NativeArray<(ulong, int)>(splatData.Length, Allocator.TempJob)
            };
            order.Schedule(splatData.Length, 4096).Complete();
            order.m_Order.Sort(new OrderComparer());

            NativeArray<InputSplatData> copy = new(order.m_SplatData, Allocator.TempJob);
            for (int i = 0; i < copy.Length; ++i)
                order.m_SplatData[i] = copy[order.m_Order[i].Item2];
            copy.Dispose();

            order.m_Order.Dispose();
        }
        
        [BurstCompile]
        struct ConvertColorJob : IJobParallelFor
        {
            public int width, height;
            [ReadOnly] public NativeArray<float4> inputData;
            [NativeDisableParallelForRestriction] public NativeArray<byte> outputData;
            public GaussianSplatAsset.ColorFormat format;
            public int formatBytesPerPixel;

            public unsafe void Execute(int y)
            {
                int srcIdx = y * width;
                byte* dstPtr = (byte*) outputData.GetUnsafePtr() + y * width * formatBytesPerPixel;
                for (int x = 0; x < width; ++x)
                {
                    float4 pix = inputData[srcIdx];

                    switch (format)
                    {
                        case GaussianSplatAsset.ColorFormat.Float32x4:
                        {
                            *(float4*) dstPtr = pix;
                        }
                            break;
                        case GaussianSplatAsset.ColorFormat.Float16x4:
                        {
                            half4 enc = new half4(pix);
                            *(half4*) dstPtr = enc;
                        }
                            break;
                        case GaussianSplatAsset.ColorFormat.Norm8x4:
                        {
                            pix = math.saturate(pix);
                            uint enc = (uint)(pix.x * 255.5f) | ((uint)(pix.y * 255.5f) << 8) | ((uint)(pix.z * 255.5f) << 16) | ((uint)(pix.w * 255.5f) << 24);
                            *(uint*) dstPtr = enc;
                        }
                            break;
                    }

                    srcIdx++;
                    dstPtr += formatBytesPerPixel;
                }
            }
        }

        static ulong EncodeFloat3ToNorm16(float3 v) // 48 bits: 16.16.16
        {
            return (ulong) (v.x * 65535.5f) | ((ulong) (v.y * 65535.5f) << 16) | ((ulong) (v.z * 65535.5f) << 32);
        }
        static uint EncodeFloat3ToNorm11(float3 v) // 32 bits: 11.10.11
        {
            return (uint) (v.x * 2047.5f) | ((uint) (v.y * 1023.5f) << 11) | ((uint) (v.z * 2047.5f) << 21);
        }
        static ushort EncodeFloat3ToNorm655(float3 v) // 16 bits: 6.5.5
        {
            return (ushort) ((uint) (v.x * 63.5f) | ((uint) (v.y * 31.5f) << 6) | ((uint) (v.z * 31.5f) << 11));
        }

        static uint EncodeQuatToNorm10(float4 v) // 32 bits: 10.10.10.2
        {
            return (uint) (v.x * 1023.5f) | ((uint) (v.y * 1023.5f) << 10) | ((uint) (v.z * 1023.5f) << 20) | ((uint) (v.w * 3.5f) << 30);
        }

        static unsafe void EmitEncodedVector(float3 v, byte* outputPtr, GaussianSplatAsset.VectorFormat format)
        {
            switch (format)
            {
                case GaussianSplatAsset.VectorFormat.Float32:
                {
                    *(float*) outputPtr = v.x;
                    *(float*) (outputPtr + 4) = v.y;
                    *(float*) (outputPtr + 8) = v.z;
                }
                    break;
                case GaussianSplatAsset.VectorFormat.Norm16:
                {
                    ulong enc = EncodeFloat3ToNorm16(math.saturate(v));
                    *(uint*) outputPtr = (uint) enc;
                    *(ushort*) (outputPtr + 4) = (ushort) (enc >> 32);
                }
                    break;
                case GaussianSplatAsset.VectorFormat.Norm11:
                {
                    uint enc = EncodeFloat3ToNorm11(math.saturate(v));
                    *(uint*) outputPtr = enc;
                }
                    break;
                case GaussianSplatAsset.VectorFormat.Norm6:
                {
                    ushort enc = EncodeFloat3ToNorm655(math.saturate(v));
                    *(ushort*) outputPtr = enc;
                }
                    break;
            }
        }

        [BurstCompile]
        struct CreatePositionsDataJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<InputSplatData> m_Input;
            public GaussianSplatAsset.VectorFormat m_Format;
            public int m_FormatSize;
            [NativeDisableParallelForRestriction] public NativeArray<byte> m_Output;

            public unsafe void Execute(int index)
            {
                byte* outputPtr = (byte*) m_Output.GetUnsafePtr() + index * m_FormatSize;
                EmitEncodedVector(m_Input[index].pos, outputPtr, m_Format);
            }
        }

        [BurstCompile]
        struct CreateOtherDataJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<InputSplatData> m_Input;
            public GaussianSplatAsset.VectorFormat m_ScaleFormat;
            public int m_FormatSize;
            [NativeDisableParallelForRestriction] public NativeArray<byte> m_Output;

            public unsafe void Execute(int index)
            {
                byte* outputPtr = (byte*) m_Output.GetUnsafePtr() + index * m_FormatSize;

                // rotation: 4 bytes
                {
                    Quaternion rotQ = m_Input[index].rot;
                    float4 rot = new float4(rotQ.x, rotQ.y, rotQ.z, rotQ.w);
                    uint enc = EncodeQuatToNorm10(rot);
                    *(uint*) outputPtr = enc;
                    outputPtr += 4;
                }

                // scale: 6, 4 or 2 bytes
                EmitEncodedVector(m_Input[index].scale, outputPtr, m_ScaleFormat);
            }
        }

        static int NextMultipleOf(int size, int multipleOf)
        {
            return (size + multipleOf - 1) / multipleOf * multipleOf;
        }

        void CreatePositionsData(NativeArray<InputSplatData> inputSplats, string filePath, ref Hash128 dataHash)
        {
            int dataLen = inputSplats.Length * GaussianSplatAsset.GetVectorSize(m_FormatPos);
            dataLen = NextMultipleOf(dataLen, 8); // serialized as ulong
            NativeArray<byte> data = new(dataLen, Allocator.TempJob);

            CreatePositionsDataJob job = new CreatePositionsDataJob
            {
                m_Input = inputSplats,
                m_Format = m_FormatPos,
                m_FormatSize = GaussianSplatAsset.GetVectorSize(m_FormatPos),
                m_Output = data
            };
            job.Schedule(inputSplats.Length, 8192).Complete();

            dataHash.Append(data);

            using var fs = new FileStream(filePath, FileMode.Create, FileAccess.Write);
            fs.Write(data);

            data.Dispose();
        }

        void CreateOtherData(NativeArray<InputSplatData> inputSplats, string filePath, ref Hash128 dataHash)//, NativeArray<int> splatSHIndices)
        {
            int formatSize = GaussianSplatAsset.GetOtherSizeNoSHIndex(m_FormatScale);
            int dataLen = inputSplats.Length * formatSize;

            dataLen = NextMultipleOf(dataLen, 8); // serialized as ulong
            NativeArray<byte> data = new(dataLen, Allocator.TempJob);

            CreateOtherDataJob job = new CreateOtherDataJob
            {
                m_Input = inputSplats,
                m_ScaleFormat = m_FormatScale,
                m_FormatSize = formatSize,
                m_Output = data
            };
            job.Schedule(inputSplats.Length, 8192).Complete();

            dataHash.Append(data);

            using var fs = new FileStream(filePath, FileMode.Create, FileAccess.Write);
            fs.Write(data);

            data.Dispose();
        }

        static int SplatIndexToTextureIndex(uint idx)
        {
            uint2 xy = GaussianUtils.DecodeMorton2D_16x16(idx);
            uint width = GaussianSplatAsset.kTextureWidth / 16;
            idx >>= 8;
            uint x = (idx % width) * 16 + xy.x;
            uint y = (idx / width) * 16 + xy.y;
            return (int)(y * GaussianSplatAsset.kTextureWidth + x);
        }

        [BurstCompile]
        struct CreateColorDataJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<InputSplatData> m_Input;
            [NativeDisableParallelForRestriction] public NativeArray<float4> m_Output;

            public void Execute(int index)
            {
                var splat = m_Input[index];
                int i = SplatIndexToTextureIndex((uint)index);
                m_Output[i] = new float4(splat.dc0.x, splat.dc0.y, splat.dc0.z, splat.opacity);
            }
        }

        void CreateColorData(NativeArray<InputSplatData> inputSplats, string filePath, ref Hash128 dataHash)
        {
            var (width, height) = GaussianSplatAsset.CalcTextureSize(inputSplats.Length);
            NativeArray<float4> data = new(width * height, Allocator.TempJob);

            CreateColorDataJob job = new CreateColorDataJob();
            job.m_Input = inputSplats;
            job.m_Output = data;
            job.Schedule(inputSplats.Length, 8192).Complete();

            dataHash.Append(data);
            dataHash.Append((int)m_FormatColor);

            GraphicsFormat gfxFormat = GaussianSplatAsset.ColorFormatToGraphics(m_FormatColor);
            int dstSize = (int)GraphicsFormatUtility.ComputeMipmapSize(width, height, gfxFormat);

            if (GraphicsFormatUtility.IsCompressedFormat(gfxFormat))
            {
                Texture2D tex = new Texture2D(width, height, GraphicsFormat.R32G32B32A32_SFloat, TextureCreationFlags.DontInitializePixels | TextureCreationFlags.DontUploadUponCreate);
                tex.SetPixelData(data, 0);
                EditorUtility.CompressTexture(tex, GraphicsFormatUtility.GetTextureFormat(gfxFormat), 100);
                NativeArray<byte> cmpData = tex.GetPixelData<byte>(0);
                using var fs = new FileStream(filePath, FileMode.Create, FileAccess.Write);
                fs.Write(cmpData);

                DestroyImmediate(tex);
            }
            else
            {
                ConvertColorJob jobConvert = new ConvertColorJob
                {
                    width = width,
                    height = height,
                    inputData = data,
                    format = m_FormatColor,
                    outputData = new NativeArray<byte>(dstSize, Allocator.TempJob),
                    formatBytesPerPixel = dstSize / width / height
                };
                jobConvert.Schedule(height, 1).Complete();
                using var fs = new FileStream(filePath, FileMode.Create, FileAccess.Write);
                fs.Write(jobConvert.outputData);
                jobConvert.outputData.Dispose();
            }

            data.Dispose();
        }
    }
}
