struct KernelConfig
{
    const int BlockRowWarps;
    const int BlockColWarps;
    const int WarpRowTiles;
    const int WarpColTiles;
    const int ChunkK;
    const int NumStages;
    const int PipelineStrategy;
    const int K;
    const int N;
};

constexpr KernelConfig octomul_4096_57344_8192 = {
    /* BlockRowWarps */ 2,
    /* BlockColWarps */ 2,
    /* WarpRowTiles */ 3,
    /* WarpColTiles */ 4,
    /* ChunkK */ 4,
    /* NumStages */ 3,
    /* PipelineStrategy */ 1,
    /* K */ 8192,
    /* N */ 57344};

constexpr KernelConfig octomul_4096_8192_8192 = {
    /* BlockRowWarps */ 3,
    /* BlockColWarps */ 2,
    /* WarpRowTiles */ 3,
    /* WarpColTiles */ 4,
    /* ChunkK */ 4,
    /* NumStages */ 2,
    /* PipelineStrategy */ 3,
    /* K */ 8192,
    /* N */ 8192};

constexpr KernelConfig octomul_4096_28672_4096 = {
    /* BlockRowWarps */ 2,
    /* BlockColWarps */ 2,
    /* WarpRowTiles */ 4,
    /* WarpColTiles */ 4,
    /* ChunkK */ 4,
    /* NumStages */ 3,
    /* PipelineStrategy */ 1,
    /* K */ 4096,
    /* N */ 28672};

constexpr KernelConfig octomul_4096_10240_8192 = {
    /* BlockRowWarps */ 2,
    /* BlockColWarps */ 3,
    /* WarpRowTiles */ 4,
    /* WarpColTiles */ 4,
    /* ChunkK */ 4,
    /* NumStages */ 2,
    /* PipelineStrategy */ 1,
    /* K */ 8192,
    /* N */ 10240};

constexpr KernelConfig octomul_4096_6144_4096 = {
    /* BlockRowWarps */ 2,
    /* BlockColWarps */ 3,
    /* WarpRowTiles */ 4,
    /* WarpColTiles */ 3,
    /* ChunkK */ 4,
    /* NumStages */ 2,
    /* PipelineStrategy */ 3,
    /* K */ 4096,
    /* N */ 6144};

constexpr KernelConfig octomul_4096_4096_4096 = {
    /* BlockRowWarps */ 2,
    /* BlockColWarps */ 2,
    /* WarpRowTiles */ 4,
    /* WarpColTiles */ 4,
    /* ChunkK */ 4,
    /* NumStages */ 3,
    /* PipelineStrategy */ 1,
    /* K */ 4096,
    /* N */ 4096};

constexpr KernelConfig octomul_2048_8192_28672 = {
    /* BlockRowWarps */ 2,
    /* BlockColWarps */ 3,
    /* WarpRowTiles */ 4,
    /* WarpColTiles */ 4,
    /* ChunkK */ 2,
    /* NumStages */ 2,
    /* PipelineStrategy */ 3,
    /* K */ 28672,
    /* N */ 8192};

constexpr KernelConfig octomul_2048_10240_8192 = {
    /* BlockRowWarps */ 2,
    /* BlockColWarps */ 3,
    /* WarpRowTiles */ 4,
    /* WarpColTiles */ 4,
    /* ChunkK */ 2,
    /* NumStages */ 2,
    /* PipelineStrategy */ 3,
    /* K */ 8192,
    /* N */ 10240};

constexpr KernelConfig octomul_2048_8192_8192 = {
    /* BlockRowWarps */ 2,
    /* BlockColWarps */ 2,
    /* WarpRowTiles */ 4,
    /* WarpColTiles */ 4,
    /* ChunkK */ 2,
    /* NumStages */ 2,
    /* PipelineStrategy */ 3,
    /* K */ 8192,
    /* N */ 8192};

constexpr KernelConfig octomul_2048_28672_4096 = {
    /* BlockRowWarps */ 2,
    /* BlockColWarps */ 3,
    /* WarpRowTiles */ 4,
    /* WarpColTiles */ 4,
    /* ChunkK */ 2,
    /* NumStages */ 3,
    /* PipelineStrategy */ 3,
    /* K */ 4096,
    /* N */ 28672};

constexpr KernelConfig octomul_2048_6144_4096 = {
    /* BlockRowWarps */ 2,
    /* BlockColWarps */ 2,
    /* WarpRowTiles */ 3,
    /* WarpColTiles */ 3,
    /* ChunkK */ 2,
    /* NumStages */ 3,
    /* PipelineStrategy */ 3,
    /* K */ 4096,
    /* N */ 6144};

constexpr KernelConfig octomul_2048_4096_4096 = {
    /* BlockRowWarps */ 3,
    /* BlockColWarps */ 2,
    /* WarpRowTiles */ 3,
    /* WarpColTiles */ 3,
    /* ChunkK */ 2,
    /* NumStages */ 3,
    /* PipelineStrategy */ 2,
    /* K */ 4096,
    /* N */ 4096};

constexpr KernelConfig octomul_1024_8192_28672 = {
    /* BlockRowWarps */ 4,
    /* BlockColWarps */ 3,
    /* WarpRowTiles */ 4,
    /* WarpColTiles */ 3,
    /* ChunkK */ 2,
    /* NumStages */ 3,
    /* PipelineStrategy */ 2,
    /* K */ 28672,
    /* N */ 8192};

constexpr KernelConfig octomul_1024_6144_4096 = {
    /* BlockRowWarps */ 2,
    /* BlockColWarps */ 2,
    /* WarpRowTiles */ 4,
    /* WarpColTiles */ 3,
    /* ChunkK */ 2,
    /* NumStages */ 3,
    /* PipelineStrategy */ 2,
    /* K */ 4096,
    /* N */ 6144};

constexpr KernelConfig octomul_4096_4096_14336 = {
    /* BlockRowWarps */ 2,
    /* BlockColWarps */ 3,
    /* WarpRowTiles */ 4,
    /* WarpColTiles */ 4,
    /* ChunkK */ 4,
    /* NumStages */ 2,
    /* PipelineStrategy */ 1,
    /* K */ 14336,
    /* N */ 4096};

constexpr KernelConfig octomul_2048_57344_8192 = {
    /* BlockRowWarps */ 2,
    /* BlockColWarps */ 3,
    /* WarpRowTiles */ 4,
    /* WarpColTiles */ 4,
    /* ChunkK */ 2,
    /* NumStages */ 2,
    /* PipelineStrategy */ 2,
    /* K */ 8192,
    /* N */ 57344};

constexpr KernelConfig octomul_1024_57344_8192 = {
    /* BlockRowWarps */ 2,
    /* BlockColWarps */ 4,
    /* WarpRowTiles */ 4,
    /* WarpColTiles */ 3,
    /* ChunkK */ 4,
    /* NumStages */ 2,
    /* PipelineStrategy */ 3,
    /* K */ 8192,
    /* N */ 57344};

constexpr KernelConfig octomul_512_6144_4096 = {
    /* BlockRowWarps */ 4,
    /* BlockColWarps */ 2,
    /* WarpRowTiles */ 3,
    /* WarpColTiles */ 4,
    /* ChunkK */ 4,
    /* NumStages */ 2,
    /* PipelineStrategy */ 3,
    /* K */ 4096,
    /* N */ 6144};

constexpr KernelConfig octomul_1024_4096_14336 = {
    /* BlockRowWarps */ 2,
    /* BlockColWarps */ 3,
    /* WarpRowTiles */ 4,
    /* WarpColTiles */ 3,
    /* ChunkK */ 4,
    /* NumStages */ 2,
    /* PipelineStrategy */ 3,
    /* K */ 14336,
    /* N */ 4096};

constexpr KernelConfig octomul_1024_28672_4096 = {
    /* BlockRowWarps */ 2,
    /* BlockColWarps */ 2,
    /* WarpRowTiles */ 4,
    /* WarpColTiles */ 4,
    /* ChunkK */ 2,
    /* NumStages */ 3,
    /* PipelineStrategy */ 3,
    /* K */ 4096,
    /* N */ 28672};

constexpr KernelConfig octomul_2048_4096_14336 = {
    /* BlockRowWarps */ 3,
    /* BlockColWarps */ 2,
    /* WarpRowTiles */ 3,
    /* WarpColTiles */ 4,
    /* ChunkK */ 2,
    /* NumStages */ 3,
    /* PipelineStrategy */ 3,
    /* K */ 14336,
    /* N */ 4096};

constexpr KernelConfig octomul_512_57344_8192 = {
    /* BlockRowWarps */ 3,
    /* BlockColWarps */ 2,
    /* WarpRowTiles */ 3,
    /* WarpColTiles */ 4,
    /* ChunkK */ 4,
    /* NumStages */ 2,
    /* PipelineStrategy */ 1,
    /* K */ 8192,
    /* N */ 57344};

constexpr KernelConfig octomul_512_8192_28672 = {
    /* BlockRowWarps */ 2,
    /* BlockColWarps */ 2,
    /* WarpRowTiles */ 4,
    /* WarpColTiles */ 4,
    /* ChunkK */ 4,
    /* NumStages */ 3,
    /* PipelineStrategy */ 2,
    /* K */ 28672,
    /* N */ 8192};

constexpr KernelConfig octomul_512_4096_4096 = {
    /* BlockRowWarps */ 2,
    /* BlockColWarps */ 3,
    /* WarpRowTiles */ 3,
    /* WarpColTiles */ 4,
    /* ChunkK */ 4,
    /* NumStages */ 2,
    /* PipelineStrategy */ 1,
    /* K */ 4096,
    /* N */ 4096};

constexpr KernelConfig octomul_512_28672_4096 = {
    /* BlockRowWarps */ 2,
    /* BlockColWarps */ 3,
    /* WarpRowTiles */ 4,
    /* WarpColTiles */ 3,
    /* ChunkK */ 4,
    /* NumStages */ 2,
    /* PipelineStrategy */ 1,
    /* K */ 4096,
    /* N */ 28672};

constexpr KernelConfig octomul_4096_8192_28672 = {
    /* BlockRowWarps */ 2,
    /* BlockColWarps */ 3,
    /* WarpRowTiles */ 4,
    /* WarpColTiles */ 4,
    /* ChunkK */ 4,
    /* NumStages */ 2,
    /* PipelineStrategy */ 2,
    /* K */ 28672,
    /* N */ 8192};

constexpr KernelConfig octomul_512_4096_14336 = {
    /* BlockRowWarps */ 2,
    /* BlockColWarps */ 2,
    /* WarpRowTiles */ 4,
    /* WarpColTiles */ 4,
    /* ChunkK */ 4,
    /* NumStages */ 3,
    /* PipelineStrategy */ 1,
    /* K */ 14336,
    /* N */ 4096};

constexpr KernelConfig octomul_512_8192_8192 = {
    /* BlockRowWarps */ 2,
    /* BlockColWarps */ 2,
    /* WarpRowTiles */ 4,
    /* WarpColTiles */ 4,
    /* ChunkK */ 4,
    /* NumStages */ 2,
    /* PipelineStrategy */ 1,
    /* K */ 8192,
    /* N */ 8192};

constexpr KernelConfig octomul_512_10240_8192 = {
    /* BlockRowWarps */ 2,
    /* BlockColWarps */ 3,
    /* WarpRowTiles */ 4,
    /* WarpColTiles */ 3,
    /* ChunkK */ 4,
    /* NumStages */ 2,
    /* PipelineStrategy */ 2,
    /* K */ 8192,
    /* N */ 10240};

constexpr KernelConfig octomul_1024_10240_8192 = {
    /* BlockRowWarps */ 2,
    /* BlockColWarps */ 2,
    /* WarpRowTiles */ 4,
    /* WarpColTiles */ 4,
    /* ChunkK */ 4,
    /* NumStages */ 2,
    /* PipelineStrategy */ 1,
    /* K */ 8192,
    /* N */ 10240};

constexpr KernelConfig octomul_1024_4096_4096 = {
    /* BlockRowWarps */ 2,
    /* BlockColWarps */ 2,
    /* WarpRowTiles */ 4,
    /* WarpColTiles */ 3,
    /* ChunkK */ 4,
    /* NumStages */ 2,
    /* PipelineStrategy */ 1,
    /* K */ 4096,
    /* N */ 4096};

constexpr KernelConfig octomul_1024_8192_8192 = {
    /* BlockRowWarps */ 2,
    /* BlockColWarps */ 2,
    /* WarpRowTiles */ 3,
    /* WarpColTiles */ 4,
    /* ChunkK */ 4,
    /* NumStages */ 2,
    /* PipelineStrategy */ 1,
    /* K */ 8192,
    /* N */ 8192};
