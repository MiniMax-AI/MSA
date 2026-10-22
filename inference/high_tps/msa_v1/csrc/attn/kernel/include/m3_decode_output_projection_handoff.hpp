#pragma once

namespace mega_atten::m3_decode::output_projection {

inline constexpr int kProblemM = 6144;
inline constexpr int kProblemN = 8;
inline constexpr int kProblemK = 2048;
inline constexpr int kTileK = 128;
inline constexpr int kActivationStageCount = kProblemK / kTileK;
inline constexpr int kHandoffStagesPerGroup = 4;
inline constexpr int kHandoffGroupCount = kActivationStageCount / kHandoffStagesPerGroup;
inline constexpr int kStageRecordWords32 = 4;
inline constexpr int kOutputProjectionTileM = 128;
inline constexpr int kOutputProjectionConsumerCount = kProblemM / kOutputProjectionTileM;
inline constexpr int kHandoffStateWords32 =
    kHandoffGroupCount * kStageRecordWords32 + kOutputProjectionConsumerCount;

static_assert(kActivationStageCount % kHandoffStagesPerGroup == 0);
static_assert(kProblemM % kOutputProjectionTileM == 0);

} // namespace mega_atten::m3_decode::output_projection
