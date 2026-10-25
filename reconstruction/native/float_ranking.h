#pragma once
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

// Exact descending finite IEEE float32 ranking, ties by ascending original ID.
// Canonicalize +/-0 because numeric comparison treats them as equal.
static std::vector<int32_t> radix_ranking(const std::vector<float>& scores){
 static_assert(sizeof(float)==4&&std::numeric_limits<float>::is_iec559,"IEEE float32 required");
 if(scores.size()>size_t(std::numeric_limits<int32_t>::max()))throw std::runtime_error("Vocabulary too large");
 std::vector<uint32_t> keys(scores.size());std::vector<int32_t> ids(scores.size()),scratch(scores.size());
 std::iota(ids.begin(),ids.end(),0);
 for(size_t i=0;i<scores.size();++i){
  if(!std::isfinite(scores[i]))throw std::runtime_error("Nonfinite logits");
  uint32_t bits;std::memcpy(&bits,&scores[i],4);if((bits&0x7fffffffu)==0)bits=0;
  uint32_t ascending=(bits&0x80000000u)?~bits:(bits^0x80000000u);keys[i]=~ascending;
 }
 for(unsigned shift=0;shift<32;shift+=8){
  std::array<size_t,256> counts{},starts{};
  for(auto id:ids)++counts[(keys[id]>>shift)&255u];
  for(size_t i=1;i<256;++i)starts[i]=starts[i-1]+counts[i-1];
  for(auto id:ids)scratch[starts[(keys[id]>>shift)&255u]++]=id;
  ids.swap(scratch);
 }
 return ids;
}
