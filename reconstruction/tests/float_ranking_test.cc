#include "float_ranking.h"
#include <algorithm>
#include <iostream>
#include <random>

int main(){
 std::mt19937 rng(20261019);
 for(int trial=0;trial<100;++trial){
  size_t n=trial==99?262144:1+rng()%4096;std::vector<float> scores(n);
  for(auto& x:scores){uint32_t bits=rng();std::memcpy(&x,&bits,4);if(!std::isfinite(x))x=0;
   if(trial%3==0)x=float(int(rng()%31)-15);
  }
  if(n>8){scores[0]=0.;scores[1]=-0.;scores[2]=std::numeric_limits<float>::denorm_min();scores[3]=-scores[2];
   scores[4]=std::numeric_limits<float>::max();scores[5]=-scores[4];scores[6]=42.;scores[7]=42.;}
  std::vector<int32_t> expected(n);std::iota(expected.begin(),expected.end(),0);
  std::stable_sort(expected.begin(),expected.end(),[&](int32_t a,int32_t b){return scores[a]>scores[b];});
  if(radix_ranking(scores)!=expected)throw std::runtime_error("Ranking differs");
 }
 for(float x:{std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}){
  bool failed=false;try{radix_ranking({x});}catch(const std::runtime_error&){failed=true;}if(!failed)throw std::runtime_error("Accepted nonfinite score");
 }
 std::cout<<"PASS 100 ranking arrays including 262144 IDs\n";
}
